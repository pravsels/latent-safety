# Based on DINO-WM https://arxiv.org/abs/2411.04983



import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Tuple, Optional
from torchvision import transforms
from scipy.spatial.transform import Rotation
from dino_wm.config import MODEL_CONFIG, get_dino_config


def batch_quat_to_rotvec(quaternions):
    """
    Convert a batch of quaternions to axis-angle using PyTorch and scipy.

    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4), where each quaternion is (w, x, y, z).

    Returns:
        axes (torch.Tensor): A tensor of shape (N, 3), representing the rotation axes.
        angles (torch.Tensor): A tensor of shape (N,), representing the rotation angles in radians.
    """
    # Convert PyTorch tensor to NumPy array
    quaternions_np = quaternions.cpu().numpy()

    # Use scipy for the quaternion-to-axis-angle conversion
    r = Rotation.from_quat(quaternions_np)
    rotvecs = r.as_rotvec()
    return rotvecs

def batch_rotvec_to_quat(rotvecs):
    """
    Convert a batch of quaternions to axis-angle using PyTorch and scipy.

    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4), where each quaternion is (w, x, y, z).

    Returns:
        axes (torch.Tensor): A tensor of shape (N, 3), representing the rotation axes.
        angles (torch.Tensor): A tensor of shape (N,), representing the rotation angles in radians.
    """
    # Convert PyTorch tensor to NumPy array
    rotvecs_np = rotvecs.cpu().numpy()

    # Use scipy for the quaternion-to-axis-angle conversion
    r = Rotation.from_rotvec(rotvecs_np)
    quaternions = r.as_quat()
    return quaternions

def normalize_acs(acs, min_ac, max_ac):
    if min_ac.device != acs.device:
        min_ac = min_ac.to(acs.device)
    if max_ac.device != acs.device:
        max_ac = max_ac.to(acs.device)
        
    norm_acs = (acs - min_ac) / (max_ac - min_ac)
    
    return norm_acs

def unnormalize_acs(acs, min_ac, max_ac):
    if min_ac.device != acs.device:
        min_ac = min_ac.to(acs.device)
    if max_ac.device != acs.device:
        max_ac = max_ac.to(acs.device)
    
    acs = (acs * (max_ac - min_ac)) + min_ac
    
    return acs

def normalize_states(states, min_state, max_state):
    if min_state.device != states.device:
        min_state = min_state.to(states.device)
    if max_state.device != states.device:
        max_state = max_state.to(states.device)
        
    norm_states = (states - min_state) / (max_state - min_state)
    
    return norm_states

def unnormalize_states(states, min_state, max_state):
    if min_state.device != states.device:
        min_state = min_state.to(states.device)
    if max_state.device != states.device:
        max_state = max_state.to(states.device)
    
    states = (states * (max_state - min_state)) + min_state
    
    return states

class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Decoder(nn.Module):
    def __init__(self, in_channels=None, out_channels=3):
        super(Decoder, self).__init__()

        # Use MODEL_CONFIG['dim'] as default if not specified
        if in_channels is None:
            in_channels = MODEL_CONFIG['dim']

        self.in_channels = in_channels
        dino_cfg = get_dino_config()
        self.grid_size = int(dino_cfg['num_patches'] ** 0.5)  # 16 for v2, 14 for v3
        
        # Two residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels),            
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels)

        )
        
        # Transposed convolutions to go from grid (16x16 or 14x14) to image size (224x224)
        self.transposed_convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # New intermediate layer
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # New additional layer with no change in resolution
            nn.ConvTranspose2d(in_channels // 8, in_channels // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 8, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.resize_transform = transforms.Resize(MODEL_CONFIG['image_size'])

        
    
    def forward(self, x):

        x = x.view(-1, self.grid_size, self.grid_size, self.in_channels)  # Reshape to spatial grid
        x = x.permute(0, 3, 1, 2)        # Pass through residual blocks
        x = self.residual_blocks(x)
        # Pass through transposed convolutions
        x = self.transposed_convs(x)
        x = self.resize_transform(x)
        
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0., 
                 num_frames: int = 2, patches_per_frame: int = 256):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register buffer instead of creating mask every forward pass
        mask = self._create_causal_mask(num_frames, patches_per_frame)
        self.register_buffer("mask", mask)
        
    def _create_causal_mask(self, num_frames: int, patches_per_frame: int) -> torch.Tensor:
        """Create causal attention mask for autoregressive prediction.
        
        Each frame can attend to itself and all previous frames, but not future frames.
        This enables: position t (seeing frames 0..t) predicts frame t+1.
        """
        total_patches = num_frames * patches_per_frame
        mask = torch.zeros(total_patches, total_patches)
        
        for i in range(num_frames):
            start_idx = i * patches_per_frame
            end_idx = (i + 1) * patches_per_frame
            
            # Allow attention within current frame
            mask[start_idx:end_idx, start_idx:end_idx] = 1
            
            # Allow attention to previous frames
            if i > 0:
                mask[start_idx:end_idx, :start_idx] = 1
                
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Use registered mask buffer
        mask = self.mask[:seq_len, :seq_len]
        dots = dots.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0., 
                 num_frames: int = 2):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout, num_frames)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class VideoTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: Tuple[int, int],
        dim: int,
        action_embed_dim: int,
        state_embed_dim: int,
        state_dim: int,
        action_dim: int, # Physical action dimension
        depth: int,
        heads: int,
        mlp_dim: int,
        num_frames: int = 2,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.device = device
        dino_cfg = get_dino_config()
        if 'hub_source' in dino_cfg:
            self.dino = torch.hub.load(dino_cfg['hub_repo'], dino_cfg['model_name'],
                                       source=dino_cfg['hub_source'], weights=dino_cfg['weights_path']).to(device)
        else:
            self.dino = torch.hub.load(dino_cfg['hub_repo'], dino_cfg['model_name']).to(device)
        self.num_patches = dino_cfg['num_patches']
        
        # Improved action embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128), # Uses physical action dimension
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, action_embed_dim),
            nn.LayerNorm(action_embed_dim)
        ).to(device)
        
        # State encoder 
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128), # Uses physical state dimension
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, state_embed_dim),
            nn.LayerNorm(state_embed_dim)
        ).to(device)
        
        total_dim = 2*dim + action_embed_dim + state_embed_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, total_dim) * 0.02)  # Spatial: patch position within frame
        self.temp_embedding = nn.Parameter(torch.randn(1, num_frames, total_dim) * 0.02)  # Temporal: frame position in sequence
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Use TransformerBlock instead of separate components
        self.transformer = nn.ModuleList([
            TransformerBlock(total_dim, heads, dim_head, mlp_dim, dropout, num_frames)
            for _ in range(depth)
        ])
        
        # Separate prediction heads
        self.wrist_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, dim)
        )
                
        
        self.front_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, dim)
        )
        
        self.state_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, state_dim)
        )

        self.failure_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, 1)
        )

        

    
    def forward(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = self.forward_features(video1, video2, states, actions)

        # Generate predictions
        pred1 = self.front_head(x)
        pred2 = self.wrist_head(x)
        state_preds = self.state_pred(x)
        failure_preds = self.failure_pred(x)
        
        return pred1, pred2, state_preds, failure_preds

    def forward_features(self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode actions and states
        action_embeddings = self.action_encoder(actions).unsqueeze(2).expand(-1, -1, self.num_patches, -1)
        state_embeddings = self.state_encoder(states).unsqueeze(2).expand(-1, -1, self.num_patches, -1)
        
        # Combine features
        batch_size, num_frames, _, _ = video1.shape
    
        x = torch.cat((video1, video2, action_embeddings, state_embeddings), dim=3)
        # Add positional embeddings: spatial (per-patch) + temporal (per-frame)
        x = x + self.pos_embedding  # Same patch positions applied to all frames
        x = x + self.temp_embedding[:, :num_frames].unsqueeze(2)  # Same frame position applied to all patches

        # Reshape for transformer
        x = rearrange(x, 'b s n d -> b (s n) d')
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)
            
        # Reshape back
        x = rearrange(x, 'b (s n) d -> b s n d', s=num_frames)
        return x

    def failure_pred(self, features):
        # features: (batch, num_frames, num_patches, dim)
        failure_preds = self.failure_head(features)  # (batch, num_frames, num_patches, 1)
        failure_preds = torch.mean(failure_preds, dim=2)  # Average over patches -> (batch, num_frames, 1)
        return failure_preds
    
    def state_pred(self, features):
        # features: (batch, num_frames, num_patches, dim)
        state_preds = self.state_head(features)  # (batch, num_frames, num_patches, state_dim)
        state_preds = torch.mean(state_preds, dim=2)  # Average over patches -> (batch, num_frames, state_dim)
        return state_preds


    @torch.no_grad()
    def get_dino_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract DINO features from video frames."""
        b, f, c, h, w = video.shape
        video = video.view(b * f, c, h, w)
        features = self.dino.forward_features(video)['x_norm_patchtokens']
        return features.view(b, f, -1, features.shape[-1])
    

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def horizontal_forward(network, x, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    x = x.reshape(-1, *input_shape)
    x = network(x)
    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x

def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist
    

class TransposedConvDecoder(nn.Module):
    def __init__(self, observation_shape=None, emb_dim=512, activation=nn.ReLU, depth=64, kernel_size=5, stride=3):
        # Use MODEL_CONFIG['image_size'] as default if not specified
        if observation_shape is None:
            observation_shape = (3, MODEL_CONFIG['image_size'][0], MODEL_CONFIG['image_size'][1])
        super().__init__()

        activation = activation()
        self.observation_shape = observation_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.emb_dim = emb_dim

        self.network = nn.Sequential(
            nn.Linear(
                emb_dim, self.depth * 32
            ),
            nn.Unflatten(1, (self.depth * 32, 1)),
            nn.Unflatten(2, (1,1)),
            nn.ConvTranspose2d(
                self.depth * 32,
                self.depth * 8,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 8,
                self.depth * 4,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 4,
                self.depth * 2,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 2,
                self.depth * 1,
                self.kernel_size,
                self.stride,
                padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 1,
                self.observation_shape[0],
                self.kernel_size,
                self.stride,
                padding=1
            ),
            nn.Upsample(size=(observation_shape[1], observation_shape[2]), mode='bilinear', align_corners=False)
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior):
        x = horizontal_forward(
            self.network, posterior, input_shape=[self.emb_dim],output_shape=self.observation_shape
        )
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        img = dist.mean.squeeze(2)
        img = einops.rearrange(img, "b t c h w -> (b t) c h w")
        return img, torch.zeros(1).to(posterior.device) # dummy placeholder