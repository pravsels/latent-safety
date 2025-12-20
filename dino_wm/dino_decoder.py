import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append('..')
from einops import rearrange
from torchvision import transforms
from dino_wm.config import MODEL_CONFIG, DECODER_CONFIG

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        # Flatten the input to (N, E) where N is the number of latents and E is the dimension of the latent.
        flatten = input.reshape(-1, self.dim)

        # Compute squared L2 distances from each latent x (N,E) to each codebook vector e_j (E,).
        # Note: codebook is stored transposed as embed=(E,K), so flatten@embed gives (N,K) dot products.
        # ||x - e||^2 = ||x||^2 - 2 x*e + ||e||^2
        # Shapes: flatten=(N,E), embed=(E,K) so dist=(N,K)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        # dist: (N, K) squared distances from each latent to each codebook entry.
        # Pick nearest code for each latent (argmin over K).
        _, embed_ind = dist.min(1)  # embed_ind: (N,) values in [0, K-1]

        # One-hot assignments, used for EMA codebook updates.
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)  # (N, K)

        # Reshape indices back to the spatial layout (all dims except the embedding dim).
        # If input is (B*T, H, W, E), this becomes (B*T, H, W).
        embed_ind = embed_ind.view(*input.shape[:-1])

        # Look up the actual embedding vectors for each index -> quantized latents with same shape as input.
        quantize = self.embed_code(embed_ind)  # (B*T, H, W, E)

        if self.training:
            # embed_onehot is N embeds and K codes
            # if we collapse this along the row dimension, we get the number of times each code is used.
            embed_onehot_sum = embed_onehot.sum(0)
            # Sum latents per code: (E, N) @ (N, K) -> (E, K)
            # Column j is sum of all latent vectors assigned to code j.
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # Distributed training support (if available)
            try:
                import distributed_fn as dist_fn
                dist_fn.all_reduce(embed_onehot_sum)
                dist_fn.all_reduce(embed_sum)
            except (ImportError, NameError):
                # Single-device training: no-op
                pass

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        # embed is stored as (E, K) for fast matmuls; F.embedding expects weight as (K, E),
        # so we transpose to look up code vectors by integer indices.
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=None,
        n_res_block=4,
        n_res_channel=128,
        emb_dim=None,
        n_embed=None,
        decay=0.99,
        quantize=False,
    ):
        # Use MODEL_CONFIG['dim'] as default for emb_dim (DINO feature dimension)
        if emb_dim is None:
            emb_dim = MODEL_CONFIG['dim']
        # Use MODEL_CONFIG['codebook_size'] as default for n_embed (VQ codebook size)
        if n_embed is None:
            n_embed = DECODER_CONFIG['codebook_size']
        # channel defaults to emb_dim (they're typically the same)
        if channel is None:
            channel = emb_dim
        super().__init__()

        self.quantize = quantize
        self.quantize_b = Quantize(emb_dim, n_embed)

        if not quantize:
            for param in self.quantize_b.parameters():
                param.requires_grad = False

        self.upsample_b = Decoder(emb_dim, emb_dim, channel, n_res_block, n_res_channel, stride=4)
        self.dec = Decoder(
            emb_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.info = f"in_channel: {in_channel}, channel: {channel}, n_res_block: {n_res_block}, n_res_channel: {n_res_channel}, emb_dim: {emb_dim}, n_embed: {n_embed}, decay: {decay}"

    def forward(self, input):
        '''
            input: (b, t, num_patches, emb_dim)
        '''
        num_patches = input.shape[2]
        num_side_patches = int(num_patches ** 0.5)    
        input = rearrange(input, "b t (h w) e -> (b t) h w e", h=num_side_patches, w=num_side_patches)

        if self.quantize:
            quant_b, diff_b, id_b = self.quantize_b(input)
        else:
            quant_b, diff_b = input, torch.zeros(1).to(input.device)

        quant_b = quant_b.permute(0, 3, 1, 2)   # (b, t, num_patches, emb_dim) -> (b, emb_dim, t, num_patches)
        diff_b = diff_b.unsqueeze(0)
        dec = self.decode(quant_b)
        # Decoder already produces its native resolution; no extra resize here.
        return dec, diff_b  # diff is 0 if no quantization

    def decode(self, quant_b):
        upsample_b = self.upsample_b(quant_b) 
        dec = self.dec(upsample_b) # quant: (128, 64, 64)
        return dec

    def decode_code(self, code_b): # not used (only used in sample.py in original repo)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        dec = self.decode(quant_b)
        return dec