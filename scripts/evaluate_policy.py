#!/usr/bin/env python3
"""
Evaluation script for trained DDPG policy on Dubins car with world model
"""

import argparse
import os
import sys
import collections
import numpy as np
import torch
import gymnasium
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Add paths (same as training)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/PyHJ'))
sys.path.append(saferl_dir)

import models
import tools
import ruamel.yaml as yaml
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
from dreamer import make_dataset
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs


def load_config():
    """Load configuration from yaml file"""
    yml = yaml.YAML(typ="safe", pure=True)
    config_path = Path(__file__).parent / "../configs.yaml"
    configs = yml.load(config_path.read_text())
    
    # Create a simple namespace object from the config
    class Config:
        pass
    
    config = Config()
    for key, value in configs['defaults'].items():
        # Convert hyphenated keys to underscored for Python attribute access
        attr_name = key.replace('-', '_')
        setattr(config, attr_name, value)
    
    return config


def setup_environment(config):
    """Setup the Dubins car environment with world model"""
    # Create environment
    env = gymnasium.make(config.task, params=[config])
    config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    
    # Load world model
    wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)
    ckpt_path = config.rssm_ckpt_path
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    wm.eval()
    
    # Load offline dataset
    offline_eps = collections.OrderedDict()
    config.batch_size = 1
    config.batch_length = 2
    tools.fill_expert_dataset_dubins(config, offline_eps)
    offline_dataset = make_dataset(offline_eps, config)
    
    # Set world model in environment
    env.set_wm(wm, offline_dataset, config)
    
    return env, wm


def create_policy(config, env):
    """Create the DDPG policy with same architecture as training"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    
    # Set activations (same as training)
    actor_activation = torch.nn.ReLU if config.actor_activation == 'ReLU' else torch.nn.Tanh
    critic_activation = torch.nn.ReLU if config.critic_activation == 'ReLU' else torch.nn.Tanh
    
    # Create critic network
    critic_net = Net(
        state_shape,
        action_shape,
        hidden_sizes=config.critic_net,
        activation=critic_activation,
        concat=True,
        device=config.device
    )
    critic = Critic(critic_net, device=config.device).to(config.device)
    # Create dummy optimizer (needed for policy initialization, won't be used)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    
    # Create actor network
    actor_net = Net(
        state_shape, 
        hidden_sizes=config.control_net, 
        activation=actor_activation, 
        device=config.device
    )
    actor = Actor(
        actor_net, 
        action_shape, 
        max_action=max_action, 
        device=config.device
    ).to(config.device)
    # Create dummy optimizer (needed for policy initialization, won't be used)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
    
    # Create policy with dummy optimizers
    policy = DDPGPolicy(
        critic,
        critic_optim,  # Needed for initialization
        tau=config.tau,
        gamma=config.gamma_pyhj,
        exploration_noise=None,  # No exploration noise for eval
        reward_normalization=config.rew_norm,
        estimation_step=config.n_step,
        action_space=env.action_space,
        actor=actor,
        actor_optim=actor_optim,  # Needed for initialization
        actor_gradient_steps=config.actor_gradient_steps,
    )
    
    return policy


def evaluate_policy(policy, env, wm, config, num_episodes=10, render=False, save_video=True):
    """Run evaluation episodes and collect statistics"""
    policy.eval()
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    # Import video saving library
    import cv2
    import os
    
    # Create videos directory
    if save_video:
        os.makedirs('evaluation_videos', exist_ok=True)
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        frames = []
        
        while not done:
            # Get action from policy (no exploration noise)
            with torch.no_grad():
                # The policy expects a Batch object with obs
                from PyHJ.data import Batch
                
                # Create batch from observation
                if isinstance(obs, np.ndarray):
                    obs_batch = Batch(obs=obs, info={})
                else:
                    obs_batch = Batch(obs=obs.cpu().numpy(), info={})
                
                # Get action using policy's forward method
                result = policy(obs_batch, model='actor')
                action = result.act
                
                # Convert to numpy if needed
                if torch.is_tensor(action):
                    action = action.cpu().numpy()
                
                # Debug: print shape on first step
                if episode_length == 0 and ep == 0:
                    print(f"Action shape from policy: {action.shape}")
                    print(f"Action value: {action}")
                
                # The environment expects a 1D array with single element
                # Squeeze all dimensions and ensure it's a 1D array
                action = np.squeeze(action)
                if action.ndim == 0:  # If it became a scalar, wrap in array
                    action = np.array([action])
                
                if episode_length == 0 and ep == 0:
                    print(f"Action after processing: {action}, shape: {action.shape}")
            
            # Decode current latent state to image for visualization
            if save_video:
                # Get the current latent state from environment
                latent = env.latent if hasattr(env, 'latent') else None
                
                if latent is not None:
                    # Decode latent to image using world model decoder
                    with torch.no_grad():
                        # Add batch dimension if needed
                        if isinstance(latent, dict):
                            # RSSM state is a dict with stoch and deter
                            feat = wm.dynamics.get_feat(latent)
                        else:
                            feat = latent
                        
                        # Ensure proper shape for decoder
                        if feat.ndim == 1:
                            feat = feat.unsqueeze(0).unsqueeze(0)
                        elif feat.ndim == 2:
                            feat = feat.unsqueeze(0)
                        
                        # Decode to image
                        decoded = wm.heads['decoder'](feat)
                        
                        # Get image from decoder output
                        if isinstance(decoded, dict) and 'image' in decoded:
                            img = decoded['image'].mode()  # Get mean of distribution
                        else:
                            img = decoded
                        
                        # Convert to numpy and proper format for video
                        img = img.squeeze().cpu().numpy()
                        
                        # Convert from (C, H, W) to (H, W, C) if needed
                        if img.ndim == 3 and img.shape[0] == 3:
                            img = np.transpose(img, (1, 2, 0))
                        
                        # Denormalize image (assuming it's in [-1, 1] or [0, 1] range)
                        img = np.clip(img, 0, 1)
                        img = (img * 255).astype(np.uint8)
                        
                        # Convert RGB to BGR for OpenCV
                        if img.shape[-1] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        # Add text overlay with episode info
                        img_copy = img.copy()
                        
                        # Add semi-transparent white background for text (tiny box)
                        overlay = img_copy.copy()
                        cv2.rectangle(overlay, (2, 2), (85, 35), (255, 255, 255), -1)
                        cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0, img_copy)
                        
                        # Add tiny black text on white background
                        font_scale = 0.25
                        thickness = 1
                        cv2.putText(img_copy, f"Step: {episode_length}", 
                                   (4, 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                        cv2.putText(img_copy, f"Reward: {episode_reward:.1f}", 
                                   (4, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                        cv2.putText(img_copy, f"Action: {action[0]:.2f}", 
                                   (4, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                        
                        frames.append(img_copy)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(info.get('success', False))
        
        # Save video for this episode
        if save_video and len(frames) > 0:
            video_path = f'evaluation_videos/episode_{ep+1:03d}_reward_{episode_reward:.2f}.mp4'
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
            
            for frame in frames:
                video.write(frame)
            
            video.release()
            print(f"Episode {ep+1}/{num_episodes}: Reward={episode_reward:.2f}, Length={episode_length} -> Saved to {video_path}")
        else:
            print(f"Episode {ep+1}/{num_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min/Max reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    if any(successes):
        print(f"Success rate: {np.mean(successes)*100:.1f}%")
    if save_video:
        print(f"\nVideos saved to: evaluation_videos/")
    print("="*60)
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'successes': successes
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DDPG policy')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to saved policy.pth file')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on (cuda:0 or cpu)')
    args = parser.parse_args()
    
    print("Loading configuration...")
    config = load_config()
    config.device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"Setting up environment: {config.task}")
    env, wm = setup_environment(config)
    
    print("Creating policy architecture...")
    policy = create_policy(config, env)
    
    print("\nPolicy keys (expected):")
    for key in list(policy.state_dict().keys())[:5]:
        print(f"  {key}")
    
    print(f"\nLoading trained policy from: {args.policy_path}")
    checkpoint = torch.load(args.policy_path, map_location=config.device)
    
    print("\nCheckpoint keys (saved):")
    for key in list(checkpoint.keys())[:5]:
        print(f"  {key}")
    
    # Try loading
    try:
        policy.load_state_dict(checkpoint, strict=True)
        print("\n✓ Policy loaded successfully!")
    except RuntimeError as e:
        print(f"\n✗ Strict loading failed, trying strict=False...")
        missing, unexpected = policy.load_state_dict(checkpoint, strict=False)
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  First few missing: {list(missing)[:3]}")
    
    policy.eval()  # Set to evaluation mode
    
    print(f"\nRunning evaluation for {args.num_episodes} episodes...")
    results = evaluate_policy(
        policy, 
        env,
        wm,  # Pass world model for video generation
        config,  # Pass config
        num_episodes=args.num_episodes,
        render=args.render,
        save_video=True  # Always save videos
    )
    
    # Optionally plot results
    if args.num_episodes > 1:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(results['rewards'], marker='o')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(results['lengths'], marker='o', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.title('Episode Lengths')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
        print("\nSaved evaluation plot to evaluation_results.png")
    
    env.close()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()