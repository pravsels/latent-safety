#!/usr/bin/env python3
"""
Uncapped Evaluation Script
Forces the world model to continue imagining beyond natural termination points.
Useful for analyzing world model behavior beyond its trained horizon.

Quick usage:

    python scripts/evaluate_uncapped.py \
        --policy_path logs/dreamer_dubins/policy.pth \
        --num_episodes 10 \
        --max_steps 100
"""

import argparse
import os
import sys
import numpy as np
import torch
import gymnasium
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import math
import io
from PIL import Image

# Add paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/PyHJ'))
sys.path.append(saferl_dir)

import models
import ruamel.yaml as yaml
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs


DEFAULT_RING_POINTS = 10
DEFAULT_RING_RADIUS_OFFSET = 0.8

def render_custom_observation(x, y, theta, config):
    """Render an RGB observation matching dataset style for a given pose."""
    fig, ax = plt.subplots()
    fig.set_size_inches(1, 1)
    ax.set_xlim([config.x_min, config.x_max])
    ax.set_ylim([config.y_min, config.y_max])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Obstacle circle
    circle = plt.Circle((config.obs_x, config.obs_y), config.obs_r, edgecolor=(1, 0, 0), facecolor='none', linewidth=2)
    ax.add_patch(circle)

    # Heading arrow
    dt = config.dt
    v = config.speed
    dx = dt * v * math.cos(theta)
    dy = dt * v * math.sin(theta)
    arena = max(config.x_max - config.x_min, config.y_max - config.y_min)
    ax.arrow(
        x, y, dx, dy,
        head_width=0.06 * arena,
        head_length=0.07 * arena,
        length_includes_head=True,
        color=(0, 0, 1),
        zorder=3,
    )

    ax.scatter([x], [y], s=10, color=(0, 0, 1), zorder=4)

    buf = io.BytesIO()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(buf, format='png', dpi=config.size[0])
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img, dtype=np.uint8)


def build_custom_init(x, y, theta, config):
    image = render_custom_observation(x, y, theta, config)
    obs_state = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    action = np.zeros((1,), dtype=np.float32)

    custom = {
        "image": image[None, None, ...],
        "obs_state": obs_state[None, None, ...],
        "action": action[None, None, ...],
        "is_first": np.array([[True]], dtype=bool),
        "is_last": np.array([[False]], dtype=bool),
        "is_terminal": np.array([[False]], dtype=bool),
        "reward": np.array([[0.0]], dtype=np.float32),
        "discount": np.array([[1.0]], dtype=np.float32),
        "privileged_state": np.array([[ [x, y, theta] ]], dtype=np.float32),
    }
    return custom


def make_ring_custom_inits(config, num_points, radius_offset=0.0):
    if num_points <= 0:
        return []

    starts = []
    radius = config.obs_r + radius_offset
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = config.obs_x + radius * math.cos(angle)
        y = config.obs_y + radius * math.sin(angle)
        theta = angle + math.pi  # Inward orientation
        starts.append(build_custom_init(x, y, theta, config))
    return starts


def _cycle_transitions(transitions):
    if not transitions:
        raise ValueError("custom transition list must not be empty.")
    while True:
        for item in transitions:
            yield item


def load_config():
    """Load configuration from yaml file"""
    yml = yaml.YAML(typ="safe", pure=True)
    config_path = Path(__file__).parent / "../configs.yaml"
    configs = yml.load(config_path.read_text())
    
    class Config:
        pass
    
    config = Config()
    for key, value in configs['defaults'].items():
        # Convert hyphenated keys to underscored
        attr_name = key.replace('-', '_')
        setattr(config, attr_name, value)
    
    return config


def setup_environment(config):
    """Setup the Dubins car environment with world model"""
    env = gymnasium.make(config.task, params=[config])
    config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    
    # Load world model
    wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)
    ckpt_path = config.rssm_ckpt_path
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    wm.eval()
    
    custom_inits = make_ring_custom_inits(config, DEFAULT_RING_POINTS, DEFAULT_RING_RADIUS_OFFSET)
    env.set_wm(wm, _cycle_transitions(custom_inits), config)
    
    return env, wm, custom_inits


def create_policy(config, env):
    """Create the DDPG policy"""
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    
    actor_activation = torch.nn.ReLU if config.actor_activation == 'ReLU' else torch.nn.Tanh
    critic_activation = torch.nn.ReLU if config.critic_activation == 'ReLU' else torch.nn.Tanh
    
    critic_net = Net(
        state_shape,
        action_shape,
        hidden_sizes=config.critic_net,
        activation=critic_activation,
        concat=True,
        device=config.device
    )
    critic = Critic(critic_net, device=config.device).to(config.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    
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
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
    
    policy = DDPGPolicy(
        critic,
        critic_optim,
        tau=config.tau,
        gamma=config.gamma_pyhj,
        exploration_noise=None,
        reward_normalization=config.rew_norm,
        estimation_step=config.n_step,
        action_space=env.action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=config.actor_gradient_steps,
    )
    
    return policy


def evaluate_uncapped(policy, env, wm, config, num_episodes=5, max_steps=100, custom_inits=None, hide_overlay=False):
    """
    Run uncapped evaluation - ignore termination and force long rollouts
    """
    policy.eval()
    
    episode_rewards = []
    episode_lengths = []
    natural_termination_steps = []  # When env would naturally terminate
    
    os.makedirs('uncapped_videos', exist_ok=True)
    
    print(f"\nForcing {max_steps} steps per episode (natural limit ~16)")
    print("Analyzing world model behavior beyond training horizon...\n")
    
    for ep in range(num_episodes):
        options = None
        if custom_inits:
            options = {"custom_init": custom_inits[ep % len(custom_inits)]}
        obs, info = env.reset(options=options)
        episode_reward = 0
        episode_length = 0
        natural_term_step = None
        
        frames = []
        
        # Force continuation beyond natural termination
        for step in range(max_steps):
            with torch.no_grad():
                from PyHJ.data import Batch
                
                obs_batch = Batch(obs=obs if isinstance(obs, np.ndarray) else obs.cpu().numpy(), info={})
                result = policy(obs_batch, model='actor')
                action = result.act
                
                if torch.is_tensor(action):
                    action = action.cpu().numpy()
                
                action = np.squeeze(action)
                if action.ndim == 0:
                    action = np.array([action])
            
            # Decode latent to video frame
            latent = env.latent if hasattr(env, 'latent') else None
            
            if latent is not None:
                with torch.no_grad():
                    feat = wm.dynamics.get_feat(latent)
                    
                    if feat.ndim == 1:
                        feat = feat.unsqueeze(0).unsqueeze(0)
                    elif feat.ndim == 2:
                        feat = feat.unsqueeze(0)
                    
                    decoded = wm.heads['decoder'](feat)
                    
                    if isinstance(decoded, dict) and 'image' in decoded:
                        img = decoded['image'].mode()
                    else:
                        img = decoded
                    
                    img = img.squeeze().cpu().numpy()
                    
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    
                    img = np.clip(img, 0, 1)
                    img = (img * 255).astype(np.uint8)
                    
                    
                    # Add overlay with text
                    if not hide_overlay:
                        img_copy = img.copy()
                        if img_copy.shape[-1] == 3:
                            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

                        # Add a black background for the text for better visibility
                        overlay = img_copy.copy()
                        cv2.rectangle(overlay, (2, 2), (50, 32), (0, 0, 0), -1) # Black background
                        cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0, img_copy)
                        
                        font_scale = 0.30
                        thickness = 1
                        text_color = (255, 255, 255)  # White text
                        
                        cv2.putText(img_copy, f"Step: {step}", 
                                   (4, 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                        cv2.putText(img_copy, f"R: {episode_reward:.1f}", 
                                   (4, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                        cv2.putText(img_copy, f"A: {-action[0]:.2f}", 
                                   (4, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                        
                        frames.append(img_copy)
                    else:
                        img_copy = img.copy()
                        if img_copy.shape[-1] == 3:
                            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                        frames.append(img_copy)
            
            # Take step (ignore termination signals)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Record natural termination point
            if (terminated or truncated) and natural_term_step is None:
                natural_term_step = step
                print(f"  Episode {ep+1}: Natural termination at step {step}, continuing to {max_steps}")
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        natural_termination_steps.append(natural_term_step if natural_term_step is not None else max_steps)
        
        # Save video
        if len(frames) > 0:
            video_path = f'uncapped_videos/episode_{ep+1:03d}_forced_{max_steps}_steps.mp4'
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
            
            for frame in frames:
                video.write(frame)
            
            video.release()
            print(f"  Episode {ep+1}/{num_episodes}: {episode_length} steps, Reward: {episode_reward:.2f}")
            print(f"  Saved: {video_path}")
    
    # Statistics
    print("\n" + "="*70)
    print("UNCAPPED EVALUATION RESULTS")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Forced steps per episode: {max_steps}")
    print(f"Natural termination (avg): {np.mean(natural_termination_steps):.1f} ± {np.std(natural_termination_steps):.1f}")
    print(f"Total reward (avg): {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Extra steps beyond natural: {max_steps - np.mean(natural_termination_steps):.1f}")
    print(f"\nVideos saved to: uncapped_videos/")
    print("="*70)
    
    return {
        'rewards': episode_rewards,
        'natural_terminations': natural_termination_steps
    }


def main():
    parser = argparse.ArgumentParser(description='Uncapped evaluation - ignore termination signals')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to policy.pth')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Force this many steps (default: 100, natural ~16)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0 or cpu)')
    parser.add_argument('--hide_overlay', action='store_true',
                        help='Hide text overlay on videos')
    args = parser.parse_args()
    
    print("Loading configuration...")
    config = load_config()
    config.device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"Setting up environment: {config.task}")
    env, wm, custom_inits = setup_environment(config)
    
    print("Creating policy...")
    policy = create_policy(config, env)
    
    print(f"Loading policy from: {args.policy_path}")
    checkpoint = torch.load(args.policy_path, map_location=config.device)
    policy.load_state_dict(checkpoint, strict=False)
    policy.eval()
    
    print(f"\nRunning uncapped evaluation: {args.num_episodes} episodes, {args.max_steps} forced steps")
    print(f"Custom starts: {len(custom_inits)} poses placed evenly around obstacle (radius offset {DEFAULT_RING_RADIUS_OFFSET}).")
    results = evaluate_uncapped(
        policy, 
        env,
        wm,
        config,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        custom_inits=custom_inits,
        hide_overlay=args.hide_overlay
    )
    
    env.close()
    print("\nUncapped evaluation complete!\n")


if __name__ == "__main__":
    main()

