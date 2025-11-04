"""
Minimal script to visualize trained world model.
Uses existing methods from dreamer_offline.py - just loads checkpoint and calls them.

Usage:
    python scripts/dreamer_inference.py --logdir logs/dreamer_dubins --checkpoint rssm_ckpt.pt
"""

import argparse
import pathlib
import sys
import os

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import ruamel.yaml as yaml
import collections

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)

import tools
from dreamer_offline import Dreamer, make_dataset
import gym

class NullLogger:
    def __init__(self, logdir, step=0): self.step = step
    def config(self, *a, **k): pass
    def scalar(self, *a, **k): pass
    def image(self, *a, **k): pass
    def video(self, *a, **k): pass
    def write(self, *a, **k): pass

def load_dreamer(config, checkpoint_path):
    """Load trained Dreamer agent from checkpoint."""
    # Build observation/action spaces 
    action_space = gym.spaces.Box(
        low=-config.turnRate, high=config.turnRate, shape=(1,), dtype=np.float32
    )
    
    bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2*np.pi]])
    low, high = bounds[:, 0], bounds[:, 1]
    midpoint, interval = (low + high) / 2.0, high - low
    
    observation_space = gym.spaces.Dict({
        'state': gym.spaces.Box(np.float32(midpoint - interval/2), np.float32(midpoint + interval/2)),
        'obs_state': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        'image': gym.spaces.Box(low=0, high=255, shape=(config.size[0], config.size[0], 3), dtype=np.uint8)
    })

    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    
    # Create dummy dataset (needed for Dreamer constructor but won't be used)
    dummy_eps = collections.OrderedDict()
    dummy_dataset = make_dataset(dummy_eps, config)
    
    # Create logger
    # logger = tools.Logger(pathlib.Path(config.logdir), 0)
    logger = NullLogger(pathlib.Path(config.logdir), 0)
    
    # Build agent
    agent = Dreamer(observation_space, action_space, config, logger, dummy_dataset).to(config.device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='latest.pt')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    logdir = pathlib.Path(args.logdir)
    
    # Load config from original training run
    config_path = logdir.parent.parent / 'configs.yaml'
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    configs = yaml_loader.load(config_path.read_text())
    
    # Rebuild config namespace
    config_dict = configs['defaults']
    config_dict['logdir'] = str(logdir)  # override logdir
    
    parser2 = argparse.ArgumentParser()
    for key, value in config_dict.items():
        arg_type = tools.args_type(value)
        parser2.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    config = parser2.parse_args([])
    
    # Load trained agent
    checkpoint_path = logdir / args.checkpoint
    agent = load_dreamer(config, checkpoint_path)
    
    # Setup output
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else logdir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate lx_plot using existing method
    print("Generating lx_plot...")
    plot_array, tp, fn, fp, tn = agent.get_eval_plot()
    
    # Save
    plt.imsave(output_dir / 'lx_plot.png', plot_array)
    print(f"Saved lx_plot to {output_dir / 'lx_plot.png'}")
    
    # Compute metrics
    total = len(tp[0]) + len(fn[0]) + len(fp[0]) + len(tn[0])
    print(f"TP: {len(tp[0])/total*100:.1f}%  TN: {len(tn[0])/total*100:.1f}%")
    print(f"FP: {len(fp[0])/total*100:.1f}%  FN: {len(fn[0])/total*100:.1f}%")


if __name__ == '__main__':
    main()

