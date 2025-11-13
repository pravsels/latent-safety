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
from datetime import datetime

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
import collections

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)

import tools
from generate_data_traj_cont import get_frame
import imageio.v2 as imageio
from dreamer_offline import Dreamer, make_dataset
import gym
from PIL import Image, ImageDraw, ImageFont

class NullLogger:
    def __init__(self, logdir, step=0): self.step = step
    def config(self, *a, **k): pass
    def scalar(self, *a, **k): pass
    def image(self, *a, **k): pass
    def video(self, *a, **k): pass
    def write(self, *a, **k): pass

def human_name(ckpt_stem, x, y, theta, horizon, action, fps):
    deg = int(round(np.degrees(theta)))
    turn = "left" if action > 1e-6 else ("right" if action < -1e-6 else "straight")
    dt = datetime.now()
    # Build "5 Nov 2025" in a portable way (no %-d needed)
    date_str = f"{dt.day} {dt.strftime('%b %Y')}"
    time_str = dt.strftime("%H-%M-%S")  # colons → hyphens for filesystem safety
    return (
        f"rollout - start x={x:.2f}, y={y:.2f}, theta={deg}deg - "
        f"action={action:+.2f} ({turn}) - horizon={int(horizon)} steps - "
        f"fps={fps} - ckpt={ckpt_stem} - {date_str} {time_str}.gif"
    )

def add_action_text(frame, action_value):
    """Add action text overlay to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    if abs(action_value) < 1e-6:
        turn = "STRAIGHT"
        color = (0, 255, 0)  # green
    elif action_value > 0:
        turn = f"LEFT {action_value:+.2f}"
        color = (255, 0, 0)  # red
    else:
        turn = f"RIGHT {action_value:+.2f}"
        color = (0, 0, 255)  # blue
    
    draw.text((10, 10), turn, fill=color)

    return np.array(img)

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

def rollout_open_loop(agent, config, x, y, theta, horizon=50, action_value=0.0):
    """
    Open-loop imagination from a chosen start pose with a simple action sequence.
    - x,y,theta: starting pose
    - horizon: number of imagined steps
    - action_value: constant turn rate to feed (in [-turnRate, turnRate])
    Returns: np.uint8 video array of shape (T, H, W, C)
    """
    # 1) One real frame to get the posterior at t=0
    img0 = get_frame(torch.tensor([x, y, theta]), config).astype(np.uint8)
    H, W = img0.shape[:2]

    images   = np.zeros((1, 1, H, W, 3), dtype=np.uint8)  # [B=1, T=1, H, W, C]
    images[0, 0] = img0
    obs_state = np.array([[[np.cos(theta), np.sin(theta)]]], dtype=np.float32)  # [1,1,2]
    actions0  = np.zeros((1, 1, 1), dtype=np.float32)       # dummy a_0
    is_first  = np.array([[[1.0]]], dtype=np.float32)
    is_term   = np.array([[[0.0]]], dtype=np.float32)

    data0 = {
        "image": images, "obs_state": obs_state, "action": actions0,
        "is_first": is_first, "is_terminal": is_term,
    }
    data0 = agent._wm.preprocess(data0)
    embed0 = agent._wm.encoder(data0)
    # posterior for the single observed step
    post0, _ = agent._wm.dynamics.observe(embed0, data0["action"], data0["is_first"])
    init = {k: v[:, -1] for k, v in post0.items()}  # last=only step

    # 2) Imagine for horizon-1 steps with constant action
    T_imag = max(0, int(horizon) - 1)
    if T_imag > 0:
        # time-first: [T, B=1, action_dim=1]
        actions = torch.full(
            (1, T_imag, 1),
            float(action_value),
            device=config.device,
            dtype=torch.float32,
        )

        use_amp = bool(getattr(agent._wm, "_use_amp", False) and torch.cuda.is_available())
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            prior = agent._wm.dynamics.imagine_with_action(actions, init)   # dict of states [T_imag, B, ...]
            print(f"[debug] prior keys: {prior.keys()}")
            print(f"[debug] prior['deter'].shape: {prior['deter'].shape}")
            feat  = agent._wm.dynamics.get_feat(prior)                      # [T_imag, B, feat]
            print(f"[debug] feat.shape: {feat.shape}")
            pred  = agent._wm.heads["decoder"](feat)["image"].mode()        # [T_imag, B, H, W, C]
            print(f"[debug] pred.shape: {pred.shape}")
    else:
        pred = torch.empty((0, 1, H, W, 3), device=config.device)

    # 3) Also decode the observed frame (reconstruction of t=0) for the first frame
    with torch.no_grad():
        recon0 = agent._wm.heads["decoder"](agent._wm.dynamics.get_feat(post0))["image"].mode()  # [B=1, T=1, H,W,C]
        recon0 = recon0[:, -1]  # [B=1, H,W,C]

    # 4) Assemble [t=0 recon] + [imagined frames]
    vid = torch.cat([recon0, pred[0, :]], dim=0)  # [T, H, W, C], T=horizon

    # 5) To numpy uint8 in [0,255]
    vid = vid.detach().cpu().float().numpy()
    vid = np.clip(vid, 0.0, 1.0)
    vid = (vid * 255).astype(np.uint8)

    for i in range(len(vid)):
        vid[i] = add_action_text(vid[i], action_value)

    return vid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='rssm_ckpt.pt')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--out_gif', type=str, default=None)
    parser.add_argument('--rollout', type=str, default=None, help='x,y,theta (e.g. "0.8,0.0,1.57")')
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--action', type=float, default=0.0, help='constant turn rate during rollout')
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--lx-plot', action='store_true', help='generate lx_plot.png')

    args = parser.parse_args()
    
    if args.horizon < 1:
        raise ValueError("--horizon must be ≥ 1")

    torch.manual_seed(0); np.random.seed(0)

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
    
    if args.lx_plot:
        print("Generating lx_plot...")
        plot_array, tp, fn, fp, tn = agent.get_eval_plot()

        plt.imsave(output_dir / 'lx_plot.png', plot_array)
        print(f"Saved lx_plot to {output_dir / 'lx_plot.png'}")
        # Compute metrics
        total = len(tp[0]) + len(fn[0]) + len(fp[0]) + len(tn[0])
        print(f"TP: {len(tp[0])/total*100:.1f}%  TN: {len(tn[0])/total*100:.1f}%")
        print(f"FP: {len(fp[0])/total*100:.1f}%  FN: {len(fn[0])/total*100:.1f}%")

    if args.rollout:
        x, y, th = map(float, args.rollout.split(','))
        #  clamp action to valid range 
        act = max(-config.turnRate, min(config.turnRate, args.action))
        vid = rollout_open_loop(agent, config, x, y, th, horizon=args.horizon, action_value=act)
        print(f"[debug] vid shape={vid.shape}, dtype={vid.dtype}")
        print(f"[debug] vid min={vid.min()}, max={vid.max()}")
        print(f"[debug] first frame shape: {vid[0].shape}")

        ckpt_stem = pathlib.Path(args.checkpoint).stem
        name = human_name(ckpt_stem, x, y, th, args.horizon, act, args.fps)
        out_gif = pathlib.Path(args.out_gif) if args.out_gif else (output_dir / name)

        print(f"[debug] vid shape={vid.shape}  (expect T={args.horizon})")
        imageio.mimsave(out_gif, list(vid), fps=args.fps)
        print(f"Saved rollout video to {out_gif}")
        mp4_path = out_gif.with_suffix('.mp4')
        imageio.mimsave(mp4_path, list(vid), fps=args.fps)
        print(f"Saved rollout video to {mp4_path}")


if __name__ == '__main__':
    main()

