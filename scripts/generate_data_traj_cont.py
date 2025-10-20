
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import io
from PIL import Image
import numpy as np
import torch
import pickle
import pathlib
import ruamel.yaml as yaml
import os
import sys

# Make parent project dirs importable (so we can import `tools` below).
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# Also add the Dreamer repo folder if tools live there.
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
import tools


def get_frame(states, config):
  """
  Render a single RGB image given the current Dubins state.

  Args:
    states (torch.Tensor shape [3]): [x, y, theta].
    config: parsed config with fields like size, bounds, obstacle params,
    dt (time step), speed, etc.

  Returns:
    np.ndarray (H, W, 3) uint8 image.
  """
  dt = config.dt
  v = config.speed

  # Create a fresh 1×1 inch figure; DPI sets the pixel resolution to `size[0]`.
  fig,ax = plt.subplots()
  plt.xlim([config.x_min, config.x_max])
  plt.ylim([config.y_min, config.y_max])
  plt.axis('off')
  fig.set_size_inches(1, 1)

  # Create and draw the obstacle (red circle outline).
  circle = patches.Circle([config.obs_x, config.obs_y], config.obs_r, edgecolor=(1,0,0), facecolor='none')
  # Add the circle patch to the axis
  ax.add_patch(circle)

  # Draw a blue velocity arrow of length v*dt pointing along heading theta.
  # Note: quiver params tuned for tiny canvas; avoid unsupported kwargs.
  plt.quiver(
    states[0], states[1], 
    dt * v * torch.cos(states[2]), 
    dt * v * torch.sin(states[2]), 
    angles='xy', scale_units='xy', minlength=0, width=0.1, scale=0.18, 
    color=(0, 0, 1), zorder=3
  )

  # Draw the agent position as a blue dot.
  plt.scatter(states[0], states[1], s=20, c=(0, 0, 1), zorder=3)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  # Rasterize figure to an in-memory PNG and decode to an RGB array.
  buf = io.BytesIO()
  plt.savefig(buf, format='png', dpi=config.size[0])
  buf.seek(0)
  img = Image.open(buf).convert('RGB')
  img_array = np.array(img)
  plt.close(fig)    # free figure resources!

  return img_array
   
def get_init_state(config):  
  """
  Sample an initial state outside the obstacle and within arena bounds
  (minus a buffer). Heading is set to roughly point back toward origin
  with added Gaussian noise.
  """
  states = torch.zeros(3)

  # Rejection-sample (x, y) outside the obstacle circle.
  while np.linalg.norm(states[:2] - np.array([config.obs_x, config.obs_y])) < config.obs_r:
    states = torch.rand(3)

    # Sample within bounds minus buffer for x and y.
    states[0] *= (config.x_max-config.buffer) - (config.x_min + config.buffer)
    states[1] *= (config.y_max-config.buffer) - (config.y_min + config.buffer)
    states[0] += config.x_min + config.buffer
    states[1] += config.y_min + config.buffer

  # Choose heading theta to roughly face toward the origin, with noise.
  states[2] = torch.atan2(-states[1], -states[0]) + np.random.normal(0, 1)
  states[2] = states[2] % (2 * np.pi)   # wrap into [0, 2π)

  return states

def gen_one_traj_img(config):
  """
  Generate a single random-action trajectory and its rendered frames.

  Returns:
    state_obs: list of observed scalars (theta_t) per step.
    acs: list of actions (turn-rate) per step.
    state_gt: list of full [x, y, theta] states per step.
    img_obs: list of RGB arrays per step (rendered frames).
    dones: list of 0/1 termination flags per step.
  """
  states = get_init_state(config)

  state_obs = []
  img_obs = []
  state_gt = []
  dones = []
  acs = []

  u_max = config.turnRate
  dt = config.dt
  v = config.speed

  for t in range(config.data_length):
    # Sample action uniformly in [-u_max, +u_max].
    ac = torch.rand(1) * 2 * u_max - u_max

    # Single-step Dubins dynamics 
    states_next = torch.rand(3)
    states_next[0] = states[0] + v * dt * torch.cos(states[2])
    states_next[1] = states[1] + v * dt * torch.sin(states[2])
    states_next[2] = states[2] + dt * ac

    # Log observations *before* applying the action (o_t, a_t pair).
    state_obs.append(states[2].numpy())   # observed heading theta
    state_gt.append(states.numpy())       # gt state for debugging

    # Compute termination: last time step OR out-of-bounds
    if t == config.data_length-1:
      dones.append(1)
    elif torch.abs(states[0]) > config.x_max - config.buffer or \
         torch.abs(states[1]) > config.y_max - config.buffer:
      # Out of bounds (relative to center, assumes symmetric limits)
      dones.append(1)
    else:
      dones.append(0)
    
    # Store action and rendered frame for this state.
    acs.append(ac)
    img_array = get_frame(states, config)
    img_obs.append(img_array)

    # Advance state.
    states = states_next

    # Early stop if done.
    if dones[-1] == 1:
      break

  return state_obs, acs, state_gt, img_obs, dones

def generate_trajs(config):
  """
  Generate `config.num_trajs` independent trajectories and dump them
  into a pickle file named by resolution (e.g., wm_demos128.pkl).
  """
  demos = []
  for i in range(config.num_trajs):
    state_obs, acs, state_gt, img_obs, dones = gen_one_traj_img(config)

    # Package one trajectory. Key names should match your training pipeline.
    demo = {
      'obs': {
        'image': img_obs,
        'state': state_obs,       # note: encoder may expect 'obs_state'
        'priv_state': state_gt,   # optional: full-state debug
      },
      'actions': acs,
      'dones': dones,
    }

    demos.append(demo)
    print('demo: ', i, "timesteps: ", len(state_obs))

    # WARNING: This ignores config.dataset_path and always writes to
    # wm_demos{size[0]}.pkl in CWD. Adjust if needed.
    with open('wm_demos'+str(config.size[0])+'.pkl', 'wb') as f:
      pickle.dump(demos, f)

def recursive_update(base, update):
    """Recursively merge dict `update` into dict `base` in-place."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

if __name__ == '__main__':
  # First, build an "empty" argparse and unknown args as remaining. 
  # This will be parsed after we derive types from YAML defaults.
  parser = argparse.ArgumentParser()
  config, remaining = parser.parse_known_args()

  # Load `../configs.yaml` using ruamel.yaml (safe loader, pure Python).
  yaml_loader = yaml.YAML(typ='safe', pure=True)
  configs = yaml_loader.load((pathlib.Path(sys.argv[0]).parent / '../configs.yaml').read_text())

  # You can stack multiple named blocks, here we only merge 'defaults'.
  name_list = ['defaults']
  defaults = {}
  for name in name_list:
    recursive_update(defaults, configs[name])

  # Now that we have a defaults dict, expose every key as a CLI flag.
  # `tools.args_type` should infer an argparse type from the default value.
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))

  # Parse remaining CLI args into a Namespace. This becomes our global config.
  final_config = parser.parse_args(remaining)

  # Generate and save trajectories according to config.
  demos = generate_trajs(final_config)

