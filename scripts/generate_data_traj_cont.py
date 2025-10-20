# generate_data_traj_cont.py 

import os, argparse
import matplotlib
matplotlib.use("Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io, sys, threading
from PIL import Image
import numpy as np
import torch, pickle, pathlib
import ruamel.yaml as yaml
import multiprocessing as mp

# Make parent project dirs importable (so we can import `tools` below).
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# Also add the Dreamer repo folder if tools live there.
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
import tools

# ---------------------------- Rendering ----------------------------

def get_frame(states, config):
  """
  Render a single RGB image given the current Dubins state.

  Args:
    states (torch.Tensor shape [3]): [x, y, theta].
    config: parsed config with fields like size, bounds, obstacle params, dt (time step), speed, etc.

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
  circle = patches.Circle([config.obs_x, config.obs_y], config.obs_r, edgecolor=(1, 0, 0), facecolor='none')
  # Add the circle patch to the axis
  ax.add_patch(circle)

  # Draw a blue velocity arrow of length v * dt pointing along heading theta.
  # Note: quiver params tuned for tiny canvas; avoid unsupported kwargs.
  plt.quiver(
    states[0], states[1], 
    dt * v * torch.cos(states[2]), 
    dt * v * torch.sin(states[2]), 
    angles='xy', scale_units='xy', width=0.1, scale=0.18, 
    color=(0, 0, 1), zorder=3
  )

  # Draw the agent position as a blue dot.
  plt.scatter(states[0], states[1], s=20, color=(0, 0, 1), zorder=3)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  # Rasterize figure to an in-memory PNG and decode to an RGB array.
  buf = io.BytesIO()
  plt.savefig(buf, format='png', dpi=config.size[0])
  buf.seek(0)
  img = Image.open(buf).convert('RGB')
  img_array = np.array(img)
  plt.close(fig)    # free figure resources!

  return img_array

# ------------------------ State / Dynamics -------------------------
   
def get_init_state(config):  
  """
  Sample an initial state outside the obstacle and within arena bounds
  (minus a buffer). Heading is set to roughly point back toward origin
  with added Gaussian noise.
  """
  states = torch.zeros(3)

  # Rejection-sample (x, y) outside the obstacle circle.
  # while loop keeps going until we have a point outisde the circle. 
  while np.linalg.norm(states[:2] - np.array([config.obs_x, config.obs_y])) < config.obs_r:
    states = torch.rand(3)

    # Sample within bounds minus buffer for x and y.
    states[0] *= (config.x_max - config.buffer) - (config.x_min + config.buffer)
    states[1] *= (config.y_max - config.buffer) - (config.y_min + config.buffer)
    states[0] += config.x_min + config.buffer
    states[1] += config.y_min + config.buffer

  # Choose heading theta to roughly face toward the origin, with noise.
  states[2] = torch.atan2(-states[1], -states[0]) + np.random.normal(0, 1)
  states[2] = states[2] % (2 * np.pi)   # wrap into [0, 2π)

  return states

# ------------------------- Serialization --------------------------

def to_python_demo(state_obs, acs, state_gt, img_obs, dones):
  """Convert tensors to compact, pickle-friendly Python/NumPy types."""
  # Scalars to float, arrays to np.float32, images already np.uint8
  obs_state = [float(th) for th in state_obs]
  actions = [float(a) for a in acs]
  priv = [np.asarray(s, dtype=np.float32) for s in state_gt]

  demo = {
    'obs': {
      'image': img_obs,     # list[np.uint8 HxWx3]
      'state': obs_state,   # list[float]
      'priv_state': priv,   # list[np.float32(3,)]
    },
    'actions': actions,     # list[float]
    'dones': [int(d) for d in dones],
  }

  return demo

# ---------------------- Trajectory generation ----------------------

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
    ac = torch.rand(()) * 2 * u_max - u_max

    # Single-step Dubins dynamics 
    states_next = torch.empty(3, dtype=states.dtype)
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

def _worker_batch(count: int, cfg, q: mp.Queue):
    """Worker: build `count` trajectories, pickle into one bytes blob, put on queue."""
    payloads = []
    for _ in range(count):
        state_obs, acs, state_gt, img_obs, dones = gen_one_traj_img(cfg)
        demo = to_python_demo(state_obs, acs, state_gt, img_obs, dones)
        payloads.append(pickle.dumps(demo, protocol=pickle.HIGHEST_PROTOCOL))
    blob = b"".join(payloads)
    q.put((blob, count))  # send (bytes, num_trajs)

def _writer_thread(q: mp.Queue, out_path: str, flush_bytes: int, total_trajs: int):
    """Single writer: append blobs; print simple progress."""
    buf = []
    sz = 0
    done = 0
    with open(out_path, "ab", buffering=4 * 1024 * 1024) as f:
        while True:
            item = q.get()  # blocks
            if item is None:  # sentinel -> drain and exit
                for b, _n in buf:
                    f.write(b)
                print(f"[writer] progress: {done}/{total_trajs} (100%)")
                return
            blob, n_trajs = item
            buf.append((blob, n_trajs))
            sz += len(blob)
            done += n_trajs
            # Print on every blob (cheap) or only on flush — your call:
            print(f"[writer] progress: {done}/{total_trajs}")
            if sz >= flush_bytes:
                for b, _n in buf:
                    f.write(b)
                buf.clear()
                sz = 0

# ----------------------------- Main --------------------------------

def recursive_update(base, update):
    """Recursively merge dict `update` into dict `base` in-place."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

if __name__ == '__main__':
    # Build config from YAML + CLI (your existing code)
    parser = argparse.ArgumentParser()
    _, remaining = parser.parse_known_args()
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    configs = yaml_loader.load((pathlib.Path(sys.argv[0]).parent / '../configs.yaml').read_text())
    defaults = {}
    for name in ['defaults']:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))

    # Parallel + queue knobs (small + sensible defaults)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count()) - 1))
    parser.add_argument('--chunk_size', type=int, default=100, help='Trajs per worker batch.')
    parser.add_argument('--flush_mb', type=int, default=16, help='Writer flush threshold (MB).')
    parser.add_argument('--queue_items', type=int, default=100, help='Max queued blobs (backpressure).')

    cfg = parser.parse_args(remaining)

    # Where to write (single file)
    out_path = cfg.dataset_path  # e.g., wm_demos128.pkl
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Setup queue + writer
    ctx = mp.get_context('spawn')  # macOS-friendly
    q: mp.Queue = ctx.Queue(maxsize=cfg.queue_items)
    writer = threading.Thread(
      target=_writer_thread,
      args=(q, out_path, int(cfg.flush_mb) * 1024 * 1024, int(cfg.num_trajs)),
      daemon=True,
    )
    writer.start()

    # Launch workers (each produces one blob with `chunk_size` trajs)
    procs = []
    remaining_trajs = int(cfg.num_trajs)
    while remaining_trajs > 0:
        take = min(cfg.chunk_size, remaining_trajs)
        p = ctx.Process(target=_worker_batch, args=(take, cfg, q))
        p.start()
        procs.append(p)
        remaining_trajs -= take

    # Wait for workers, then stop writer
    for p in procs:
        p.join()

    q.put(None)  # sentinel to stop writer after draining
    writer.join()

    print(f"Written dataset (streamed): {out_path}")
