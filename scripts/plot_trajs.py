#!/usr/bin/env python3
import argparse, pickle, random, pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import ruamel.yaml as yaml

def iter_pickles(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def load_cfg_bounds(config_path, block="defaults"):
    y = yaml.YAML(typ="safe", pure=True)
    cfgs = y.load(pathlib.Path(config_path).read_text())
    c = cfgs[block]
    return {
        "x_min": float(c["x_min"]), "x_max": float(c["x_max"]),
        "y_min": float(c["y_min"]), "y_max": float(c["y_max"]),
        "obs_x": float(c["obs_x"]), "obs_y": float(c["obs_y"]),
        "obs_r": float(c["obs_r"]),
    }

def main():
    ap = argparse.ArgumentParser(description="Plot random N trajectories using bounds from configs.yaml.")
    ap.add_argument("--dataset", default="./wm_demos128.pkl")
    ap.add_argument("--config",  default="./configs.yaml")
    ap.add_argument("--block",   default="defaults")
    ap.add_argument("--num",     type=int, default=100)
    ap.add_argument("--seed",    type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    arena = load_cfg_bounds(args.config, args.block)

    # Reservoir sampling (one pass)
    reservoir, seen = [], 0
    for traj in iter_pickles(args.dataset):
        priv = traj["obs"].get("priv_state")
        if not priv:
            continue
        xs = [float(s[0]) for s in priv]
        ys = [float(s[1]) for s in priv]
        if len(reservoir) < args.num:
            reservoir.append((xs, ys))
        else:
            j = random.randint(0, seen)  # inclusive
            if j < args.num:
                reservoir[j] = (xs, ys)
        seen += 1

    # Bigger plot
    fig, ax = plt.subplots(figsize=(9, 9))

    # Purple boundary rectangle
    ax.add_patch(Rectangle(
        (arena["x_min"], arena["y_min"]),
        arena["x_max"] - arena["x_min"],
        arena["y_max"] - arena["y_min"],
        fill=False, linewidth=2.0, edgecolor="purple"
    ))

    # Red obstacle
    ax.add_patch(Circle(
        (arena["obs_x"], arena["obs_y"]),
        arena["obs_r"], fill=False, linewidth=1.8, edgecolor="red"
    ))

    # Plot trajectories
    for xs, ys in reservoir:
        ax.plot(xs, ys, linewidth=0.9, alpha=0.7)

    ax.set_xlim(arena["x_min"]-0.5, arena["x_max"]+0.5)
    ax.set_ylim(arena["y_min"]-0.5, arena["y_max"]+0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Random {len(reservoir)} trajectories (from {seen} total)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # no grid lines
    # ax.grid(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
