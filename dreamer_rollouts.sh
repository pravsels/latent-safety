#!/usr/bin/env bash
set -euo pipefail

# --------- EDIT THESE IF NEEDED ----------
LOGDIR="logs/dreamer_dubins"
CKPT="rssm_ckpt.pt"
FPS=16
HORIZON=60
OUTPUT_DIR="${LOGDIR}/visualizations"
PY="python"   # or path to your python
SCRIPT="scripts/dreamer_inference.py"
# -----------------------------------------

# Starts are "x,y,theta" (theta in radians)
STARTS=(
  "1.0,0.0,1.5708"   # x=1.0, y=0.0, 90 deg
  "1.0,0.0,0.0"      # facing +x
  "0.0,1.0,3.1416"   # facing -x
  "-1.0,0.0,1.5708"  # left side, facing up
)

# Constant turn rates to test (within [-turnRate, turnRate])
ACTIONS=(
  "0.0"    # straight
  "0.6"    # gentle left
  "-0.6"   # gentle right
)

# Optional: different horizons (uncomment to try multiple)
HORIZONS=(
  "${HORIZON}"
  # "80"
  # "100"
)

mkdir -p "${OUTPUT_DIR}"

for start in "${STARTS[@]}"; do
  for action in "${ACTIONS[@]}"; do
    for T in "${HORIZONS[@]}"; do
      echo "Running rollout: start=${start}  action=${action}  horizon=${T}"
      ${PY} "${SCRIPT}" \
        --logdir "${LOGDIR}" \
        --checkpoint "${CKPT}" \
        --output_dir "${OUTPUT_DIR}" \
        --rollout "${start}" \
        --action "${action}" \
        --horizon "${T}" \
        --fps "${FPS}"
      # Note: GIF filename is auto-humanized by your script (no --out_gif needed).
    done
  done
done

