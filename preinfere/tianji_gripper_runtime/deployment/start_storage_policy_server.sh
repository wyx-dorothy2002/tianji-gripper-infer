#!/usr/bin/env bash

set -euo pipefail

OPENPI_ROOT="/home/user/workspace/TJ-gripper_infer-main"
CKPT_DIR="/home/user/workspace/checkpoints/storage/pi05_tianji_pick_and_place_0720_night_0721_day_night_50k_bs256_lr5e5_8gpu_dualarm/50000"
CONFIG_NAME="storage"
PORT="${PORT:-8000}"

cd "$OPENPI_ROOT"
export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"

exec uv run scripts/serve_policy.py \
  --port "$PORT" \
  --default-prompt "Put all the objects on the desk into the storage box" \
  policy:checkpoint \
  --policy.config "$CONFIG_NAME" \
  --policy.dir "$CKPT_DIR"
