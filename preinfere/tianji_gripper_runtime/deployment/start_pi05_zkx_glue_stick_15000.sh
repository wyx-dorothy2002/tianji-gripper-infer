#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/user/workspace/openpi"
CKPT_DIR="$ROOT_DIR/ckpts/pi05_ziyi_sensor_30k_bs256_lr5e5/15000"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/packages/openpi-client/src:${PYTHONPATH:-}"

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_ziyi_sensor \
  --policy.dir "$CKPT_DIR"
