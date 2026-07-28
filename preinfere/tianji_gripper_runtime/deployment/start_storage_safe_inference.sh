#!/usr/bin/env bash

set -euo pipefail

INFER_ROOT="/home/user/workspace/TJ-gripper_infer-main"

cd "$INFER_ROOT"
export PYTHONPATH="$INFER_ROOT/src:$INFER_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"

# Explicitly enable Safe Mode. The robot remains paused until N is pressed for
# each policy step; Space holds and Q exits.
exec uv run python \
  preinfere/tianji_gripper_runtime/deployment/infer_tianji_gripper.py \
  --config preinfere/tianji_gripper_runtime/configs/infer.yaml \
  --safe-mode \
  --execution-horizon 1
