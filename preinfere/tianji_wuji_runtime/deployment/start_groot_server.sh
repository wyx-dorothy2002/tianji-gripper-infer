#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_REPO="${GR00T_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CKPT="${CKPT:-${1:-}}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${CKPT}" ]]; then
  echo "Usage: CKPT=/path/to/checkpoint EMBODIMENT_TAG=YOUR_TAG bash deployment/start_groot_server.sh"
  echo "   or: bash deployment/start_groot_server.sh /path/to/checkpoint"
  exit 2
fi

if [[ ! -d "${GR00T_REPO}" ]]; then
  echo "GR00T_REPO does not exist: ${GR00T_REPO}" >&2
  exit 2
fi

if [[ "${CKPT}" == /* && ! -e "${CKPT}" ]]; then
  echo "Checkpoint path does not exist: ${CKPT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export NO_ALBUMENTATIONS_UPDATE=1

echo "[groot-server] repo: ${GR00T_REPO}"
echo "[groot-server] checkpoint: ${CKPT}"
echo "[groot-server] embodiment tag: ${EMBODIMENT_TAG}"
echo "[groot-server] device: ${DEVICE}"
echo "[groot-server] endpoint: ${HOST}:${PORT}"
echo "[groot-server] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd "${GR00T_REPO}"
exec uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "${CKPT}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --device "${DEVICE}" \
  --host "${HOST}" \
  --port "${PORT}"

