#!/usr/bin/env python3
"""Extract a replay-compatible [H,54] GT action chunk from a LeRobot dataset."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_wuji_runtime.runtime import schema


SEGMENT_SLICES = {
    "left_arm": schema.LEFT_ARM_SLICE,
    "right_arm": schema.RIGHT_ARM_SLICE,
    "left_hand": schema.LEFT_HAND_SLICE,
    "right_hand": schema.RIGHT_HAND_SLICE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default="/mnt/data/qdhe/workspace/datasets/local_lerobot_dataset",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default=str(RUNTIME_ROOT / "infer_logs" / "dataset_action_chunks"),
    )
    parser.add_argument(
        "--pad-last",
        action="store_true",
        help="Repeat the last frame if the requested chunk reaches past the episode end.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if args.start_step < 0:
        raise ValueError("--start-step must be non-negative")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")

    info = _read_json(dataset_path / "meta" / "info.json")
    episodes = _read_jsonl(dataset_path / "meta" / "episodes.jsonl")
    episode_meta = _find_episode_meta(episodes, args.episode_index)
    parquet_path = _resolve_episode_path(dataset_path, info, args.episode_index)
    df = pd.read_parquet(parquet_path)

    if "action" not in df.columns:
        raise ValueError(f"{parquet_path} does not contain an 'action' column")
    if args.start_step >= len(df):
        raise ValueError(
            f"start step {args.start_step} is outside episode length {len(df)}"
        )

    indices = _resolve_indices(
        start_step=args.start_step,
        num_steps=args.num_steps,
        episode_length=len(df),
        pad_last=args.pad_last,
    )
    action_chunk = _stack_vector_column(df, "action", indices, schema.ACTION_DIM)
    state_chunk = (
        _stack_vector_column(df, "observation.state", indices, schema.STATE_DIM)
        if "observation.state" in df.columns
        else None
    )

    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    action_path = run_dir / "action_chunk.npy"
    np.save(action_path, action_chunk)
    state_path = None
    if state_chunk is not None:
        state_path = run_dir / "state_chunk.npy"
        np.save(state_path, state_chunk)

    metadata = {
        "dataset_path": str(dataset_path),
        "parquet_path": str(parquet_path),
        "episode_index": args.episode_index,
        "episode_name": None if episode_meta is None else episode_meta.get("episode_name"),
        "episode_tasks": [] if episode_meta is None else episode_meta.get("tasks", []),
        "episode_length": len(df),
        "start_step": args.start_step,
        "num_steps": args.num_steps,
        "indices": indices,
        "action_path": str(action_path),
        "state_path": None if state_path is None else str(state_path),
        "action_shape": list(action_chunk.shape),
        "state_shape": None if state_chunk is None else list(state_chunk.shape),
        "schema": schema.schema_metadata(),
        "segment_stats": _segment_stats(action_chunk),
        "first_action": _split_to_lists(action_chunk[0]),
        "last_action": _split_to_lists(action_chunk[-1]),
    }
    _write_json(run_dir / "metadata.json", metadata)

    print(f"[extract] action: {action_path}")
    if state_path is not None:
        print(f"[extract] state:  {state_path}")
    print(f"[extract] shape:  {action_chunk.shape}")
    print(f"[extract] task:   {metadata['episode_tasks']}")
    print(f"[extract] range:  step {indices[0]} -> {indices[-1]}")
    print("[extract] replay dry-run:")
    print(
        "  python tianji_wuji_runtime/deployment/replay_policy_check.py "
        f"--action-file {action_path} --duration 0.05 --dry-run --safe-mode"
    )
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_episode_meta(
    episodes: list[dict[str, Any]],
    episode_index: int,
) -> dict[str, Any] | None:
    for item in episodes:
        if int(item.get("episode_index", -1)) == episode_index:
            return item
    return None


def _resolve_episode_path(dataset_path: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    episode_chunk = episode_index // chunks_size
    path = dataset_path / data_template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _resolve_indices(
    *,
    start_step: int,
    num_steps: int,
    episode_length: int,
    pad_last: bool,
) -> list[int]:
    end = start_step + num_steps
    if end <= episode_length:
        return list(range(start_step, end))
    if not pad_last:
        raise ValueError(
            f"requested steps [{start_step}, {end}) exceed episode length {episode_length}; "
            "pass --pad-last to repeat the final frame"
        )
    indices = list(range(start_step, episode_length))
    indices.extend([episode_length - 1] * (num_steps - len(indices)))
    return indices


def _stack_vector_column(
    df: pd.DataFrame,
    column: str,
    indices: list[int],
    width: int,
) -> np.ndarray:
    values = [np.asarray(df[column].iloc[idx], dtype=np.float32) for idx in indices]
    arr = np.stack(values, axis=0)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{column} chunk must have shape [H,{width}], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{column} chunk contains NaN or Inf")
    return arr.astype(np.float32, copy=False)


def _split_to_lists(action_54: np.ndarray) -> dict[str, list[float]]:
    return {
        name: np.asarray(action_54[segment_slice], dtype=np.float32).tolist()
        for name, segment_slice in SEGMENT_SLICES.items()
    }


def _segment_stats(action_chunk: np.ndarray) -> dict[str, dict[str, float]]:
    stats = {}
    for name, segment_slice in SEGMENT_SLICES.items():
        arr = action_chunk[:, segment_slice]
        if len(action_chunk) > 1:
            max_step = float(np.max(np.abs(np.diff(arr, axis=0))))
        else:
            max_step = 0.0
        stats[name] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "max_abs_step": max_step,
        }
    return stats


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
