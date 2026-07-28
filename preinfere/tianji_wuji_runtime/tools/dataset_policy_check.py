#!/usr/bin/env python3
"""Compare policy predictions against GT actions on selected LeRobot dataset frames."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from gr00t.data.dataset.lerobot_episode_loader import LANG_KEYS, LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag

from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.groot_policy_client import GrootPolicyClient, PolicyServerError


SEGMENTS = {
    "left_arm": schema.LEFT_ARM_SLICE,
    "right_arm": schema.RIGHT_ARM_SLICE,
    "left_hand": schema.LEFT_HAND_SLICE,
    "right_hand": schema.RIGHT_HAND_SLICE,
}


@dataclass
class SegmentMetrics:
    mae: float
    rmse: float
    max_abs: float
    first_step_mae: float
    first_step_rmse: float
    pred_first: list[float]
    gt_first: list[float]
    diff_first: list[float]


@dataclass
class SampleResult:
    episode_index: int
    dataset_episode_index: int
    episode_name: str | None
    step_index: int
    task: str
    horizon: int
    policy_latency_ms: float | None
    overall: SegmentMetrics
    segments: dict[str, SegmentMetrics]
    pred_action_path: str
    gt_action_path: str
    input_state_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default="/mnt/data/qdhe/workspace/datasets/local_lerobot_dataset",
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--policy-timeout-ms", type=int, default=30000)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--task", default=None, help="Override dataset task text.")
    parser.add_argument("--episodes", default="0,8,11")
    parser.add_argument("--fractions", default="0.1,0.5,0.8")
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Explicit sample as EPISODE:STEP. Can be passed multiple times.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-padding", action="store_true")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--output-dir",
        default=str(RUNTIME_ROOT / "infer_logs" / "dataset_policy_check"),
    )
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = GrootPolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
    )
    if not client.ping():
        raise PolicyServerError(f"cannot ping policy server {args.policy_host}:{args.policy_port}")
    modality_configs = client.get_modality_config()
    action_horizon = len(modality_configs["action"].delta_indices)
    embodiment = EmbodimentTag.resolve(args.embodiment_tag)

    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    samples = _resolve_samples(
        loader,
        action_horizon=action_horizon,
        explicit_samples=args.sample,
        episodes=_parse_ints(args.episodes),
        fractions=_parse_floats(args.fractions),
        max_samples=args.max_samples,
        allow_padding=args.allow_padding,
    )
    if not samples:
        raise ValueError("no valid samples selected")

    run_dir = Path(args.output_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    results: list[SampleResult] = []
    for sample_idx, (episode_index, step_index) in enumerate(samples):
        print(f"[dataset-check] sample {sample_idx}: episode={episode_index}, step={step_index}")
        result = _evaluate_sample(
            client=client,
            loader=loader,
            modality_configs=modality_configs,
            embodiment=embodiment,
            episode_index=episode_index,
            step_index=step_index,
            horizon=action_horizon,
            run_dir=run_dir,
            sample_idx=sample_idx,
            task_override=args.task,
            allow_padding=args.allow_padding,
            save_images=args.save_images,
        )
        results.append(result)
        print(_format_result_line(result))

    summary = {
        "dataset_path": args.dataset_path,
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "embodiment_tag": embodiment.value,
        "action_horizon": action_horizon,
        "modality": {
            name: {
                "keys": list(cfg.modality_keys),
                "delta_indices": list(cfg.delta_indices),
                "action_configs": [str(c) for c in cfg.action_configs or []],
            }
            for name, cfg in modality_configs.items()
        },
        "results": [_to_jsonable(asdict(result)) for result in results],
        "aggregate": _aggregate(results),
    }
    _write_json(run_dir / "summary.json", summary)
    print(f"[dataset-check] summary: {run_dir / 'summary.json'}")
    print("[dataset-check] aggregate:")
    for key, value in summary["aggregate"].items():
        print(f"  {key}: {value:.6f}")
    return 0


def _evaluate_sample(
    *,
    client: GrootPolicyClient,
    loader: LeRobotEpisodeLoader,
    modality_configs: dict[str, Any],
    embodiment: EmbodimentTag,
    episode_index: int,
    step_index: int,
    horizon: int,
    run_dir: Path,
    sample_idx: int,
    task_override: str | None,
    allow_padding: bool,
    save_images: bool,
) -> SampleResult:
    episode_meta = loader.episodes_metadata[episode_index]
    dataset_episode_index = int(episode_meta["episode_index"])
    episode_name = episode_meta.get("episode_name")
    df = loader._load_parquet_data(dataset_episode_index)
    if "language" in modality_configs:
        lang_key = modality_configs["language"].modality_keys[0]
        if lang_key in LANG_KEYS:
            df["language." + lang_key] = loader.create_language_from_meta(
                episode_meta,
                len(df),
                lang_key,
            )
    actual_length = min(len(df), int(episode_meta["length"]))
    df = df.iloc[:actual_length]

    video_indices = _indices_for(step_index, modality_configs["video"].delta_indices, actual_length, allow_padding)
    state_indices = _indices_for(step_index, modality_configs["state"].delta_indices, actual_length, allow_padding)
    action_indices = _indices_for(step_index, modality_configs["action"].delta_indices, actual_length, allow_padding)

    images = loader._load_video_data(dataset_episode_index, np.asarray(video_indices, dtype=np.int64))
    observation = {
        "video": {
            key: np.stack([np.asarray(frame, dtype=np.uint8) for frame in images[key]], axis=0)[
                None, ...
            ]
            for key in modality_configs["video"].modality_keys
        },
        "state": {
            key: _stack_numeric_column(df, f"state.{key}", state_indices)[None, ...]
            for key in modality_configs["state"].modality_keys
        },
        "language": {},
    }
    language_key = modality_configs["language"].modality_keys[0]
    task = task_override or _read_language(df, language_key, step_index, episode_meta)
    observation["language"][language_key] = [[task]]

    gt_chunk = np.concatenate(
        [
            _stack_numeric_column(df, f"action.{key}", action_indices)
            for key in modality_configs["action"].modality_keys
        ],
        axis=1,
    ).astype(np.float32)
    pred_chunk = client.predict_action_chunk(observation, min_horizon=horizon)[:horizon]
    schema.validate_flat_vector(pred_chunk[0], dim=schema.ACTION_DIM, name="pred[0]")
    if pred_chunk.shape != gt_chunk.shape:
        raise ValueError(f"pred/gt shape mismatch: pred={pred_chunk.shape}, gt={gt_chunk.shape}")

    sample_dir = run_dir / f"sample_{sample_idx:03d}_ep{episode_index:06d}_step{step_index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=False)
    pred_path = sample_dir / "pred_action.npy"
    gt_path = sample_dir / "gt_action.npy"
    state_path = sample_dir / "input_state.npy"
    np.save(pred_path, pred_chunk)
    np.save(gt_path, gt_chunk)
    input_state = np.concatenate(
        [
            observation["state"][key][0, -1]
            for key in modality_configs["state"].modality_keys
        ],
        axis=0,
    ).astype(np.float32)
    np.save(state_path, input_state)
    if save_images:
        from PIL import Image

        image_dir = sample_dir / "images"
        image_dir.mkdir(exist_ok=True)
        for key, arr in observation["video"].items():
            Image.fromarray(arr[0, -1].astype(np.uint8), mode="RGB").save(image_dir / f"{key}.png")

    result = SampleResult(
        episode_index=episode_index,
        dataset_episode_index=dataset_episode_index,
        episode_name=episode_name,
        step_index=step_index,
        task=task,
        horizon=horizon,
        policy_latency_ms=client.last_latency_ms,
        overall=_metrics(pred_chunk, gt_chunk),
        segments={name: _metrics(pred_chunk[:, sl], gt_chunk[:, sl]) for name, sl in SEGMENTS.items()},
        pred_action_path=str(pred_path),
        gt_action_path=str(gt_path),
        input_state_path=str(state_path),
    )
    _write_json(sample_dir / "metrics.json", _to_jsonable(asdict(result)))
    return result


def _resolve_samples(
    loader: LeRobotEpisodeLoader,
    *,
    action_horizon: int,
    explicit_samples: list[str],
    episodes: list[int],
    fractions: list[float],
    max_samples: int | None,
    allow_padding: bool,
) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    if explicit_samples:
        for spec in explicit_samples:
            if ":" not in spec:
                raise ValueError(f"--sample must be EPISODE:STEP, got {spec!r}")
            ep, step = spec.split(":", 1)
            samples.append((int(ep), int(step)))
    else:
        for episode_index in episodes:
            if episode_index < 0 or episode_index >= len(loader):
                continue
            length = loader.get_episode_length(episode_index)
            max_start = length - action_horizon
            if max_start < 0 and not allow_padding:
                continue
            usable = max(0, max_start)
            for fraction in fractions:
                step = int(round(float(np.clip(fraction, 0.0, 1.0)) * usable))
                samples.append((episode_index, step))

    unique = []
    seen = set()
    for ep, step in samples:
        if ep < 0 or ep >= len(loader):
            raise IndexError(f"episode {ep} out of range 0..{len(loader)-1}")
        length = loader.get_episode_length(ep)
        if not allow_padding and step + action_horizon > length:
            raise ValueError(
                f"sample {ep}:{step} needs horizon {action_horizon}, but episode length is {length}"
            )
        if (ep, step) not in seen:
            unique.append((ep, step))
            seen.add((ep, step))
        if max_samples is not None and len(unique) >= max_samples:
            break
    return unique


def _indices_for(
    step_index: int,
    delta_indices: list[int],
    length: int,
    allow_padding: bool,
) -> list[int]:
    indices = [step_index + delta for delta in delta_indices]
    if allow_padding:
        return [max(0, min(idx, length - 1)) for idx in indices]
    bad = [idx for idx in indices if idx < 0 or idx >= length]
    if bad:
        raise IndexError(f"indices out of range for length {length}: {bad}")
    return indices


def _stack_numeric_column(df, column: str, indices: list[int]) -> np.ndarray:
    values = df[column].iloc[indices]
    return np.vstack([np.asarray(value, dtype=np.float32) for value in values])


def _read_language(df, language_key: str, step_index: int, episode_meta: dict[str, Any]) -> str:
    column = f"language.{language_key}"
    if column in df.columns:
        value = df[column].iloc[step_index]
        if isinstance(value, str):
            return value
    tasks = episode_meta.get("tasks") or []
    if tasks:
        return str(tasks[0])
    raise ValueError("no task text available; pass --task")


def _metrics(pred: np.ndarray, gt: np.ndarray) -> SegmentMetrics:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    diff = pred - gt
    first = diff[0]
    return SegmentMetrics(
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff**2))),
        max_abs=float(np.max(np.abs(diff))),
        first_step_mae=float(np.mean(np.abs(first))),
        first_step_rmse=float(np.sqrt(np.mean(first**2))),
        pred_first=pred[0].astype(float).tolist(),
        gt_first=gt[0].astype(float).tolist(),
        diff_first=first.astype(float).tolist(),
    )


def _aggregate(results: list[SampleResult]) -> dict[str, float]:
    out: dict[str, float] = {
        "overall_mae": float(np.mean([r.overall.mae for r in results])),
        "overall_rmse": float(np.mean([r.overall.rmse for r in results])),
        "overall_first_step_mae": float(np.mean([r.overall.first_step_mae for r in results])),
    }
    for segment in SEGMENTS:
        out[f"{segment}_mae"] = float(np.mean([r.segments[segment].mae for r in results]))
        out[f"{segment}_first_step_mae"] = float(
            np.mean([r.segments[segment].first_step_mae for r in results])
        )
    return out


def _format_result_line(result: SampleResult) -> str:
    parts = [
        f"overall_mae={result.overall.mae:.4f}",
        f"first_mae={result.overall.first_step_mae:.4f}",
    ]
    for name in SEGMENTS:
        parts.append(f"{name}_mae={result.segments[name].mae:.4f}")
    return "[dataset-check] " + ", ".join(parts)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
