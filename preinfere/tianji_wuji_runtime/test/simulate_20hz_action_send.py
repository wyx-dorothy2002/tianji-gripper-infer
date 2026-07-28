#!/usr/bin/env python3
"""Simulate sending a saved [T,54] action chunk at a fixed control frequency."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.action_adapter import ActionAdapter, DualArmHandAction
from tianji_wuji_runtime.runtime.robot_interface import RobotConnectionConfig, make_robot
from tianji_wuji_runtime.runtime.safety import SafetyConfig, SafetyLayer


DEFAULT_ACTION_FILE = (
    RUNTIME_ROOT
    / "test"
    / "action_replay"
    / "episode_000011_full"
    / "action_chunk.npy"
)


def _parse_optional_joint_list(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size != schema.LEFT_HAND_DOF:
        raise ValueError(
            f"hand home pose must provide {schema.LEFT_HAND_DOF} comma-separated values, "
            f"got {values.size}"
        )
    return tuple(float(v) for v in values.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-file", default=str(DEFAULT_ACTION_FILE))
    parser.add_argument(
        "--robot-limits",
        default=str(RUNTIME_ROOT / "configs" / "robot_limits.yaml"),
    )
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--robot-backend", default="fake")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--left-hand-serial", default=None)
    parser.add_argument("--right-hand-serial", default=None)
    parser.add_argument("--left-hand-home", default=None)
    parser.add_argument("--right-hand-home", default=None)
    parser.add_argument("--hand-lowpass-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--tianji-sdk-root", default=None)
    parser.add_argument("--tianji-config-path", default=None)
    parser.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    parser.add_argument(
        "--with-safety",
        action="store_true",
        help="Apply the YAML-backed runtime safety layer before sending actions.",
    )
    parser.add_argument("--max-arm-joint-step", type=float, default=None)
    parser.add_argument("--max-hand-joint-step", type=float, default=None)
    parser.add_argument("--max-arm-velocity", type=float, default=None)
    parser.add_argument("--max-hand-velocity", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default=str(RUNTIME_ROOT / "test" / "action_replay" / "send_logs"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hz <= 0:
        raise ValueError("--hz must be positive")
    if args.start_step < 0:
        raise ValueError("--start-step must be non-negative")

    dt = 1.0 / float(args.hz)
    raw_chunk = _load_action_chunk(Path(args.action_file), args.start_step, args.max_steps)
    adapter = ActionAdapter(action_mode=args.action_mode)
    robot = make_robot(
        RobotConnectionConfig(
            backend=args.robot_backend,
            robot_ip=args.robot_ip,
            left_hand_serial=args.left_hand_serial,
            right_hand_serial=args.right_hand_serial,
            left_hand_home=_parse_optional_joint_list(args.left_hand_home),
            right_hand_home=_parse_optional_joint_list(args.right_hand_home),
            hand_lowpass_cutoff_hz=args.hand_lowpass_cutoff_hz,
            tianji_sdk_root=args.tianji_sdk_root,
            tianji_config_path=args.tianji_config_path,
        )
    )

    actions = adapter.split_chunk(raw_chunk)
    safety_events: list[dict[str, object]] = []

    run_dir = Path(args.output_dir).expanduser().resolve() / datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "send_trace.jsonl"

    robot.connect()
    try:
        state_before_chunk = robot.get_state()
        send_actions = actions
        if args.with_safety:
            safety_config = SafetyConfig.from_yaml(args.robot_limits)
            if args.max_arm_joint_step is not None:
                safety_config.arm_max_step = args.max_arm_joint_step
            if args.max_hand_joint_step is not None:
                safety_config.hand_max_step = args.max_hand_joint_step
            if args.max_arm_velocity is not None:
                safety_config.arm_max_velocity = np.full(
                    schema.LEFT_ARM_DOF,
                    args.max_arm_velocity,
                    dtype=np.float32,
                )
            if args.max_hand_velocity is not None:
                safety_config.hand_max_velocity = np.full(
                    schema.LEFT_HAND_DOF,
                    args.max_hand_velocity,
                    dtype=np.float32,
                )
            safety = SafetyLayer(safety_config, adapter)
            send_actions, safety_events = safety.process_chunk(
                state_before_chunk,
                actions,
                dt,
            )

        send_times = _send_at_fixed_rate(
            robot=robot,
            actions=send_actions,
            raw_chunk=raw_chunk,
            trace_path=trace_path,
            adapter=adapter,
            dt=dt,
            source_start_step=args.start_step,
        )
    finally:
        robot.hold_position()
        robot.disconnect()

    summary = _summary(
        args=args,
        dt=dt,
        raw_chunk=raw_chunk,
        send_times=send_times,
        safety_events=safety_events,
        trace_path=trace_path,
    )
    _write_json(run_dir / "summary.json", summary)

    print(f"[send-sim] action file: {args.action_file}")
    print(f"[send-sim] action shape: {raw_chunk.shape}")
    print(f"[send-sim] target hz: {args.hz:.3f}, target dt: {dt * 1000.0:.3f} ms")
    print(f"[send-sim] sent steps: {len(send_times)}")
    print(f"[send-sim] mean interval: {summary['timing']['mean_interval_ms']:.3f} ms")
    print(f"[send-sim] max interval: {summary['timing']['max_interval_ms']:.3f} ms")
    print(f"[send-sim] overruns: {summary['timing']['overrun_count']}")
    print(f"[send-sim] with_safety: {args.with_safety}, safety_events: {len(safety_events)}")
    print(f"[send-sim] logs: {run_dir}")
    return 0


def _load_action_chunk(path: Path, start_step: int, max_steps: int | None) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != schema.ACTION_DIM:
        raise ValueError(f"action file must have shape [T,54], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("action file contains NaN or Inf")
    if start_step >= arr.shape[0]:
        raise ValueError(f"start step {start_step} is outside action length {arr.shape[0]}")
    end_step = arr.shape[0] if max_steps is None else min(arr.shape[0], start_step + max_steps)
    return arr[start_step:end_step].astype(np.float32, copy=False)


def _send_at_fixed_rate(
    *,
    robot,
    actions: list[DualArmHandAction],
    raw_chunk: np.ndarray,
    trace_path: Path,
    adapter: ActionAdapter,
    dt: float,
    source_start_step: int,
) -> list[float]:
    send_times: list[float] = []
    loop_start = time.perf_counter()
    next_deadline = loop_start

    for step_idx, action in enumerate(actions):
        now = time.perf_counter()
        sleep_s = next_deadline - now
        if sleep_s > 0:
            time.sleep(sleep_s)

        step_start = time.perf_counter()
        state_before = robot.get_state()
        robot.send_action(action)
        state_after = robot.get_state()
        step_end = time.perf_counter()
        send_times.append(step_start)

        safe_flat = adapter.merge_action(action)
        _append_jsonl(
            trace_path,
            {
                "source_step": source_start_step + step_idx,
                "step_in_run": step_idx,
                "monotonic_time": step_start,
                "elapsed_since_start_ms": (step_start - loop_start) * 1000.0,
                "send_latency_ms": (step_end - step_start) * 1000.0,
                "raw_action": raw_chunk[step_idx].tolist(),
                "sent_action": safe_flat.tolist(),
                "sent_segments": _split_to_lists(safe_flat),
                "state_before": state_before.as_flat().tolist(),
                "state_after": state_after.as_flat().tolist(),
            },
        )
        next_deadline += dt

    return send_times


def _summary(
    *,
    args: argparse.Namespace,
    dt: float,
    raw_chunk: np.ndarray,
    send_times: list[float],
    safety_events: list[dict[str, object]],
    trace_path: Path,
) -> dict[str, Any]:
    intervals = np.diff(np.asarray(send_times, dtype=np.float64))
    if len(intervals) == 0:
        interval_stats = {
            "mean_interval_ms": 0.0,
            "min_interval_ms": 0.0,
            "max_interval_ms": 0.0,
            "std_interval_ms": 0.0,
            "overrun_count": 0,
        }
    else:
        tolerance = 0.002
        interval_stats = {
            "mean_interval_ms": float(np.mean(intervals) * 1000.0),
            "min_interval_ms": float(np.min(intervals) * 1000.0),
            "max_interval_ms": float(np.max(intervals) * 1000.0),
            "std_interval_ms": float(np.std(intervals) * 1000.0),
            "overrun_count": int(np.sum(intervals > dt + tolerance)),
        }
    return {
        "action_file": args.action_file,
        "action_shape": list(raw_chunk.shape),
        "start_step": args.start_step,
        "max_steps": args.max_steps,
        "target_hz": args.hz,
        "target_dt_ms": dt * 1000.0,
        "robot_backend": args.robot_backend,
        "action_mode": args.action_mode,
        "with_safety": args.with_safety,
        "safety_event_count": len(safety_events),
        "trace_path": str(trace_path),
        "timing": interval_stats,
        "schema": schema.schema_metadata(),
    }


def _split_to_lists(action_54: np.ndarray) -> dict[str, list[float]]:
    return {
        "left_arm": action_54[schema.LEFT_ARM_SLICE].tolist(),
        "right_arm": action_54[schema.RIGHT_ARM_SLICE].tolist(),
        "left_hand": action_54[schema.LEFT_HAND_SLICE].tolist(),
        "right_hand": action_54[schema.RIGHT_HAND_SLICE].tolist(),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
