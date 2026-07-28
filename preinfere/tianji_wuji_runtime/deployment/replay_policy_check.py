#!/usr/bin/env python3
"""Replay a saved [H,54] action chunk through adapter/safety/executor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.action_adapter import ActionAdapter
from tianji_wuji_runtime.runtime.executor import ActionExecutor
from tianji_wuji_runtime.runtime.recorder import Recorder
from tianji_wuji_runtime.runtime.robot_interface import RobotConnectionConfig, make_robot
from tianji_wuji_runtime.runtime.safety import SafetyConfig, SafetyLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-limits",
        default=str(RUNTIME_ROOT / "configs" / "robot_limits.yaml"),
    )
    parser.add_argument("--action-file", required=True)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-dir", default=str(RUNTIME_ROOT / "infer_logs" / "replay"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-arm-joint-step", type=float, default=10.0)
    parser.add_argument("--max-hand-joint-step", type=float, default=4.5)
    parser.add_argument("--robot-backend", default="fake")
    parser.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_chunk = np.load(args.action_file)
    raw_chunk = np.asarray(raw_chunk, dtype=np.float32)
    if raw_chunk.ndim != 2 or raw_chunk.shape[1] != schema.ACTION_DIM:
        raise ValueError(f"action file must have shape [H,54], got {raw_chunk.shape}")
    if not np.all(np.isfinite(raw_chunk)):
        raise ValueError("action file contains NaN or Inf")

    adapter = ActionAdapter(action_mode=args.action_mode)
    robot = make_robot(RobotConnectionConfig(backend=args.robot_backend))
    safety_config = SafetyConfig.from_yaml(args.robot_limits)
    safety_config.arm_max_step = args.max_arm_joint_step
    safety_config.hand_max_step = args.max_hand_joint_step
    safety = SafetyLayer(safety_config, adapter)
    recorder = Recorder(
        args.record_dir,
        config={
            "entrypoint": "replay_policy_check.py",
            "inference_mode": "sync_replay",
            "action_file": args.action_file,
            "duration": args.duration,
            "control_frequency_hz": 1.0 / args.duration,
            "safe_mode": args.safe_mode,
            "dry_run": args.dry_run,
            "max_steps": args.max_steps,
        },
        adapter=adapter,
    )
    executor = ActionExecutor(robot, adapter=adapter, recorder=recorder)

    robot.connect()
    try:
        current_state = robot.get_state()
        current_state_flat = current_state.as_flat()
        actions = adapter.split_chunk(raw_chunk)
        safe_actions, safety_events = safety.process_chunk(current_state, actions, args.duration)
        recorder.save_chunk(
            observation={
                "state": {"replay_current_state": current_state_flat[None, None, :]},
                "video": {},
                "language": {"replay": [["replay policy check"]]},
            },
            raw_chunk=raw_chunk,
            safe_actions=safe_actions,
            safety_events=safety_events,
            inference_latency_ms=None,
        )
        n_steps = args.max_steps or (1 if args.safe_mode else len(safe_actions))
        executor.execute_chunk(
            safe_actions[:n_steps],
            args.duration,
            dry_run=args.dry_run,
            chunk_index=0,
            raw_chunk=raw_chunk,
            safety_events=safety_events,
        )
        print(f"[replay] action chunk shape: {raw_chunk.shape}")
        print(f"[replay] executed steps: {n_steps}, dry_run={args.dry_run}")
        print(f"[replay] safety events: {len(safety_events)}")
        print(f"[replay] logs: {recorder.run_dir}")
    finally:
        robot.hold_position()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
