#!/usr/bin/env python3
"""One-shot dry-run inference without sending hardware commands."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_wuji_runtime.runtime.action_adapter import ActionAdapter
from tianji_wuji_runtime.runtime.camera_manager import CameraManager
from tianji_wuji_runtime.runtime.groot_policy_client import GrootPolicyClient, PolicyServerError
from tianji_wuji_runtime.runtime.observation_builder import ObservationBuilder
from tianji_wuji_runtime.runtime.recorder import Recorder
from tianji_wuji_runtime.runtime.robot_interface import RobotConnectionConfig, make_robot
from tianji_wuji_runtime.runtime.safety import SafetyConfig, SafetyLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-limits",
        default=str(RUNTIME_ROOT / "configs" / "robot_limits.yaml"),
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--policy-timeout-ms", type=int, default=15000)
    parser.add_argument("--task", required=True)
    parser.add_argument("--state-source", choices=["fake"], default="fake")
    parser.add_argument("--image-source", default=None)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--record-dir", default=str(RUNTIME_ROOT / "infer_logs" / "dry_run"))
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--max-arm-joint-step", type=float, default=10.0)
    parser.add_argument("--max-hand-joint-step", type=float, default=4.5)
    parser.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = ActionAdapter(action_mode=args.action_mode)
    policy = GrootPolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
    )
    if not policy.ping():
        raise PolicyServerError(f"cannot ping policy server {args.policy_host}:{args.policy_port}")
    modality_configs = policy.get_modality_config()
    obs_builder = ObservationBuilder(modality_configs)
    cameras = CameraManager.from_cli_specs(
        args.camera,
        required_keys=obs_builder.required_camera_keys,
        image_source=args.image_source,
        allow_dummy=True,
    )
    robot = make_robot(RobotConnectionConfig(backend="fake"))
    safety_config = SafetyConfig.from_yaml(args.robot_limits)
    safety_config.arm_max_step = args.max_arm_joint_step
    safety_config.hand_max_step = args.max_hand_joint_step
    safety = SafetyLayer(safety_config, adapter)
    recorder = Recorder(
        args.record_dir,
        config={
            "entrypoint": "dry_run_infer.py",
            "inference_mode": "sync",
            "policy_host": args.policy_host,
            "policy_port": args.policy_port,
            "task": args.task,
            "state_source": args.state_source,
            "image_source": args.image_source,
            "execution_horizon": args.execution_horizon,
            "duration": args.duration,
            "control_frequency_hz": 1.0 / args.duration,
        },
        adapter=adapter,
    )

    robot.connect()
    cameras.connect_all()
    try:
        state = robot.get_state()
        images = cameras.read()
        observation = obs_builder.build(state, images, args.task)
        raw_chunk = policy.predict_action_chunk(observation, min_horizon=args.execution_horizon)
        actions = adapter.split_chunk(raw_chunk)
        safe_actions, safety_events = safety.process_chunk(state, actions, args.duration)
        recorder.save_chunk(
            observation=observation,
            raw_chunk=raw_chunk,
            safe_actions=safe_actions,
            safety_events=safety_events,
            inference_latency_ms=policy.last_latency_ms,
        )
        first = safe_actions[0]
        print(f"[dry-run] action chunk shape: {raw_chunk.shape}")
        print(
            "[dry-run] split shapes: "
            f"left_arm={first.left_arm_q.shape}, right_arm={first.right_arm_q.shape}, "
            f"left_hand={first.left_hand_q.shape}, right_hand={first.right_hand_q.shape}"
        )
        print(f"[dry-run] safety events: {len(safety_events)}")
        print(f"[dry-run] logs: {recorder.run_dir}")
    finally:
        robot.hold_position()
        cameras.disconnect_all()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
