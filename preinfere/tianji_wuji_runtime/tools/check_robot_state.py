#!/usr/bin/env python3
"""Read the structured robot state once and print the canonical flat view."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.robot_interface import RobotConnectionConfig, make_robot


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-backend", default="fake")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--left-hand-serial", default=None)
    parser.add_argument("--right-hand-serial", default=None)
    parser.add_argument("--left-hand-home", default=None)
    parser.add_argument("--right-hand-home", default=None)
    parser.add_argument("--hand-lowpass-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--tianji-sdk-root", default=None)
    parser.add_argument("--tianji-config-path", default=None)
    args = parser.parse_args()
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
    robot.connect()
    try:
        state = robot.get_state()
        flat = state.as_flat()
        print("structured segments:")
        for name, values in state.as_dict().items():
            print(f"  {name}: len={len(values)}")
        print(f"flat state shape: {flat.shape}")
        print(flat)
    finally:
        robot.hold_position()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
