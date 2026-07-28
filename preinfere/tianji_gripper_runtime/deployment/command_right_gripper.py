#!/usr/bin/env python3
"""Command the right gripper directly without starting policy inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import yaml


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
DEFAULT_CONFIG_PATH = RUNTIME_ROOT / "configs" / "infer.yaml"

sys.path.insert(0, str(REPO_ROOT))

from tianji_gripper_runtime.runtime.robot_interface import RobotConnectionConfig, make_robot


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping, got {type(data)!r}")
    return data


def _get_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _get_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None:
        return int(default)
    if isinstance(value, str) and value.lower().startswith("0x"):
        return int(value, 16)
    return int(value)


def _parse_optional_scalar(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        arr = np.fromstring(value, sep=",", dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"expected one gripper scalar, got {arr.size}")
    return (float(arr[0]),)


def _build_robot_config(config: dict[str, object]) -> RobotConnectionConfig:
    return RobotConnectionConfig(
        backend=str(config.get("robot_backend", "fake")),
        robot_ip=config.get("robot_ip"),
        left_arm_ip=config.get("left_arm_ip"),
        left_arm_port=config.get("left_arm_port"),
        right_arm_ip=config.get("right_arm_ip"),
        right_arm_port=config.get("right_arm_port"),
        gripper_backend=str(config.get("gripper_backend", "fake")),
        left_gripper_backend="fake",
        right_gripper_backend=str(config.get("right_gripper_backend") or config.get("gripper_backend", "fake")),
        left_gripper_ip=config.get("left_gripper_ip"),
        left_gripper_port=config.get("left_gripper_port"),
        right_gripper_ip=config.get("right_gripper_ip"),
        right_gripper_port=config.get("right_gripper_port"),
        left_gripper_serial=config.get("left_gripper_serial"),
        right_gripper_serial=config.get("right_gripper_serial"),
        left_gripper_home=None,
        right_gripper_home=_parse_optional_scalar(config.get("right_gripper_home")),
        gripper_sdk_module=config.get("gripper_sdk_module"),
        gripper_sdk_class=config.get("gripper_sdk_class"),
        tianji_sdk_root=config.get("tianji_sdk_root"),
        tianji_config_path=config.get("tianji_config_path"),
        gripper_485_arm=str(config.get("gripper_485_arm", "A")),
        gripper_485_com=_get_int(config, "gripper_485_com", 1),
        gripper_direct_onset=bool(config.get("gripper_direct_onset", True)),
        gripper_rs05_target_id=_get_int(config, "gripper_rs05_target_id", 0x7F),
        gripper_rs05_master_id=_get_int(config, "gripper_rs05_master_id", 0xFD),
        gripper_can_id_byteorder=str(config.get("gripper_can_id_byteorder", "little")),
        gripper_standard_id_bytes=_get_int(config, "gripper_standard_id_bytes", 4),
        gripper_enter_motor=bool(config.get("gripper_enter_motor", True)),
        gripper_stop_on_disconnect=bool(config.get("gripper_stop_on_disconnect", False)),
        gripper_kp=_get_float(config, "gripper_kp", 80.0),
        gripper_kd=_get_float(config, "gripper_kd", 1.0),
        gripper_torque_nm=_get_float(config, "gripper_torque_nm", 0.0),
        gripper_min_pos_rad=_get_float(config, "gripper_min_pos_rad", -5.5),
        gripper_max_pos_rad=_get_float(config, "gripper_max_pos_rad", 1.2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--action", choices=["read", "close", "open", "move"], default="read")
    parser.add_argument("--target", type=float, default=None, help="Target gripper position in rad for --action move.")
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=1.0,
        help="Seconds to keep resending the command before reading feedback.",
    )
    parser.add_argument(
        "--command-hz",
        type=float,
        default=20.0,
        help="How often to resend the target while holding.",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config).expanduser())
    robot_config = _build_robot_config(config)
    robot = make_robot(robot_config)
    robot.connect()
    try:
        current = robot.right_gripper.get_position()
        print(f"right gripper current(rad): {current.tolist()}")

        if args.action == "read":
            return 0

        if args.action == "close":
            target = robot_config.gripper_min_pos_rad
        elif args.action == "open":
            target = robot_config.gripper_max_pos_rad
        else:
            if args.target is None:
                raise ValueError("--target is required when --action move")
            target = float(args.target)

        print(f"commanding right gripper target(rad): {target:.6f}")
        target_arr = np.asarray([target], dtype=np.float32)
        hold_sec = max(args.hold_sec, 0.0)
        period_sec = 0.05 if args.command_hz <= 0 else max(1.0 / args.command_hz, 0.01)
        deadline = time.monotonic() + hold_sec
        send_count = 0
        while True:
            robot.right_gripper.send_position(target_arr)
            send_count += 1
            if time.monotonic() >= deadline:
                break
            time.sleep(period_sec)
        after = robot.right_gripper.get_position()
        print(f"command frames sent: {send_count}")
        print(f"right gripper after(rad): {after.tolist()}")
    finally:
        robot.hold_position()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
