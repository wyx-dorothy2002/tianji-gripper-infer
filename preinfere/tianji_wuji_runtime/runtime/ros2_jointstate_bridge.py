"""System-Python ROS 2 JointState bridge for PlotJuggler.

Reads JSON lines from stdin:
{"target": [...14 floats...], "position": [...14 floats...]}
and publishes:
  /gr00t_runtime/target
  /gr00t_runtime/position
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


JOINT_NAMES = [
    *[f"left_arm_joint_{idx}" for idx in range(1, 8)],
    *[f"right_arm_joint_{idx}" for idx in range(1, 8)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-prefix", default="/gr00t_runtime")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init(args=None)
    node = rclpy.create_node("gr00t_runtime_jointstate_bridge")
    prefix = args.topic_prefix.rstrip("/")
    target_pub = node.create_publisher(JointState, f"{prefix}/target", 10)
    position_pub = node.create_publisher(JointState, f"{prefix}/position", 10)
    print(
        f"[ros2-jointstate-bridge] publishing {prefix}/target and {prefix}/position",
        file=sys.stderr,
        flush=True,
    )

    try:
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                target = _coerce_positions(payload.get("target"))
                position = _coerce_positions(payload.get("position"))
            except Exception as exc:  # noqa: BLE001
                print(f"[ros2-jointstate-bridge] dropped bad sample: {exc}", file=sys.stderr)
                continue

            stamp = node.get_clock().now().to_msg()
            target_pub.publish(_make_joint_state(JointState, target, stamp))
            position_pub.publish(_make_joint_state(JointState, position, stamp))
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def _coerce_positions(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} joint values")
    return [float(item) for item in value]


def _make_joint_state(joint_state_type: type, positions: list[float], stamp: object) -> object:
    msg = joint_state_type()
    msg.header.stamp = stamp
    msg.name = list(JOINT_NAMES)
    msg.position = positions
    return msg


if __name__ == "__main__":
    raise SystemExit(main())
