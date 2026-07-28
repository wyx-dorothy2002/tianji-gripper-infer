"""Canonical state/action layouts for Tianji gripper runtimes.

Supported control modes:
  - right_arm_right_gripper: right_arm(7) + right_gripper(1)
  - dual_arm_dual_gripper: right_arm(7) + left_arm(7) + right_gripper(1) + left_gripper(1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


RIGHT_ARM_DOF = 7
LEFT_ARM_DOF = 7
RIGHT_GRIPPER_DOF = 1
LEFT_GRIPPER_DOF = 1

RIGHT_ONLY_STATE_DIM = RIGHT_ARM_DOF + RIGHT_GRIPPER_DOF
FULL_STATE_DIM = RIGHT_ARM_DOF + LEFT_ARM_DOF + RIGHT_GRIPPER_DOF + LEFT_GRIPPER_DOF

STATE_DIM = RIGHT_ONLY_STATE_DIM
ACTION_DIM = RIGHT_ONLY_STATE_DIM

RIGHT_ARM_SLICE = slice(0, RIGHT_ARM_DOF)
LEFT_ARM_SLICE = slice(RIGHT_ARM_SLICE.stop, RIGHT_ARM_SLICE.stop + LEFT_ARM_DOF)
RIGHT_GRIPPER_SLICE = slice(LEFT_ARM_SLICE.stop, LEFT_ARM_SLICE.stop + RIGHT_GRIPPER_DOF)
LEFT_GRIPPER_SLICE = slice(RIGHT_GRIPPER_SLICE.stop, RIGHT_GRIPPER_SLICE.stop + LEFT_GRIPPER_DOF)

ACTION_SCHEMA_VERSION = "tianji_dual_arm_dual_gripper_v1"
ACTION_ORDER_VERSION = ACTION_SCHEMA_VERSION

STATE_UNIT = "deg"
ACTION_UNIT = "deg"
ACTION_MODE = "absolute"

RIGHT_ARM_KEYS = [f"right_joint_{i}.pos" for i in range(1, RIGHT_ARM_DOF + 1)]
LEFT_ARM_KEYS = [f"left_joint_{i}.pos" for i in range(1, LEFT_ARM_DOF + 1)]
RIGHT_GRIPPER_KEYS = ["right_gripper.pos"]
LEFT_GRIPPER_KEYS = ["left_gripper.pos"]

RIGHT_ONLY_STATE_KEYS = RIGHT_ARM_KEYS + RIGHT_GRIPPER_KEYS
FULL_STATE_KEYS = RIGHT_ARM_KEYS + LEFT_ARM_KEYS + RIGHT_GRIPPER_KEYS + LEFT_GRIPPER_KEYS


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    dof: int
    vector_slice: slice
    keys: tuple[str, ...]


SEGMENTS: Mapping[str, SegmentSpec] = {
    "right_arm": SegmentSpec("right_arm", RIGHT_ARM_DOF, RIGHT_ARM_SLICE, tuple(RIGHT_ARM_KEYS)),
    "left_arm": SegmentSpec("left_arm", LEFT_ARM_DOF, LEFT_ARM_SLICE, tuple(LEFT_ARM_KEYS)),
    "right_gripper": SegmentSpec(
        "right_gripper",
        RIGHT_GRIPPER_DOF,
        RIGHT_GRIPPER_SLICE,
        tuple(RIGHT_GRIPPER_KEYS),
    ),
    "left_gripper": SegmentSpec(
        "left_gripper",
        LEFT_GRIPPER_DOF,
        LEFT_GRIPPER_SLICE,
        tuple(LEFT_GRIPPER_KEYS),
    ),
}

GROUP_ALIASES: Mapping[str, str] = {
    "right_arm": "right_arm",
    "right_arm_q": "right_arm",
    "right_arm_joint": "right_arm",
    "right_arm_joint_position": "right_arm",
    "left_arm": "left_arm",
    "left_arm_q": "left_arm",
    "left_arm_joint": "left_arm",
    "left_arm_joint_position": "left_arm",
    "right_gripper": "right_gripper",
    "right_gripper_q": "right_gripper",
    "right_gripper_position": "right_gripper",
    "left_gripper": "left_gripper",
    "left_gripper_q": "left_gripper",
    "left_gripper_position": "left_gripper",
    "gripper": "right_gripper",
}


def state_dim_for_mode(control_mode: str) -> int:
    if control_mode == "dual_arm_dual_gripper":
        return FULL_STATE_DIM
    if control_mode == "right_arm_right_gripper":
        return RIGHT_ONLY_STATE_DIM
    raise ValueError(f"unsupported control mode: {control_mode!r}")


def action_dim_for_mode(control_mode: str) -> int:
    return state_dim_for_mode(control_mode)


def state_keys_for_mode(control_mode: str) -> list[str]:
    if control_mode == "dual_arm_dual_gripper":
        return FULL_STATE_KEYS.copy()
    if control_mode == "right_arm_right_gripper":
        return RIGHT_ONLY_STATE_KEYS.copy()
    raise ValueError(f"unsupported control mode: {control_mode!r}")


def validate_flat_vector(
    value: np.ndarray,
    *,
    dim: int,
    name: str,
    finite: bool = True,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
    if finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def schema_metadata(*, control_mode: str = "right_arm_right_gripper") -> dict[str, object]:
    return {
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_order_version": ACTION_ORDER_VERSION,
        "control_mode": control_mode,
        "state_dim": state_dim_for_mode(control_mode),
        "action_dim": action_dim_for_mode(control_mode),
        "full_state_dim": FULL_STATE_DIM,
        "right_only_state_dim": RIGHT_ONLY_STATE_DIM,
        "state_unit": STATE_UNIT,
        "action_unit": ACTION_UNIT,
        "action_mode": ACTION_MODE,
        "order": {
            "right_arm": [RIGHT_ARM_SLICE.start, RIGHT_ARM_SLICE.stop],
            "left_arm": [LEFT_ARM_SLICE.start, LEFT_ARM_SLICE.stop],
            "right_gripper": [RIGHT_GRIPPER_SLICE.start, RIGHT_GRIPPER_SLICE.stop],
            "left_gripper": [LEFT_GRIPPER_SLICE.start, LEFT_GRIPPER_SLICE.stop],
        },
        "state_keys": state_keys_for_mode(control_mode),
        "full_state_keys": FULL_STATE_KEYS.copy(),
    }
