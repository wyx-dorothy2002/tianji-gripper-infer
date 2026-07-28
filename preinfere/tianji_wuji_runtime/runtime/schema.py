"""Canonical 54-DoF state/action schema for Tianji Wuji runtime.

This schema is aligned with:
  /mnt/data/qdhe/workspace/datasets/local_lerobot_dataset/meta/modality.json

Order:
  [0:7]   left_arm_joint
  [7:14]  right_arm_joint
  [14:34] left_hand
  [34:54] right_hand
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


LEFT_ARM_DOF = 7
LEFT_HAND_DOF = 20
RIGHT_ARM_DOF = 7
RIGHT_HAND_DOF = 20

STATE_DIM = LEFT_ARM_DOF + RIGHT_ARM_DOF + LEFT_HAND_DOF + RIGHT_HAND_DOF
ACTION_DIM = STATE_DIM

LEFT_ARM_SLICE = slice(0, LEFT_ARM_DOF)
RIGHT_ARM_SLICE = slice(LEFT_ARM_DOF, LEFT_ARM_DOF + RIGHT_ARM_DOF)
LEFT_HAND_SLICE = slice(LEFT_ARM_DOF + RIGHT_ARM_DOF, LEFT_ARM_DOF + RIGHT_ARM_DOF + LEFT_HAND_DOF)
RIGHT_HAND_SLICE = slice(
    LEFT_ARM_DOF + RIGHT_ARM_DOF + LEFT_HAND_DOF,
    LEFT_ARM_DOF + RIGHT_ARM_DOF + LEFT_HAND_DOF + RIGHT_HAND_DOF,
)

ACTION_SCHEMA_VERSION = "tianji_dual_arm_dexterous_hand_v1"
ACTION_ORDER_VERSION = ACTION_SCHEMA_VERSION

# Runtime units. These must match the trained checkpoint and the hardware controller.
# The current Tianji host-runtime integration standardizes on degrees.
STATE_UNIT = "deg"
ACTION_UNIT = "deg"
ACTION_MODE = "absolute"  # "absolute" joint targets unless explicitly changed.

LEFT_ARM_KEYS = [f"left_joint_{i}.pos" for i in range(1, LEFT_ARM_DOF + 1)]
RIGHT_ARM_KEYS = [f"right_joint_{i}.pos" for i in range(1, RIGHT_ARM_DOF + 1)]
LEFT_HAND_KEYS = [
    f"left_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]
RIGHT_HAND_KEYS = [
    f"right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]

STATE_KEYS = LEFT_ARM_KEYS + RIGHT_ARM_KEYS + LEFT_HAND_KEYS + RIGHT_HAND_KEYS
ACTION_KEYS = STATE_KEYS.copy()


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    dof: int
    vector_slice: slice
    keys: tuple[str, ...]


SEGMENTS: Mapping[str, SegmentSpec] = {
    "left_arm": SegmentSpec("left_arm", LEFT_ARM_DOF, LEFT_ARM_SLICE, tuple(LEFT_ARM_KEYS)),
    "right_arm": SegmentSpec("right_arm", RIGHT_ARM_DOF, RIGHT_ARM_SLICE, tuple(RIGHT_ARM_KEYS)),
    "left_hand": SegmentSpec("left_hand", LEFT_HAND_DOF, LEFT_HAND_SLICE, tuple(LEFT_HAND_KEYS)),
    "right_hand": SegmentSpec("right_hand", RIGHT_HAND_DOF, RIGHT_HAND_SLICE, tuple(RIGHT_HAND_KEYS)),
}

GROUP_ALIASES: Mapping[str, str] = {
    "left_arm": "left_arm",
    "left_arm_joint": "left_arm",
    "left_arm_q": "left_arm",
    "left_arm_joint_position": "left_arm",
    "right_arm": "right_arm",
    "right_arm_joint": "right_arm",
    "right_arm_q": "right_arm",
    "right_arm_joint_position": "right_arm",
    "left_hand": "left_hand",
    "left_hand_q": "left_hand",
    "left_hand_joint_position": "left_hand",
    "right_hand": "right_hand",
    "right_hand_q": "right_hand",
    "right_hand_joint_position": "right_hand",
}


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


def schema_metadata() -> dict[str, object]:
    return {
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_order_version": ACTION_ORDER_VERSION,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_unit": STATE_UNIT,
        "action_unit": ACTION_UNIT,
        "action_mode": ACTION_MODE,
        "order": {
            "left_arm": [LEFT_ARM_SLICE.start, LEFT_ARM_SLICE.stop],
            "right_arm": [RIGHT_ARM_SLICE.start, RIGHT_ARM_SLICE.stop],
            "left_hand": [LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop],
            "right_hand": [RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop],
        },
        "state_keys": STATE_KEYS,
        "action_keys": ACTION_KEYS,
    }
