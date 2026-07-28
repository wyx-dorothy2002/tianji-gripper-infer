"""Structured robot state for the dual-arm dual-hand runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import schema


class RobotStateError(ValueError):
    """Raised when a structured robot state is malformed."""


@dataclass
class DualArmHandState:
    """Segmented state matching the canonical 54-DoF dataset schema."""

    left_arm_q: np.ndarray
    right_arm_q: np.ndarray
    left_hand_q: np.ndarray
    right_hand_q: np.ndarray

    def __post_init__(self) -> None:
        self.left_arm_q = _as_segment(self.left_arm_q, schema.LEFT_ARM_DOF, "left_arm_q")
        self.right_arm_q = _as_segment(self.right_arm_q, schema.RIGHT_ARM_DOF, "right_arm_q")
        self.left_hand_q = _as_segment(self.left_hand_q, schema.LEFT_HAND_DOF, "left_hand_q")
        self.right_hand_q = _as_segment(self.right_hand_q, schema.RIGHT_HAND_DOF, "right_hand_q")

    @classmethod
    def from_flat(cls, state_54: np.ndarray) -> "DualArmHandState":
        arr = schema.validate_flat_vector(state_54, dim=schema.STATE_DIM, name="state_54")
        return cls(
            left_arm_q=arr[schema.LEFT_ARM_SLICE],
            right_arm_q=arr[schema.RIGHT_ARM_SLICE],
            left_hand_q=arr[schema.LEFT_HAND_SLICE],
            right_hand_q=arr[schema.RIGHT_HAND_SLICE],
        )

    def as_flat(self) -> np.ndarray:
        return np.concatenate(
            [
                self.left_arm_q,
                self.right_arm_q,
                self.left_hand_q,
                self.right_hand_q,
            ],
            axis=0,
        ).astype(np.float32)

    def copy(self) -> "DualArmHandState":
        return DualArmHandState(
            left_arm_q=self.left_arm_q.copy(),
            right_arm_q=self.right_arm_q.copy(),
            left_hand_q=self.left_hand_q.copy(),
            right_hand_q=self.right_hand_q.copy(),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "left_arm": self.left_arm_q.tolist(),
            "right_arm": self.right_arm_q.tolist(),
            "left_hand": self.left_hand_q.tolist(),
            "right_hand": self.right_hand_q.tolist(),
        }


def ensure_state(value: DualArmHandState | np.ndarray) -> DualArmHandState:
    if isinstance(value, DualArmHandState):
        return value
    return DualArmHandState.from_flat(value)


def _as_segment(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (dim,):
        raise RobotStateError(f"{name} must have shape ({dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise RobotStateError(f"{name} contains NaN or Inf")
    return arr
