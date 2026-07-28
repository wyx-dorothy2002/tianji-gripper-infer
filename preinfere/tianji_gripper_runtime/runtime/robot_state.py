"""Structured robot state for Tianji gripper runtimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import schema


class RobotStateError(ValueError):
    """Raised when a structured robot state is malformed."""


@dataclass
class RightArmGripperState:
    right_arm_q: np.ndarray
    right_gripper_q: np.ndarray
    left_arm_q: np.ndarray | None = None
    left_gripper_q: np.ndarray | None = None
    include_left: bool = False

    def __post_init__(self) -> None:
        self.right_arm_q = _as_segment(
            self.right_arm_q,
            schema.RIGHT_ARM_DOF,
            "right_arm_q",
        )
        self.right_gripper_q = _as_segment(
            self.right_gripper_q,
            schema.RIGHT_GRIPPER_DOF,
            "right_gripper_q",
        )
        if self.include_left and self.left_arm_q is None:
            raise RobotStateError("left_arm_q is required when include_left=True")
        if self.left_arm_q is None:
            self.left_arm_q = np.zeros(schema.LEFT_ARM_DOF, dtype=np.float32)
        self.left_arm_q = _as_segment(
            self.left_arm_q,
            schema.LEFT_ARM_DOF,
            "left_arm_q",
        )
        if self.include_left and self.left_gripper_q is None:
            raise RobotStateError("left_gripper_q is required when include_left=True")
        if self.left_gripper_q is None:
            self.left_gripper_q = np.zeros(schema.LEFT_GRIPPER_DOF, dtype=np.float32)
        self.left_gripper_q = _as_segment(
            self.left_gripper_q,
            schema.LEFT_GRIPPER_DOF,
            "left_gripper_q",
        )

    @classmethod
    def from_flat(cls, state: np.ndarray) -> "RightArmGripperState":
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.shape == (schema.RIGHT_ONLY_STATE_DIM,):
            return cls(
                right_arm_q=arr[schema.RIGHT_ARM_SLICE],
                right_gripper_q=arr[7:8],
                include_left=False,
            )
        if arr.shape == (schema.FULL_STATE_DIM,):
            return cls(
                right_arm_q=arr[schema.RIGHT_ARM_SLICE],
                left_arm_q=arr[schema.LEFT_ARM_SLICE],
                right_gripper_q=arr[schema.RIGHT_GRIPPER_SLICE],
                left_gripper_q=arr[schema.LEFT_GRIPPER_SLICE],
                include_left=True,
            )
        raise RobotStateError(
            f"state must have shape ({schema.RIGHT_ONLY_STATE_DIM},) or "
            f"({schema.FULL_STATE_DIM},), got {arr.shape}"
        )

    def as_flat(self, *, control_mode: str | None = None) -> np.ndarray:
        if control_mode is None:
            control_mode = "dual_arm_dual_gripper" if self.include_left else "right_arm_right_gripper"
        if control_mode == "dual_arm_dual_gripper":
            return np.concatenate(
                [
                    self.right_arm_q,
                    self.left_arm_q,
                    self.right_gripper_q,
                    self.left_gripper_q,
                ],
                axis=0,
            ).astype(np.float32)
        if control_mode == "right_arm_right_gripper":
            return np.concatenate([self.right_arm_q, self.right_gripper_q], axis=0).astype(np.float32)
        raise RobotStateError(f"unsupported control mode: {control_mode!r}")

    def copy(self) -> "RightArmGripperState":
        return RightArmGripperState(
            right_arm_q=self.right_arm_q.copy(),
            right_gripper_q=self.right_gripper_q.copy(),
            left_arm_q=self.left_arm_q.copy(),
            left_gripper_q=self.left_gripper_q.copy(),
            include_left=self.include_left,
        )

    def as_dict(self) -> dict[str, list[float]]:
        data = {
            "right_arm": self.right_arm_q.tolist(),
            "right_gripper": self.right_gripper_q.tolist(),
        }
        if self.include_left:
            data["left_arm"] = self.left_arm_q.tolist()
            data["left_gripper"] = self.left_gripper_q.tolist()
        return data


def ensure_state(value: RightArmGripperState | np.ndarray) -> RightArmGripperState:
    if isinstance(value, RightArmGripperState):
        return value
    return RightArmGripperState.from_flat(value)


def _as_segment(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (dim,):
        raise RobotStateError(f"{name} must have shape ({dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise RobotStateError(f"{name} contains NaN or Inf")
    return arr.copy()
