"""Action conversion for Tianji right-arm / dual-arm gripper schemas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import schema
from .gripper_normalization import GripperCalibration
from .gripper_normalization import convert_gripper_units
from .robot_state import RightArmGripperState, ensure_state


class ActionAdapterError(RuntimeError):
    """Raised when a policy action cannot be adapted safely."""


@dataclass
class RightArmGripperAction:
    right_arm_q: np.ndarray
    right_gripper_q: np.ndarray
    left_arm_q: np.ndarray | None = None
    left_gripper_q: np.ndarray | None = None
    control_left_arm: bool = False
    control_left_gripper: bool = False

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
        if self.control_left_arm and self.left_arm_q is None:
            raise ActionAdapterError("left_arm_q is required when control_left_arm=True")
        if self.left_arm_q is None:
            self.left_arm_q = np.zeros(schema.LEFT_ARM_DOF, dtype=np.float32)
        self.left_arm_q = _as_segment(
            self.left_arm_q,
            schema.LEFT_ARM_DOF,
            "left_arm_q",
        )
        if self.control_left_gripper and self.left_gripper_q is None:
            raise ActionAdapterError("left_gripper_q is required when control_left_gripper=True")
        if self.left_gripper_q is None:
            self.left_gripper_q = np.zeros(schema.LEFT_GRIPPER_DOF, dtype=np.float32)
        self.left_gripper_q = _as_segment(
            self.left_gripper_q,
            schema.LEFT_GRIPPER_DOF,
            "left_gripper_q",
        )

    def copy(self) -> "RightArmGripperAction":
        return RightArmGripperAction(
            right_arm_q=self.right_arm_q.copy(),
            right_gripper_q=self.right_gripper_q.copy(),
            left_arm_q=self.left_arm_q.copy(),
            left_gripper_q=self.left_gripper_q.copy(),
            control_left_arm=self.control_left_arm,
            control_left_gripper=self.control_left_gripper,
        )

    def as_dict(self) -> dict[str, list[float] | bool]:
        return {
            "right_arm": self.right_arm_q.tolist(),
            "left_arm": self.left_arm_q.tolist(),
            "right_gripper": self.right_gripper_q.tolist(),
            "left_gripper": self.left_gripper_q.tolist(),
            "control_left_arm": self.control_left_arm,
            "control_left_gripper": self.control_left_gripper,
        }

    def control_mode(self) -> str:
        if self.control_left_arm or self.control_left_gripper:
            return "dual_arm_dual_gripper"
        return "right_arm_right_gripper"


def _as_segment(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (dim,):
        raise ActionAdapterError(f"{name} must have shape ({dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ActionAdapterError(f"{name} contains NaN or Inf")
    return arr.copy()


def _convert_units(arr: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    if from_unit == to_unit:
        return arr.astype(np.float32, copy=True)
    if from_unit == "rad" and to_unit in {"deg", "degree", "degrees"}:
        return np.rad2deg(arr).astype(np.float32)
    if from_unit in {"deg", "degree", "degrees"} and to_unit == "rad":
        return np.deg2rad(arr).astype(np.float32)
    return arr.astype(np.float32, copy=True)


class ActionAdapter:
    """Split, merge, and unit-convert action vectors for the selected control mode."""

    def __init__(
        self,
        *,
        policy_unit: str = schema.ACTION_UNIT,
        control_unit: str = schema.ACTION_UNIT,
        policy_arm_unit: str | None = None,
        control_arm_unit: str | None = None,
        policy_gripper_unit: str | None = None,
        control_gripper_unit: str | None = None,
        action_mode: str = schema.ACTION_MODE,
        schema_version: str = schema.ACTION_SCHEMA_VERSION,
        control_mode: str = "right_arm_right_gripper",
        right_gripper_calibration: GripperCalibration | None = None,
        left_gripper_calibration: GripperCalibration | None = None,
    ) -> None:
        if action_mode not in {"absolute", "delta"}:
            raise ValueError(f"action_mode must be 'absolute' or 'delta', got {action_mode!r}")
        self.policy_unit = policy_unit
        self.control_unit = control_unit
        self.policy_arm_unit = policy_arm_unit or policy_unit
        self.control_arm_unit = control_arm_unit or control_unit
        self.policy_gripper_unit = policy_gripper_unit or policy_unit
        self.control_gripper_unit = control_gripper_unit or control_unit
        self.action_mode = action_mode
        self.schema_version = schema_version
        self.control_mode = control_mode
        self.right_gripper_calibration = right_gripper_calibration
        self.left_gripper_calibration = left_gripper_calibration
        self.action_dim = schema.action_dim_for_mode(control_mode)

    def split_action(self, action: np.ndarray) -> RightArmGripperAction:
        arr = schema.validate_flat_vector(action, dim=self.action_dim, name="action")
        if self.control_mode == "dual_arm_dual_gripper":
            return RightArmGripperAction(
                right_arm_q=_convert_units(
                    arr[schema.RIGHT_ARM_SLICE],
                    self.policy_arm_unit,
                    self.control_arm_unit,
                ),
                left_arm_q=_convert_units(
                    arr[schema.LEFT_ARM_SLICE],
                    self.policy_arm_unit,
                    self.control_arm_unit,
                ),
                right_gripper_q=convert_gripper_units(
                    arr[schema.RIGHT_GRIPPER_SLICE],
                    self.policy_gripper_unit,
                    self.control_gripper_unit,
                    self.right_gripper_calibration,
                ),
                left_gripper_q=convert_gripper_units(
                    arr[schema.LEFT_GRIPPER_SLICE],
                    self.policy_gripper_unit,
                    self.control_gripper_unit,
                    self.left_gripper_calibration,
                ),
                control_left_arm=True,
                control_left_gripper=True,
            )
        return RightArmGripperAction(
            right_arm_q=_convert_units(
                arr[0:7],
                self.policy_arm_unit,
                self.control_arm_unit,
            ),
            right_gripper_q=convert_gripper_units(
                arr[7:8],
                self.policy_gripper_unit,
                self.control_gripper_unit,
                self.right_gripper_calibration,
            ),
        )

    def split_state(self, state: RightArmGripperState | np.ndarray) -> RightArmGripperState:
        return ensure_state(state).copy()

    def split_chunk(self, action_chunk: np.ndarray) -> list[RightArmGripperAction]:
        arr = np.asarray(action_chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ActionAdapterError(f"action_chunk must be 2-D [H, D], got {arr.shape}")
        if arr.shape[1] != self.action_dim:
            raise ActionAdapterError(
                f"action_chunk second dimension must be {self.action_dim}, got {arr.shape[1]}"
            )
        return [self.split_action(step) for step in arr]

    def merge_action(self, action: RightArmGripperAction) -> np.ndarray:
        if action.control_left_arm or action.control_left_gripper:
            return np.concatenate(
                [
                    action.right_arm_q,
                    action.left_arm_q,
                    action.right_gripper_q,
                    action.left_gripper_q,
                ],
                axis=0,
            ).astype(np.float32)
        return np.concatenate([action.right_arm_q, action.right_gripper_q], axis=0).astype(
            np.float32
        )

    def merge_chunk(self, actions: list[RightArmGripperAction]) -> np.ndarray:
        if not actions:
            return np.empty((0, self.action_dim), dtype=np.float32)
        return np.stack([self.merge_action(action) for action in actions], axis=0)

    def merge_state(self, state: RightArmGripperState | np.ndarray) -> np.ndarray:
        checked = ensure_state(state)
        return checked.as_flat(control_mode=self.control_mode)

    def to_absolute(
        self,
        current_state: RightArmGripperState | np.ndarray,
        action: RightArmGripperAction,
    ) -> RightArmGripperAction:
        if self.action_mode == "absolute":
            return action.copy()
        current = self.split_state(current_state)
        return RightArmGripperAction(
            right_arm_q=current.right_arm_q + action.right_arm_q,
            right_gripper_q=current.right_gripper_q + action.right_gripper_q,
            left_arm_q=current.left_arm_q + action.left_arm_q,
            left_gripper_q=current.left_gripper_q + action.left_gripper_q,
            control_left_arm=action.control_left_arm,
            control_left_gripper=action.control_left_gripper,
        )

    def metadata(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "control_mode": self.control_mode,
            "policy_unit": self.policy_unit,
            "control_unit": self.control_unit,
            "policy_arm_unit": self.policy_arm_unit,
            "control_arm_unit": self.control_arm_unit,
            "policy_gripper_unit": self.policy_gripper_unit,
            "control_gripper_unit": self.control_gripper_unit,
            "action_mode": self.action_mode,
        }
