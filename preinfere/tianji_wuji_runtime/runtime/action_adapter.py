"""Action conversion for the canonical 54-DoF dual-arm dual-hand schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import schema
from .robot_state import DualArmHandState, ensure_state


class ActionAdapterError(RuntimeError):
    """Raised when a policy action cannot be adapted safely."""


@dataclass
class DualArmHandAction:
    left_arm_q: np.ndarray
    left_hand_q: np.ndarray
    right_arm_q: np.ndarray
    right_hand_q: np.ndarray

    def __post_init__(self) -> None:
        self.left_arm_q = _as_segment(self.left_arm_q, schema.LEFT_ARM_DOF, "left_arm_q")
        self.left_hand_q = _as_segment(self.left_hand_q, schema.LEFT_HAND_DOF, "left_hand_q")
        self.right_arm_q = _as_segment(self.right_arm_q, schema.RIGHT_ARM_DOF, "right_arm_q")
        self.right_hand_q = _as_segment(self.right_hand_q, schema.RIGHT_HAND_DOF, "right_hand_q")

    def copy(self) -> "DualArmHandAction":
        return DualArmHandAction(
            left_arm_q=self.left_arm_q.copy(),
            left_hand_q=self.left_hand_q.copy(),
            right_arm_q=self.right_arm_q.copy(),
            right_hand_q=self.right_hand_q.copy(),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "left_arm": self.left_arm_q.tolist(),
            "left_hand": self.left_hand_q.tolist(),
            "right_arm": self.right_arm_q.tolist(),
            "right_hand": self.right_hand_q.tolist(),
        }


def _as_segment(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (dim,):
        raise ActionAdapterError(f"{name} must have shape ({dim},), got {arr.shape}")
    return arr


def _convert_units(arr: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    if from_unit == to_unit:
        return arr.astype(np.float32, copy=True)
    if from_unit == "rad" and to_unit in {"deg", "degree", "degrees"}:
        return np.rad2deg(arr).astype(np.float32)
    if from_unit in {"deg", "degree", "degrees"} and to_unit == "rad":
        return np.deg2rad(arr).astype(np.float32)
    # Encoder and normalized units need calibration values from hardware/data owners.
    return arr.astype(np.float32, copy=True)


class ActionAdapter:
    """Split, merge, and unit-convert 54-DoF action vectors."""

    def __init__(
        self,
        *,
        policy_unit: str = schema.ACTION_UNIT,
        control_unit: str = schema.ACTION_UNIT,
        action_mode: str = schema.ACTION_MODE,
        schema_version: str = schema.ACTION_SCHEMA_VERSION,
    ) -> None:
        if action_mode not in {"absolute", "delta"}:
            raise ValueError(f"action_mode must be 'absolute' or 'delta', got {action_mode!r}")
        self.policy_unit = policy_unit
        self.control_unit = control_unit
        self.action_mode = action_mode
        self.schema_version = schema_version

    def split_action(self, action_54: np.ndarray) -> DualArmHandAction:
        arr = schema.validate_flat_vector(action_54, dim=schema.ACTION_DIM, name="action_54")
        arr = _convert_units(arr, self.policy_unit, self.control_unit)
        return DualArmHandAction(
            left_arm_q=arr[schema.LEFT_ARM_SLICE],
            left_hand_q=arr[schema.LEFT_HAND_SLICE],
            right_arm_q=arr[schema.RIGHT_ARM_SLICE],
            right_hand_q=arr[schema.RIGHT_HAND_SLICE],
        )

    def split_state(self, state: DualArmHandState | np.ndarray) -> DualArmHandState:
        return ensure_state(state).copy()

    def split_chunk(self, action_chunk: np.ndarray) -> list[DualArmHandAction]:
        arr = np.asarray(action_chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ActionAdapterError(f"action_chunk must be 2-D [H, 54], got {arr.shape}")
        if arr.shape[1] != schema.ACTION_DIM:
            raise ActionAdapterError(
                f"action_chunk second dimension must be {schema.ACTION_DIM}, got {arr.shape[1]}"
            )
        return [self.split_action(step) for step in arr]

    def merge_action(self, action: DualArmHandAction) -> np.ndarray:
        return np.concatenate(
            [
                action.left_arm_q,
                action.right_arm_q,
                action.left_hand_q,
                action.right_hand_q,
            ],
            axis=0,
        ).astype(np.float32)

    def merge_chunk(self, actions: list[DualArmHandAction]) -> np.ndarray:
        if not actions:
            return np.empty((0, schema.ACTION_DIM), dtype=np.float32)
        return np.stack([self.merge_action(action) for action in actions], axis=0)

    def merge_state(self, state: DualArmHandState | np.ndarray) -> np.ndarray:
        return ensure_state(state).as_flat()

    def to_absolute(
        self,
        current_state: DualArmHandState | np.ndarray,
        action: DualArmHandAction,
    ) -> DualArmHandAction:
        if self.action_mode == "absolute":
            return action.copy()
        current = self.split_state(current_state)
        return DualArmHandAction(
            left_arm_q=current.left_arm_q + action.left_arm_q,
            left_hand_q=current.left_hand_q + action.left_hand_q,
            right_arm_q=current.right_arm_q + action.right_arm_q,
            right_hand_q=current.right_hand_q + action.right_hand_q,
        )

    def metadata(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "policy_unit": self.policy_unit,
            "control_unit": self.control_unit,
            "action_mode": self.action_mode,
        }
