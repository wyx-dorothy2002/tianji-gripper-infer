"""Control-side action overrides for constrained runtime modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import schema
from .action_adapter import DualArmHandAction
from .robot_state import DualArmHandState, ensure_state


class ControlOverrideError(ValueError):
    """Raised when a control override target is malformed."""


@dataclass
class LeftSideFreezeTarget:
    """Fixed left arm + left hand command used for right-side-only execution."""

    left_arm_q: np.ndarray
    left_hand_q: np.ndarray
    source: str

    def __post_init__(self) -> None:
        self.left_arm_q = _as_vector(
            self.left_arm_q,
            dim=schema.LEFT_ARM_DOF,
            name="left_freeze_arm",
        )
        self.left_hand_q = _as_vector(
            self.left_hand_q,
            dim=schema.LEFT_HAND_DOF,
            name="left_freeze_hand",
        )

    @classmethod
    def from_state(
        cls,
        state: DualArmHandState | np.ndarray,
        *,
        source: str = "current_state",
    ) -> "LeftSideFreezeTarget":
        current = ensure_state(state)
        return cls(
            left_arm_q=current.left_arm_q.copy(),
            left_hand_q=current.left_hand_q.copy(),
            source=source,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "left_arm": self.left_arm_q.tolist(),
            "left_hand": self.left_hand_q.tolist(),
            "schema": {
                "state_unit": schema.STATE_UNIT,
                "action_unit": schema.ACTION_UNIT,
                "order": {
                    "left_arm": [schema.LEFT_ARM_SLICE.start, schema.LEFT_ARM_SLICE.stop],
                    "left_hand": [schema.LEFT_HAND_SLICE.start, schema.LEFT_HAND_SLICE.stop],
                },
            },
        }


def apply_left_side_freeze(
    actions: list[DualArmHandAction],
    target: LeftSideFreezeTarget,
) -> tuple[list[DualArmHandAction], list[dict[str, object]]]:
    """Replace left-side commands while preserving policy-generated right-side commands."""
    frozen_actions: list[DualArmHandAction] = []
    events: list[dict[str, object]] = []
    for step_idx, action in enumerate(actions):
        frozen_actions.append(
            DualArmHandAction(
                left_arm_q=target.left_arm_q.copy(),
                left_hand_q=target.left_hand_q.copy(),
                right_arm_q=action.right_arm_q.copy(),
                right_hand_q=action.right_hand_q.copy(),
            )
        )
        events.append(
            {
                "type": "control_override",
                "override": "freeze_left_side",
                "segments": ["left_arm", "left_hand"],
                "source": target.source,
                "step_in_chunk": step_idx,
            }
        )
    return frozen_actions, events


def _as_vector(value: np.ndarray, *, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (dim,):
        raise ControlOverrideError(f"{name} must have shape ({dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ControlOverrideError(f"{name} contains NaN or Inf")
    return arr.copy()
