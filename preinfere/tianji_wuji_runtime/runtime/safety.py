"""Software safety layer for 54-DoF action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import schema
from .action_adapter import ActionAdapter, DualArmHandAction
from .robot_state import DualArmHandState


class SafetyError(RuntimeError):
    """Raised when safety policy refuses further execution."""


@dataclass
class SafetyConfig:
    left_arm_joint_min: np.ndarray
    left_arm_joint_max: np.ndarray
    left_hand_joint_min: np.ndarray
    left_hand_joint_max: np.ndarray
    right_arm_joint_min: np.ndarray
    right_arm_joint_max: np.ndarray
    right_hand_joint_min: np.ndarray
    right_hand_joint_max: np.ndarray
    arm_max_step: float
    hand_max_step: float
    arm_max_velocity: np.ndarray | None = None
    hand_max_velocity: np.ndarray | None = None
    enable_arm_joint_limit: bool = True
    enable_hand_joint_limit: bool = False
    enable_arm_delta_clip: bool = True
    enable_hand_delta_clip: bool = False
    enable_arm_velocity_limit: bool = True
    enable_hand_velocity_limit: bool = False
    enable_filter: bool = False
    filter_alpha: float = 0.35
    max_consecutive_events: int = 20

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SafetyConfig":
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        def _vector(name: str, dim: int) -> np.ndarray:
            value = raw.get(name)
            if value is None:
                raise ValueError(f"missing required safety config field: {name}")
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != (dim,):
                raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
            return arr

        def _optional_limit(name: str, dim: int) -> np.ndarray | None:
            value = raw.get(name)
            if value is None:
                return None
            if np.isscalar(value):
                return np.full(dim, float(value), dtype=np.float32)
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != (dim,):
                raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
            return arr

        return cls(
            left_arm_joint_min=_vector("left_arm_joint_min", schema.LEFT_ARM_DOF),
            left_arm_joint_max=_vector("left_arm_joint_max", schema.LEFT_ARM_DOF),
            left_hand_joint_min=_vector("left_hand_joint_min", schema.LEFT_HAND_DOF),
            left_hand_joint_max=_vector("left_hand_joint_max", schema.LEFT_HAND_DOF),
            right_arm_joint_min=_vector("right_arm_joint_min", schema.RIGHT_ARM_DOF),
            right_arm_joint_max=_vector("right_arm_joint_max", schema.RIGHT_ARM_DOF),
            right_hand_joint_min=_vector("right_hand_joint_min", schema.RIGHT_HAND_DOF),
            right_hand_joint_max=_vector("right_hand_joint_max", schema.RIGHT_HAND_DOF),
            arm_max_step=float(raw.get("arm_max_step", 10.0)),
            hand_max_step=float(raw.get("hand_max_step", 4.5)),
            arm_max_velocity=_optional_limit("arm_max_velocity", schema.LEFT_ARM_DOF),
            hand_max_velocity=_optional_limit("hand_max_velocity", schema.LEFT_HAND_DOF),
            enable_arm_joint_limit=bool(raw.get("enable_arm_joint_limit", True)),
            enable_hand_joint_limit=bool(raw.get("enable_hand_joint_limit", False)),
            enable_arm_delta_clip=bool(raw.get("enable_arm_delta_clip", True)),
            enable_hand_delta_clip=bool(raw.get("enable_hand_delta_clip", False)),
            enable_arm_velocity_limit=bool(raw.get("enable_arm_velocity_limit", True)),
            enable_hand_velocity_limit=bool(raw.get("enable_hand_velocity_limit", False)),
        )

    @classmethod
    def permissive(
        cls,
        *,
        arm_max_step: float = 10.0,
        hand_max_step: float = 4.5,
        arm_max_velocity: float | None = None,
        hand_max_velocity: float | None = None,
    ) -> "SafetyConfig":
        # Marvin M6-CCS independent arm joint limits from the vendor manual.
        arm_min = np.array([-178.0, -120.0, -178.0, -145.0, -178.0, -60.0, -90.0], dtype=np.float32)
        arm_max = np.array([178.0, 120.0, 178.0, 60.0, 178.0, 60.0, 90.0], dtype=np.float32)
        # Keep hand limits broad until Wuji hand hardware/data units are finalized.
        hand_min = np.full(schema.LEFT_HAND_DOF, -360.0, dtype=np.float32)
        hand_max = np.full(schema.LEFT_HAND_DOF, 360.0, dtype=np.float32)
        arm_vel = (
            None
            if arm_max_velocity is None
            else np.full(schema.LEFT_ARM_DOF, arm_max_velocity, dtype=np.float32)
        )
        hand_vel = (
            None
            if hand_max_velocity is None
            else np.full(schema.LEFT_HAND_DOF, hand_max_velocity, dtype=np.float32)
        )
        return cls(
            left_arm_joint_min=arm_min.copy(),
            left_arm_joint_max=arm_max.copy(),
            left_hand_joint_min=hand_min.copy(),
            left_hand_joint_max=hand_max.copy(),
            right_arm_joint_min=arm_min.copy(),
            right_arm_joint_max=arm_max.copy(),
            right_hand_joint_min=hand_min.copy(),
            right_hand_joint_max=hand_max.copy(),
            arm_max_step=float(arm_max_step),
            hand_max_step=float(hand_max_step),
            arm_max_velocity=arm_vel,
            hand_max_velocity=hand_vel,
        )


class SafetyLayer:
    def __init__(self, config: SafetyConfig, adapter: ActionAdapter) -> None:
        self.config = config
        self.adapter = adapter
        self.previous_action: DualArmHandAction | None = None
        self._consecutive_events = 0

    def check_finite(self, action: DualArmHandAction) -> None:
        for name, arr in _segments(action):
            if not np.all(np.isfinite(arr)):
                raise SafetyError(f"{name} action contains NaN or Inf")

    def clamp_joint_limits(
        self,
        action: DualArmHandAction,
    ) -> tuple[DualArmHandAction, list[dict[str, object]]]:
        cfg = self.config
        events: list[dict[str, object]] = []
        clipped = action.copy()
        limit_specs = {
            "left_arm": (
                cfg.enable_arm_joint_limit,
                cfg.left_arm_joint_min,
                cfg.left_arm_joint_max,
                clipped.left_arm_q,
            ),
            "left_hand": (
                cfg.enable_hand_joint_limit,
                cfg.left_hand_joint_min,
                cfg.left_hand_joint_max,
                clipped.left_hand_q,
            ),
            "right_arm": (
                cfg.enable_arm_joint_limit,
                cfg.right_arm_joint_min,
                cfg.right_arm_joint_max,
                clipped.right_arm_q,
            ),
            "right_hand": (
                cfg.enable_hand_joint_limit,
                cfg.right_hand_joint_min,
                cfg.right_hand_joint_max,
                clipped.right_hand_q,
            ),
        }
        for name, (enabled, lo, hi, arr) in limit_specs.items():
            if not enabled:
                continue
            before = arr.copy()
            arr[:] = np.clip(arr, lo, hi)
            if not np.allclose(before, arr):
                events.append({"type": "joint_limit_clamp", "segment": name})
        return clipped, events

    def clip_delta(
        self,
        current_state: DualArmHandState | np.ndarray,
        action: DualArmHandAction,
    ) -> tuple[DualArmHandAction, list[dict[str, object]]]:
        current = self.adapter.split_state(current_state)
        return self._clip_against_reference(current, action)

    def limit_velocity(
        self,
        previous_action: DualArmHandAction,
        action: DualArmHandAction,
        dt: float,
    ) -> tuple[DualArmHandAction, list[dict[str, object]]]:
        if dt <= 0:
            return action.copy(), []
        cfg = self.config
        clipped = action.copy()
        events: list[dict[str, object]] = []
        for segment, enabled, limit in (
            ("left_arm", cfg.enable_arm_velocity_limit, cfg.arm_max_velocity),
            ("right_arm", cfg.enable_arm_velocity_limit, cfg.arm_max_velocity),
            ("left_hand", cfg.enable_hand_velocity_limit, cfg.hand_max_velocity),
            ("right_hand", cfg.enable_hand_velocity_limit, cfg.hand_max_velocity),
        ):
            if not enabled:
                continue
            if limit is None:
                continue
            prev = getattr(previous_action, f"{segment}_q")
            cur = getattr(clipped, f"{segment}_q")
            max_delta = np.asarray(limit, dtype=np.float32) * float(dt)
            before = cur.copy()
            cur[:] = prev + np.clip(cur - prev, -max_delta, max_delta)
            if not np.allclose(before, cur):
                events.append({"type": "velocity_clip", "segment": segment})
        return clipped, events

    def process_chunk(
        self,
        current_state: DualArmHandState | np.ndarray,
        actions: list[DualArmHandAction],
        dt: float,
    ) -> tuple[list[DualArmHandAction], list[dict[str, object]]]:
        current = self.adapter.split_state(current_state)
        processed: list[DualArmHandAction] = []
        all_events: list[dict[str, object]] = []
        reference: DualArmHandState | DualArmHandAction = current

        for step_idx, action in enumerate(actions):
            self.check_finite(action)
            candidate = self.adapter.to_absolute(current, action)
            step_events: list[dict[str, object]] = []

            candidate, events = self.clamp_joint_limits(candidate)
            step_events.extend(events)

            candidate, events = self._clip_against_reference(reference, candidate)
            step_events.extend(events)

            if self.previous_action is not None:
                candidate, events = self.limit_velocity(self.previous_action, candidate, dt)
                step_events.extend(events)

            if self.config.enable_filter and self.previous_action is not None:
                candidate = self._filter(self.previous_action, candidate)

            for event in step_events:
                event["step_in_chunk"] = step_idx
            all_events.extend(step_events)
            processed.append(candidate)
            reference = candidate
            self.previous_action = candidate.copy()

            self._consecutive_events = self._consecutive_events + 1 if step_events else 0
            if self._consecutive_events >= self.config.max_consecutive_events:
                raise SafetyError(f"too many consecutive safety events: {self._consecutive_events}")

        return processed, all_events

    def _clip_against_reference(
        self,
        reference: DualArmHandState | DualArmHandAction,
        action: DualArmHandAction,
    ) -> tuple[DualArmHandAction, list[dict[str, object]]]:
        cfg = self.config
        clipped = action.copy()
        events: list[dict[str, object]] = []
        for segment, enabled, max_step in (
            ("left_arm", cfg.enable_arm_delta_clip, cfg.arm_max_step),
            ("right_arm", cfg.enable_arm_delta_clip, cfg.arm_max_step),
            ("left_hand", cfg.enable_hand_delta_clip, cfg.hand_max_step),
            ("right_hand", cfg.enable_hand_delta_clip, cfg.hand_max_step),
        ):
            if not enabled:
                continue
            ref = getattr(reference, f"{segment}_q")
            cur = getattr(clipped, f"{segment}_q")
            before = cur.copy()
            cur[:] = ref + np.clip(cur - ref, -float(max_step), float(max_step))
            if not np.allclose(before, cur):
                events.append({"type": "delta_clip", "segment": segment, "max_step": max_step})
        return clipped, events

    def _filter(
        self,
        previous_action: DualArmHandAction,
        action: DualArmHandAction,
    ) -> DualArmHandAction:
        alpha = float(np.clip(self.config.filter_alpha, 0.0, 1.0))
        return DualArmHandAction(
            left_arm_q=alpha * action.left_arm_q + (1.0 - alpha) * previous_action.left_arm_q,
            left_hand_q=alpha * action.left_hand_q + (1.0 - alpha) * previous_action.left_hand_q,
            right_arm_q=alpha * action.right_arm_q + (1.0 - alpha) * previous_action.right_arm_q,
            right_hand_q=alpha * action.right_hand_q + (1.0 - alpha) * previous_action.right_hand_q,
        )


def _segments(action: DualArmHandAction):
    yield "left_arm", action.left_arm_q
    yield "left_hand", action.left_hand_q
    yield "right_arm", action.right_arm_q
    yield "right_hand", action.right_hand_q
