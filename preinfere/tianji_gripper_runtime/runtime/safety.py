"""Software safety layer for Tianji gripper action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import schema
from .action_adapter import ActionAdapter, RightArmGripperAction
from .robot_state import RightArmGripperState


class SafetyError(RuntimeError):
    """Raised when safety policy refuses further execution."""


@dataclass
class SafetyConfig:
    right_arm_joint_min: np.ndarray
    right_arm_joint_max: np.ndarray
    left_arm_joint_min: np.ndarray
    left_arm_joint_max: np.ndarray
    right_gripper_min: np.ndarray
    right_gripper_max: np.ndarray
    left_gripper_min: np.ndarray
    left_gripper_max: np.ndarray
    arm_max_step: float
    gripper_max_step: float
    arm_max_velocity: np.ndarray | None = None
    gripper_max_velocity: np.ndarray | None = None
    enable_arm_joint_limit: bool = True
    enable_gripper_limit: bool = True
    enable_arm_delta_clip: bool = True
    enable_gripper_delta_clip: bool = True
    enable_arm_velocity_limit: bool = True
    enable_gripper_velocity_limit: bool = False
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

        def _vector_with_fallback(name: str, fallback_name: str, dim: int) -> np.ndarray:
            value = raw.get(name, raw.get(fallback_name))
            if value is None:
                raise ValueError(
                    f"missing required safety config field: {name} (and fallback {fallback_name})"
                )
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != (dim,):
                raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
            return arr

        return cls(
            right_arm_joint_min=_vector("right_arm_joint_min", schema.RIGHT_ARM_DOF),
            right_arm_joint_max=_vector("right_arm_joint_max", schema.RIGHT_ARM_DOF),
            left_arm_joint_min=_vector_with_fallback(
                "left_arm_joint_min",
                "right_arm_joint_min",
                schema.LEFT_ARM_DOF,
            ),
            left_arm_joint_max=_vector_with_fallback(
                "left_arm_joint_max",
                "right_arm_joint_max",
                schema.LEFT_ARM_DOF,
            ),
            right_gripper_min=_vector("right_gripper_min", schema.RIGHT_GRIPPER_DOF),
            right_gripper_max=_vector("right_gripper_max", schema.RIGHT_GRIPPER_DOF),
            left_gripper_min=_vector_with_fallback(
                "left_gripper_min",
                "right_gripper_min",
                schema.LEFT_GRIPPER_DOF,
            ),
            left_gripper_max=_vector_with_fallback(
                "left_gripper_max",
                "right_gripper_max",
                schema.LEFT_GRIPPER_DOF,
            ),
            arm_max_step=float(raw.get("arm_max_step", 10.0)),
            gripper_max_step=float(raw.get("gripper_max_step", 0.1)),
            arm_max_velocity=_optional_limit("arm_max_velocity", schema.RIGHT_ARM_DOF),
            gripper_max_velocity=_optional_limit(
                "gripper_max_velocity",
                schema.RIGHT_GRIPPER_DOF,
            ),
            enable_arm_joint_limit=bool(raw.get("enable_arm_joint_limit", True)),
            enable_gripper_limit=bool(raw.get("enable_gripper_limit", True)),
            enable_arm_delta_clip=bool(raw.get("enable_arm_delta_clip", True)),
            enable_gripper_delta_clip=bool(raw.get("enable_gripper_delta_clip", True)),
            enable_arm_velocity_limit=bool(raw.get("enable_arm_velocity_limit", True)),
            enable_gripper_velocity_limit=bool(raw.get("enable_gripper_velocity_limit", False)),
            enable_filter=bool(raw.get("enable_filter", False)),
            filter_alpha=float(raw.get("filter_alpha", 0.35)),
            max_consecutive_events=int(raw.get("max_consecutive_events", 20)),
        )


class SafetyLayer:
    def __init__(self, config: SafetyConfig, adapter: ActionAdapter) -> None:
        self.config = config
        self.adapter = adapter
        self.previous_action: RightArmGripperAction | None = None
        self._consecutive_events = 0

    def reset_consecutive_events(self) -> None:
        """Start a new operator-approved Safe Mode step.

        This resets only the event escalation counter. Previous commanded
        positions are intentionally retained for velocity limiting and
        filtering, and every hard joint/gripper limit remains enabled.
        """

        self._consecutive_events = 0

    def process_chunk(
        self,
        current_state: RightArmGripperState | np.ndarray,
        actions: list[RightArmGripperAction],
        dt: float,
    ) -> tuple[list[RightArmGripperAction], list[dict[str, object]]]:
        current = self.adapter.split_state(current_state)
        processed: list[RightArmGripperAction] = []
        all_events: list[dict[str, object]] = []
        reference: RightArmGripperState | RightArmGripperAction = current

        for step_idx, action in enumerate(actions):
            self._check_finite(action)
            candidate = self.adapter.to_absolute(current, action)
            step_events: list[dict[str, object]] = []

            candidate, events = self._clamp_limits(candidate)
            step_events.extend(events)

            candidate, events = self._clip_against_reference(reference, candidate)
            step_events.extend(events)

            if self.previous_action is not None:
                candidate, events = self._limit_velocity(self.previous_action, candidate, dt)
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

    def _check_finite(self, action: RightArmGripperAction) -> None:
        for name, arr in _segments(action):
            if not np.all(np.isfinite(arr)):
                raise SafetyError(f"{name} action contains NaN or Inf")

    def _clamp_limits(
        self,
        action: RightArmGripperAction,
    ) -> tuple[RightArmGripperAction, list[dict[str, object]]]:
        cfg = self.config
        clipped = action.copy()
        events: list[dict[str, object]] = []
        for name, enabled, lo, hi, arr in (
            (
                "right_arm",
                cfg.enable_arm_joint_limit,
                cfg.right_arm_joint_min,
                cfg.right_arm_joint_max,
                clipped.right_arm_q,
            ),
            (
                "left_arm",
                cfg.enable_arm_joint_limit and clipped.control_left_arm,
                cfg.left_arm_joint_min,
                cfg.left_arm_joint_max,
                clipped.left_arm_q,
            ),
            (
                "right_gripper",
                cfg.enable_gripper_limit,
                cfg.right_gripper_min,
                cfg.right_gripper_max,
                clipped.right_gripper_q,
            ),
            (
                "left_gripper",
                cfg.enable_gripper_limit and clipped.control_left_gripper,
                cfg.left_gripper_min,
                cfg.left_gripper_max,
                clipped.left_gripper_q,
            ),
        ):
            if not enabled:
                continue
            before = arr.copy()
            arr[:] = np.clip(arr, lo, hi)
            if not np.allclose(before, arr):
                events.append({"type": "joint_limit_clamp", "segment": name})
        return clipped, events

    def _clip_against_reference(
        self,
        reference: RightArmGripperState | RightArmGripperAction,
        action: RightArmGripperAction,
    ) -> tuple[RightArmGripperAction, list[dict[str, object]]]:
        cfg = self.config
        clipped = action.copy()
        events: list[dict[str, object]] = []
        for segment, enabled, max_step in (
            ("right_arm", cfg.enable_arm_delta_clip, cfg.arm_max_step),
            ("left_arm", cfg.enable_arm_delta_clip and action.control_left_arm, cfg.arm_max_step),
            ("right_gripper", cfg.enable_gripper_delta_clip, cfg.gripper_max_step),
            (
                "left_gripper",
                cfg.enable_gripper_delta_clip and action.control_left_gripper,
                cfg.gripper_max_step,
            ),
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

    def _limit_velocity(
        self,
        previous_action: RightArmGripperAction,
        action: RightArmGripperAction,
        dt: float,
    ) -> tuple[RightArmGripperAction, list[dict[str, object]]]:
        if dt <= 0:
            return action.copy(), []
        cfg = self.config
        clipped = action.copy()
        events: list[dict[str, object]] = []
        for segment, enabled, limit in (
            ("right_arm", cfg.enable_arm_velocity_limit, cfg.arm_max_velocity),
            ("left_arm", cfg.enable_arm_velocity_limit and action.control_left_arm, cfg.arm_max_velocity),
            ("right_gripper", cfg.enable_gripper_velocity_limit, cfg.gripper_max_velocity),
            (
                "left_gripper",
                cfg.enable_gripper_velocity_limit and action.control_left_gripper,
                cfg.gripper_max_velocity,
            ),
        ):
            if not enabled or limit is None:
                continue
            prev = getattr(previous_action, f"{segment}_q")
            cur = getattr(clipped, f"{segment}_q")
            max_delta = np.asarray(limit, dtype=np.float32) * float(dt)
            before = cur.copy()
            cur[:] = prev + np.clip(cur - prev, -max_delta, max_delta)
            if not np.allclose(before, cur):
                events.append({"type": "velocity_clip", "segment": segment})
        return clipped, events

    def _filter(
        self,
        previous_action: RightArmGripperAction,
        action: RightArmGripperAction,
    ) -> RightArmGripperAction:
        alpha = float(np.clip(self.config.filter_alpha, 0.0, 1.0))
        return RightArmGripperAction(
            right_arm_q=alpha * action.right_arm_q + (1.0 - alpha) * previous_action.right_arm_q,
            left_arm_q=alpha * action.left_arm_q + (1.0 - alpha) * previous_action.left_arm_q,
            right_gripper_q=(
                alpha * action.right_gripper_q
                + (1.0 - alpha) * previous_action.right_gripper_q
            ),
            left_gripper_q=(
                alpha * action.left_gripper_q
                + (1.0 - alpha) * previous_action.left_gripper_q
            ),
            control_left_arm=action.control_left_arm,
            control_left_gripper=action.control_left_gripper,
        )


def _segments(action: RightArmGripperAction):
    yield "right_arm", action.right_arm_q
    if action.control_left_arm:
        yield "left_arm", action.left_arm_q
    yield "right_gripper", action.right_gripper_q
    if action.control_left_gripper:
        yield "left_gripper", action.left_gripper_q
