"""Synchronous slow interpolated dispatch for model actions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

import numpy as np
import yaml

from .action_adapter import ActionAdapter
from .action_adapter import RightArmGripperAction
from .robot_interface import RightArmGripperRobot
from .robot_state import RightArmGripperState
from .safety import SafetyLayer


class SlowDispatchError(RuntimeError):
    """Raised when slow model-action dispatch cannot complete."""


class SlowDispatchCancelledError(SlowDispatchError):
    """Raised when an operator stop is requested during interpolation."""


@dataclass(frozen=True)
class SlowDispatchConfig:
    """YAML-configurable timing settings for slow model-action dispatch."""

    duration_sec: float = 1.0
    frequency_hz: float = 20.0
    interpolation_mode: str = "cubic"

    @classmethod
    def from_yaml(cls, path: str | Path) -> SlowDispatchConfig:
        with Path(path).open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file) or {}
        if not isinstance(raw, dict):
            raise SlowDispatchError(f"slow dispatch config must be a mapping, got {type(raw)!r}")
        defaults = cls()
        return cls(
            duration_sec=float(raw.get("slow_dispatch_duration_sec", defaults.duration_sec)),
            frequency_hz=float(raw.get("slow_dispatch_frequency_hz", defaults.frequency_hz)),
            interpolation_mode=str(
                raw.get("slow_dispatch_interpolation_mode", defaults.interpolation_mode)
            ),
        )


class SlowInterpolatedDispatcher:
    """Convert one model action into a slow, interpolated robot command."""

    def __init__(
        self,
        robot: RightArmGripperRobot,
        *,
        adapter: ActionAdapter,
        safety: SafetyLayer | None = None,
        duration_sec: float = 1.0,
        frequency_hz: float = 20.0,
        interpolation_mode: str = "cubic",
        event_logger: Callable[..., None] | None = None,
    ) -> None:
        self.robot = robot
        self.adapter = adapter
        self.safety = safety
        self.duration_sec = _validate_positive(duration_sec, "duration_sec", allow_zero=True)
        self.frequency_hz = _validate_positive(frequency_hz, "frequency_hz")
        if interpolation_mode not in {"linear", "cubic"}:
            raise ValueError("interpolation_mode must be one of {'linear', 'cubic'}")
        self.interpolation_mode = interpolation_mode
        self.event_logger = event_logger
        self._dispatch_lock = threading.RLock()

    @classmethod
    def from_yaml(
        cls,
        robot: RightArmGripperRobot,
        *,
        adapter: ActionAdapter,
        config_path: str | Path,
        safety: SafetyLayer | None = None,
        event_logger: Callable[..., None] | None = None,
    ) -> SlowInterpolatedDispatcher:
        config = SlowDispatchConfig.from_yaml(config_path)
        return cls(
            robot,
            adapter=adapter,
            safety=safety,
            duration_sec=config.duration_sec,
            frequency_hz=config.frequency_hz,
            interpolation_mode=config.interpolation_mode,
            event_logger=event_logger,
        )

    def dispatch(
        self,
        model_action: np.ndarray,
        *,
        duration_sec: float | None = None,
        frequency_hz: float | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> RightArmGripperAction:
        """Dispatch one flat model action and return its final safe target."""
        duration = self.duration_sec if duration_sec is None else _validate_positive(
            duration_sec,
            "duration_sec",
            allow_zero=True,
        )
        frequency = self.frequency_hz if frequency_hz is None else _validate_positive(
            frequency_hz,
            "frequency_hz",
        )
        with self._dispatch_lock:
            self._stop_if_requested(stop_callback)
            current_state = self.robot.get_state()
            action = self.adapter.split_action(model_action)
            if self.safety is not None:
                safe_actions, _events = self.safety.process_chunk(
                    current_state,
                    [action],
                    duration,
                )
                action = safe_actions[0]
            target = self.adapter.to_absolute(current_state, action)
            return self._dispatch_target(
                current_state,
                target,
                duration=duration,
                frequency=frequency,
                stop_callback=stop_callback,
            )

    def dispatch_control_action(
        self,
        target: RightArmGripperAction,
        *,
        duration_sec: float | None = None,
        frequency_hz: float | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> RightArmGripperAction:
        """Slowly dispatch an already converted absolute control target.

        This is intended for operator actions such as reset/home. ``target``
        must already use the robot control units (arm degrees and gripper
        radians in the default runtime) and is not passed through the model
        action adapter or model safety clipping.
        """
        duration = self.duration_sec if duration_sec is None else _validate_positive(
            duration_sec,
            "duration_sec",
            allow_zero=True,
        )
        frequency = self.frequency_hz if frequency_hz is None else _validate_positive(
            frequency_hz,
            "frequency_hz",
        )
        with self._dispatch_lock:
            self._stop_if_requested(stop_callback)
            current_state = self.robot.get_state()
            return self._dispatch_target(
                current_state,
                target.copy(),
                duration=duration,
                frequency=frequency,
                stop_callback=stop_callback,
            )

    def _dispatch_target(
        self,
        current_state: RightArmGripperState,
        target: RightArmGripperAction,
        *,
        duration: float,
        frequency: float,
        stop_callback: Callable[[], bool] | None,
    ) -> RightArmGripperAction:
        start = _state_as_action(current_state, target)
        substeps = max(1, math.ceil(duration * frequency))
        period = duration / float(substeps)
        dispatch_start = time.perf_counter()
        self._log(
            "slow_dispatch_start",
            duration_sec=duration,
            frequency_hz=frequency,
            substeps=substeps,
            target=target.as_dict(),
        )

        last_sent = start
        try:
            for step_index in range(substeps):
                self._stop_if_requested(stop_callback)
                alpha = _interpolation_alpha(
                    (step_index + 1) / float(substeps),
                    self.interpolation_mode,
                )
                last_sent = _interpolate_action(start, target, alpha)
                self.robot.send_action(last_sent)
                self._log(
                    "slow_dispatch_step",
                    step_index=step_index,
                    substeps=substeps,
                    alpha=alpha,
                    action=last_sent.as_dict(),
                )
                if step_index < substeps - 1:
                    _sleep_until(dispatch_start + (step_index + 1) * period, stop_callback)
        except SlowDispatchCancelledError:
            self.robot.hold_position()
            raise
        except Exception:
            self.robot.hold_position()
            raise
        self._log(
            "slow_dispatch_end",
            elapsed_sec=time.perf_counter() - dispatch_start,
            action=last_sent.as_dict(),
        )
        return target.copy()

    def dispatch_chunk(
        self,
        model_actions: np.ndarray,
        *,
        duration_sec: float | None = None,
        frequency_hz: float | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> list[RightArmGripperAction]:
        """Dispatch a model action chunk one target at a time."""
        actions = np.asarray(model_actions, dtype=np.float32)
        if actions.ndim != 2:
            raise SlowDispatchError(f"model_actions must be 2-D [H, D], got {actions.shape}")
        return [
            self.dispatch(
                action,
                duration_sec=duration_sec,
                frequency_hz=frequency_hz,
                stop_callback=stop_callback,
            )
            for action in actions
        ]

    def _stop_if_requested(self, stop_callback: Callable[[], bool] | None) -> None:
        if stop_callback is not None and stop_callback():
            self.robot.hold_position()
            raise SlowDispatchCancelledError("slow model-action dispatch was cancelled")

    def _log(self, event: str, **payload: object) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger(event, **payload)
        except Exception:
            return


def _state_as_action(
    state: RightArmGripperState,
    target: RightArmGripperAction,
) -> RightArmGripperAction:
    return RightArmGripperAction(
        right_arm_q=state.right_arm_q,
        left_arm_q=state.left_arm_q,
        right_gripper_q=state.right_gripper_q,
        left_gripper_q=state.left_gripper_q,
        control_left_arm=target.control_left_arm,
        control_left_gripper=target.control_left_gripper,
    )


def _interpolate_action(
    start: RightArmGripperAction,
    target: RightArmGripperAction,
    alpha: float,
) -> RightArmGripperAction:
    return RightArmGripperAction(
        right_arm_q=_interpolate_array(start.right_arm_q, target.right_arm_q, alpha),
        left_arm_q=_interpolate_array(start.left_arm_q, target.left_arm_q, alpha),
        right_gripper_q=_interpolate_array(start.right_gripper_q, target.right_gripper_q, alpha),
        left_gripper_q=_interpolate_array(start.left_gripper_q, target.left_gripper_q, alpha),
        control_left_arm=target.control_left_arm,
        control_left_gripper=target.control_left_gripper,
    )


def _interpolate_array(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return (start + float(alpha) * (target - start)).astype(np.float32)


def _interpolation_alpha(s: float, mode: str) -> float:
    s = min(max(float(s), 0.0), 1.0)
    if mode == "linear":
        return s
    if mode == "cubic":
        return 3.0 * s * s - 2.0 * s * s * s
    raise ValueError(f"unsupported interpolation mode: {mode!r}")


def _sleep_until(
    deadline: float,
    stop_callback: Callable[[], bool] | None,
    quantum: float = 0.005,
) -> None:
    while True:
        if stop_callback is not None and stop_callback():
            raise SlowDispatchCancelledError("slow model-action dispatch was cancelled")
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        time.sleep(min(quantum, remaining))


def _validate_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    checked = float(value)
    if not np.isfinite(checked) or (checked < 0.0 if allow_zero else checked <= 0.0):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparator}, got {value!r}")
    return checked
