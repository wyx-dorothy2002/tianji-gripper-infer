"""Fixed-period action chunk executor for right-arm gripper control."""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from .action_adapter import ActionAdapter, RightArmGripperAction
from .recorder import Recorder
from .robot_interface import RightArmGripperRobot, RobotError


class ActionExecutor:
    def __init__(
        self,
        robot: RightArmGripperRobot,
        *,
        adapter: ActionAdapter,
        recorder: Recorder | None = None,
        event_logger: Callable[..., None] | None = None,
        debug_callback: Callable[..., None] | None = None,
        arm_interpolation_hz: float | None = None,
        arm_interpolation_mode: str = "cubic",
        slow_dispatch_enabled: bool = False,
        slow_dispatch_duration_sec: float = 1.0,
        slow_dispatch_frequency_hz: float = 20.0,
        slow_dispatch_interpolation_mode: str = "cubic",
    ) -> None:
        self.robot = robot
        self.adapter = adapter
        self.recorder = recorder
        self.event_logger = event_logger
        self.debug_callback = debug_callback
        if arm_interpolation_mode not in {"cubic", "linear"}:
            raise ValueError("arm_interpolation_mode must be one of {'cubic', 'linear'}")
        self.arm_interpolation_hz = arm_interpolation_hz
        self.arm_interpolation_mode = arm_interpolation_mode
        if slow_dispatch_duration_sec < 0.0:
            raise ValueError("slow_dispatch_duration_sec must be non-negative")
        if slow_dispatch_frequency_hz <= 0.0:
            raise ValueError("slow_dispatch_frequency_hz must be positive")
        if slow_dispatch_interpolation_mode not in {"cubic", "linear"}:
            raise ValueError("slow_dispatch_interpolation_mode must be one of {'cubic', 'linear'}")
        self.slow_dispatch_enabled = bool(slow_dispatch_enabled)
        self.slow_dispatch_duration_sec = float(slow_dispatch_duration_sec)
        self.slow_dispatch_frequency_hz = float(slow_dispatch_frequency_hz)
        self.slow_dispatch_interpolation_mode = slow_dispatch_interpolation_mode
        self._last_sent_action: RightArmGripperAction | None = None

    def execute_chunk(
        self,
        actions: list[RightArmGripperAction],
        dt: float,
        *,
        dry_run: bool = False,
        chunk_index: int = 0,
        raw_chunk: np.ndarray | None = None,
        safety_events: list[dict[str, object]] | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> RightArmGripperAction | None:
        last_executed_action: RightArmGripperAction | None = None
        for step_idx, action in enumerate(actions):
            if stop_callback is not None and stop_callback():
                self._log("executor_stop_requested", chunk_index=chunk_index, step_in_chunk=step_idx)
                self.robot.hold_position()
                break
            step_start = time.perf_counter()
            state_before = None
            state_after = None
            executed = False
            error: str | None = None
            try:
                state_before = self.robot.get_state()
                if not dry_run:
                    interpolation_dt = (
                        self.slow_dispatch_duration_sec if self.slow_dispatch_enabled else dt
                    )
                    interpolation_hz = (
                        self.slow_dispatch_frequency_hz
                        if self.slow_dispatch_enabled
                        else self.arm_interpolation_hz
                    )
                    sent_action = self._send_action_with_optional_interpolation(
                        action=action,
                        interpolation_dt=interpolation_dt,
                        interpolation_hz=interpolation_hz,
                        interpolate_grippers=self.slow_dispatch_enabled,
                        step_start=step_start,
                        state_before=state_before,
                        stop_callback=stop_callback,
                    )
                    executed = True
                    last_executed_action = sent_action.copy()
                state_after = self.robot.get_state()
            except RobotError as exc:
                error = repr(exc)
                self.robot.hold_position()
                raise
            finally:
                elapsed = time.perf_counter() - step_start
                self._log(
                    "executor_step",
                    chunk_index=chunk_index,
                    step_in_chunk=step_idx,
                    dt_sec=float(dt),
                    control_latency_ms=elapsed * 1000.0,
                    period_overrun_ms=max((elapsed - float(dt)) * 1000.0, 0.0),
                    executed=executed,
                    dry_run=dry_run,
                    error=error,
                    action=action.as_dict(),
                )
                if self.recorder is not None:
                    raw_action = raw_chunk[step_idx] if raw_chunk is not None else None
                    self.recorder.record_step(
                        chunk_index=chunk_index,
                        step_in_chunk=step_idx,
                        state_before=state_before,
                        raw_action=raw_action,
                        safe_action=action,
                        executed=executed,
                        state_after=state_after,
                        control_latency_ms=elapsed * 1000.0,
                        safety_events=[
                            event
                            for event in safety_events or []
                            if event.get("step_in_chunk") == step_idx
                        ],
                    )
                else:
                    raw_action = raw_chunk[step_idx] if raw_chunk is not None else None
                if self.debug_callback is not None:
                    try:
                        self.debug_callback(
                            chunk_index=chunk_index,
                            step_in_chunk=step_idx,
                            raw_action=raw_action,
                            safe_action=action,
                            state_before=state_before,
                            state_after=state_after,
                            executed=executed,
                            error=error,
                            safety_events=[
                                event
                                for event in safety_events or []
                                if event.get("step_in_chunk") == step_idx
                            ],
                        )
                    except Exception:
                        pass
            remaining = float(dt) - (time.perf_counter() - step_start)
            if remaining > 0:
                _interruptible_sleep(remaining, stop_callback)
        return last_executed_action

    def _send_action_with_optional_interpolation(
        self,
        *,
        action: RightArmGripperAction,
        interpolation_dt: float,
        interpolation_hz: float | None,
        interpolate_grippers: bool,
        step_start: float,
        state_before: object,
        stop_callback: Callable[[], bool] | None,
    ) -> RightArmGripperAction:
        substeps = _interpolation_substeps(dt=interpolation_dt, hz=interpolation_hz)
        if substeps <= 1:
            self.robot.send_action(action)
            self._last_sent_action = action.copy()
            return action

        sub_dt = float(interpolation_dt) / float(substeps)
        start_action = self._last_sent_action
        start_right = (
            np.asarray(state_before.right_arm_q, dtype=np.float32)
            if start_action is None
            else start_action.right_arm_q
        )
        start_left = (
            np.asarray(state_before.left_arm_q, dtype=np.float32)
            if start_action is None or not start_action.control_left_arm
            else start_action.left_arm_q
        )
        start_right_gripper = (
            np.asarray(state_before.right_gripper_q, dtype=np.float32)
            if start_action is None
            else start_action.right_gripper_q
        )
        start_left_gripper = (
            np.asarray(state_before.left_gripper_q, dtype=np.float32)
            if start_action is None or not start_action.control_left_gripper
            else start_action.left_gripper_q
        )
        interpolation_mode = (
            self.slow_dispatch_interpolation_mode
            if self.slow_dispatch_enabled
            else self.arm_interpolation_mode
        )
        sent_action = action
        for interp_idx in range(substeps):
            if stop_callback is not None and stop_callback():
                break
            s = float(interp_idx + 1) / float(substeps)
            alpha = _interpolation_alpha(s, interpolation_mode)
            sent_action = RightArmGripperAction(
                right_arm_q=start_right + alpha * (action.right_arm_q - start_right),
                left_arm_q=start_left + alpha * (action.left_arm_q - start_left),
                right_gripper_q=(
                    start_right_gripper
                    + alpha * (action.right_gripper_q - start_right_gripper)
                    if interpolate_grippers
                    else action.right_gripper_q.copy()
                ),
                left_gripper_q=(
                    start_left_gripper
                    + alpha * (action.left_gripper_q - start_left_gripper)
                    if interpolate_grippers
                    else action.left_gripper_q.copy()
                ),
                control_left_arm=action.control_left_arm,
                control_left_gripper=action.control_left_gripper,
            )
            self.robot.send_action(sent_action)
            self._last_sent_action = sent_action.copy()
            if interp_idx < substeps - 1:
                _sleep_until(step_start + (interp_idx + 1) * sub_dt, stop_callback)
        return sent_action

    def _log(self, event: str, **payload: object) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger(event, **payload)
        except Exception:
            return


def _interruptible_sleep(
    seconds: float,
    stop_callback: Callable[[], bool] | None,
    quantum: float = 0.01,
) -> None:
    deadline = time.perf_counter() + seconds
    while True:
        if stop_callback is not None and stop_callback():
            return
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(quantum, remaining))


def _sleep_until(
    deadline: float,
    stop_callback: Callable[[], bool] | None,
    quantum: float = 0.001,
) -> None:
    while True:
        if stop_callback is not None and stop_callback():
            return
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(quantum, remaining))


def _interpolation_substeps(*, dt: float, hz: float | None) -> int:
    if hz is None or hz <= 0.0 or dt <= 0.0:
        return 1
    return max(int(round(float(dt) * float(hz))), 1)


def _interpolation_alpha(s: float, mode: str) -> float:
    s = min(max(float(s), 0.0), 1.0)
    if mode == "cubic":
        return 3.0 * s * s - 2.0 * s * s * s
    if mode == "linear":
        return s
    raise ValueError(f"unsupported arm interpolation mode: {mode!r}")
