"""Fixed-period action chunk executor."""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from .action_adapter import ActionAdapter, DualArmHandAction
from .recorder import Recorder
from .robot_interface import DualArmHandRobot, RobotError
from .ros2_jointstate_publisher import Ros2JointStatePublisher


class ActionExecutor:
    def __init__(
        self,
        robot: DualArmHandRobot,
        *,
        adapter: ActionAdapter,
        recorder: Recorder | None = None,
        event_logger: Callable[..., None] | None = None,
        ros_publisher: Ros2JointStatePublisher | None = None,
        arm_interpolation_hz: float | None = None,
        arm_interpolation_mode: str = "cubic",
    ) -> None:
        self.robot = robot
        self.adapter = adapter
        self.recorder = recorder
        self.event_logger = event_logger
        self.ros_publisher = ros_publisher
        if arm_interpolation_mode not in {"cubic", "linear"}:
            raise ValueError(
                "arm_interpolation_mode must be one of {'cubic', 'linear'}, "
                f"got {arm_interpolation_mode!r}"
            )
        self.arm_interpolation_hz = arm_interpolation_hz
        self.arm_interpolation_mode = arm_interpolation_mode
        self._last_sent_action: DualArmHandAction | None = None

    def execute_chunk(
        self,
        actions: list[DualArmHandAction],
        dt: float,
        *,
        dry_run: bool = False,
        chunk_index: int = 0,
        raw_chunk: np.ndarray | None = None,
        safety_events: list[dict[str, object]] | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> DualArmHandAction | None:
        last_executed_action: DualArmHandAction | None = None
        for step_idx, action in enumerate(actions):
            if stop_callback is not None and stop_callback():
                self._log(
                    "executor_stop_requested",
                    chunk_index=chunk_index,
                    step_in_chunk=step_idx,
                )
                self.robot.hold_position()
                break
            step_start = time.perf_counter()
            state_before_t0: float | None = None
            state_before_t1: float | None = None
            send_t0: float | None = None
            send_t1: float | None = None
            state_after_t0: float | None = None
            state_after_t1: float | None = None
            state_before = None
            state_after = None
            executed = False
            error: str | None = None
            try:
                state_before_t0 = time.perf_counter()
                state_before = self.robot.get_state()
                state_before_t1 = time.perf_counter()
                if not dry_run:
                    send_t0 = time.perf_counter()
                    sent_action = self._send_action_with_optional_arm_interpolation(
                        action=action,
                        dt=dt,
                        step_start=step_start,
                        state_before=state_before,
                        stop_callback=stop_callback,
                    )
                    send_t1 = time.perf_counter()
                    executed = True
                    last_executed_action = sent_action.copy()
                state_after_t0 = time.perf_counter()
                state_after = self.robot.get_state()
                state_after_t1 = time.perf_counter()
                if self.ros_publisher is not None:
                    self.ros_publisher.publish(
                        target_action=action,
                        current_state=state_after,
                    )
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
                    step_start_monotonic=step_start,
                    step_end_monotonic=step_start + elapsed,
                    control_latency_ms=elapsed * 1000.0,
                    period_overrun_ms=max((elapsed - float(dt)) * 1000.0, 0.0),
                    executed=executed,
                    dry_run=dry_run,
                    error=error,
                    state_before_latency_ms=_elapsed_ms(state_before_t0, state_before_t1),
                    send_latency_ms=_elapsed_ms(send_t0, send_t1),
                    state_after_latency_ms=_elapsed_ms(state_after_t0, state_after_t1),
                    action=_action_debug_summary(action),
                    arm_interpolation=_interpolation_debug_summary(
                        dt=dt,
                        hz=self.arm_interpolation_hz,
                        mode=self.arm_interpolation_mode,
                    ),
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
            remaining = float(dt) - elapsed
            if remaining > 0:
                _interruptible_sleep(remaining, stop_callback)
            else:
                print(f"[executor] warning: control step overran by {-remaining * 1000.0:.2f} ms")
        return last_executed_action

    def _send_action_with_optional_arm_interpolation(
        self,
        *,
        action: DualArmHandAction,
        dt: float,
        step_start: float,
        state_before: object,
        stop_callback: Callable[[], bool] | None,
    ) -> DualArmHandAction:
        substeps = _interpolation_substeps(dt=dt, hz=self.arm_interpolation_hz)
        if substeps <= 1:
            self.robot.send_action(action)
            self._last_sent_action = action.copy()
            return action

        if not hasattr(state_before, "left_arm_q") or not hasattr(state_before, "right_arm_q"):
            self.robot.send_action(action)
            self._last_sent_action = action.copy()
            return action

        sub_dt = float(dt) / float(substeps)
        sent_action = action
        start_action = self._last_sent_action
        start_left = (
            np.asarray(state_before.left_arm_q, dtype=np.float32)
            if start_action is None
            else start_action.left_arm_q
        )
        start_right = (
            np.asarray(state_before.right_arm_q, dtype=np.float32)
            if start_action is None
            else start_action.right_arm_q
        )
        for interp_idx in range(substeps):
            if stop_callback is not None and stop_callback():
                break
            s = float(interp_idx + 1) / float(substeps)
            alpha = _interpolation_alpha(s, self.arm_interpolation_mode)
            sent_action = _interpolate_arm_action(
                start_left=start_left,
                start_right=start_right,
                target=action,
                alpha=alpha,
            )
            self.robot.send_action(sent_action)
            self._last_sent_action = sent_action.copy()
            if interp_idx < substeps - 1:
                target_time = step_start + (interp_idx + 1) * sub_dt
                _sleep_until(target_time, stop_callback)
        return sent_action

    def _log(self, event: str, **payload: object) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger(event, **payload)
        except Exception:
            # Logging must never interrupt the control path.
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


def _interpolate_arm_action(
    *,
    start_left: np.ndarray,
    start_right: np.ndarray,
    target: DualArmHandAction,
    alpha: float,
) -> DualArmHandAction:
    return DualArmHandAction(
        left_arm_q=(start_left + alpha * (target.left_arm_q - start_left)).astype(np.float32),
        left_hand_q=target.left_hand_q.copy(),
        right_arm_q=(start_right + alpha * (target.right_arm_q - start_right)).astype(np.float32),
        right_hand_q=target.right_hand_q.copy(),
    )


def _interpolation_debug_summary(
    *,
    dt: float,
    hz: float | None,
    mode: str,
) -> dict[str, object]:
    substeps = _interpolation_substeps(dt=dt, hz=hz)
    return {
        "enabled": substeps > 1,
        "mode": mode,
        "hz": hz,
        "substeps": substeps,
    }


def _elapsed_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _action_debug_summary(action: DualArmHandAction) -> dict[str, object]:
    right_arm = action.right_arm_q
    right_hand = action.right_hand_q
    return {
        "right_arm": right_arm.tolist(),
        "right_hand": right_hand.tolist(),
        "right_arm_l2": float((right_arm * right_arm).sum() ** 0.5),
        "right_hand_l2": float((right_hand * right_hand).sum() ** 0.5),
    }
