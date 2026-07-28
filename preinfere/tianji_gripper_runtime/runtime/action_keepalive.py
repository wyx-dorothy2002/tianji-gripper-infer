"""Keep sending the last executed right-arm gripper action between chunks."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time

from .action_adapter import RightArmGripperAction
from .robot_interface import RobotError


EventLogger = Callable[..., None]


class ActionKeepalive:
    def __init__(self, robot, *, event_logger: EventLogger | None = None) -> None:
        self.robot = robot
        self.event_logger = event_logger
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._send_count = 0
        self._started_monotonic: float | None = None

    def start(self, action: RightArmGripperAction, interval_sec: float) -> None:
        if interval_sec <= 0:
            raise ValueError("keepalive interval must be positive")
        self.stop()
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._error = None
        self._send_count = 0
        self._started_monotonic = time.perf_counter()
        action_copy = action.copy()
        self._log("keepalive_start", interval_sec=float(interval_sec), action=action_copy.as_dict())
        thread = threading.Thread(
            target=self._run,
            args=(action_copy, float(interval_sec), stop_event),
            name="right-arm-gripper-keepalive",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None:
            return
        self._stop_event.set()
        if thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            send_count = self._send_count
            started_monotonic = self._started_monotonic
        self._log(
            "keepalive_stop",
            sent_count=send_count,
            duration_ms=None
            if started_monotonic is None
            else (time.perf_counter() - started_monotonic) * 1000.0,
            thread_alive_after_join=thread.is_alive(),
        )

    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    def raise_if_failed(self) -> None:
        error = self._error
        if error is None:
            return
        self._error = None
        if isinstance(error, RobotError):
            raise error
        raise RobotError(f"action keepalive failed: {error}") from error

    def _run(
        self,
        action: RightArmGripperAction,
        interval_sec: float,
        stop_event: threading.Event,
    ) -> None:
        send_index = 0
        while not stop_event.is_set():
            try:
                self.robot.send_action(action)
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
                stop_event.set()
                self._log("keepalive_error", send_index=send_index, error=repr(exc))
                break
            with self._lock:
                self._send_count = send_index + 1
            self._log("keepalive_send", send_index=send_index, action=action.as_dict())
            send_index += 1
            stop_event.wait(interval_sec)

    def _log(self, event: str, **payload: object) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger(event, **payload)
        except Exception:
            return
