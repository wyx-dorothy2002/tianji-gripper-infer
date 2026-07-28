"""Keyboard polling and runtime state machine."""

from __future__ import annotations

from enum import Enum
import select
import sys
import termios
import tty


class RuntimeState(str, Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class RuntimeStateMachine:
    def __init__(
        self,
        *,
        auto_start: bool = False,
        safe_mode: bool = False,
        manual_gripper_open_key: str = "o",
    ) -> None:
        self.state = RuntimeState.RUNNING if auto_start else RuntimeState.STOPPED
        self.safe_mode = safe_mode
        normalized_key = (manual_gripper_open_key or "o").strip().lower()
        self.manual_gripper_open_key = normalized_key[:1] or "o"
        self.quit_requested = False
        self.step_requested = False
        self.recording = True
        self.home_requested = False
        self.manual_gripper_open_requested = False

    def update(self, key: str | None) -> None:
        if key is None:
            return
        key = key.lower()
        if key == " ":
            self.state = RuntimeState.STOPPED
        elif key == "r" and self.state != RuntimeState.ERROR:
            self.state = RuntimeState.RUNNING
        elif key == "p" and self.state != RuntimeState.ERROR:
            self.state = RuntimeState.PAUSED
        elif key == "h":
            self.home_requested = True
        elif key == self.manual_gripper_open_key:
            self.manual_gripper_open_requested = True
        elif key == "q":
            self.quit_requested = True
        elif key == "n" and self.safe_mode and self.state != RuntimeState.ERROR:
            self.step_requested = True
            self.state = RuntimeState.RUNNING
        elif key == "s":
            self.recording = True
        elif key == "d":
            self.recording = False

    def pause_after_safe_chunk(self) -> None:
        if self.safe_mode and self.state == RuntimeState.RUNNING:
            self.state = RuntimeState.PAUSED
            self.step_requested = False

    def to_error(self) -> None:
        self.state = RuntimeState.ERROR

    def consume_manual_gripper_open_request(self) -> bool:
        requested = self.manual_gripper_open_requested
        self.manual_gripper_open_requested = False
        return requested


class KeyboardController:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stdin.isatty()
        self._old_settings = None

    def __enter__(self) -> "KeyboardController":
        if self.enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return None
        return sys.stdin.read(1)
