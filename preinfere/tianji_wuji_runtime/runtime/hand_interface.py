"""Dexterous hand SDK boundary for Tianji Wuji runtime."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from . import schema


class HandError(RuntimeError):
    """Raised by hand SDK adapters."""


@dataclass
class HandConnectionConfig:
    side: str
    ip: str | None = None
    port: int | None = None
    serial_number: str | None = None
    home_position: np.ndarray | None = None
    lowpass_cutoff_hz: float = 5.0
    unit: str = schema.STATE_UNIT


class HandInterface:
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        pass

    def shared_resource_id(self) -> object:
        return id(self)

    def get_joint_state(self) -> np.ndarray:
        """Return 20-dim joint positions."""
        raise NotImplementedError

    def send_joint_position(self, q: np.ndarray) -> None:
        raise NotImplementedError

    def hold_position(self) -> None:
        raise NotImplementedError

    def go_home(self) -> None:
        raise NotImplementedError


class FakeHandInterface(HandInterface):
    """In-memory hand used for dry-run and integration tests."""

    def __init__(self, config: HandConnectionConfig) -> None:
        self.config = config
        self.connected = False
        self.q = np.zeros(schema.LEFT_HAND_DOF, dtype=np.float32)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_state(self) -> np.ndarray:
        return self.q.astype(np.float32, copy=True)

    def send_joint_position(self, q: np.ndarray) -> None:
        arr = np.asarray(q, dtype=np.float32)
        if arr.shape != (schema.LEFT_HAND_DOF,):
            raise HandError(f"{self.config.side} hand command must have shape (20,), got {arr.shape}")
        self.q = arr.copy()

    def hold_position(self) -> None:
        return None

    def go_home(self) -> None:
        self.q[:] = 0.0


class WujiDirectHandInterface(HandInterface):
    """Host-side Wuji dexterous hand integration using wujihandpy."""

    def __init__(self, config: HandConnectionConfig) -> None:
        self.config = config
        self.connected = False
        self._module: Any | None = None
        self._hand: Any | None = None
        self._controller: Any | None = None
        self._last_command = np.zeros(schema.LEFT_HAND_DOF, dtype=np.float32)

    def connect(self) -> None:
        module = _load_wujihandpy()
        try:
            hand = module.Hand(serial_number=self.config.serial_number or None)
            disable_check = getattr(hand, "disable_thread_safe_check", None)
            if callable(disable_check):
                disable_check()
            hand.write_joint_enabled(True)
            controller = hand.realtime_controller(
                enable_upstream=False,
                filter=module.filter.LowPass(cutoff_freq=float(self.config.lowpass_cutoff_hz)),
            )
        except Exception as exc:  # noqa: BLE001
            raise HandError(f"failed to connect {self.config.side} Wuji hand: {exc}") from exc
        self._module = module
        self._hand = hand
        self._controller = controller
        self.connected = True
        try:
            self._last_command = self.get_joint_state()
        except Exception:
            self._last_command = np.zeros(schema.LEFT_HAND_DOF, dtype=np.float32)
        try:
            # Prime realtime communication with the current hardware pose, matching
            # the validated direct-driver behavior from the legacy stack.
            self.send_joint_position(self._last_command)
        except Exception as exc:  # noqa: BLE001
            raise HandError(
                f"failed to prime realtime control for {self.config.side} Wuji hand: {exc}"
            ) from exc

    def disconnect(self) -> None:
        hand = self._hand
        self._module = None
        self._hand = None
        self._controller = None
        self.connected = False
        if hand is None:
            return
        try:
            hand.write_joint_enabled(False)
        except Exception as exc:  # noqa: BLE001
            raise HandError(f"failed to disable {self.config.side} Wuji hand: {exc}") from exc

    def get_joint_state(self) -> np.ndarray:
        controller = self._controller
        hand = self._hand
        if hand is None:
            raise HandError(f"{self.config.side} Wuji hand is not connected")

        readers = []
        if controller is not None:
            readers.append(getattr(controller, "get_joint_actual_position", None))
        readers.append(getattr(hand, "get_joint_actual_position", None))
        readers.append(getattr(hand, "read_joint_actual_position", None))

        last_error: Exception | None = None
        for reader in readers:
            if not callable(reader):
                continue
            try:
                arr = _as_hand_joint_vector(reader(), name=f"{self.config.side}_hand_state")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            self._last_command = arr.copy()
            return arr

        if last_error is not None:
            raise HandError(
                f"failed to read {self.config.side} Wuji hand actual position: {last_error}"
            ) from last_error
        raise HandError(
            f"{self.config.side} Wuji hand SDK does not expose a readable actual-position API"
        )

    def send_joint_position(self, q: np.ndarray) -> None:
        controller = self._controller
        if controller is None:
            raise HandError(f"{self.config.side} Wuji hand is not connected")
        arr = _as_hand_joint_vector(q, name=f"{self.config.side}_hand_command")
        try:
            controller.set_joint_target_position(arr.reshape(5, 4))
        except Exception as exc:  # noqa: BLE001
            raise HandError(f"failed to send {self.config.side} Wuji hand command: {exc}") from exc
        self._last_command = arr

    def hold_position(self) -> None:
        try:
            target = self.get_joint_state()
        except Exception:
            target = self._last_command.copy()
        self.send_joint_position(target)

    def go_home(self) -> None:
        if self.config.home_position is None:
            self.hold_position()
            return
        self.send_joint_position(self.config.home_position)


def _load_wujihandpy() -> Any:
    try:
        return import_module("wujihandpy")
    except Exception as exc:  # noqa: BLE001
        raise HandError(
            "failed to import wujihandpy; install the Wuji hand SDK in the host runtime env"
        ) from exc


def _as_hand_joint_vector(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < schema.LEFT_HAND_DOF:
        raise HandError(f"{name} must provide at least {schema.LEFT_HAND_DOF} values, got {arr.size}")
    arr = arr[: schema.LEFT_HAND_DOF]
    if not np.all(np.isfinite(arr)):
        raise HandError(f"{name} contains NaN or Inf")
    return arr.astype(np.float32, copy=True)
