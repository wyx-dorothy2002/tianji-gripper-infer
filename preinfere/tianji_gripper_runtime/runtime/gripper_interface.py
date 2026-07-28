"""Single-DoF gripper SDK boundary for the right-arm runtime."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import time
from typing import Any

import numpy as np

from . import schema


class GripperError(RuntimeError):
    """Raised by gripper SDK adapters."""


@dataclass
class GripperConnectionConfig:
    side: str
    ip: str | None = None
    port: int | None = None
    serial_number: str | None = None
    home_position: np.ndarray | None = None
    sdk_module: str | None = None
    sdk_class: str | None = None
    unit: str = schema.STATE_UNIT
    end_channel_arm: str = "A"
    end_channel_com: int = 1
    end_channel_direct: bool = True
    rs05_target_id: int = 0x7F
    rs05_master_id: int = 0xFD
    rs05_can_id_byteorder: str = "little"
    rs05_standard_id_bytes: int = 4
    rs05_enter_motor: bool = True
    rs05_stop_on_disconnect: bool = False
    rs05_kp: float = 80.0
    rs05_kd: float = 1.0
    rs05_torque_nm: float = 0.0
    rs05_min_pos_rad: float = -5.5
    rs05_max_pos_rad: float = 1.2


class GripperInterface:
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        pass

    def shared_resource_id(self) -> object:
        return id(self)

    def get_position(self) -> np.ndarray:
        """Return a one-dimensional gripper position array."""
        raise NotImplementedError

    def send_position(self, q: np.ndarray) -> None:
        raise NotImplementedError

    def hold_position(self) -> None:
        raise NotImplementedError

    def go_home(self) -> None:
        raise NotImplementedError


class FakeGripperInterface(GripperInterface):
    """In-memory gripper used for dry-runs and until the real SDK is wired."""

    def __init__(self, config: GripperConnectionConfig) -> None:
        self.config = config
        self.connected = False
        self.q = np.zeros(schema.RIGHT_GRIPPER_DOF, dtype=np.float32)

    def connect(self) -> None:
        self.connected = True
        if self.config.home_position is not None:
            self.q = _as_gripper_position(self.config.home_position, name="home_position")

    def disconnect(self) -> None:
        self.connected = False

    def get_position(self) -> np.ndarray:
        return self.q.copy()

    def send_position(self, q: np.ndarray) -> None:
        self.q = _as_gripper_position(q, name=f"{self.config.side}_gripper_command")

    def hold_position(self) -> None:
        return None

    def go_home(self) -> None:
        if self.config.home_position is not None:
            self.send_position(self.config.home_position)


class GenericSdkGripperInterface(GripperInterface):
    """Small adapter placeholder for a vendor gripper SDK.

    The vendor object may expose common methods such as connect/disconnect,
    get_position/read_position/get_state, and send_position/set_position/move_to.
    If its API differs, subclass GripperInterface and plug it in here.
    """

    def __init__(self, config: GripperConnectionConfig) -> None:
        self.config = config
        self._device: Any | None = None
        self._last_command = np.zeros(schema.RIGHT_GRIPPER_DOF, dtype=np.float32)

    def connect(self) -> None:
        if not self.config.sdk_module or not self.config.sdk_class:
            raise GripperError(
                "real gripper backend requires sdk_module and sdk_class; "
                "use fake gripper until the vendor SDK binding is known"
            )
        try:
            module = import_module(self.config.sdk_module)
            cls = getattr(module, self.config.sdk_class)
            self._device = cls(
                side=self.config.side,
                ip=self.config.ip,
                port=self.config.port,
                serial_number=self.config.serial_number,
            )
        except TypeError:
            self._device = cls()
        except Exception as exc:  # noqa: BLE001
            raise GripperError(
                f"failed to create {self.config.side} gripper SDK object: {exc}"
            ) from exc
        connect = getattr(self._device, "connect", None)
        if callable(connect):
            connect()
        try:
            self._last_command = self.get_position()
        except Exception:
            if self.config.home_position is not None:
                self._last_command = _as_gripper_position(
                    self.config.home_position,
                    name="home_position",
                )

    def disconnect(self) -> None:
        device = self._device
        self._device = None
        if device is None:
            return
        disconnect = getattr(device, "disconnect", None)
        if callable(disconnect):
            disconnect()

    def get_position(self) -> np.ndarray:
        device = self._require_device()
        for name in ("get_position", "read_position", "get_state", "read_state"):
            reader = getattr(device, name, None)
            if callable(reader):
                arr = _as_gripper_position(reader(), name=f"{self.config.side}_gripper_state")
                self._last_command = arr.copy()
                return arr
        return self._last_command.copy()

    def send_position(self, q: np.ndarray) -> None:
        device = self._require_device()
        arr = _as_gripper_position(q, name=f"{self.config.side}_gripper_command")
        for name in ("send_position", "set_position", "move_to", "command_position"):
            writer = getattr(device, name, None)
            if callable(writer):
                writer(float(arr[0]))
                self._last_command = arr.copy()
                return
        raise GripperError(
            f"{self.config.side} gripper SDK object has no recognized position command method"
        )

    def hold_position(self) -> None:
        self.send_position(self.get_position())

    def go_home(self) -> None:
        if self.config.home_position is None:
            self.hold_position()
            return
        self.send_position(self.config.home_position)

    def _require_device(self) -> Any:
        if self._device is None:
            raise GripperError(f"{self.config.side} gripper is not connected")
        return self._device


class Rs05MitEndChannelGripperInterface(GripperInterface):
    """RS05 MIT position backend using the shared Tianji/Marvin end channel."""

    def __init__(self, config: GripperConnectionConfig, end_channel: Any) -> None:
        if end_channel is None:
            raise GripperError("rs05_mit_end_channel backend requires a shared arm end channel")
        self.config = config
        self.end_channel = end_channel
        self._connected = False
        self._last_command = np.zeros(schema.RIGHT_GRIPPER_DOF, dtype=np.float32)
        self.last_feedback_position_rad: float | None = None
        self.last_feedback_velocity_rad_s: float | None = None
        self.last_feedback_torque_nm: float | None = None
        self.last_feedback_mode_state: int | None = None
        self.last_feedback_has_fault: int | None = None
        self.last_feedback_has_warning: int | None = None
        self.last_feedback_temperature_c: float | None = None
        if config.home_position is not None:
            self._last_command = _as_gripper_position(config.home_position, name="home_position")

    def connect(self) -> None:
        if self.config.rs05_enter_motor:
            for command in ("clear-error", "enter-motor"):
                self._send_frame(
                    _rs05_mit_command_hex(
                        command,
                        target_id=self.config.rs05_target_id,
                        master_id=self.config.rs05_master_id,
                        can_id_byteorder=self.config.rs05_can_id_byteorder,
                        standard_id_bytes=self.config.rs05_standard_id_bytes,
                    )
                )
                time.sleep(0.02)
            time.sleep(0.08)
        feedback = self._read_latest_feedback()
        if feedback is not None:
            self._last_command = np.asarray([feedback], dtype=np.float32)
        self._connected = True

    def disconnect(self) -> None:
        if self._connected and self.config.rs05_stop_on_disconnect:
            self._send_frame(
                _rs05_mit_command_hex(
                    "stop",
                    target_id=self.config.rs05_target_id,
                    master_id=self.config.rs05_master_id,
                    can_id_byteorder=self.config.rs05_can_id_byteorder,
                    standard_id_bytes=self.config.rs05_standard_id_bytes,
                )
            )
        self._connected = False

    def get_position(self) -> np.ndarray:
        feedback = self._read_latest_feedback()
        if feedback is not None:
            self._last_command = np.asarray([feedback], dtype=np.float32)
        return self._last_command.copy()

    def send_position(self, q: np.ndarray) -> None:
        arr = _as_gripper_position(q, name=f"{self.config.side}_gripper_command")
        target = float(
            np.clip(arr[0], self.config.rs05_min_pos_rad, self.config.rs05_max_pos_rad)
        )
        payload = _rs05_mit_control_payload(
            position_rad=target,
            velocity_rad_s=0.0,
            kp=self.config.rs05_kp,
            kd=self.config.rs05_kd,
            torque_nm=self.config.rs05_torque_nm,
        )
        self._send_frame(
            _rs05_mit_marvin_hex(
                can_id=self.config.rs05_target_id,
                payload=payload,
                can_id_byteorder=self.config.rs05_can_id_byteorder,
                standard_id_bytes=self.config.rs05_standard_id_bytes,
            )
        )
        self._last_command = np.asarray([target], dtype=np.float32)

    def hold_position(self) -> None:
        self.send_position(self._last_command)

    def go_home(self) -> None:
        if self.config.home_position is None:
            self.hold_position()
            return
        self.send_position(self.config.home_position)

    def shared_resource_id(self) -> object:
        return (
            id(self.end_channel),
            self.config.side,
            self.config.end_channel_arm,
            self.config.end_channel_com,
        )

    def _send_frame(self, frame_hex: str) -> None:
        try:
            self.end_channel.send_end_channel_data(
                self.config.end_channel_arm,
                frame_hex,
                self.config.end_channel_com,
                self.config.end_channel_direct,
            )
        except Exception as exc:  # noqa: BLE001
            raise GripperError(
                f"failed to send {self.config.side} gripper end-channel frame: {exc}"
            ) from exc

    def _read_latest_feedback(self) -> float | None:
        reader = getattr(self.end_channel, "get_end_channel_data", None)
        if not callable(reader):
            return None
        latest: float | None = None
        for _ in range(8):
            try:
                size, _, hex_text = reader(
                    self.config.end_channel_arm,
                    self.config.end_channel_com,
                    self.config.end_channel_direct,
                )
            except Exception:
                break
            if int(size) <= 0 or not hex_text:
                break
            frame = _parse_hex_bytes(hex_text)
            payload = frame[4:12] if len(frame) >= 12 else frame
            decoded = _decode_rs05_mit_feedback_payload(payload)
            velocity = decoded.get("velocity_rad_s")
            torque = decoded.get("torque_nm")
            mode_state = decoded.get("mode_state")
            has_fault = decoded.get("has_fault")
            has_warning = decoded.get("has_warning")
            temperature_c = decoded.get("temperature_c")
            position = decoded.get("position_rad")
            if position is not None:
                latest = float(position)
                self.last_feedback_position_rad = latest
            if velocity is not None:
                self.last_feedback_velocity_rad_s = float(velocity)
            if torque is not None:
                self.last_feedback_torque_nm = float(torque)
            if mode_state is not None:
                self.last_feedback_mode_state = int(mode_state)
            if has_fault is not None:
                self.last_feedback_has_fault = int(has_fault)
            if has_warning is not None:
                self.last_feedback_has_warning = int(has_warning)
            if temperature_c is not None:
                self.last_feedback_temperature_c = float(temperature_c)
        return latest


def make_gripper(
    config: GripperConnectionConfig,
    *,
    backend: str,
    end_channel: Any | None = None,
) -> GripperInterface:
    if backend in {"fake", "tianji"}:
        return FakeGripperInterface(config)
    if backend in {"generic_sdk", "sdk"}:
        return GenericSdkGripperInterface(config)
    if backend in {"rs05_mit_end_channel", "tianji_end_channel", "marvin_end_channel"}:
        return Rs05MitEndChannelGripperInterface(config, end_channel)
    raise GripperError(f"unknown gripper backend {backend!r}")


def _as_gripper_position(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < schema.RIGHT_GRIPPER_DOF:
        raise GripperError(f"{name} must provide at least one value, got {arr.size}")
    arr = arr[: schema.RIGHT_GRIPPER_DOF]
    if not np.all(np.isfinite(arr)):
        raise GripperError(f"{name} contains NaN or Inf")
    return arr.astype(np.float32, copy=True)


def _float_to_uint(value: float, min_value: float, max_value: float, bits: int) -> int:
    value = min(max(float(value), min_value), max_value)
    span = max_value - min_value
    return int((value - min_value) * ((1 << bits) - 1) / span)


def _uint_to_float(value: int, min_value: float, max_value: float, bits: int) -> float:
    span = max_value - min_value
    return float(value) * span / float((1 << bits) - 1) + min_value


def _rs05_mit_control_payload(
    position_rad: float,
    velocity_rad_s: float,
    kp: float,
    kd: float,
    torque_nm: float,
) -> bytes:
    p_int = _float_to_uint(position_rad, -12.5, 12.5, 16)
    v_int = _float_to_uint(velocity_rad_s, -50.0, 50.0, 12)
    kp_int = _float_to_uint(kp, 0.0, 500.0, 12)
    kd_int = _float_to_uint(kd, 0.0, 5.0, 12)
    t_int = _float_to_uint(torque_nm, -5.5, 5.5, 12)
    return bytes(
        [
            (p_int >> 8) & 0xFF,
            p_int & 0xFF,
            (v_int >> 4) & 0xFF,
            ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F),
            kp_int & 0xFF,
            (kd_int >> 4) & 0xFF,
            ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F),
            t_int & 0xFF,
        ]
    )


def _decode_rs05_mit_feedback_payload(payload: bytes) -> dict[str, float | int]:
    if len(payload) < 8:
        return {}
    p_int = (int(payload[1]) << 8) | int(payload[2])
    v_int = (int(payload[3]) << 4) | (int(payload[4]) >> 4)
    t_int = ((int(payload[4]) & 0x0F) << 8) | int(payload[5])
    state_temp = int(payload[6])
    temperature_raw = ((state_temp & 0x0F) << 8) | int(payload[7])
    return {
        "motor_id": int(payload[0]),
        "position_rad": _uint_to_float(p_int, -12.5, 12.5, 16),
        "velocity_rad_s": _uint_to_float(v_int, -50.0, 50.0, 12),
        "torque_nm": _uint_to_float(t_int, -5.5, 5.5, 12),
        "mode_state": (state_temp >> 6) & 0x03,
        "has_fault": (state_temp >> 5) & 0x01,
        "has_warning": (state_temp >> 4) & 0x01,
        "temperature_raw": temperature_raw,
        "temperature_c": temperature_raw / 10.0,
    }


def _rs05_mit_marvin_hex(
    can_id: int,
    payload: bytes,
    can_id_byteorder: str = "little",
    standard_id_bytes: int = 4,
) -> str:
    if len(payload) != 8:
        raise GripperError("RS05 MIT payload must be exactly 8 bytes")
    if standard_id_bytes not in (2, 4):
        raise GripperError("RS05 standard CAN ID prefix must be 2 or 4 bytes")
    prefix = (int(can_id) & 0x7FF).to_bytes(
        int(standard_id_bytes),
        byteorder=can_id_byteorder,
        signed=False,
    )
    return _hex_bytes(prefix + payload)


def _rs05_mit_command_hex(
    command: str,
    target_id: int,
    master_id: int,
    can_id_byteorder: str = "little",
    standard_id_bytes: int = 4,
) -> str:
    del master_id
    payload_by_command = {
        "read-error": "FF FF FF FF FF FF 00 FB",
        "enter-motor": "FF FF FF FF FF FF FF FC",
        "set-zero": "FF FF FF FF FF FF FF FE",
        "clear-error": "FF FF FF FF FF FF FF FB",
        "stop": "FF FF FF FF FF FF FF FD",
    }
    normalized = command.strip().lower().replace("_", "-")
    if normalized == "query-id":
        normalized = "read-error"
    if normalized not in payload_by_command:
        raise GripperError(f"unsupported RS05 MIT command: {command}")
    return _rs05_mit_marvin_hex(
        can_id=target_id,
        payload=_parse_hex_bytes(payload_by_command[normalized]),
        can_id_byteorder=can_id_byteorder,
        standard_id_bytes=standard_id_bytes,
    )


def _parse_hex_bytes(text: str) -> bytes:
    cleaned = str(text).replace(",", " ").replace("0x", " ")
    parts = [part for part in cleaned.split() if part]
    try:
        return bytes(int(part, 16) & 0xFF for part in parts)
    except ValueError as exc:
        raise GripperError(f"invalid hex bytes: {text!r}") from exc


def _hex_bytes(data: bytes) -> str:
    return " ".join(f"{value:02X}" for value in data)
