"""Unified dual-arm dual-hand robot interface."""

from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np

from . import schema
from .action_adapter import DualArmHandAction
from .arm_interface import ArmConnectionConfig, ArmInterface, FakeArmInterface, TianjiArmInterface
from .hand_interface import (
    FakeHandInterface,
    HandConnectionConfig,
    HandInterface,
    WujiDirectHandInterface,
)
from .robot_state import DualArmHandState
from .tianji_arm_system import TianjiDualArmSystem, TianjiHostConfig


class RobotError(RuntimeError):
    """Unified runtime robot error."""


@dataclass
class RobotConnectionConfig:
    backend: str = "fake"
    robot_ip: str | None = None
    left_arm_ip: str | None = None
    left_arm_port: int | None = None
    right_arm_ip: str | None = None
    right_arm_port: int | None = None
    left_hand_ip: str | None = None
    left_hand_port: int | None = None
    right_hand_ip: str | None = None
    right_hand_port: int | None = None
    left_hand_serial: str | None = None
    right_hand_serial: str | None = None
    left_hand_home: tuple[float, ...] | None = None
    right_hand_home: tuple[float, ...] | None = None
    hand_lowpass_cutoff_hz: float = 5.0
    tianji_sdk_root: str | None = None
    tianji_config_path: str | None = None


class DualArmHandRobot:
    """Composes four SDK interfaces behind a single safe contract."""

    def __init__(
        self,
        left_arm: ArmInterface,
        left_hand: HandInterface,
        right_arm: ArmInterface,
        right_hand: HandInterface,
    ) -> None:
        self.left_arm = left_arm
        self.left_hand = left_hand
        self.right_arm = right_arm
        self.right_hand = right_hand
        self._connected = False
        self._io_lock = threading.RLock()

    def connect(self) -> None:
        with self._io_lock:
            try:
                self._call_once(
                    (self.left_arm, self.left_hand, self.right_arm, self.right_hand),
                    "connect",
                )
                self._connected = True
            except Exception as exc:  # noqa: BLE001 - SDKs throw mixed exception types.
                self.hold_position()
                raise RobotError(f"failed to connect robot interfaces: {exc}") from exc

    def disconnect(self) -> None:
        with self._io_lock:
            self._call_once(
                (self.left_arm, self.left_hand, self.right_arm, self.right_hand),
                "disconnect",
                ignore_errors=True,
            )
            self._connected = False

    def is_connected(self) -> bool:
        with self._io_lock:
            return self._connected

    def get_state(self) -> DualArmHandState:
        with self._io_lock:
            try:
                return DualArmHandState(
                    left_arm_q=self.left_arm.get_joint_state(),
                    right_arm_q=self.right_arm.get_joint_state(),
                    left_hand_q=self.left_hand.get_joint_state(),
                    right_hand_q=self.right_hand.get_joint_state(),
                )
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to read robot state: {exc}") from exc

    def send_action(self, action: DualArmHandAction) -> None:
        if not isinstance(action, DualArmHandAction):
            raise RobotError(f"send_action expects DualArmHandAction, got {type(action)!r}")
        with self._io_lock:
            try:
                self.left_arm.stage_joint_position(action.left_arm_q)
                self.left_hand.send_joint_position(action.left_hand_q)
                self.right_arm.stage_joint_position(action.right_arm_q)
                self.right_hand.send_joint_position(action.right_hand_q)
                self._flush_arm_commands()
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to send robot action: {exc}") from exc

    def hold_position(self) -> None:
        with self._io_lock:
            self._call_once(
                (self.left_arm, self.left_hand, self.right_arm, self.right_hand),
                "hold_position",
                ignore_errors=True,
            )

    def go_home(self) -> None:
        with self._io_lock:
            try:
                self._call_once(
                    (self.left_arm, self.left_hand, self.right_arm, self.right_hand),
                    "go_home",
                )
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to go home: {exc}") from exc

    def _flush_arm_commands(self) -> None:
        seen: set[object] = set()
        for arm in (self.left_arm, self.right_arm):
            resource_id = arm.shared_resource_id()
            if resource_id in seen:
                continue
            seen.add(resource_id)
            arm.flush_staged_commands()

    @staticmethod
    def _call_once(
        parts: tuple[object, ...], method_name: str, *, ignore_errors: bool = False
    ) -> None:
        seen: set[object] = set()
        for part in parts:
            resource_id = getattr(part, "shared_resource_id", lambda: id(part))()
            if resource_id in seen:
                continue
            seen.add(resource_id)
            try:
                getattr(part, method_name)()
            except Exception:
                if not ignore_errors:
                    raise


def make_robot(config: RobotConnectionConfig) -> DualArmHandRobot:
    left_hand_home = _optional_hand_home(config.left_hand_home, side="left")
    right_hand_home = _optional_hand_home(config.right_hand_home, side="right")
    if config.backend == "fake":
        return DualArmHandRobot(
            left_arm=FakeArmInterface(
                ArmConnectionConfig("left", ip=config.left_arm_ip, port=config.left_arm_port)
            ),
            left_hand=FakeHandInterface(
                HandConnectionConfig(
                    "left",
                    ip=config.left_hand_ip,
                    port=config.left_hand_port,
                    home_position=left_hand_home,
                )
            ),
            right_arm=FakeArmInterface(
                ArmConnectionConfig("right", ip=config.right_arm_ip, port=config.right_arm_port)
            ),
            right_hand=FakeHandInterface(
                HandConnectionConfig(
                    "right",
                    ip=config.right_hand_ip,
                    port=config.right_hand_port,
                    home_position=right_hand_home,
                )
            ),
        )
    if config.backend == "wuji_host":
        return DualArmHandRobot(
            left_arm=FakeArmInterface(
                ArmConnectionConfig("left", ip=config.left_arm_ip, port=config.left_arm_port)
            ),
            left_hand=WujiDirectHandInterface(
                HandConnectionConfig(
                    "left",
                    ip=config.left_hand_ip,
                    port=config.left_hand_port,
                    serial_number=config.left_hand_serial,
                    home_position=left_hand_home,
                    lowpass_cutoff_hz=config.hand_lowpass_cutoff_hz,
                )
            ),
            right_arm=FakeArmInterface(
                ArmConnectionConfig("right", ip=config.right_arm_ip, port=config.right_arm_port)
            ),
            right_hand=WujiDirectHandInterface(
                HandConnectionConfig(
                    "right",
                    ip=config.right_hand_ip,
                    port=config.right_hand_port,
                    serial_number=config.right_hand_serial,
                    home_position=right_hand_home,
                    lowpass_cutoff_hz=config.hand_lowpass_cutoff_hz,
                )
            ),
        )
    if config.backend in {"tianji", "tianji_host"}:
        robot_ip = config.robot_ip or config.left_arm_ip or config.right_arm_ip
        if robot_ip is None:
            raise RobotError("tianji backend requires --robot-ip or one arm IP to be set")
        arm_system = TianjiDualArmSystem(
            TianjiHostConfig(
                robot_ip=robot_ip,
                sdk_root=config.tianji_sdk_root,
                config_path=config.tianji_config_path,
            )
        )
        return DualArmHandRobot(
            left_arm=TianjiArmInterface(
                ArmConnectionConfig(
                    "left",
                    ip=robot_ip,
                    sdk_root=config.tianji_sdk_root,
                    config_path=config.tianji_config_path,
                ),
                system=arm_system,
            ),
            left_hand=FakeHandInterface(
                HandConnectionConfig(
                    "left",
                    ip=config.left_hand_ip,
                    port=config.left_hand_port,
                    serial_number=config.left_hand_serial,
                    home_position=left_hand_home,
                )
            ),
            right_arm=TianjiArmInterface(
                ArmConnectionConfig(
                    "right",
                    ip=robot_ip,
                    sdk_root=config.tianji_sdk_root,
                    config_path=config.tianji_config_path,
                ),
                system=arm_system,
            ),
            right_hand=FakeHandInterface(
                HandConnectionConfig(
                    "right",
                    ip=config.right_hand_ip,
                    port=config.right_hand_port,
                    serial_number=config.right_hand_serial,
                    home_position=right_hand_home,
                )
            ),
        )
    if config.backend == "tianji_wuji_host":
        robot_ip = config.robot_ip or config.left_arm_ip or config.right_arm_ip
        if robot_ip is None:
            raise RobotError("tianji_wuji_host backend requires --robot-ip or one arm IP to be set")
        arm_system = TianjiDualArmSystem(
            TianjiHostConfig(
                robot_ip=robot_ip,
                sdk_root=config.tianji_sdk_root,
                config_path=config.tianji_config_path,
            )
        )
        return DualArmHandRobot(
            left_arm=TianjiArmInterface(
                ArmConnectionConfig(
                    "left",
                    ip=robot_ip,
                    sdk_root=config.tianji_sdk_root,
                    config_path=config.tianji_config_path,
                ),
                system=arm_system,
            ),
            left_hand=WujiDirectHandInterface(
                HandConnectionConfig(
                    "left",
                    ip=config.left_hand_ip,
                    port=config.left_hand_port,
                    serial_number=config.left_hand_serial,
                    home_position=left_hand_home,
                    lowpass_cutoff_hz=config.hand_lowpass_cutoff_hz,
                )
            ),
            right_arm=TianjiArmInterface(
                ArmConnectionConfig(
                    "right",
                    ip=robot_ip,
                    sdk_root=config.tianji_sdk_root,
                    config_path=config.tianji_config_path,
                ),
                system=arm_system,
            ),
            right_hand=WujiDirectHandInterface(
                HandConnectionConfig(
                    "right",
                    ip=config.right_hand_ip,
                    port=config.right_hand_port,
                    serial_number=config.right_hand_serial,
                    home_position=right_hand_home,
                    lowpass_cutoff_hz=config.hand_lowpass_cutoff_hz,
                )
            ),
        )
    else:
        raise RobotError(
            f"robot backend {config.backend!r} is not implemented in this local runtime. "
            "Wire the vendor SDK by implementing ArmInterface/HandInterface."
        )


def _optional_hand_home(value: tuple[float, ...] | None, *, side: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (schema.LEFT_HAND_DOF,):
        raise RobotError(
            f"{side}_hand_home must have {schema.LEFT_HAND_DOF} values, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise RobotError(f"{side}_hand_home contains NaN or Inf")
    return arr
