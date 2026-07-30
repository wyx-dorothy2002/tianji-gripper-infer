"""Unified Tianji dual-arm + dual-gripper robot interface."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

import numpy as np

from tianji_wuji_runtime.runtime.arm_interface import (
    ArmConnectionConfig,
    ArmInterface,
    FakeArmInterface,
    TianjiArmInterface,
)
from tianji_wuji_runtime.runtime.tianji_arm_system import TianjiDualArmSystem, TianjiHostConfig

from .action_adapter import RightArmGripperAction
from .gripper_interface import GripperConnectionConfig, GripperInterface, make_gripper
from .robot_state import RightArmGripperState


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
    left_gripper_ip: str | None = None
    left_gripper_port: int | None = None
    right_gripper_ip: str | None = None
    right_gripper_port: int | None = None
    left_gripper_serial: str | None = None
    right_gripper_serial: str | None = None
    left_gripper_home: tuple[float, ...] | None = None
    right_gripper_home: tuple[float, ...] | None = None
    gripper_backend: str = "fake"
    left_gripper_backend: str | None = None
    right_gripper_backend: str | None = None
    gripper_sdk_module: str | None = None
    gripper_sdk_class: str | None = None
    tianji_sdk_root: str | None = None
    tianji_config_path: str | None = None
    tianji_vel_ratio: int = 80
    tianji_acc_ratio: int = 70
    tianji_settle_sec: float = 0.5
    tianji_tool_arm_a_kinematics: tuple[float, ...] | None = None
    tianji_tool_arm_a_dynamics: tuple[float, ...] | None = None
    tianji_tool_arm_b_kinematics: tuple[float, ...] | None = None
    tianji_tool_arm_b_dynamics: tuple[float, ...] | None = None
    tianji_state3_joint_k_a: tuple[float, ...] | None = None
    tianji_state3_joint_d_a: tuple[float, ...] | None = None
    tianji_state3_joint_k_b: tuple[float, ...] | None = None
    tianji_state3_joint_d_b: tuple[float, ...] | None = None
    tianji_home_joints_a: tuple[float, ...] | None = None
    tianji_home_joints_b: tuple[float, ...] | None = None
    gripper_485_arm: str = "A"
    gripper_485_com: int = 1
    left_gripper_485_arm: str | None = None
    left_gripper_485_com: int | None = None
    right_gripper_485_arm: str | None = None
    right_gripper_485_com: int | None = None
    gripper_direct_onset: bool = True
    gripper_rs05_target_id: int = 0x7F
    gripper_rs05_master_id: int = 0xFD
    gripper_can_id_byteorder: str = "little"
    gripper_standard_id_bytes: int = 4
    gripper_enter_motor: bool = True
    gripper_stop_on_disconnect: bool = False
    gripper_kp: float = 80.0
    gripper_kd: float = 1.0
    gripper_torque_nm: float = 0.0
    gripper_min_pos_rad: float = -5.5
    gripper_max_pos_rad: float = 1.2
    gripper_torque_protection_enabled: bool = False
    gripper_torque_protection_mode: str = "torque"
    gripper_torque_filter_alpha: float = 0.3
    gripper_torque_threshold_nm: float = 1.0
    gripper_torque_release_threshold_nm: float = 0.2
    gripper_torque_count_threshold: int = 5
    gripper_torque_extra_tighten_rad: float = 0.0
    gripper_holding_kp: float = 4.0
    gripper_holding_kd: float = 0.1
    gripper_closing_direction: float = 1.0
    gripper_torque_direction_deadband_rad: float = 0.01
    # 左右夹爪可以使用不同的抓取力矩标定。空值表示继续使用上面的公共参数，
    # 以兼容已有配置和只控制单夹爪的调用方。
    left_gripper_torque_threshold_nm: float | None = None
    left_gripper_torque_release_threshold_nm: float | None = None
    left_gripper_torque_count_threshold: int | None = None
    left_gripper_torque_extra_tighten_rad: float | None = None
    left_gripper_holding_kp: float | None = None
    left_gripper_holding_kd: float | None = None
    right_gripper_torque_threshold_nm: float | None = None
    right_gripper_torque_release_threshold_nm: float | None = None
    right_gripper_torque_count_threshold: int | None = None
    right_gripper_torque_extra_tighten_rad: float | None = None
    right_gripper_holding_kp: float | None = None
    right_gripper_holding_kd: float | None = None
    command_left_side: bool = True
    left_arm_freeze_q: tuple[float, ...] | None = None
    left_gripper_freeze_q: tuple[float, ...] | None = None


class RightArmGripperRobot:
    """Composes Tianji arms and left/right gripper interfaces.

    In right-only mode, the left arm and left gripper are held at their frozen
    connected positions. In dual mode, all four effectors are commanded.
    """

    def __init__(
        self,
        left_arm: ArmInterface,
        right_arm: ArmInterface,
        left_gripper: GripperInterface,
        right_gripper: GripperInterface,
        *,
        event_logger: Callable[..., None] | None = None,
        command_left_side: bool = True,
        left_arm_freeze_q: np.ndarray | None = None,
        left_gripper_freeze_q: np.ndarray | None = None,
    ) -> None:
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.left_gripper = left_gripper
        self.right_gripper = right_gripper
        self.event_logger = event_logger
        self.command_left_side = command_left_side
        if command_left_side and (left_arm_freeze_q is not None or left_gripper_freeze_q is not None):
            raise RobotError(
                "left-side freeze targets are not allowed in dual-arm mode; "
                "Arm A and the left gripper must use live feedback"
            )
        self._left_arm_freeze_q = None if left_arm_freeze_q is None else np.asarray(left_arm_freeze_q, dtype=np.float32).copy()
        self._left_gripper_freeze_q = None if left_gripper_freeze_q is None else np.asarray(left_gripper_freeze_q, dtype=np.float32).copy()
        self._connected = False
        self._io_lock = threading.RLock()

    def connect(self) -> None:
        with self._io_lock:
            try:
                self._call_once(
                    (self.left_arm, self.right_arm, self.left_gripper, self.right_gripper),
                    "connect",
                )
                if not self.command_left_side:
                    if self._left_arm_freeze_q is None or self._left_gripper_freeze_q is None:
                        self.capture_freeze_targets()
                    self._hold_right_side_only(ignore_errors=False)
                self._connected = True
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to connect robot interfaces: {exc}") from exc

    def disconnect(self) -> None:
        with self._io_lock:
            self._call_once(
                (self.left_arm, self.right_arm, self.left_gripper, self.right_gripper),
                "disconnect",
                ignore_errors=True,
            )
            self._connected = False

    def is_connected(self) -> bool:
        with self._io_lock:
            return self._connected

    def get_state(self) -> RightArmGripperState:
        with self._io_lock:
            try:
                return RightArmGripperState(
                    right_arm_q=self.right_arm.get_joint_state(),
                    left_arm_q=self._get_left_arm_state_for_observation(),
                    right_gripper_q=self.right_gripper.get_position(),
                    left_gripper_q=self._get_left_gripper_state_for_observation(),
                    include_left=True,
                )
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to read robot state: {exc}") from exc

    def get_pi_state_15(self) -> np.ndarray:
        """Return the pi0.5 training state order: right_arm(7), left_arm(7), right_gripper(1)."""
        with self._io_lock:
            try:
                right_arm_q = self.right_arm.get_joint_state()
                left_arm_q = self._get_left_arm_state_for_observation()
                right_gripper_q = self.right_gripper.get_position()
                return np.concatenate([right_arm_q, left_arm_q, right_gripper_q], axis=0).astype(
                    np.float32
                )
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to read pi 15D robot state: {exc}") from exc

    def get_pi_state_16(self) -> np.ndarray:
        """Return the 16D training state order: right_arm(7), left_arm(7), right_gripper(1), left_gripper(1)."""
        with self._io_lock:
            try:
                right_arm_q = self.right_arm.get_joint_state()
                left_arm_q = self._get_left_arm_state_for_observation()
                right_gripper_q = self.right_gripper.get_position()
                left_gripper_q = self._get_left_gripper_state_for_observation()
                return np.concatenate(
                    [right_arm_q, left_arm_q, right_gripper_q, left_gripper_q],
                    axis=0,
                ).astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to read pi 16D robot state: {exc}") from exc

    def send_action(self, action: RightArmGripperAction) -> None:
        if not isinstance(action, RightArmGripperAction):
            raise RobotError(f"send_action expects RightArmGripperAction, got {type(action)!r}")
        if self.command_left_side and not (action.control_left_arm and action.control_left_gripper):
            raise RobotError(
                "dual-arm robot requires every action to control the left arm and left gripper"
            )
        if not self.command_left_side and (action.control_left_arm or action.control_left_gripper):
            raise RobotError("right-only robot received an action that controls the left side")
        with self._io_lock:
            try:
                if action.control_left_arm:
                    self.left_arm.stage_joint_position(action.left_arm_q)
                self.right_arm.stage_joint_position(action.right_arm_q)
                self._flush_arm_commands()

                left_gripper_target = None
                if action.control_left_gripper:
                    left_gripper_target = action.left_gripper_q
                    self.left_gripper.send_position(left_gripper_target)
                self.right_gripper.send_position(action.right_gripper_q)
                self._log_gripper_output_signal(
                    left_gripper_target=left_gripper_target,
                    right_gripper_target=action.right_gripper_q,
                )
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to send robot action: {exc}") from exc

    def hold_position(self) -> None:
        with self._io_lock:
            if not self.command_left_side:
                self._hold_right_side_only(ignore_errors=True)
                return
            self._call_once(
                (self.left_arm, self.right_arm, self.left_gripper, self.right_gripper),
                "hold_position",
                ignore_errors=True,
            )

    def go_home(self) -> None:
        with self._io_lock:
            try:
                self._call_once(
                    (self.left_arm, self.right_arm, self.left_gripper, self.right_gripper),
                    "go_home",
                )
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to go home: {exc}") from exc

    def freeze_targets(self) -> dict[str, list[float] | None]:
        return {
            "left_arm": None
            if self._left_arm_freeze_q is None
            else self._left_arm_freeze_q.tolist(),
            "left_gripper": None
            if self._left_gripper_freeze_q is None
            else self._left_gripper_freeze_q.tolist(),
        }

    def capture_freeze_targets(self) -> dict[str, list[float] | None]:
        if self.command_left_side:
            raise RobotError("cannot capture left-side freeze targets in dual-arm mode")
        with self._io_lock:
            try:
                self._left_arm_freeze_q = np.asarray(self.left_arm.get_joint_state(), dtype=np.float32).copy()
                self._left_gripper_freeze_q = np.asarray(self.left_gripper.get_position(), dtype=np.float32).copy()
            except Exception as exc:  # noqa: BLE001
                self.hold_position()
                raise RobotError(f"failed to capture left-side freeze targets: {exc}") from exc
            return self.freeze_targets()

    def _get_left_arm_state_for_observation(self) -> np.ndarray:
        if not self.command_left_side and self._left_arm_freeze_q is not None:
            return self._left_arm_freeze_q.copy()
        return self.left_arm.get_joint_state()

    def _get_left_gripper_state_for_observation(self) -> np.ndarray:
        if not self.command_left_side and self._left_gripper_freeze_q is not None:
            return self._left_gripper_freeze_q.copy()
        return self.left_gripper.get_position()

    def _hold_right_side_only(self, *, ignore_errors: bool) -> None:
        try:
            self.right_arm.stage_joint_position(self.right_arm.get_joint_state())
            self._flush_arm_commands()
        except Exception:
            if not ignore_errors:
                raise
        try:
            self.right_gripper.hold_position()
        except Exception:
            if not ignore_errors:
                raise

    def _resolve_left_arm_target(self, action: RightArmGripperAction) -> np.ndarray:
        if action.control_left_arm:
            return action.left_arm_q
        left_q = self._left_arm_freeze_q
        if left_q is None:
            left_q = self.left_arm.get_joint_state()
            self._left_arm_freeze_q = left_q.copy()
        return left_q

    def _resolve_left_gripper_target(self, action: RightArmGripperAction) -> np.ndarray | None:
        if action.control_left_gripper:
            return action.left_gripper_q
        if self._left_gripper_freeze_q is not None:
            return self._left_gripper_freeze_q
        return None

    def _flush_arm_commands(self) -> None:
        seen: set[object] = set()
        for arm in (self.left_arm, self.right_arm):
            resource_id = arm.shared_resource_id()
            if resource_id in seen:
                continue
            seen.add(resource_id)
            arm.flush_staged_commands()

    def _log_gripper_output_signal(
        self,
        *,
        left_gripper_target: np.ndarray | None,
        right_gripper_target: np.ndarray,
    ) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger(
                "gripper_output_signal",
                left_gripper_target=None
                if left_gripper_target is None
                else np.asarray(left_gripper_target, dtype=np.float32).reshape(-1).tolist(),
                right_gripper_target=np.asarray(
                    right_gripper_target, dtype=np.float32
                ).reshape(-1).tolist(),
            )
        except Exception:
            return

    @staticmethod
    def _call_once(
        parts: tuple[object, ...],
        method_name: str,
        *,
        ignore_errors: bool = False,
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


def make_robot(config: RobotConnectionConfig) -> RightArmGripperRobot:
    if config.backend == "fake":
        left_gripper, right_gripper = _make_grippers(config, end_channel=None)
        return RightArmGripperRobot(
            left_arm=FakeArmInterface(
                ArmConnectionConfig("left", ip=config.left_arm_ip, port=config.left_arm_port)
            ),
            right_arm=FakeArmInterface(
                ArmConnectionConfig("right", ip=config.right_arm_ip, port=config.right_arm_port)
            ),
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            command_left_side=config.command_left_side,
            left_arm_freeze_q=_optional_left_arm_freeze(config.left_arm_freeze_q),
            left_gripper_freeze_q=_optional_gripper_home(config.left_gripper_freeze_q),
        )
    if config.backend in {"tianji", "tianji_gripper_host"}:
        robot_ip = config.robot_ip or config.left_arm_ip or config.right_arm_ip
        if robot_ip is None:
            raise RobotError("tianji_gripper_host backend requires --robot-ip or one arm IP")
        arm_system = TianjiDualArmSystem(
            TianjiHostConfig(
                robot_ip=robot_ip,
                sdk_root=config.tianji_sdk_root,
                config_path=config.tianji_config_path,
                vel_ratio=config.tianji_vel_ratio,
                acc_ratio=config.tianji_acc_ratio,
                settle_sec=config.tianji_settle_sec,
                tool_arm_a_kinematics=config.tianji_tool_arm_a_kinematics,
                tool_arm_a_dynamics=config.tianji_tool_arm_a_dynamics,
                tool_arm_b_kinematics=config.tianji_tool_arm_b_kinematics,
                tool_arm_b_dynamics=config.tianji_tool_arm_b_dynamics,
                state3_joint_k_a=config.tianji_state3_joint_k_a,
                state3_joint_d_a=config.tianji_state3_joint_d_a,
                state3_joint_k_b=config.tianji_state3_joint_k_b,
                state3_joint_d_b=config.tianji_state3_joint_d_b,
                home_joints_a=config.tianji_home_joints_a,
                home_joints_b=config.tianji_home_joints_b,
            )
        )
        left_gripper, right_gripper = _make_grippers(config, end_channel=arm_system)
        return RightArmGripperRobot(
            left_arm=TianjiArmInterface(
                ArmConnectionConfig(
                    "left",
                    ip=robot_ip,
                    sdk_root=config.tianji_sdk_root,
                    config_path=config.tianji_config_path,
                ),
                system=arm_system,
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
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            command_left_side=config.command_left_side,
            left_arm_freeze_q=_optional_left_arm_freeze(config.left_arm_freeze_q),
            left_gripper_freeze_q=_optional_gripper_home(config.left_gripper_freeze_q),
        )
    raise RobotError(f"robot backend {config.backend!r} is not implemented")


def _make_grippers(
    config: RobotConnectionConfig,
    *,
    end_channel: object | None,
) -> tuple[GripperInterface, GripperInterface]:
    left_485_arm = config.left_gripper_485_arm or ("A" if config.left_gripper_backend else config.gripper_485_arm)
    left_485_com = (
        config.left_gripper_485_com
        if config.left_gripper_485_com is not None
        else config.gripper_485_com
    )
    right_485_arm = config.right_gripper_485_arm or config.gripper_485_arm
    right_485_com = (
        config.right_gripper_485_com
        if config.right_gripper_485_com is not None
        else config.gripper_485_com
    )
    left_gripper = make_gripper(
        GripperConnectionConfig(
            "left",
            ip=config.left_gripper_ip,
            port=config.left_gripper_port,
            serial_number=config.left_gripper_serial,
            home_position=_optional_gripper_home(config.left_gripper_home),
            sdk_module=config.gripper_sdk_module,
            sdk_class=config.gripper_sdk_class,
            end_channel_arm=left_485_arm,
            end_channel_com=left_485_com,
            end_channel_direct=config.gripper_direct_onset,
            rs05_target_id=config.gripper_rs05_target_id,
            rs05_master_id=config.gripper_rs05_master_id,
            rs05_can_id_byteorder=config.gripper_can_id_byteorder,
            rs05_standard_id_bytes=config.gripper_standard_id_bytes,
            rs05_enter_motor=config.gripper_enter_motor,
            rs05_stop_on_disconnect=config.gripper_stop_on_disconnect,
            rs05_kp=config.gripper_kp,
            rs05_kd=config.gripper_kd,
            rs05_torque_nm=config.gripper_torque_nm,
            rs05_min_pos_rad=config.gripper_min_pos_rad,
            rs05_max_pos_rad=config.gripper_max_pos_rad,
            rs05_torque_protection_enabled=config.gripper_torque_protection_enabled,
            rs05_torque_protection_mode=config.gripper_torque_protection_mode,
            rs05_torque_filter_alpha=config.gripper_torque_filter_alpha,
            rs05_torque_threshold_nm=_override(
                config.left_gripper_torque_threshold_nm,
                config.gripper_torque_threshold_nm,
            ),
            rs05_torque_release_threshold_nm=_override(
                config.left_gripper_torque_release_threshold_nm,
                config.gripper_torque_release_threshold_nm,
            ),
            rs05_torque_count_threshold=_override(
                config.left_gripper_torque_count_threshold,
                config.gripper_torque_count_threshold,
            ),
            rs05_torque_extra_tighten_rad=_override(
                config.left_gripper_torque_extra_tighten_rad,
                config.gripper_torque_extra_tighten_rad,
            ),
            rs05_holding_kp=_override(
                config.left_gripper_holding_kp,
                config.gripper_holding_kp,
            ),
            rs05_holding_kd=_override(
                config.left_gripper_holding_kd,
                config.gripper_holding_kd,
            ),
            rs05_closing_direction=config.gripper_closing_direction,
            rs05_torque_direction_deadband_rad=config.gripper_torque_direction_deadband_rad,
        ),
        backend="fake" if not config.command_left_side else config.left_gripper_backend or config.gripper_backend,
        end_channel=end_channel,
    )
    right_gripper = make_gripper(
        GripperConnectionConfig(
            "right",
            ip=config.right_gripper_ip,
            port=config.right_gripper_port,
            serial_number=config.right_gripper_serial,
            home_position=_optional_gripper_home(config.right_gripper_home),
            sdk_module=config.gripper_sdk_module,
            sdk_class=config.gripper_sdk_class,
            end_channel_arm=right_485_arm,
            end_channel_com=right_485_com,
            end_channel_direct=config.gripper_direct_onset,
            rs05_target_id=config.gripper_rs05_target_id,
            rs05_master_id=config.gripper_rs05_master_id,
            rs05_can_id_byteorder=config.gripper_can_id_byteorder,
            rs05_standard_id_bytes=config.gripper_standard_id_bytes,
            rs05_enter_motor=config.gripper_enter_motor,
            rs05_stop_on_disconnect=config.gripper_stop_on_disconnect,
            rs05_kp=config.gripper_kp,
            rs05_kd=config.gripper_kd,
            rs05_torque_nm=config.gripper_torque_nm,
            rs05_min_pos_rad=config.gripper_min_pos_rad,
            rs05_max_pos_rad=config.gripper_max_pos_rad,
            rs05_torque_protection_enabled=config.gripper_torque_protection_enabled,
            rs05_torque_protection_mode=config.gripper_torque_protection_mode,
            rs05_torque_filter_alpha=config.gripper_torque_filter_alpha,
            rs05_torque_threshold_nm=_override(
                config.right_gripper_torque_threshold_nm,
                config.gripper_torque_threshold_nm,
            ),
            rs05_torque_release_threshold_nm=_override(
                config.right_gripper_torque_release_threshold_nm,
                config.gripper_torque_release_threshold_nm,
            ),
            rs05_torque_count_threshold=_override(
                config.right_gripper_torque_count_threshold,
                config.gripper_torque_count_threshold,
            ),
            rs05_torque_extra_tighten_rad=_override(
                config.right_gripper_torque_extra_tighten_rad,
                config.gripper_torque_extra_tighten_rad,
            ),
            rs05_holding_kp=_override(
                config.right_gripper_holding_kp,
                config.gripper_holding_kp,
            ),
            rs05_holding_kd=_override(
                config.right_gripper_holding_kd,
                config.gripper_holding_kd,
            ),
            rs05_closing_direction=config.gripper_closing_direction,
            rs05_torque_direction_deadband_rad=config.gripper_torque_direction_deadband_rad,
        ),
        backend=config.right_gripper_backend or config.gripper_backend,
        end_channel=end_channel,
    )
    return left_gripper, right_gripper


def _override(side_value, common_value):
    """优先使用单侧夹爪配置；未填写时回退到原有公共配置。"""
    return common_value if side_value is None else side_value


def _optional_gripper_home(value: tuple[float, ...] | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (1,):
        raise RobotError(f"gripper home pose must have shape (1,), got {arr.shape}")
    return arr


def _optional_left_arm_freeze(value: tuple[float, ...] | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (7,):
        raise RobotError(f"left arm freeze pose must have shape (7,), got {arr.shape}")
    return arr
