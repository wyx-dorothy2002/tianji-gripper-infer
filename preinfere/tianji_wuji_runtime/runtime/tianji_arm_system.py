"""Host-side Tianji dual-arm integration using the Marvin SDK."""

from __future__ import annotations

from contextlib import suppress
import ctypes
from dataclasses import dataclass
from importlib import import_module
from itertools import pairwise
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
from typing import Any

import numpy as np

from . import schema

# Settling delay between SDK state transitions during the idle-reset/enable handshake.
# Mirrors the known-good DexProj state=3 probe path.
_IDLE_RESET_SETTLE_SEC = 0.5
_JOINT_VEL_RATIO = 80
_JOINT_ACC_RATIO = 70
_MARVIN_FEEDBACK_SAMPLES = 5
_MARVIN_FEEDBACK_INTERVAL_SEC = 0.1
_MARVIN_FEEDBACK_VERIFY_ATTEMPTS = 3
_MARVIN_HOME_JOINTS_DEG = [21.8, -41.0, -4.74, -63.67, 10.15, 14.72, 7.68]


DEFAULT_TIANJI_SDK_ROOT = Path(
    "/home/user/workspace/TJ-gripper-codex-gripper-fixes-20260617-reclone/reference/MARVIN_SDK"
)
DEFAULT_TIANJI_CONFIG_PATH = (
    DEFAULT_TIANJI_SDK_ROOT / "DEMO_PYTHON/ccs_m3.MvKDCfg"
)


@dataclass
class TianjiHostConfig:
    robot_ip: str
    sdk_root: str | None = None
    config_path: str | None = None
    unit: str = schema.STATE_UNIT
    vel_ratio: int = _JOINT_VEL_RATIO
    acc_ratio: int = _JOINT_ACC_RATIO
    settle_sec: float = _IDLE_RESET_SETTLE_SEC
    tool_arm_a_kinematics: tuple[float, ...] | None = None
    tool_arm_a_dynamics: tuple[float, ...] | None = None
    tool_arm_b_kinematics: tuple[float, ...] | None = None
    tool_arm_b_dynamics: tuple[float, ...] | None = None
    home_joints_a: tuple[float, ...] | None = None
    home_joints_b: tuple[float, ...] | None = None


class TianjiHostError(RuntimeError):
    """Raised when the Tianji host-side integration cannot proceed safely."""


class TianjiDualArmSystem:
    """Single shared controller for the Tianji left/right arm pair."""

    def __init__(self, config: TianjiHostConfig) -> None:
        self.config = config
        self._controller: Any | None = None
        self._connected = False
        self._resource_id = object()
        self._staged_left: np.ndarray | None = None
        self._staged_right: np.ndarray | None = None
        self._hold_left: np.ndarray | None = None
        self._hold_right: np.ndarray | None = None
        self._command_lock = threading.RLock()

    def shared_resource_id(self) -> object:
        return self._resource_id

    def connect(self) -> None:
        if self._connected:
            return
        self._validate_config()
        controller_cls = _load_tianji_controller(self._resolve_sdk_root())
        config_path = self._resolve_config_path()
        self._controller = controller_cls(
            robot_ip=self.config.robot_ip,
            config_path=str(config_path),
            dry_run=False,
            read_only=False,
            feedback_handshake=False,
            prefer_last_ik_reference=False,
            ik_subprocess_isolate=False,
        )
        self._prepare_joint_control()
        self._connected = True

    def disconnect(self) -> None:
        controller = self._controller
        self._controller = None
        self._connected = False
        self._clear_staged_commands()
        self._hold_left = None
        self._hold_right = None
        if controller is None:
            return
        try:
            controller.disable_and_release()
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to release Tianji controller: {exc}") from exc

    def get_joint_state(self, side: str) -> np.ndarray:
        left, right = self.get_current_joints()
        values = left if side == "left" else right
        return _as_arm_array(values, name=f"{side}_arm_state")

    def stage_joint_position(self, side: str, q: np.ndarray) -> None:
        arr = _as_arm_array(q, name=f"{side}_arm_command")
        with self._command_lock:
            if side == "left":
                self._staged_left = arr
            else:
                self._staged_right = arr

    def flush_staged_commands(self) -> None:
        controller = self._require_controller()
        with self._command_lock:
            left_cmd = self._staged_left
            right_cmd = self._staged_right
            hold_left = None if self._hold_left is None else self._hold_left.copy()
            hold_right = None if self._hold_right is None else self._hold_right.copy()
            self._staged_left = None
            self._staged_right = None

        if left_cmd is None and right_cmd is None:
            return

        try:
            controller.move_to_joints_direct(
                left_joints=None if left_cmd is None else left_cmd.tolist(),
                right_joints=None if right_cmd is None else right_cmd.tolist(),
            )
        except TypeError:
            if left_cmd is None or right_cmd is None:
                if hold_left is None or hold_right is None:
                    hold_left, hold_right = self.get_current_joints()
                if left_cmd is None:
                    left_cmd = hold_left
                if right_cmd is None:
                    right_cmd = hold_right
            try:
                controller.move_to_joints_direct(
                    left_joints=left_cmd.tolist(),
                    right_joints=right_cmd.tolist(),
                )
            except Exception as exc:  # noqa: BLE001
                raise TianjiHostError(f"failed to send Tianji joint command: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to send Tianji joint command: {exc}") from exc
        if left_cmd is not None:
            self._hold_left = left_cmd.copy()
        if right_cmd is not None:
            self._hold_right = right_cmd.copy()

    def hold_position(self) -> None:
        self._clear_staged_commands()
        try:
            left, right = self.get_current_joints()
        except TianjiHostError:
            if self._hold_left is None or self._hold_right is None:
                raise
            left = self._hold_left.copy()
            right = self._hold_right.copy()
        try:
            self._require_controller().move_to_joints_direct(
                left_joints=left.tolist(),
                right_joints=right.tolist(),
            )
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to hold Tianji arms: {exc}") from exc
        self._update_hold_targets(left, right)

    def go_home(self) -> None:
        controller = self._require_controller()
        self._clear_staged_commands()
        try:
            robot = getattr(controller, "robot", None)
            if (
                robot is not None
                and self.config.home_joints_a is not None
                and self.config.home_joints_b is not None
            ):
                _write_marvin_batch(
                    robot,
                    lambda: robot.set_joint_cmd_pose(
                        arm="A",
                        joints=list(self.config.home_joints_a or ()),
                    ),
                    lambda: robot.set_joint_cmd_pose(
                        arm="B",
                        joints=list(self.config.home_joints_b or ()),
                    ),
                )
            else:
                controller.move_to_init(wait=False, duration=3.0, dt=0.01, sides="both")
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to send Tianji home trajectory: {exc}") from exc

    def get_current_joints(self) -> tuple[np.ndarray, np.ndarray]:
        controller = self._require_controller()
        try:
            left, right = controller.get_current_joints()
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to read Tianji joint feedback: {exc}") from exc
        left_arr = _as_arm_array(left, name="left_arm_state")
        right_arr = _as_arm_array(
            right,
            name="right_arm_state",
        )
        self._update_hold_targets(left_arr, right_arr)
        return left_arr, right_arr

    def send_end_channel_data(
        self,
        arm: str,
        frame_hex: str,
        com: int,
        direct: bool = True,
    ) -> tuple[bool, object]:
        """Write a CAN/485 frame through the Tianji/Marvin end-effector channel."""

        sdk_robot = self._require_sdk_robot()
        if direct:
            frame = _parse_hex_bytes(frame_hex)
            if not frame or len(frame) > 256:
                raise TianjiHostError(
                    f"end-channel frame length must be 1..256 bytes, got {len(frame)}"
                )
            buffer = (ctypes.c_ubyte * 256)()
            for index, value in enumerate(frame):
                buffer[index] = value
            native = getattr(sdk_robot, "robot", None)
            if native is None:
                raise TianjiHostError("Tianji SDK object does not expose native end-channel API")
            arm_upper = str(arm).upper()
            if arm_upper == "A":
                native.OnSetChDataA.argtypes = [
                    ctypes.POINTER(ctypes.c_ubyte),
                    ctypes.c_long,
                    ctypes.c_long,
                ]
                native.OnSetChDataA.restype = ctypes.c_bool
                return True, native.OnSetChDataA(
                    buffer,
                    ctypes.c_long(len(frame)),
                    ctypes.c_long(int(com)),
                )
            if arm_upper == "B":
                native.OnSetChDataB.argtypes = [
                    ctypes.POINTER(ctypes.c_ubyte),
                    ctypes.c_long,
                    ctypes.c_long,
                ]
                native.OnSetChDataB.restype = ctypes.c_bool
                return True, native.OnSetChDataB(
                    buffer,
                    ctypes.c_long(len(frame)),
                    ctypes.c_long(int(com)),
                )
            raise TianjiHostError(f"unsupported end-channel arm: {arm}")

        set_485_data = getattr(sdk_robot, "set_485_data", None)
        if not callable(set_485_data):
            raise TianjiHostError("Tianji SDK object does not expose set_485_data")
        return set_485_data(str(arm).upper(), frame_hex, len(frame_hex.split()), int(com))

    def get_end_channel_data(
        self,
        arm: str,
        com: int,
        direct: bool = True,
    ) -> tuple[int, int | None, str]:
        """Read a CAN/485 feedback frame from the Tianji/Marvin end-effector channel."""

        sdk_robot = self._require_sdk_robot()
        if not direct:
            get_485_data = getattr(sdk_robot, "get_485_data", None)
            if not callable(get_485_data):
                raise TianjiHostError("Tianji SDK object does not expose get_485_data")
            size, hex_text = get_485_data(str(arm).upper(), int(com))
            return int(size), None, _trim_hex(hex_text, int(size))

        native = getattr(sdk_robot, "robot", None)
        if native is None:
            raise TianjiHostError("Tianji SDK object does not expose native end-channel API")
        buffer = (ctypes.c_ubyte * 256)()
        ret_ch = ctypes.c_long(int(com))
        arm_upper = str(arm).upper()
        if arm_upper == "A":
            native.OnGetChDataA.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.POINTER(ctypes.c_long),
            ]
            native.OnGetChDataA.restype = ctypes.c_long
            size = int(native.OnGetChDataA(buffer, ctypes.byref(ret_ch)))
        elif arm_upper == "B":
            native.OnGetChDataB.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.POINTER(ctypes.c_long),
            ]
            native.OnGetChDataB.restype = ctypes.c_long
            size = int(native.OnGetChDataB(buffer, ctypes.byref(ret_ch)))
        else:
            raise TianjiHostError(f"unsupported end-channel arm: {arm}")
        return size, int(ret_ch.value), _trim_hex(_hex_bytes(bytes(buffer)), size)

    def _prepare_joint_control(self) -> None:
        controller = self._require_controller()
        try:
            self._apply_configured_tool_info(controller)
            self._clear_robot_errors_and_prime_feedback(controller)
            self._idle_reset_arms(controller)
            try:
                controller.set_impedance_mode(
                    mode="joint",
                    vel_ratio=int(self.config.vel_ratio),
                    acc_ratio=int(self.config.acc_ratio),
                )
            except TypeError as exc:
                # Keep compatibility with the legacy Tianji controller, whose
                # set_impedance_mode method only accepts ``mode``.
                if "unexpected keyword" not in str(exc):
                    raise
                controller.set_impedance_mode(mode="joint")
            time.sleep(max(float(self.config.settle_sec), 0.0))
            self._set_joint_velocity_acceleration(controller)
            left, right = controller.get_current_joints()
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(
                "Tianji controller connected but joint-control preparation failed "
                f"(set mode / clear error / feedback verify path): {exc}"
            ) from exc
        self._update_hold_targets(
            _as_arm_array(left, name="left_arm_feedback_verify"),
            _as_arm_array(right, name="right_arm_feedback_verify"),
        )

    def _require_controller(self) -> Any:
        if self._controller is None:
            raise TianjiHostError("Tianji controller is not connected")
        return self._controller

    def _require_sdk_robot(self) -> Any:
        sdk_robot = getattr(self._require_controller(), "robot", None)
        if sdk_robot is None:
            raise TianjiHostError("Tianji controller does not expose the legacy robot SDK object")
        return sdk_robot

    def _clear_staged_commands(self) -> None:
        with self._command_lock:
            self._staged_left = None
            self._staged_right = None

    def _clear_robot_errors_and_prime_feedback(self, controller: Any) -> None:
        robot = getattr(controller, "robot", None)
        if robot is None:
            return
        clear_set = getattr(robot, "clear_set", None)
        clear_error = getattr(robot, "clear_error", None)
        send_cmd = getattr(robot, "send_cmd", None)
        if not callable(clear_set) or not callable(clear_error) or not callable(send_cmd):
            return
        clear_set()
        clear_error("A")
        clear_error("B")
        send_cmd()

    def _apply_configured_tool_info(self, controller: Any) -> None:
        robot = getattr(controller, "robot", None)
        if robot is None:
            return
        clear_set = getattr(robot, "clear_set", None)
        set_tool = getattr(robot, "set_tool", None)
        send_cmd = getattr(robot, "send_cmd", None)
        if not all(callable(fn) for fn in (clear_set, set_tool, send_cmd)):
            return

        configured = (
            (
                "A",
                self.config.tool_arm_a_kinematics,
                self.config.tool_arm_a_dynamics,
            ),
            (
                "B",
                self.config.tool_arm_b_kinematics,
                self.config.tool_arm_b_dynamics,
            ),
        )
        setters = []
        for arm, kinematics, dynamics in configured:
            if kinematics is None and dynamics is None:
                continue
            if kinematics is None or dynamics is None:
                raise TianjiHostError(
                    f"configured tool info for arm {arm} requires both kinematics and dynamics"
                )
            print(
                f"[runtime] applying configured tool info for arm {arm}: "
                f"dyn={list(dynamics)} kine={list(kinematics)}"
            )
            setters.append(
                lambda arm=arm, kinematics=kinematics, dynamics=dynamics: set_tool(
                    arm=arm,
                    kineParams=list(kinematics),
                    dynamicParams=list(dynamics),
                )
            )
        if setters:
            _write_marvin_batch(robot, *setters)
            time.sleep(max(float(self.config.settle_sec), 0.0))

    def _set_joint_velocity_acceleration(self, controller: Any) -> None:
        robot = getattr(controller, "robot", None)
        if robot is None:
            return
        clear_set = getattr(robot, "clear_set", None)
        set_vel_acc = getattr(robot, "set_vel_acc", None)
        send_cmd = getattr(robot, "send_cmd", None)
        if not all(callable(fn) for fn in (clear_set, set_vel_acc, send_cmd)):
            return

        clear_set()
        set_vel_acc(
            arm="A",
            velRatio=int(self.config.vel_ratio),
            AccRatio=int(self.config.acc_ratio),
        )
        set_vel_acc(
            arm="B",
            velRatio=int(self.config.vel_ratio),
            AccRatio=int(self.config.acc_ratio),
        )
        send_cmd()

    def _idle_reset_arms(self, controller: Any) -> None:
        """Force both arms through state=0 + clear_error before enabling state=3.

        Jumping straight to state=3 only re-arms an arm that is already in a clean
        idle state. After an abnormal exit or a latched protective stop, the SDK
        keeps accepting joint commands while the arm never actually servos. Driving
        state=0 then clearing faults first matches the known-good DexProj probe path
        and lets the subsequent set_impedance_mode (state=3) actually take effect.
        """
        robot = getattr(controller, "robot", None)
        if robot is None:
            return
        clear_set = getattr(robot, "clear_set", None)
        set_state = getattr(robot, "set_state", None)
        clear_error = getattr(robot, "clear_error", None)
        send_cmd = getattr(robot, "send_cmd", None)
        if not all(callable(fn) for fn in (clear_set, set_state, clear_error, send_cmd)):
            return

        clear_set()
        set_state(arm="A", state=0)
        set_state(arm="B", state=0)
        send_cmd()
        time.sleep(max(float(self.config.settle_sec), 0.0))

        clear_set()
        clear_error("A")
        clear_error("B")
        send_cmd()
        time.sleep(max(float(self.config.settle_sec), 0.0))

    def _validate_config(self) -> None:
        if not 1 <= int(self.config.vel_ratio) <= 100:
            raise TianjiHostError("Tianji vel_ratio must be in [1, 100]")
        if not 1 <= int(self.config.acc_ratio) <= 100:
            raise TianjiHostError("Tianji acc_ratio must be in [1, 100]")
        if float(self.config.settle_sec) < 0:
            raise TianjiHostError("Tianji settle_sec must be non-negative")
        vectors = (
            ("tool_arm_a_kinematics", self.config.tool_arm_a_kinematics, 6),
            ("tool_arm_a_dynamics", self.config.tool_arm_a_dynamics, 10),
            ("tool_arm_b_kinematics", self.config.tool_arm_b_kinematics, 6),
            ("tool_arm_b_dynamics", self.config.tool_arm_b_dynamics, 10),
            ("home_joints_a", self.config.home_joints_a, schema.LEFT_ARM_DOF),
            ("home_joints_b", self.config.home_joints_b, schema.RIGHT_ARM_DOF),
        )
        for name, values, expected_len in vectors:
            if values is not None and len(values) != expected_len:
                raise TianjiHostError(
                    f"Tianji {name} must contain {expected_len} values, got {len(values)}"
                )

    def _update_hold_targets(self, left: np.ndarray, right: np.ndarray) -> None:
        with self._command_lock:
            self._hold_left = np.asarray(left, dtype=np.float32).copy()
            self._hold_right = np.asarray(right, dtype=np.float32).copy()

    def _resolve_sdk_root(self) -> Path:
        root = Path(self.config.sdk_root) if self.config.sdk_root else DEFAULT_TIANJI_SDK_ROOT
        if not root.exists():
            raise TianjiHostError(
                f"Tianji SDK root does not exist: {root}. Pass --tianji-sdk-root explicitly."
            )
        return root

    def _resolve_config_path(self) -> Path:
        if self.config.config_path is not None:
            path = Path(self.config.config_path)
        else:
            path = DEFAULT_TIANJI_CONFIG_PATH
        if not path.exists():
            raise TianjiHostError(
                f"Tianji kinematics config does not exist: {path}. "
                "Pass --tianji-config-path explicitly."
            )
        return path


def _load_tianji_controller(sdk_root: Path) -> type[Any]:
    _ensure_ament_index_stub(sdk_root)
    root_str = str(sdk_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    loaded = sys.modules.get("tianji_output")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", "") or ""
        if loaded_file and not Path(loaded_file).resolve().is_relative_to(sdk_root.resolve()):
            sys.modules.pop("tianji_output", None)
    try:
        module = import_module("tianji_output")
    except Exception as exc:  # noqa: BLE001
        try:
            return _load_marvin_controller(sdk_root)
        except Exception as marvin_exc:  # noqa: BLE001
            raise TianjiHostError(
                "failed to import either legacy Tianji SDK package or Marvin controller "
                f"adapter from {sdk_root}: tianji_output error={exc}; marvin error={marvin_exc}"
            ) from marvin_exc
    controller = getattr(module, "TianjiArmController", None)
    if controller is None:
        raise TianjiHostError("legacy tianji_output package does not export TianjiArmController")
    return controller


def _load_marvin_controller(sdk_root: Path) -> type[Any]:
    sdk_python_parent = _resolve_marvin_sdk_parent(sdk_root)
    fx_robot = _import_marvin_fx_robot(sdk_python_parent)

    class MarvinControllerAdapter:
        def __init__(
            self,
            robot_ip: str,
            config_path: str,
            dry_run: bool = False,
            read_only: bool = False,
            feedback_handshake: bool = False,
            prefer_last_ik_reference: bool = False,
            ik_subprocess_isolate: bool = False,
        ) -> None:
            del (
                config_path,
                dry_run,
                read_only,
                feedback_handshake,
                prefer_last_ik_reference,
                ik_subprocess_isolate,
            )
            print(f"[runtime] loading Marvin SDK from {sdk_python_parent}")
            self.robot = fx_robot.Marvin_Robot()
            self._dcss = fx_robot.DCSS()
            print(f"[runtime] connecting Marvin robot at {robot_ip}")
            connected = self.robot.connect(robot_ip)
            if connected not in (True, 1):
                raise TianjiHostError(
                    f"Marvin connect failed for ip={robot_ip}: {connected!r}"
                )
            try:
                time.sleep(_IDLE_RESET_SETTLE_SEC)
                print("[runtime] verifying Marvin A/B feedback")
                _verify_marvin_feedback_with_retry(
                    self.robot,
                    self._dcss,
                    samples=_MARVIN_FEEDBACK_SAMPLES,
                    interval_sec=_MARVIN_FEEDBACK_INTERVAL_SEC,
                    attempts=_MARVIN_FEEDBACK_VERIFY_ATTEMPTS,
                )
                print("[runtime] Marvin A/B feedback ready")
            except Exception:
                with suppress(Exception):
                    self.robot.release_robot()
                raise

        def get_current_joints(self) -> tuple[list[float], list[float]]:
            feedback = _read_marvin_feedback(self.robot, self._dcss)
            return (
                _extract_marvin_feedback_joints(feedback, arm_index=0),
                _extract_marvin_feedback_joints(feedback, arm_index=1),
            )

        def move_to_joints_direct(
            self,
            *,
            left_joints: list[float] | None,
            right_joints: list[float] | None,
        ) -> None:
            commands: dict[str, list[float]] = {}
            if left_joints is not None:
                commands["A"] = left_joints
            if right_joints is not None:
                commands["B"] = right_joints
            if not commands:
                return
            before_feedback = _read_marvin_feedback(self.robot, self._dcss)
            setters = [
                lambda arm=arm, joints=joints: self.robot.set_joint_cmd_pose(
                    arm=arm,
                    joints=joints,
                )
                for arm, joints in commands.items()
            ]
            _write_marvin_batch(self.robot, *setters)
            time.sleep(0.05)
            after_feedback = _read_marvin_feedback(self.robot, self._dcss)
            _print_marvin_motion_debug(
                before_feedback=before_feedback,
                after_feedback=after_feedback,
                left_target=left_joints,
                right_target=right_joints,
            )

        def move_to_init(
            self,
            wait: bool = False,
            duration: float = 3.0,
            dt: float = 0.01,
            sides: str = "both",
        ) -> None:
            del wait, duration, dt, sides
            _write_marvin_batch(
                self.robot,
                lambda: self.robot.set_joint_cmd_pose(
                    arm="A",
                    joints=list(_MARVIN_HOME_JOINTS_DEG),
                ),
                lambda: self.robot.set_joint_cmd_pose(
                    arm="B",
                    joints=list(_MARVIN_HOME_JOINTS_DEG),
                ),
            )

        def set_impedance_mode(
            self,
            mode: str = "joint",
            vel_ratio: int = _JOINT_VEL_RATIO,
            acc_ratio: int = _JOINT_ACC_RATIO,
        ) -> None:
            if mode != "joint":
                raise TianjiHostError(f"Marvin adapter only supports joint impedance mode, got {mode!r}")
            if not 1 <= int(vel_ratio) <= 100 or not 1 <= int(acc_ratio) <= 100:
                raise TianjiHostError(
                    "Marvin joint velocity and acceleration ratios must be in [1, 100]"
                )
            # Match MarvinTeachRuntime.enter_udp_joint_impedance_mode exactly:
            # state=3 + joint impedance + velocity/acceleration in one batch.
            # Do not call set_drag_space here. It is a drag-teach setting, is
            # unnecessary for joint-command inference, and some SDK builds do
            # not declare its ctypes bool return type correctly.
            setters = [
                lambda: self.robot.set_state(arm="A", state=3),
                lambda: self.robot.set_state(arm="B", state=3),
                lambda: self.robot.set_impedance_type(arm="A", type=1),
                lambda: self.robot.set_impedance_type(arm="B", type=1),
                lambda: self.robot.set_vel_acc(
                    arm="A",
                    velRatio=int(vel_ratio),
                    AccRatio=int(acc_ratio),
                ),
                lambda: self.robot.set_vel_acc(
                    arm="B",
                    velRatio=int(vel_ratio),
                    AccRatio=int(acc_ratio),
                ),
            ]
            try:
                _write_marvin_batch(self.robot, *setters)
            except Exception as exc:  # noqa: BLE001
                raise TianjiHostError(f"failed to finish Marvin impedance-mode setup: {exc}") from exc
            time.sleep(_IDLE_RESET_SETTLE_SEC)

        def disable_and_release(self) -> None:
            try:
                _write_marvin_batch(
                    self.robot,
                    lambda: self.robot.set_state(arm="A", state=0),
                    lambda: self.robot.set_state(arm="B", state=0),
                )
                time.sleep(_IDLE_RESET_SETTLE_SEC)
            finally:
                self.robot.release_robot()

    return MarvinControllerAdapter


def _resolve_marvin_sdk_parent(sdk_root: Path) -> Path:
    root = sdk_root.expanduser().resolve()
    candidates: list[Path] = []
    if (root / "src/marvin.py").exists():
        candidates.extend(
            [
                root / "reference/MARVIN_SDK",
                root / "reference/MARVIN_SDK_UBUNTU2204",
            ]
        )
    if (root / "SDK_PYTHON").exists():
        candidates.append(root)
    if root.name == "SDK_PYTHON":
        candidates.append(root.parent)
    for sdk_parent in candidates:
        if (sdk_parent / "SDK_PYTHON/fx_robot.py").exists():
            return sdk_parent
    raise TianjiHostError(
        "could not resolve Marvin SDK layout. Expected `SDK_PYTHON/fx_robot.py` under "
        f"{root}, a direct SDK_PYTHON path, or a repository with reference/MARVIN_SDK"
    )


def _import_marvin_fx_robot(sdk_python_parent: Path) -> ModuleType:
    parent = sdk_python_parent.expanduser().resolve()
    module_path = parent / "SDK_PYTHON/fx_robot.py"
    if not module_path.exists():
        raise TianjiHostError(f"Marvin SDK module does not exist: {module_path}")
    parent_str = str(parent)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
    loaded = sys.modules.get("SDK_PYTHON.fx_robot")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        if loaded_file.is_relative_to(parent):
            return loaded
        sys.modules.pop("SDK_PYTHON.fx_robot", None)
        sys.modules.pop("SDK_PYTHON", None)
    try:
        return import_module("SDK_PYTHON.fx_robot")
    except Exception as exc:  # noqa: BLE001
        raise TianjiHostError(f"failed to import Marvin SDK module {module_path}: {exc}") from exc


def _read_marvin_feedback(robot: Any, dcss: Any) -> dict[str, Any]:
    feedback = robot.subscribe(dcss)
    if not isinstance(feedback, dict):
        raise TianjiHostError(f"unexpected Marvin feedback type: {type(feedback)!r}")
    return feedback


def _verify_marvin_feedback(
    robot: Any,
    dcss: Any,
    *,
    samples: int,
    interval_sec: float,
) -> None:
    frame_history = {0: [], 1: []}
    valid_joint_samples = {0: 0, 1: 0}
    sample_count = max(int(samples), 1)
    for sample_index in range(sample_count):
        feedback = _read_marvin_feedback(robot, dcss)
        outputs = feedback.get("outputs", [])
        for arm_index in (0, 1):
            output = outputs[arm_index] if arm_index < len(outputs) else {}
            try:
                frame_serial = int(output.get("frame_serial", 0))
            except (AttributeError, TypeError, ValueError):
                frame_serial = 0
            try:
                joints = [float(value) for value in output.get("fb_joint_pos", [])]
            except (AttributeError, TypeError, ValueError):
                joints = []
            if len(joints) == schema.LEFT_ARM_DOF:
                valid_joint_samples[arm_index] += 1
            frame_history[arm_index].append(frame_serial)
        if sample_index + 1 < sample_count:
            time.sleep(max(float(interval_sec), 0.0))

    missing = []
    for arm_index, arm_name in ((0, "A"), (1, "B")):
        nonzero_frames = [value for value in frame_history[arm_index] if value != 0]
        frame_updated = any(
            current != previous
            for previous, current in pairwise(nonzero_frames)
        )
        if not frame_updated and valid_joint_samples[arm_index] <= 0:
            missing.append(arm_name)
    if missing:
        raise TianjiHostError(
            "Marvin feedback stream did not provide frame updates or valid joint "
            f"feedback after connect for arm(s): {', '.join(missing)}"
        )


def _verify_marvin_feedback_with_retry(
    robot: Any,
    dcss: Any,
    *,
    samples: int,
    interval_sec: float,
    attempts: int,
) -> None:
    last_error: TianjiHostError | None = None
    for attempt in range(1, max(int(attempts), 1) + 1):
        try:
            _verify_marvin_feedback(
                robot,
                dcss,
                samples=samples,
                interval_sec=interval_sec,
            )
            return
        except TianjiHostError as exc:
            last_error = exc
            if attempt < max(int(attempts), 1):
                print(f"[runtime] {exc}; retrying feedback verify ({attempt}/{attempts})")
                time.sleep(max(float(interval_sec) * 2.0, _IDLE_RESET_SETTLE_SEC))
    if last_error is not None:
        raise last_error


def _write_marvin_batch(robot: Any, *setters: Any) -> None:
    _require_marvin_success("clear_set", robot.clear_set())
    for index, setter in enumerate(setters, start=1):
        _require_marvin_success(f"setter_{index}", setter())
    _require_marvin_success("send_cmd", robot.send_cmd())


def _require_marvin_success(name: str, result: Any) -> None:
    if result is None or result is True:
        return
    if isinstance(result, int) and not isinstance(result, bool) and result in (0, 1):
        return
    raise TianjiHostError(f"Marvin SDK call failed: {name} -> {result!r}")


def _extract_marvin_feedback_joints(feedback: dict[str, Any], arm_index: int) -> list[float]:
    outputs = feedback.get("outputs", [])
    if not isinstance(outputs, list) or not (0 <= arm_index < len(outputs)):
        raise TianjiHostError(f"Marvin feedback is missing outputs[{arm_index}]")
    joints = outputs[arm_index].get("fb_joint_pos")
    if not isinstance(joints, list) or len(joints) != schema.LEFT_ARM_DOF:
        raise TianjiHostError(
            f"Marvin feedback outputs[{arm_index}]['fb_joint_pos'] must have length "
            f"{schema.LEFT_ARM_DOF}, got {joints!r}"
        )
    try:
        return [float(value) for value in joints]
    except (TypeError, ValueError) as exc:
        raise TianjiHostError(f"invalid Marvin feedback joint values: {joints!r}") from exc


def _marvin_feedback_snapshot(feedback: dict[str, Any], arm_index: int) -> dict[str, Any]:
    states = feedback.get("states", [])
    outputs = feedback.get("outputs", [])
    inputs = feedback.get("inputs", [])
    state = states[arm_index] if arm_index < len(states) else {}
    output = outputs[arm_index] if arm_index < len(outputs) else {}
    cmd_input = inputs[arm_index] if arm_index < len(inputs) else {}
    joints = output.get("fb_joint_pos", [])
    return {
        "cur_state": int(state.get("cur_state", 0)),
        "cmd_state": int(state.get("cmd_state", 0)),
        "err_code": int(state.get("err_code", 0)),
        "frame_serial": int(output.get("frame_serial", 0)),
        "imp_type": int(cmd_input.get("imp_type", 0)),
        "joint_pos_deg": [float(v) for v in joints[: schema.LEFT_ARM_DOF]],
    }


def _print_marvin_motion_debug(
    *,
    before_feedback: dict[str, Any],
    after_feedback: dict[str, Any],
    left_target: list[float] | None,
    right_target: list[float] | None,
) -> None:
    before_b = _marvin_feedback_snapshot(before_feedback, 1)
    after_b = _marvin_feedback_snapshot(after_feedback, 1)
    before_a = _marvin_feedback_snapshot(before_feedback, 0)
    after_a = _marvin_feedback_snapshot(after_feedback, 0)
    print(
        "[marvin-debug] "
        f"A state {before_a['cur_state']}->{after_a['cur_state']} "
        f"err {after_a['err_code']} frame {before_a['frame_serial']}->{after_a['frame_serial']} "
        f"fb_j1 {before_a['joint_pos_deg'][0] if before_a['joint_pos_deg'] else 'na'}"
        f"->{after_a['joint_pos_deg'][0] if after_a['joint_pos_deg'] else 'na'} "
        f"fb_j4 {before_a['joint_pos_deg'][3] if len(before_a['joint_pos_deg']) > 3 else 'na'}"
        f"->{after_a['joint_pos_deg'][3] if len(after_a['joint_pos_deg']) > 3 else 'na'} "
        f"target_j1 {float(left_target[0]) if left_target else 'na'} "
        f"target_j4 {float(left_target[3]) if left_target and len(left_target) > 3 else 'na'} | "
        f"B state {before_b['cur_state']}->{after_b['cur_state']} "
        f"err {after_b['err_code']} cmd_state {after_b['cmd_state']} imp {after_b['imp_type']} "
        f"frame {before_b['frame_serial']}->{after_b['frame_serial']} "
        f"fb_j1 {before_b['joint_pos_deg'][0] if before_b['joint_pos_deg'] else 'na'}"
        f"->{after_b['joint_pos_deg'][0] if after_b['joint_pos_deg'] else 'na'} "
        f"fb_j4 {before_b['joint_pos_deg'][3] if len(before_b['joint_pos_deg']) > 3 else 'na'}"
        f"->{after_b['joint_pos_deg'][3] if len(after_b['joint_pos_deg']) > 3 else 'na'} "
        f"target_j1 {float(right_target[0]) if right_target else 'na'} "
        f"target_j4 {float(right_target[3]) if right_target and len(right_target) > 3 else 'na'}"
    )


def _ensure_ament_index_stub(sdk_root: Path) -> None:
    try:
        import_module("ament_index_python.packages")
        return
    except Exception:
        pass

    package_root = sdk_root / "tianji_output"

    def get_package_share_directory(_: str) -> str:
        return str(package_root)

    ament_module = ModuleType("ament_index_python")
    packages_module = ModuleType("ament_index_python.packages")
    packages_module.get_package_share_directory = get_package_share_directory
    ament_module.packages = packages_module
    sys.modules.setdefault("ament_index_python", ament_module)
    sys.modules["ament_index_python.packages"] = packages_module


def _as_arm_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (schema.LEFT_ARM_DOF,):
        raise TianjiHostError(f"{name} must have shape ({schema.LEFT_ARM_DOF},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise TianjiHostError(f"{name} contains NaN or Inf")
    return arr


def _parse_hex_bytes(text: str) -> bytes:
    cleaned = str(text).replace(",", " ").replace("0x", " ")
    parts = [part for part in cleaned.split() if part]
    try:
        return bytes(int(part, 16) & 0xFF for part in parts)
    except ValueError as exc:
        raise TianjiHostError(f"invalid end-channel hex frame: {text!r}") from exc


def _hex_bytes(data: bytes) -> str:
    return " ".join(f"{value:02X}" for value in data)


def _trim_hex(hex_text: str, size: int) -> str:
    if size <= 0:
        return ""
    parts = str(hex_text).split()
    return " ".join(parts[:size])
