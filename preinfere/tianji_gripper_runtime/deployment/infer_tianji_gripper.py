#!/usr/bin/env python3
"""Real-runtime GR00T inference entrypoint for Tianji right-arm gripper tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import time

import numpy as np
import yaml


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
DEFAULT_INFER_CONFIG_PATH = RUNTIME_ROOT / "configs" / "infer.yaml"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_gripper_runtime.runtime import schema
from tianji_gripper_runtime.runtime.action_adapter import (
    ActionAdapter,
    ActionAdapterError,
    RightArmGripperAction,
)
from tianji_gripper_runtime.runtime.action_keepalive import ActionKeepalive
from tianji_gripper_runtime.runtime.executor import ActionExecutor
from tianji_gripper_runtime.runtime.gripper_normalization import GripperCalibration
from tianji_gripper_runtime.runtime.groot_policy_client import GrootPolicyClient, PolicyServerError
from tianji_gripper_runtime.runtime.observation_builder import (
    ObservationBuilder,
    ObservationError,
    PiObservationBuilder,
    validate_policy_inputs,
)
from tianji_gripper_runtime.runtime.pi_policy_client import (
    PiPolicyClient,
    PiPolicyDependencyError,
    PiPolicyServerError,
)
from tianji_gripper_runtime.runtime.recorder import Recorder
from tianji_gripper_runtime.runtime.robot_interface import (
    RobotConnectionConfig,
    RobotError,
    make_robot,
)
from tianji_gripper_runtime.runtime.safety import SafetyConfig, SafetyError, SafetyLayer
from tianji_gripper_runtime.runtime.slow_dispatch import (
    SlowDispatchCancelledError,
    SlowInterpolatedDispatcher,
)
from tianji_wuji_runtime.runtime.camera_manager import CameraError, CameraManager
from tianji_wuji_runtime.runtime.event_logger import RuntimeEventLogger
from tianji_wuji_runtime.runtime.keyboard import (
    KeyboardController,
    RuntimeState,
    RuntimeStateMachine,
)


def _parse_optional_scalar(raw: str | None, *, name: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size != 1:
        raise ValueError(f"{name} must provide one value, got {values.size}")
    return (float(values[0]),)


def _parse_optional_vector(raw: str | None, *, name: str, dim: int) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size != dim:
        raise ValueError(f"{name} must provide {dim} comma-separated values, got {values.size}")
    return tuple(float(value) for value in values.tolist())


def _wait_for_policy_client(args: argparse.Namespace):
    deadline = time.monotonic() + float(args.policy_wait_sec)
    next_log_time = 0.0
    last_error: Exception | None = None
    while True:
        try:
            if args.policy_type == "pi":
                client = PiPolicyClient(
                    host=args.policy_host,
                    port=args.policy_port,
                    timeout_ms=args.policy_timeout_ms,
                    action_layout=args.pi_action_layout,
                    control_mode=args.control_mode,
                )
            else:
                client = GrootPolicyClient(
                    host=args.policy_host,
                    port=args.policy_port,
                    timeout_ms=args.policy_timeout_ms,
                )
            if client.ping():
                return client
        except PiPolicyDependencyError:
            # Retrying cannot repair a missing package in the current interpreter.
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        now = time.monotonic()
        if now >= deadline:
            raise PolicyServerError(
                f"policy server did not respond at {args.policy_host}:{args.policy_port} "
                f"within {args.policy_wait_sec:.1f}s; last error: {last_error}"
            )
        if now >= next_log_time:
            print(
                "[runtime] waiting for policy server "
                f"{args.policy_host}:{args.policy_port} from host {socket.gethostname()}"
            )
            next_log_time = now + 5.0
        time.sleep(min(1.0, max(deadline - now, 0.0)))


def _apply_manual_gripper_open(
    actions: list[RightArmGripperAction],
    *,
    target_gripper_q: float,
) -> list[RightArmGripperAction]:
    for action in actions:
        action.right_gripper_q[:] = np.float32(target_gripper_q)
        if action.control_left_gripper:
            action.left_gripper_q[:] = np.float32(target_gripper_q)
    return actions


def _build_reset_action(
    *,
    left_arm_q: tuple[float, ...],
    right_arm_q: tuple[float, ...],
    gripper1_motor_rad: float,
    gripper2_motor_rad: float,
) -> RightArmGripperAction:
    """Build the configured dual-arm/dual-gripper operator reset target."""
    return RightArmGripperAction(
        right_arm_q=np.asarray(right_arm_q, dtype=np.float32),
        left_arm_q=np.asarray(left_arm_q, dtype=np.float32),
        right_gripper_q=np.asarray([gripper2_motor_rad], dtype=np.float32),
        left_gripper_q=np.asarray([gripper1_motor_rad], dtype=np.float32),
        control_left_arm=True,
        control_left_gripper=True,
    )


def _send_prestart_gripper_open(
    robot,
    *,
    target_gripper_q: float,
    control_gripper_unit: str,
    control_left_side: bool,
    resend_count: int = 10,
    resend_period_sec: float = 0.05,
) -> None:
    target = np.asarray([target_gripper_q], dtype=np.float32)
    for _ in range(max(int(resend_count), 1)):
        if control_left_side:
            robot.left_gripper.send_position(target)
        robot.right_gripper.send_position(target)
        time.sleep(max(float(resend_period_sec), 0.0))
    print(
        "[runtime] prestart manual open: "
        f"{'both grippers' if control_left_side else 'right gripper'} commanded to max "
        f"({target_gripper_q:.4f} {control_gripper_unit})"
    )


def _print_gripper_debug(
    *,
    chunk_index: int,
    step_in_chunk: int,
    raw_action: np.ndarray | None,
    safe_action: RightArmGripperAction,
    state_before,
    state_after,
    executed: bool,
    error: str | None,
    safety_events: list[dict[str, object]],
) -> None:
    raw_right_gripper = None
    raw_left_gripper = None
    if raw_action is not None:
        arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if arr.size >= schema.FULL_STATE_DIM:
            raw_right_gripper = float(arr[schema.RIGHT_GRIPPER_SLICE][0])
            raw_left_gripper = float(arr[schema.LEFT_GRIPPER_SLICE][0])
        elif arr.size >= schema.RIGHT_ONLY_STATE_DIM:
            raw_right_gripper = float(arr[7])
    right_fb_before = None if state_before is None else float(state_before.right_gripper_q[0])
    right_fb_after = None if state_after is None else float(state_after.right_gripper_q[0])
    left_fb_before = None if state_before is None else float(state_before.left_gripper_q[0])
    left_fb_after = None if state_after is None else float(state_after.left_gripper_q[0])
    safety_summary = ",".join(str(event.get("type", "?")) for event in safety_events) or "-"
    print(
        "[gripper] "
        f"chunk={chunk_index} step={step_in_chunk} "
        f"right=(fb_before={right_fb_before!s} raw={raw_right_gripper!s} "
        f"safe={float(safe_action.right_gripper_q[0]):.4f} fb_after={right_fb_after!s}) "
        f"left=(fb_before={left_fb_before!s} raw={raw_left_gripper!s} "
        f"safe={float(safe_action.left_gripper_q[0]):.4f} fb_after={left_fb_after!s}) "
        f"executed={executed} "
        f"safety={safety_summary} error={error or '-'}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_INFER_CONFIG_PATH),
        help=f"YAML/JSON config file. Defaults to {DEFAULT_INFER_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--robot-limits",
        default=str(RUNTIME_ROOT / "configs" / "robot_limits.yaml"),
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-type", choices=["groot", "pi"], default="pi")
    parser.add_argument(
        "--control-mode",
        choices=["right_arm_right_gripper", "dual_arm_dual_gripper"],
        default="dual_arm_dual_gripper",
        help="Execution mode. The provided pi0.5 checkpoint supports right_arm_right_gripper safely.",
    )
    parser.add_argument(
        "--pi-action-layout",
        choices=[
            "ziyi_15d_right_left_right_gripper",
            "ziyi_16d_right_left_dual_gripper",
        ],
        default="ziyi_16d_right_left_dual_gripper",
    )
    parser.add_argument("--policy-timeout-ms", type=int, default=15000)
    parser.add_argument("--policy-wait-sec", type=float, default=60.0)
    parser.add_argument("--robot-arm-state-unit", default="deg")
    parser.add_argument("--policy-arm-state-unit", default="deg")
    parser.add_argument("--robot-gripper-state-unit", default="rad")
    parser.add_argument("--policy-gripper-state-unit", default="rad")
    parser.add_argument("--gripper-close-motor-rad", type=float, default=None)
    parser.add_argument("--gripper-open-motor-rad", type=float, default=None)
    parser.add_argument("--gripper2-close-motor-rad", type=float, default=None)
    parser.add_argument("--gripper2-open-motor-rad", type=float, default=None)
    parser.add_argument("--task", default="pick up anything")
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--chunk-sleep", type=float, default=0.0)
    parser.add_argument("--arm-interpolation-hz", type=float, default=None)
    parser.add_argument("--arm-interpolation-mode", choices=["cubic", "linear"], default="cubic")
    parser.add_argument("--slow-dispatch-duration-sec", type=float, default=1.0)
    parser.add_argument("--slow-dispatch-frequency-hz", type=float, default=20.0)
    parser.add_argument(
        "--slow-dispatch-interpolation-mode",
        choices=["cubic", "linear"],
        default="cubic",
    )
    parser.add_argument("--reset-left-arm-q", default="140,-90,-90,-120,0,0,0")
    parser.add_argument("--reset-right-arm-q", default="-140,-90,90,-120,0,0,0")
    parser.add_argument("--reset-gripper1-motor-rad", type=float, default=-5.0616)
    parser.add_argument("--reset-gripper2-motor-rad", type=float, default=-5.2817)
    parser.add_argument("--slow-dispatch-enabled", action="store_true")
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-dir", default=str(RUNTIME_ROOT / "infer_logs"))
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--image-source", default=None)
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument("--camera-capture-fps", type=float, default=60.0)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--head-stereo-crop", choices=["left", "right"], default=None)
    parser.add_argument("--max-camera-age-ms", type=float, default=150.0)
    parser.add_argument("--camera-warmup-sec", type=float, default=3.0)
    parser.add_argument("--robot-backend", default="fake")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--left-arm-ip", default=None)
    parser.add_argument("--left-arm-port", type=int, default=None)
    parser.add_argument("--right-arm-ip", default=None)
    parser.add_argument("--right-arm-port", type=int, default=None)
    parser.add_argument("--gripper-backend", default="fake")
    parser.add_argument("--left-gripper-backend", default=None)
    parser.add_argument("--right-gripper-backend", default=None)
    parser.add_argument("--left-gripper-ip", default=None)
    parser.add_argument("--left-gripper-port", type=int, default=None)
    parser.add_argument("--right-gripper-ip", default=None)
    parser.add_argument("--right-gripper-port", type=int, default=None)
    parser.add_argument("--left-gripper-serial", default=None)
    parser.add_argument("--right-gripper-serial", default=None)
    parser.add_argument("--left-gripper-home", default=None)
    parser.add_argument("--right-gripper-home", default=None)
    parser.add_argument(
        "--left-arm-freeze-q",
        default=None,
        help="Optional fixed left arm pose in deg, e.g. '90,-90,-90,0,0,0,0'.",
    )
    parser.add_argument(
        "--left-gripper-freeze-q",
        default=None,
        help="Optional fixed left gripper pose in rad for right-only inference.",
    )
    parser.add_argument("--gripper-sdk-module", default=None)
    parser.add_argument("--gripper-sdk-class", default=None)
    parser.add_argument("--gripper-485-arm", choices=["A", "B"], default="A")
    parser.add_argument("--gripper-485-com", type=int, default=1)
    parser.add_argument("--left-gripper-485-arm", choices=["A", "B"], default=None)
    parser.add_argument("--left-gripper-485-com", type=int, default=None)
    parser.add_argument("--right-gripper-485-arm", choices=["A", "B"], default=None)
    parser.add_argument("--right-gripper-485-com", type=int, default=None)
    parser.add_argument("--gripper-direct-onset", action="store_true")
    parser.add_argument("--gripper-rs05-target-id", type=lambda value: int(value, 0), default=0x7F)
    parser.add_argument("--gripper-rs05-master-id", type=lambda value: int(value, 0), default=0xFD)
    parser.add_argument("--gripper-can-id-byteorder", choices=["little", "big"], default="little")
    parser.add_argument("--gripper-standard-id-bytes", type=int, choices=[2, 4], default=4)
    parser.add_argument("--gripper-enter-motor", action="store_true")
    parser.add_argument("--gripper-stop-on-disconnect", action="store_true")
    parser.add_argument("--gripper-kp", type=float, default=80.0)
    parser.add_argument("--gripper-kd", type=float, default=1.0)
    parser.add_argument("--gripper-torque-nm", type=float, default=0.0)
    parser.add_argument("--gripper-min-pos-rad", type=float, default=-5.5)
    parser.add_argument("--gripper-max-pos-rad", type=float, default=1.2)
    parser.add_argument("--gripper-torque-protection-enabled", action="store_true")
    parser.add_argument(
        "--gripper-torque-protection-mode",
        choices=["torque", "disabled"],
        default="torque",
    )
    parser.add_argument("--gripper-torque-filter-alpha", type=float, default=0.3)
    parser.add_argument("--gripper-torque-threshold-nm", type=float, default=1.0)
    parser.add_argument("--gripper-torque-release-threshold-nm", type=float, default=0.2)
    parser.add_argument("--gripper-torque-count-threshold", type=int, default=5)
    parser.add_argument("--gripper-torque-extra-tighten-rad", type=float, default=0.0)
    parser.add_argument("--gripper-holding-kp", type=float, default=4.0)
    parser.add_argument("--gripper-holding-kd", type=float, default=0.1)
    parser.add_argument("--gripper-closing-direction", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--gripper-torque-direction-deadband-rad", type=float, default=0.01)
    parser.add_argument("--gripper-open-key", default="o")
    parser.add_argument(
        "--gripper-open-limit",
        choices=["min", "max"],
        default="min",
        help="Which configured gripper limit corresponds to fully open for the current hardware.",
    )
    parser.add_argument("--tianji-sdk-root", default=None)
    parser.add_argument("--tianji-config-path", default=None)
    parser.add_argument("--tianji-vel-ratio", type=int, default=80)
    parser.add_argument("--tianji-acc-ratio", type=int, default=70)
    parser.add_argument("--tianji-settle-sec", type=float, default=0.5)
    parser.add_argument("--tianji-tool-arm-a-kinematics", default=None)
    parser.add_argument("--tianji-tool-arm-a-dynamics", default=None)
    parser.add_argument("--tianji-tool-arm-b-kinematics", default=None)
    parser.add_argument("--tianji-tool-arm-b-dynamics", default=None)
    parser.add_argument("--tianji-state3-joint-k-a", default=None)
    parser.add_argument("--tianji-state3-joint-d-a", default=None)
    parser.add_argument("--tianji-state3-joint-k-b", default=None)
    parser.add_argument("--tianji-state3-joint-d-b", default=None)
    parser.add_argument("--tianji-home-joints-a", default=None)
    parser.add_argument("--tianji-home-joints-b", default=None)
    parser.add_argument("--max-arm-joint-step", type=float, default=10.0)
    parser.add_argument("--max-gripper-step", type=float, default=0.1)
    parser.add_argument("--max-arm-velocity", type=float, default=None)
    parser.add_argument("--max-gripper-velocity", type=float, default=None)
    parser.add_argument("--ignore-gripper-safety", action="store_true")
    parser.add_argument("--use-filter", action="store_true")
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    parser.add_argument("--policy-unit", default="deg")
    parser.add_argument("--control-unit", default="deg")
    parser.add_argument("--policy-arm-unit", default=None)
    parser.add_argument("--control-arm-unit", default=None)
    parser.add_argument("--policy-gripper-unit", default=None)
    parser.add_argument("--control-gripper-unit", default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    return parser


def _load_cli_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"config file does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a top-level mapping")
    return raw


def _normalize_config_key(key: str) -> str:
    return key.strip().replace("-", "_")


def _value_to_cli_tokens(*, key: str, value: object, action: argparse.Action) -> list[str]:
    flag = next(opt for opt in action.option_strings if opt.startswith("--"))
    if isinstance(action, argparse._AppendAction):
        if not isinstance(value, list):
            raise ValueError(f"config key {key!r} must be a list")
        tokens: list[str] = []
        for item in value:
            tokens.extend([flag, str(item)])
        return tokens
    if isinstance(action, argparse._StoreTrueAction):
        if not isinstance(value, bool):
            raise ValueError(f"config key {key!r} must be true/false")
        return [flag] if value else []
    if isinstance(value, (list, dict)):
        raise ValueError(f"config key {key!r} must be a scalar for {flag}")
    # Use --flag=value so negative numeric vectors (for example a B-arm home
    # pose beginning with -90) cannot be mistaken for another CLI option.
    return [f"{flag}={value}"]


def _config_to_argv(
    config_values: dict[str, object],
    *,
    parser: argparse.ArgumentParser,
) -> list[str]:
    actions = {
        action.dest: action
        for action in parser._actions
        if action.option_strings and action.dest not in {"help", "config"}
    }
    argv: list[str] = []
    for raw_key, value in config_values.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"config keys must be strings, got {type(raw_key)!r}")
        key = _normalize_config_key(raw_key)
        action = actions.get(key)
        if action is None:
            known = ", ".join(sorted(actions))
            raise ValueError(f"unknown config key {raw_key!r}; known keys: {known}")
        if value is None:
            continue
        argv.extend(_value_to_cli_tokens(key=raw_key, value=value, action=action))
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_INFER_CONFIG_PATH))
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    parser = _build_parser()
    config_argv: list[str] = []
    if bootstrap_args.config is not None:
        config_values = _load_cli_config(Path(bootstrap_args.config).expanduser())
        config_argv = _config_to_argv(config_values, parser=parser)
    return parser.parse_args(config_argv + argv)


def main() -> int:
    args = parse_args()
    if args.execution_horizon <= 0:
        raise ValueError("--execution-horizon must be positive")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.chunk_sleep < 0:
        raise ValueError("--chunk-sleep must be non-negative")
    if args.slow_dispatch_duration_sec < 0:
        raise ValueError("--slow-dispatch-duration-sec must be non-negative")
    if args.slow_dispatch_frequency_hz <= 0:
        raise ValueError("--slow-dispatch-frequency-hz must be positive")
    reset_left_arm_q = _parse_optional_vector(
        args.reset_left_arm_q, name="--reset-left-arm-q", dim=7
    )
    reset_right_arm_q = _parse_optional_vector(
        args.reset_right_arm_q, name="--reset-right-arm-q", dim=7
    )
    if reset_left_arm_q is None or reset_right_arm_q is None:
        raise ValueError("reset arm targets must be configured")
    if args.no_keyboard and not args.auto_start:
        raise ValueError("--no-keyboard requires --auto-start")
    if (
        args.control_mode == "dual_arm_dual_gripper"
        and args.pi_action_layout != "ziyi_16d_right_left_dual_gripper"
    ):
        raise ValueError(
            "--control-mode dual_arm_dual_gripper requires "
            "--pi-action-layout ziyi_16d_right_left_dual_gripper"
        )
    if args.control_mode == "dual_arm_dual_gripper" and (
        args.left_arm_freeze_q is not None or args.left_gripper_freeze_q is not None
    ):
        raise ValueError(
            "left-side freeze options are invalid in dual_arm_dual_gripper mode; "
            "remove left_arm_freeze_q/left_gripper_freeze_q so live feedback is used"
        )
    if not args.dry_run and args.robot_backend == "fake":
        print("[runtime] warning: robot-backend=fake; commands update only in-memory state")

    adapter = ActionAdapter(
        policy_unit=args.policy_unit,
        control_unit=args.control_unit,
        policy_arm_unit=args.policy_arm_unit,
        control_arm_unit=args.control_arm_unit,
        policy_gripper_unit=args.policy_gripper_unit,
        control_gripper_unit=args.control_gripper_unit,
        action_mode=args.action_mode,
        control_mode=args.control_mode,
        right_gripper_calibration=GripperCalibration(
            close_motor_rad=args.gripper2_close_motor_rad,
            open_motor_rad=args.gripper2_open_motor_rad,
            label="right gripper (gripper2)",
        ),
        left_gripper_calibration=GripperCalibration(
            close_motor_rad=args.gripper_close_motor_rad,
            open_motor_rad=args.gripper_open_motor_rad,
            label="left gripper (gripper)",
        ),
    )
    policy = _wait_for_policy_client(args)
    modality_configs = policy.get_modality_config()
    if args.policy_type == "pi":
        obs_builder = PiObservationBuilder(
            robot_arm_state_unit=args.robot_arm_state_unit,
            policy_arm_state_unit=args.policy_arm_state_unit,
            robot_gripper_state_unit=args.robot_gripper_state_unit,
            policy_gripper_state_unit=args.policy_gripper_state_unit,
            right_gripper_calibration=GripperCalibration(
                close_motor_rad=args.gripper2_close_motor_rad,
                open_motor_rad=args.gripper2_open_motor_rad,
                label="right gripper (gripper2)",
            ),
            left_gripper_calibration=GripperCalibration(
                close_motor_rad=args.gripper_close_motor_rad,
                open_motor_rad=args.gripper_open_motor_rad,
                label="left gripper (gripper)",
            ),
        )
    else:
        obs_builder = ObservationBuilder(
            modality_configs,
            robot_arm_state_unit=args.robot_arm_state_unit,
            policy_arm_state_unit=args.policy_arm_state_unit,
            robot_gripper_state_unit=args.robot_gripper_state_unit,
            policy_gripper_state_unit=args.policy_gripper_state_unit,
            right_gripper_calibration=GripperCalibration(
                close_motor_rad=args.gripper2_close_motor_rad,
                open_motor_rad=args.gripper2_open_motor_rad,
                label="right gripper (gripper2)",
            ),
            left_gripper_calibration=GripperCalibration(
                close_motor_rad=args.gripper_close_motor_rad,
                open_motor_rad=args.gripper_open_motor_rad,
                label="left gripper (gripper)",
            ),
        )

    cameras = CameraManager.from_cli_specs(
        args.camera,
        required_keys=obs_builder.required_camera_keys,
        image_source=args.image_source,
        allow_dummy=args.dry_run,
        width=args.camera_width,
        height=args.camera_height,
        capture_fps=args.camera_capture_fps,
        fps=args.camera_fps,
        head_stereo_crop=args.head_stereo_crop,
    )
    robot = make_robot(
        RobotConnectionConfig(
            backend=args.robot_backend,
            robot_ip=args.robot_ip,
            left_arm_ip=args.left_arm_ip,
            left_arm_port=args.left_arm_port,
            right_arm_ip=args.right_arm_ip,
            right_arm_port=args.right_arm_port,
            gripper_backend=args.gripper_backend,
            left_gripper_backend=args.left_gripper_backend,
            right_gripper_backend=args.right_gripper_backend,
            left_gripper_ip=args.left_gripper_ip,
            left_gripper_port=args.left_gripper_port,
            right_gripper_ip=args.right_gripper_ip,
            right_gripper_port=args.right_gripper_port,
            left_gripper_serial=args.left_gripper_serial,
            right_gripper_serial=args.right_gripper_serial,
            left_gripper_home=_parse_optional_scalar(
                args.left_gripper_home,
                name="--left-gripper-home",
            ),
            right_gripper_home=_parse_optional_scalar(
                args.right_gripper_home,
                name="--right-gripper-home",
            ),
            gripper_sdk_module=args.gripper_sdk_module,
            gripper_sdk_class=args.gripper_sdk_class,
            tianji_sdk_root=args.tianji_sdk_root,
            tianji_config_path=args.tianji_config_path,
            tianji_vel_ratio=args.tianji_vel_ratio,
            tianji_acc_ratio=args.tianji_acc_ratio,
            tianji_settle_sec=args.tianji_settle_sec,
            tianji_tool_arm_a_kinematics=_parse_optional_vector(
                args.tianji_tool_arm_a_kinematics,
                name="--tianji-tool-arm-a-kinematics",
                dim=6,
            ),
            tianji_tool_arm_a_dynamics=_parse_optional_vector(
                args.tianji_tool_arm_a_dynamics,
                name="--tianji-tool-arm-a-dynamics",
                dim=10,
            ),
            tianji_tool_arm_b_kinematics=_parse_optional_vector(
                args.tianji_tool_arm_b_kinematics,
                name="--tianji-tool-arm-b-kinematics",
                dim=6,
            ),
            tianji_tool_arm_b_dynamics=_parse_optional_vector(
                args.tianji_tool_arm_b_dynamics,
                name="--tianji-tool-arm-b-dynamics",
                dim=10,
            ),
            tianji_state3_joint_k_a=_parse_optional_vector(
                args.tianji_state3_joint_k_a,
                name="--tianji-state3-joint-k-a",
                dim=schema.LEFT_ARM_DOF,
            ),
            tianji_state3_joint_d_a=_parse_optional_vector(
                args.tianji_state3_joint_d_a,
                name="--tianji-state3-joint-d-a",
                dim=schema.LEFT_ARM_DOF,
            ),
            tianji_state3_joint_k_b=_parse_optional_vector(
                args.tianji_state3_joint_k_b,
                name="--tianji-state3-joint-k-b",
                dim=schema.RIGHT_ARM_DOF,
            ),
            tianji_state3_joint_d_b=_parse_optional_vector(
                args.tianji_state3_joint_d_b,
                name="--tianji-state3-joint-d-b",
                dim=schema.RIGHT_ARM_DOF,
            ),
            tianji_home_joints_a=_parse_optional_vector(
                args.tianji_home_joints_a,
                name="--tianji-home-joints-a",
                dim=schema.LEFT_ARM_DOF,
            ),
            tianji_home_joints_b=_parse_optional_vector(
                args.tianji_home_joints_b,
                name="--tianji-home-joints-b",
                dim=schema.RIGHT_ARM_DOF,
            ),
            gripper_485_arm=args.gripper_485_arm,
            gripper_485_com=args.gripper_485_com,
            left_gripper_485_arm=args.left_gripper_485_arm,
            left_gripper_485_com=args.left_gripper_485_com,
            right_gripper_485_arm=args.right_gripper_485_arm,
            right_gripper_485_com=args.right_gripper_485_com,
            gripper_direct_onset=args.gripper_direct_onset,
            gripper_rs05_target_id=args.gripper_rs05_target_id,
            gripper_rs05_master_id=args.gripper_rs05_master_id,
            gripper_can_id_byteorder=args.gripper_can_id_byteorder,
            gripper_standard_id_bytes=args.gripper_standard_id_bytes,
            gripper_enter_motor=args.gripper_enter_motor,
            gripper_stop_on_disconnect=args.gripper_stop_on_disconnect,
            gripper_kp=args.gripper_kp,
            gripper_kd=args.gripper_kd,
            gripper_torque_nm=args.gripper_torque_nm,
            gripper_min_pos_rad=args.gripper_min_pos_rad,
            gripper_max_pos_rad=args.gripper_max_pos_rad,
            gripper_torque_protection_enabled=args.gripper_torque_protection_enabled,
            gripper_torque_protection_mode=args.gripper_torque_protection_mode,
            gripper_torque_filter_alpha=args.gripper_torque_filter_alpha,
            gripper_torque_threshold_nm=args.gripper_torque_threshold_nm,
            gripper_torque_release_threshold_nm=args.gripper_torque_release_threshold_nm,
            gripper_torque_count_threshold=args.gripper_torque_count_threshold,
            gripper_torque_extra_tighten_rad=args.gripper_torque_extra_tighten_rad,
            gripper_holding_kp=args.gripper_holding_kp,
            gripper_holding_kd=args.gripper_holding_kd,
            gripper_closing_direction=args.gripper_closing_direction,
            gripper_torque_direction_deadband_rad=args.gripper_torque_direction_deadband_rad,
            command_left_side=args.control_mode == "dual_arm_dual_gripper",
            left_arm_freeze_q=_parse_optional_vector(
                args.left_arm_freeze_q,
                name="--left-arm-freeze-q",
                dim=schema.LEFT_ARM_DOF,
            ),
            left_gripper_freeze_q=_parse_optional_scalar(
                args.left_gripper_freeze_q,
                name="--left-gripper-freeze-q",
            ),
        )
    )
    safety_config = SafetyConfig.from_yaml(args.robot_limits)
    safety_config.arm_max_step = args.max_arm_joint_step
    safety_config.gripper_max_step = args.max_gripper_step
    if args.max_arm_velocity is not None:
        safety_config.arm_max_velocity = np.full(
            schema.RIGHT_ARM_DOF,
            args.max_arm_velocity,
            dtype=np.float32,
        )
    if args.max_gripper_velocity is not None:
        safety_config.gripper_max_velocity = np.full(
            schema.RIGHT_GRIPPER_DOF,
            args.max_gripper_velocity,
            dtype=np.float32,
        )
    if args.ignore_gripper_safety:
        safety_config.enable_gripper_limit = False
        safety_config.enable_gripper_delta_clip = False
        safety_config.enable_gripper_velocity_limit = False
    safety_config.enable_filter = args.use_filter
    safety = SafetyLayer(safety_config, adapter)
    if args.gripper_open_limit == "min":
        manual_gripper_open_target = float(
            max(
                float(args.gripper_min_pos_rad),
                float(safety_config.right_gripper_min[0]),
            )
        )
    else:
        manual_gripper_open_target = float(
            min(
                float(args.gripper_max_pos_rad),
                float(safety_config.right_gripper_max[0]),
            )
        )
    recorder = Recorder(
        args.record_dir,
        config={
            "entrypoint": "infer_tianji_gripper.py",
            "inference_mode": "sync",
            "policy_type": args.policy_type,
            "control_mode": args.control_mode,
            "pi_action_layout": args.pi_action_layout,
            "policy_host": args.policy_host,
            "policy_port": args.policy_port,
            "task": args.task,
            "execution_horizon": args.execution_horizon,
            "duration": args.duration,
            "chunk_sleep": args.chunk_sleep,
            "slow_dispatch_duration_sec": args.slow_dispatch_duration_sec,
            "slow_dispatch_frequency_hz": args.slow_dispatch_frequency_hz,
            "slow_dispatch_interpolation_mode": args.slow_dispatch_interpolation_mode,
            "slow_dispatch_enabled": args.slow_dispatch_enabled,
            "dry_run": args.dry_run,
            "ignore_gripper_safety": args.ignore_gripper_safety,
            "robot_backend": args.robot_backend,
            "gripper_backend": args.gripper_backend,
            "left_gripper_backend": args.left_gripper_backend,
            "right_gripper_backend": args.right_gripper_backend,
            "left_gripper_485_arm": args.left_gripper_485_arm,
            "left_gripper_485_com": args.left_gripper_485_com,
            "right_gripper_485_arm": args.right_gripper_485_arm,
            "right_gripper_485_com": args.right_gripper_485_com,
            "camera": args.camera,
            "gripper_open_key": args.gripper_open_key,
            "gripper_open_limit": args.gripper_open_limit,
            "manual_gripper_open_target": manual_gripper_open_target,
            "modality": {
                name: {
                    "keys": list(cfg.modality_keys),
                    "delta_indices": list(cfg.delta_indices),
                }
                for name, cfg in modality_configs.items()
            },
        },
        adapter=adapter,
    )
    runtime_event_logger = RuntimeEventLogger(recorder.run_dir / "runtime_events.jsonl")
    log_event = runtime_event_logger.log
    robot.event_logger = log_event
    def _print_runtime_gripper_debug(**payload: object) -> None:
        _print_gripper_debug(
            **payload,
        )
        statuses = [("right", robot.right_gripper)]
        if args.control_mode == "dual_arm_dual_gripper":
            statuses.append(("left", robot.left_gripper))
        for side, status in statuses:
            print(
                "[gripper-status] "
                f"side={side} mode={getattr(status, 'last_feedback_mode_state', None)!s} "
                f"fault={getattr(status, 'last_feedback_has_fault', None)!s} "
                f"warn={getattr(status, 'last_feedback_has_warning', None)!s} "
                f"torque_nm={getattr(status, 'last_feedback_torque_nm', None)!s} "
                f"temp_c={getattr(status, 'last_feedback_temperature_c', None)!s}"
            )

    executor = ActionExecutor(
        robot,
        adapter=adapter,
        recorder=recorder,
        event_logger=log_event,
        debug_callback=_print_runtime_gripper_debug,
        arm_interpolation_hz=args.arm_interpolation_hz,
        arm_interpolation_mode=args.arm_interpolation_mode,
        slow_dispatch_enabled=args.slow_dispatch_enabled,
        slow_dispatch_duration_sec=args.slow_dispatch_duration_sec,
        slow_dispatch_frequency_hz=args.slow_dispatch_frequency_hz,
        slow_dispatch_interpolation_mode=args.slow_dispatch_interpolation_mode,
    )
    reset_dispatcher = SlowInterpolatedDispatcher(
        robot,
        adapter=adapter,
        duration_sec=args.slow_dispatch_duration_sec,
        frequency_hz=args.slow_dispatch_frequency_hz,
        interpolation_mode=args.slow_dispatch_interpolation_mode,
        event_logger=log_event,
    )
    keepalive = ActionKeepalive(robot, event_logger=log_event)
    state_machine = RuntimeStateMachine(
        auto_start=args.auto_start,
        safe_mode=args.safe_mode,
        manual_gripper_open_key=args.gripper_open_key,
    )
    manual_gripper_open_active = False
    manual_gripper_open_tolerance = 1e-3

    print(f"[runtime] logs: {recorder.run_dir}")
    print(
        "[runtime] controls: R run/resume, E pause (P also works), B reset, Space hold, "
        f"{args.gripper_open_key.upper()} prestart/override gripper max open, "
        "H home, N next safe chunk, Q quit"
    )
    chunk_count = 0
    previous_runtime_state: RuntimeState | None = None
    try:
        log_event("robot_connect_start", backend=args.robot_backend, robot_ip=args.robot_ip)
        robot.connect()
        initial_robot_state = robot.get_state()
        print(
            "[runtime] live initial arm feedback: "
            f"A/left={initial_robot_state.left_arm_q.tolist()} "
            f"B/right={initial_robot_state.right_arm_q.tolist()}"
        )
        log_event(
            "robot_connect_end",
            freeze_targets=robot.freeze_targets(),
            initial_state=initial_robot_state.as_dict(),
        )
        cameras.connect_all()
        cameras.start_streaming()
        cameras.wait_until_ready(timeout_sec=args.camera_warmup_sec)
        with KeyboardController(enabled=not args.no_keyboard) as keyboard:

            def stop_requested() -> bool:
                state_machine.update(keyboard.poll())
                if state_machine.home_requested:
                    keepalive.stop()
                    robot.go_home()
                    state_machine.home_requested = False
                if state_machine.quit_requested:
                    keepalive.stop()
                    robot.hold_position()
                    return True
                return state_machine.state != RuntimeState.RUNNING

            while True:
                state_machine.update(keyboard.poll())
                if state_machine.home_requested:
                    keepalive.stop()
                    robot.go_home()
                    state_machine.home_requested = False
                if state_machine.quit_requested:
                    keepalive.stop()
                    robot.hold_position()
                    break
                if state_machine.consume_reset_request():
                    if args.control_mode != "dual_arm_dual_gripper":
                        print("[runtime] B reset requires --control-mode dual_arm_dual_gripper")
                        log_event(
                            "reset_rejected",
                            reason="reset_requires_dual_arm_dual_gripper",
                            control_mode=args.control_mode,
                        )
                    else:
                        keepalive.stop()
                        print(
                            "[runtime] B reset: moving A/B arms and both grippers "
                            f"with slow interpolation ({args.slow_dispatch_duration_sec:.2f}s, "
                            f"{args.slow_dispatch_frequency_hz:.1f}Hz)"
                        )
                        try:
                            reset_dispatcher.dispatch_control_action(
                                _build_reset_action(
                                    left_arm_q=reset_left_arm_q,
                                    right_arm_q=reset_right_arm_q,
                                    gripper1_motor_rad=args.reset_gripper1_motor_rad,
                                    gripper2_motor_rad=args.reset_gripper2_motor_rad,
                                ),
                                stop_callback=lambda: state_machine.quit_requested,
                            )
                            state_machine.state = RuntimeState.PAUSED
                            log_event(
                                "reset_reached",
                                trigger_key="b",
                                target={
                                    "left_arm": list(reset_left_arm_q),
                                    "right_arm": list(reset_right_arm_q),
                                    "gripper1_motor_rad": args.reset_gripper1_motor_rad,
                                    "gripper2_motor_rad": args.reset_gripper2_motor_rad,
                                },
                            )
                            print("[runtime] B reset reached; press R to resume model inference")
                        except SlowDispatchCancelledError:
                            print("[runtime] B reset cancelled; robot is holding position")
                        except RobotError as exc:
                            print(f"[runtime] B reset failed: {exc}")
                            log_event("reset_error", error=str(exc))
                    previous_runtime_state = state_machine.state
                    continue
                if (
                    args.control_mode == "right_arm_right_gripper"
                    and state_machine.state == RuntimeState.RUNNING
                    and previous_runtime_state != RuntimeState.RUNNING
                    and args.left_arm_freeze_q is None
                    and args.left_gripper_freeze_q is None
                ):
                    freeze_targets = robot.capture_freeze_targets()
                    print(
                        "[runtime] captured left-side freeze targets at run start: "
                        f"left_arm={freeze_targets['left_arm']} "
                        f"left_gripper={freeze_targets['left_gripper']}"
                    )
                    log_event(
                        "left_freeze_targets_captured",
                        trigger="run_start",
                        freeze_targets=freeze_targets,
                    )
                previous_runtime_state = state_machine.state
                if state_machine.state != RuntimeState.RUNNING:
                    keepalive.stop()
                    if state_machine.consume_manual_gripper_open_request():
                        _send_prestart_gripper_open(
                            robot,
                            target_gripper_q=manual_gripper_open_target,
                            control_gripper_unit=args.control_gripper_unit,
                            control_left_side=args.control_mode == "dual_arm_dual_gripper",
                        )
                        log_event(
                            "manual_gripper_open_prestart",
                            chunk_index=chunk_count,
                            target_gripper_q=manual_gripper_open_target,
                            control_gripper_unit=args.control_gripper_unit,
                            trigger_key=args.gripper_open_key,
                        )
                    time.sleep(0.01)
                    continue

                chunk_index = chunk_count
                raw_chunk = None
                observation = None
                safe_actions = []
                safety_events = []
                try:
                    keepalive.raise_if_failed()
                    state_t0 = time.perf_counter()
                    robot_state = robot.get_state()
                    state_t1 = time.perf_counter()
                    reference_time = 0.5 * (state_t0 + state_t1)
                    frames = cameras.snapshot_latest(
                        reference_time=reference_time,
                        max_age_ms=args.max_camera_age_ms,
                    )
                    images = {key: frame.image for key, frame in frames.items()}
                    robot_state = validate_policy_inputs(
                        robot_state,
                        images,
                        required_camera_keys=obs_builder.required_camera_keys,
                    )
                    if state_machine.consume_manual_gripper_open_request():
                        manual_gripper_open_active = True
                        print(
                            "[runtime] manual override armed: opening gripper to max "
                            f"({manual_gripper_open_target:.4f} {args.control_gripper_unit})"
                        )
                        log_event(
                            "manual_gripper_open_armed",
                            chunk_index=chunk_index,
                            target_gripper_q=manual_gripper_open_target,
                            control_gripper_unit=args.control_gripper_unit,
                            trigger_key=args.gripper_open_key,
                        )
                    right_gripper_open = np.isclose(
                        float(robot_state.right_gripper_q[0]),
                        manual_gripper_open_target,
                        atol=manual_gripper_open_tolerance,
                    )
                    left_gripper_open = (
                        args.control_mode != "dual_arm_dual_gripper"
                        or np.isclose(
                            float(robot_state.left_gripper_q[0]),
                            manual_gripper_open_target,
                            atol=manual_gripper_open_tolerance,
                        )
                    )
                    if manual_gripper_open_active and right_gripper_open and left_gripper_open:
                        manual_gripper_open_active = False
                        log_event(
                            "manual_gripper_open_reached",
                            chunk_index=chunk_index,
                            target_gripper_q=manual_gripper_open_target,
                            control_gripper_unit=args.control_gripper_unit,
                        )
                    if args.policy_type == "pi":
                        state_pi = (
                            robot.get_pi_state_16()
                            if getattr(policy, "policy_state_dim", 15) == 16
                            else robot.get_pi_state_15()
                        )
                        observation = obs_builder.build(
                            state_pi=state_pi,
                            images=images,
                            task=args.task,
                        )
                    else:
                        observation = obs_builder.build(robot_state, images, args.task)
                    raw_chunk = policy.predict_action_chunk(
                        observation,
                        min_horizon=args.execution_horizon,
                    )
                    # The policy normally returns its full 50-step training
                    # horizon even when Safe Mode requests one step. Limit the
                    # chunk before safety processing so --execution-horizon 1
                    # really validates and executes exactly one approved step.
                    actions = adapter.split_chunk(raw_chunk)[: args.execution_horizon]
                    if manual_gripper_open_active:
                        actions = _apply_manual_gripper_open(
                            actions,
                            target_gripper_q=manual_gripper_open_target,
                        )
                        log_event(
                            "manual_gripper_open",
                            chunk_index=chunk_index,
                            target_gripper_q=manual_gripper_open_target,
                            control_gripper_unit=args.control_gripper_unit,
                            trigger_key=args.gripper_open_key,
                        )
                    if args.safe_mode:
                        # Each N key press is an independent operator-approved
                        # step. Hard limits remain active, while an expected
                        # delta clip from previous approved steps must not
                        # accumulate into a false 20-event shutdown.
                        safety.reset_consecutive_events()
                    safe_actions, safety_events = safety.process_chunk(
                        robot_state,
                        actions,
                        args.duration,
                    )
                    if safety_events:
                        event_summary: dict[str, int] = {}
                        for event in safety_events:
                            key = f"{event.get('type', '?')}:{event.get('segment', '?')}"
                            event_summary[key] = event_summary.get(key, 0) + 1
                        print(
                            f"[safety] chunk={chunk_index} processed_steps={len(actions)} "
                            f"events={event_summary}"
                        )
                        log_event(
                            "safety_events",
                            chunk_index=chunk_index,
                            processed_steps=len(actions),
                            events=safety_events,
                            summary=event_summary,
                        )
                    keepalive.stop()
                    last_executed_action = executor.execute_chunk(
                        safe_actions,
                        args.duration,
                        dry_run=args.dry_run,
                        chunk_index=chunk_index,
                        raw_chunk=raw_chunk,
                        safety_events=safety_events,
                        stop_callback=stop_requested,
                    )
                    chunk_count += 1
                    state_machine.pause_after_safe_chunk()
                    reached_max_chunks = (
                        args.max_chunks is not None and chunk_count >= args.max_chunks
                    )
                    if (
                        last_executed_action is not None
                        and state_machine.state == RuntimeState.RUNNING
                        and not reached_max_chunks
                    ):
                        keepalive.start(last_executed_action, args.duration)
                    recorder.save_chunk(
                        observation=observation,
                        raw_chunk=raw_chunk,
                        safe_actions=safe_actions,
                        safety_events=safety_events,
                        inference_latency_ms=policy.last_latency_ms,
                    )
                    if not reached_max_chunks and args.chunk_sleep > 0:
                        time.sleep(args.chunk_sleep)
                    if reached_max_chunks:
                        break
                except (
                    ActionAdapterError,
                    CameraError,
                    ObservationError,
                    PolicyServerError,
                    PiPolicyServerError,
                    RobotError,
                    SafetyError,
                ) as exc:
                    print(f"[runtime] ERROR: {exc}")
                    try:
                        current_gripper = float(robot.get_state().right_gripper_q[0])
                    except Exception:  # noqa: BLE001
                        current_gripper = None
                    print(
                        "[gripper] runtime_error "
                        f"chunk={chunk_index} fb_now={current_gripper!s} "
                        f"manual_open_active={manual_gripper_open_active} error={exc}"
                    )
                    log_event(
                        "runtime_error",
                        chunk_index=chunk_index,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    keepalive.stop()
                    robot.hold_position()
                    state_machine.to_error()
                    if args.no_keyboard:
                        return 2
    except KeyboardInterrupt:
        print("\n[runtime] interrupted")
    finally:
        keepalive.stop()
        robot.hold_position()
        cameras.stop_streaming()
        cameras.disconnect_all()
        robot.disconnect()
        runtime_event_logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
