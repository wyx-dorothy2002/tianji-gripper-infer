#!/usr/bin/env python3
"""Real-runtime GR00T inference entrypoint for Tianji Wuji."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import yaml


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
DEFAULT_LEFT_FREEZE_TARGET_PATH = RUNTIME_ROOT / "configs" / "left_freeze_defaults.json"
DEFAULT_INFER_CONFIG_PATH = RUNTIME_ROOT / "configs" / "infer.yaml"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.action_adapter import (
    ActionAdapter,
    ActionAdapterError,
    DualArmHandAction,
)
from tianji_wuji_runtime.runtime.action_keepalive import ActionKeepalive
from tianji_wuji_runtime.runtime.camera_manager import CameraError, CameraManager
from tianji_wuji_runtime.runtime.control_overrides import (
    ControlOverrideError,
    LeftSideFreezeTarget,
    apply_left_side_freeze,
)
from tianji_wuji_runtime.runtime.event_logger import RuntimeEventLogger
from tianji_wuji_runtime.runtime.executor import ActionExecutor
from tianji_wuji_runtime.runtime.groot_policy_client import GrootPolicyClient, PolicyServerError
from tianji_wuji_runtime.runtime.keyboard import (
    KeyboardController,
    RuntimeState,
    RuntimeStateMachine,
)
from tianji_wuji_runtime.runtime.observation_builder import (
    ObservationBuilder,
    ObservationError,
    validate_policy_inputs,
)
from tianji_wuji_runtime.runtime.recorder import Recorder
from tianji_wuji_runtime.runtime.robot_interface import (
    RobotConnectionConfig,
    RobotError,
    make_robot,
)
from tianji_wuji_runtime.runtime.ros2_jointstate_publisher import Ros2JointStatePublisher
from tianji_wuji_runtime.runtime.safety import SafetyConfig, SafetyError, SafetyLayer


def _parse_optional_joint_list(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size != schema.LEFT_HAND_DOF:
        raise ValueError(
            f"hand home pose must provide {schema.LEFT_HAND_DOF} comma-separated values, "
            f"got {values.size}"
        )
    return tuple(float(v) for v in values.tolist())


def _parse_optional_vector(raw: str | None, *, dim: int, name: str) -> np.ndarray | None:
    if raw is None:
        return None
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size != dim:
        raise ValueError(f"{name} must provide {dim} comma-separated values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    return values.astype(np.float32, copy=True)


def _load_left_freeze_target(path: Path) -> LeftSideFreezeTarget:
    if not path.exists():
        raise ControlOverrideError(f"left freeze target file does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ControlOverrideError(f"failed to read left freeze target file {path}: {exc}") from exc
    try:
        return LeftSideFreezeTarget(
            left_arm_q=np.asarray(raw["left_arm"], dtype=np.float32),
            left_hand_q=np.asarray(raw["left_hand"], dtype=np.float32),
            source=f"file:{path}",
        )
    except KeyError as exc:
        raise ControlOverrideError(
            f"left freeze target file {path} must contain left_arm and left_hand"
        ) from exc


def _resolve_freeze_segment(
    *,
    cli_value: np.ndarray | None,
    file_value: np.ndarray | None,
    current_value: np.ndarray | None,
    name: str,
) -> np.ndarray:
    if cli_value is not None:
        return cli_value.copy()
    if file_value is not None:
        return file_value.copy()
    if current_value is not None:
        return np.asarray(current_value, dtype=np.float32).copy()
    raise ControlOverrideError(f"no freeze target available for {name}")


def _freeze_segment_source(*, cli_value: str | None, use_current: bool) -> str:
    if cli_value is not None:
        return "cli"
    if use_current:
        return "current_state"
    return "default_json"


def _freeze_target_source(
    args: argparse.Namespace, file_target: LeftSideFreezeTarget | None
) -> str:
    if args.freeze_left_arm is not None or args.freeze_left_hand is not None:
        base = (
            "current_state"
            if args.freeze_left_use_current
            else (file_target.source if file_target else "none")
        )
        return f"cli_override+{base}"
    if args.freeze_left_use_current:
        return "current_state"
    if file_target is not None:
        return file_target.source
    return "unknown"


def _tcp_endpoint_open(host: str, port: int, *, timeout_sec: float = 0.5) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return True, "tcp endpoint is accepting connections"
    except OSError as exc:
        return False, str(exc)


def _wait_for_policy_client(args: argparse.Namespace) -> GrootPolicyClient:
    wait_sec = max(float(args.policy_wait_sec), 0.0)
    deadline = time.perf_counter() + wait_sec
    last_status = "not checked"
    next_log_time = 0.0

    print(
        "[runtime] waiting for policy server "
        f"{args.policy_host}:{args.policy_port} (timeout {wait_sec:.1f}s)"
    )
    while True:
        endpoint_open, status = _tcp_endpoint_open(args.policy_host, args.policy_port)
        last_status = status
        if endpoint_open:
            policy = GrootPolicyClient(
                host=args.policy_host,
                port=args.policy_port,
                timeout_ms=args.policy_timeout_ms,
            )
            if policy.ping():
                print(f"[runtime] policy server ready: {args.policy_host}:{args.policy_port}")
                return policy
            last_status = "tcp endpoint is open, but policy ping failed"

        now = time.perf_counter()
        if now >= deadline:
            raise PolicyServerError(
                "cannot ping policy server "
                f"{args.policy_host}:{args.policy_port} after {wait_sec:.1f}s; "
                f"last status: {last_status}. Start the policy server in another terminal, "
                "for example: `uv run python gr00t/eval/run_gr00t_server.py "
                "--model-path <CHECKPOINT_PATH> --embodiment-tag <TAG> --host 0.0.0.0 --port 5555`"
            )
        if now >= next_log_time:
            remaining = max(deadline - now, 0.0)
            print(
                "[runtime] policy server not ready yet: "
                f"{last_status}; retrying for {remaining:.1f}s"
            )
            next_log_time = now + 5.0
        time.sleep(min(1.0, max(deadline - now, 0.0)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_INFER_CONFIG_PATH),
        help=(
            "YAML/JSON config file. Defaults to "
            f"{DEFAULT_INFER_CONFIG_PATH}. Keys should use argument dest names such as "
            "robot_ip, camera, camera_fps. CLI flags still work and override scalar values."
        ),
    )
    parser.add_argument(
        "--robot-limits",
        default=str(RUNTIME_ROOT / "configs" / "robot_limits.yaml"),
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--policy-timeout-ms", type=int, default=15000)
    parser.add_argument(
        "--policy-wait-sec",
        type=float,
        default=60.0,
        help="Seconds to wait for the policy server before connecting robot/cameras.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument(
        "--arm-interpolation-hz",
        type=float,
        default=None,
        help=(
            "Optional high-rate arm command interpolation frequency. "
            "For example, 200 expands each 20 Hz policy step into smooth arm targets."
        ),
    )
    parser.add_argument(
        "--arm-interpolation-mode",
        choices=["cubic", "linear"],
        default="cubic",
        help="Interpolation curve for --arm-interpolation-hz.",
    )
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-dir", default=str(RUNTIME_ROOT / "infer_logs"))
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--image-source", default=None)
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=20.0,
        help="Python camera-worker processing frequency; does not change device capture FPS.",
    )
    parser.add_argument(
        "--camera-capture-fps",
        type=float,
        default=60.0,
        help="Native V4L2 device capture frequency configured before the first frame read.",
    )
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--head-stereo-crop", choices=["left", "right"], default=None)
    parser.add_argument("--max-camera-age-ms", type=float, default=150.0)
    parser.add_argument("--camera-warmup-sec", type=float, default=3.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--left-arm-ip", default=None)
    parser.add_argument("--left-arm-port", type=int, default=None)
    parser.add_argument("--right-arm-ip", default=None)
    parser.add_argument("--right-arm-port", type=int, default=None)
    parser.add_argument("--left-hand-ip", default=None)
    parser.add_argument("--left-hand-port", type=int, default=None)
    parser.add_argument("--right-hand-ip", default=None)
    parser.add_argument("--right-hand-port", type=int, default=None)
    parser.add_argument("--left-hand-serial", default=None)
    parser.add_argument("--right-hand-serial", default=None)
    parser.add_argument("--left-hand-home", default=None)
    parser.add_argument("--right-hand-home", default=None)
    parser.add_argument("--hand-lowpass-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--tianji-sdk-root", default=None)
    parser.add_argument("--tianji-config-path", default=None)
    parser.add_argument("--max-arm-joint-step", type=float, default=10.0)
    parser.add_argument("--max-hand-joint-step", type=float, default=4.5)
    parser.add_argument("--max-arm-velocity", type=float, default=None)
    parser.add_argument("--max-hand-velocity", type=float, default=None)
    parser.add_argument("--use-filter", action="store_true")
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--robot-backend", default="fake")
    parser.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    parser.add_argument("--policy-unit", default="deg")
    parser.add_argument("--control-unit", default="deg")
    parser.add_argument(
        "--ros2-publish-jointstate",
        action="store_true",
        help="Publish per-step target and position JointState topics for PlotJuggler.",
    )
    parser.add_argument(
        "--ros2-jointstate-prefix",
        default="/gr00t_runtime",
        help="ROS 2 topic prefix used by --ros2-publish-jointstate.",
    )
    parser.add_argument(
        "--freeze-left-side",
        action="store_true",
        help=(
            "Right-side-only control mode: keep left arm and left hand fixed while "
            "still feeding/predicting the full 54-DoF policy contract."
        ),
    )
    parser.add_argument(
        "--freeze-left-arm",
        default=None,
        help=(
            "Optional 7 comma-separated left-arm target for --freeze-left-side. "
            "If omitted, the target comes from --freeze-left-target-path unless "
            "--freeze-left-use-current is set."
        ),
    )
    parser.add_argument(
        "--freeze-left-hand",
        default=None,
        help=(
            "Optional 20 comma-separated left-hand target for --freeze-left-side. "
            "If omitted, the target comes from --freeze-left-target-path unless "
            "--freeze-left-use-current is set."
        ),
    )
    parser.add_argument(
        "--freeze-left-target-path",
        default=str(DEFAULT_LEFT_FREEZE_TARGET_PATH),
        help="Default left-side freeze target JSON used by --freeze-left-side.",
    )
    parser.add_argument(
        "--freeze-left-use-current",
        action="store_true",
        help="Use the current left arm + left hand state as the freeze target instead of JSON defaults.",
    )
    parser.add_argument("--max-chunks", type=int, default=None)
    return parser


def _load_cli_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"config file does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"failed to read config file {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a top-level mapping")
    return raw


def _normalize_config_key(key: str) -> str:
    return key.strip().replace("-", "_")


def _value_to_cli_tokens(
    *,
    key: str,
    value: object,
    action: argparse.Action,
) -> list[str]:
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
    if isinstance(action, argparse._StoreFalseAction):
        if not isinstance(value, bool):
            raise ValueError(f"config key {key!r} must be true/false")
        return [] if value else [flag]
    if isinstance(value, (list, dict)):
        raise ValueError(f"config key {key!r} must be a scalar for {flag}")
    return [flag, str(value)]


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
    if args.arm_interpolation_hz is not None and args.arm_interpolation_hz <= 0:
        raise ValueError("--arm-interpolation-hz must be positive when provided")
    if args.camera_fps <= 0:
        raise ValueError("--camera-fps must be positive")
    if args.camera_capture_fps <= 0:
        raise ValueError("--camera-capture-fps must be positive")
    if args.no_keyboard and not args.auto_start:
        raise ValueError("--no-keyboard requires --auto-start so execution is explicit")
    if not args.freeze_left_side and (
        args.freeze_left_arm is not None
        or args.freeze_left_hand is not None
        or args.freeze_left_use_current
    ):
        raise ValueError(
            "--freeze-left-arm/--freeze-left-hand/--freeze-left-use-current "
            "require --freeze-left-side"
        )
    if args.freeze_left_side and args.action_mode != "absolute":
        raise ValueError("--freeze-left-side expects --action-mode absolute")
    if args.safe_mode and args.execution_horizon > 2:
        print("[runtime] warning: initial safe execution should use --execution-horizon 1 or 2")
    if not args.dry_run and args.robot_backend == "fake":
        print("[runtime] warning: robot-backend=fake; commands update only the in-memory robot")

    adapter = ActionAdapter(
        policy_unit=args.policy_unit,
        control_unit=args.control_unit,
        action_mode=args.action_mode,
    )
    policy = _wait_for_policy_client(args)
    modality_configs = policy.get_modality_config()
    obs_builder = ObservationBuilder(modality_configs)

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
            left_hand_ip=args.left_hand_ip,
            left_hand_port=args.left_hand_port,
            right_hand_ip=args.right_hand_ip,
            right_hand_port=args.right_hand_port,
            left_hand_serial=args.left_hand_serial,
            right_hand_serial=args.right_hand_serial,
            left_hand_home=_parse_optional_joint_list(args.left_hand_home),
            right_hand_home=_parse_optional_joint_list(args.right_hand_home),
            hand_lowpass_cutoff_hz=args.hand_lowpass_cutoff_hz,
            tianji_sdk_root=args.tianji_sdk_root,
            tianji_config_path=args.tianji_config_path,
        )
    )
    safety_config = SafetyConfig.from_yaml(args.robot_limits)
    safety_config.arm_max_step = args.max_arm_joint_step
    safety_config.hand_max_step = args.max_hand_joint_step
    if args.max_arm_velocity is not None:
        safety_config.arm_max_velocity = np.full(
            schema.LEFT_ARM_DOF, args.max_arm_velocity, dtype=np.float32
        )
    if args.max_hand_velocity is not None:
        safety_config.hand_max_velocity = np.full(
            schema.LEFT_HAND_DOF, args.max_hand_velocity, dtype=np.float32
        )
    safety_config.enable_filter = args.use_filter
    safety = SafetyLayer(safety_config, adapter)
    recorder = Recorder(
        args.record_dir,
        config={
            "entrypoint": "infer_tianji_wuji.py",
            "inference_mode": "sync",
            "policy_host": args.policy_host,
            "policy_port": args.policy_port,
            "task": args.task,
            "execution_horizon": args.execution_horizon,
            "duration": args.duration,
            "control_frequency_hz": 1.0 / args.duration,
            "arm_interpolation_hz": args.arm_interpolation_hz,
            "arm_interpolation_mode": args.arm_interpolation_mode,
            "safe_mode": args.safe_mode,
            "dry_run": args.dry_run,
            "camera": args.camera,
            "camera_capture_fps": args.camera_capture_fps,
            "camera_processing_fps": args.camera_fps,
            "image_source": args.image_source,
            "robot_backend": args.robot_backend,
            "control_overrides": {
                "freeze_left_side": args.freeze_left_side,
                "freeze_left_arm_source": (
                    _freeze_segment_source(
                        cli_value=args.freeze_left_arm,
                        use_current=args.freeze_left_use_current,
                    )
                ),
                "freeze_left_hand_source": (
                    _freeze_segment_source(
                        cli_value=args.freeze_left_hand,
                        use_current=args.freeze_left_use_current,
                    )
                ),
                "freeze_left_target_path": args.freeze_left_target_path,
            },
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
    runtime_events_path = recorder.run_dir / "runtime_events.jsonl"
    runtime_event_logger = RuntimeEventLogger(runtime_events_path)
    log_event = runtime_event_logger.log
    ros_publisher = (
        Ros2JointStatePublisher(topic_prefix=args.ros2_jointstate_prefix)
        if args.ros2_publish_jointstate
        else None
    )

    executor = ActionExecutor(
        robot,
        adapter=adapter,
        recorder=recorder,
        event_logger=log_event,
        ros_publisher=ros_publisher,
        arm_interpolation_hz=args.arm_interpolation_hz,
        arm_interpolation_mode=args.arm_interpolation_mode,
    )
    keepalive = ActionKeepalive(robot, event_logger=log_event)
    state_machine = RuntimeStateMachine(auto_start=args.auto_start, safe_mode=args.safe_mode)

    print(f"[runtime] logs: {recorder.run_dir}")
    print(f"[runtime] events: {runtime_events_path}")
    print(f"[runtime] inference mode: sync, action step: {args.duration:.4f}s")
    print("[runtime] controls: R run, P pause, Space hold, H home, N next safe chunk, Q quit")
    log_event(
        "runtime_start",
        run_dir=str(recorder.run_dir),
        execution_horizon=args.execution_horizon,
        duration_sec=args.duration,
        arm_interpolation_hz=args.arm_interpolation_hz,
        arm_interpolation_mode=args.arm_interpolation_mode,
        camera_capture_fps=args.camera_capture_fps,
        camera_processing_fps=args.camera_fps,
        max_camera_age_ms=args.max_camera_age_ms,
        freeze_left_side=args.freeze_left_side,
        dry_run=args.dry_run,
        ros2_publish_jointstate=args.ros2_publish_jointstate,
        ros2_jointstate_prefix=args.ros2_jointstate_prefix,
    )

    chunk_count = 0
    left_freeze_target: LeftSideFreezeTarget | None = None
    try:
        log_event("robot_connect_start", backend=args.robot_backend, robot_ip=args.robot_ip)
        robot_connect_t0 = time.perf_counter()
        robot.connect()
        log_event(
            "robot_connect_end",
            latency_ms=(time.perf_counter() - robot_connect_t0) * 1000.0,
        )
        if args.freeze_left_side:
            if args.freeze_left_use_current:
                freeze_state_t0 = time.perf_counter()
                current_state = robot.get_state()
                freeze_state_t1 = time.perf_counter()
                log_event(
                    "freeze_left_state_read",
                    latency_ms=(freeze_state_t1 - freeze_state_t0) * 1000.0,
                )
            else:
                current_state = None
            file_target = (
                None
                if args.freeze_left_use_current
                else _load_left_freeze_target(Path(args.freeze_left_target_path))
            )
            freeze_arm = _parse_optional_vector(
                args.freeze_left_arm,
                dim=schema.LEFT_ARM_DOF,
                name="--freeze-left-arm",
            )
            freeze_hand = _parse_optional_vector(
                args.freeze_left_hand,
                dim=schema.LEFT_HAND_DOF,
                name="--freeze-left-hand",
            )
            left_freeze_target = LeftSideFreezeTarget(
                left_arm_q=(
                    _resolve_freeze_segment(
                        cli_value=freeze_arm,
                        file_value=None if file_target is None else file_target.left_arm_q,
                        current_value=None if current_state is None else current_state.left_arm_q,
                        name="left_arm",
                    )
                ),
                left_hand_q=(
                    _resolve_freeze_segment(
                        cli_value=freeze_hand,
                        file_value=None if file_target is None else file_target.left_hand_q,
                        current_value=None if current_state is None else current_state.left_hand_q,
                        name="left_hand",
                    )
                ),
                source=_freeze_target_source(args, file_target),
            )
            _write_json(recorder.run_dir / "left_freeze_target.json", left_freeze_target.as_dict())
            print(
                "[runtime] freeze-left-side enabled: "
                f"left_arm={left_freeze_target.left_arm_q.tolist()}, "
                f"left_hand={left_freeze_target.left_hand_q.tolist()}"
            )
            log_event(
                "freeze_left_target",
                source=left_freeze_target.source,
                left_arm=left_freeze_target.left_arm_q.tolist(),
                left_hand=left_freeze_target.left_hand_q.tolist(),
            )
        log_event(
            "camera_connect_start",
            cameras=list(args.camera),
            width=args.camera_width,
            height=args.camera_height,
            capture_fps=args.camera_capture_fps,
            processing_fps=args.camera_fps,
            warmup_sec=args.camera_warmup_sec,
        )
        camera_connect_t0 = time.perf_counter()
        cameras.connect_all()
        log_event(
            "camera_connect_end",
            latency_ms=(time.perf_counter() - camera_connect_t0) * 1000.0,
        )
        log_event("camera_stream_start")
        cameras.start_streaming()
        camera_ready_t0 = time.perf_counter()
        warmup_frames = cameras.wait_until_ready(timeout_sec=args.camera_warmup_sec)
        camera_ready_ref = time.perf_counter()
        log_event(
            "camera_ready",
            latency_ms=(camera_ready_ref - camera_ready_t0) * 1000.0,
            frames=_frame_debug_summary(warmup_frames, camera_ready_ref),
        )
        with KeyboardController(enabled=not args.no_keyboard) as keyboard:

            def stop_requested() -> bool:
                state_machine.update(keyboard.poll())
                if state_machine.home_requested:
                    log_event("keyboard_home_requested")
                    keepalive.stop()
                    robot.go_home()
                    state_machine.home_requested = False
                if state_machine.quit_requested:
                    log_event("keyboard_quit_requested")
                    keepalive.stop()
                    robot.hold_position()
                    return True
                return state_machine.state != RuntimeState.RUNNING

            while True:
                """
                R      让程序跑起来，可以继续推理执行
                P      暂停，但程序不退出
                Space  停止当前运行状态，更像急停式 hold 的软件入口
                H      命令机器人回 home
                Q      退出整个 runtime，并断开资源
                N      safe-mode 下放行下一段，执行完又暂停
                """
                state_machine.update(keyboard.poll())
                if state_machine.home_requested:
                    log_event("keyboard_home_requested", chunk_index=chunk_count)
                    keepalive.stop()
                    robot.go_home()
                    state_machine.home_requested = False
                if state_machine.quit_requested:
                    log_event("keyboard_quit_requested", chunk_index=chunk_count)
                    keepalive.stop()
                    robot.hold_position()
                    break
                if state_machine.state != RuntimeState.RUNNING:
                    keepalive.stop()
                    time.sleep(0.01)
                    continue

                chunk_index = chunk_count
                raw_chunk = None
                observation = None
                safe_actions = []
                safety_events = []
                chunk_saved = False
                try:
                    log_event(
                        "chunk_loop_start",
                        chunk_index=chunk_index,
                        keepalive_running=keepalive.is_running(),
                    )
                    keepalive.raise_if_failed()
                    state_t0 = time.perf_counter()
                    robot_state = robot.get_state()
                    state_t1 = time.perf_counter()
                    reference_time = 0.5 * (state_t0 + state_t1)
                    log_event(
                        "robot_state_read",
                        chunk_index=chunk_index,
                        state_read_start_monotonic=state_t0,
                        state_read_end_monotonic=state_t1,
                        reference_monotonic=reference_time,
                        latency_ms=(state_t1 - state_t0) * 1000.0,
                    )
                    snapshot_t0 = time.perf_counter()
                    frames = cameras.snapshot_latest(
                        reference_time=reference_time,
                        max_age_ms=args.max_camera_age_ms,
                    )
                    snapshot_t1 = time.perf_counter()
                    log_event(
                        "camera_snapshot",
                        chunk_index=chunk_index,
                        reference_monotonic=reference_time,
                        snapshot_latency_ms=(snapshot_t1 - snapshot_t0) * 1000.0,
                        max_camera_age_ms=args.max_camera_age_ms,
                        frames=_frame_debug_summary(frames, reference_time),
                    )
                    images = {key: frame.image for key, frame in frames.items()}
                    observation_t0 = time.perf_counter()
                    robot_state = validate_policy_inputs(
                        robot_state,
                        images,
                        required_camera_keys=obs_builder.required_camera_keys,
                    )
                    observation = obs_builder.build(robot_state, images, args.task)
                    observation_t1 = time.perf_counter()
                    log_event(
                        "observation_ready",
                        chunk_index=chunk_index,
                        latency_ms=(observation_t1 - observation_t0) * 1000.0,
                        age_from_state_reference_ms=(observation_t1 - reference_time) * 1000.0,
                        image_keys=sorted(images),
                    )
                    policy_t0 = time.perf_counter()
                    log_event(
                        "policy_predict_start",
                        chunk_index=chunk_index,
                        age_from_state_reference_ms=(policy_t0 - reference_time) * 1000.0,
                    )
                    raw_chunk = policy.predict_action_chunk(
                        observation,
                        min_horizon=args.execution_horizon,
                    )
                    policy_t1 = time.perf_counter()
                    log_event(
                        "policy_predict_end",
                        chunk_index=chunk_index,
                        latency_ms=(policy_t1 - policy_t0) * 1000.0,
                        client_latency_ms=policy.last_latency_ms,
                        age_from_state_reference_ms=(policy_t1 - reference_time) * 1000.0,
                        raw_action_shape=list(np.asarray(raw_chunk).shape),
                    )
                    actions = adapter.split_chunk(raw_chunk)
                    if actions:
                        log_event(
                            "policy_actions_split",
                            chunk_index=chunk_index,
                            policy_action_count=len(actions),
                            executed_horizon=args.execution_horizon,
                            first_policy_action=_action_debug_summary(actions[0]),
                            last_policy_action=_action_debug_summary(actions[-1]),
                        )
                    override_events: list[dict[str, object]] = []
                    if left_freeze_target is not None:
                        actions, override_events = apply_left_side_freeze(
                            actions,
                            left_freeze_target,
                        )
                        log_event(
                            "left_freeze_applied",
                            chunk_index=chunk_index,
                            event_count=len(override_events),
                        )
                    safety_t0 = time.perf_counter()
                    safe_actions, safety_events = safety.process_chunk(
                        robot_state,
                        actions,
                        args.duration,
                    )
                    safety_t1 = time.perf_counter()
                    safety_events = override_events + safety_events
                    executable_count = min(args.execution_horizon, len(safe_actions))
                    log_event(
                        "safety_processed",
                        chunk_index=chunk_index,
                        latency_ms=(safety_t1 - safety_t0) * 1000.0,
                        policy_action_count=len(actions),
                        safe_action_count=len(safe_actions),
                        executed_horizon=args.execution_horizon,
                        executable_count=executable_count,
                        safety_event_count=len(safety_events),
                        safety_events_by_type=_event_counts(safety_events),
                        first_safe_action=(
                            None if not safe_actions else _action_debug_summary(safe_actions[0])
                        ),
                        last_executable_action=(
                            None
                            if executable_count <= 0
                            else _action_debug_summary(safe_actions[executable_count - 1])
                        ),
                    )
                    log_event(
                        "chunk_execute_prepare",
                        chunk_index=chunk_index,
                        keepalive_running_before_stop=keepalive.is_running(),
                        executable_count=executable_count,
                    )
                    keepalive.stop()
                    execute_t0 = time.perf_counter()
                    log_event(
                        "chunk_execute_start",
                        chunk_index=chunk_index,
                        action_count=executable_count,
                        age_from_state_reference_ms=(execute_t0 - reference_time) * 1000.0,
                    )
                    last_executed_action = executor.execute_chunk(
                        safe_actions[: args.execution_horizon],
                        args.duration,
                        dry_run=args.dry_run,
                        chunk_index=chunk_index,
                        raw_chunk=raw_chunk,
                        safety_events=safety_events,
                        stop_callback=stop_requested,
                    )
                    execute_t1 = time.perf_counter()
                    log_event(
                        "chunk_execute_end",
                        chunk_index=chunk_index,
                        latency_ms=(execute_t1 - execute_t0) * 1000.0,
                        last_executed_action=(
                            None
                            if last_executed_action is None
                            else _action_debug_summary(last_executed_action)
                        ),
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
                    else:
                        if last_executed_action is None:
                            keepalive_reason = "no_last_executed_action"
                        elif state_machine.state != RuntimeState.RUNNING:
                            keepalive_reason = f"state_{state_machine.state.value}"
                        elif reached_max_chunks:
                            keepalive_reason = "reached_max_chunks"
                        else:
                            keepalive_reason = "unknown"
                        log_event(
                            "keepalive_not_started",
                            chunk_index=chunk_index,
                            reason=keepalive_reason,
                            reached_max_chunks=reached_max_chunks,
                        )
                    chunk_dir = recorder.save_chunk(
                        observation=observation,
                        raw_chunk=raw_chunk,
                        safe_actions=safe_actions,
                        safety_events=safety_events,
                        inference_latency_ms=policy.last_latency_ms,
                    )
                    chunk_saved = True
                    log_event(
                        "chunk_saved",
                        chunk_index=chunk_index,
                        chunk_dir=str(chunk_dir),
                    )
                    if not reached_max_chunks:
                        log_event(
                            "chunk_replan_delay",
                            chunk_index=chunk_index,
                            sleep_sec=0.05,
                        )
                        time.sleep(0.05)
                    if reached_max_chunks:
                        break
                except (
                    ActionAdapterError,
                    CameraError,
                    ControlOverrideError,
                    ObservationError,
                    PolicyServerError,
                    RobotError,
                    SafetyError,
                ) as exc:
                    if (
                        not chunk_saved
                        and observation is not None
                        and raw_chunk is not None
                    ):
                        try:
                            chunk_dir = recorder.save_chunk(
                                observation=observation,
                                raw_chunk=raw_chunk,
                                safe_actions=safe_actions,
                                safety_events=safety_events,
                                inference_latency_ms=policy.last_latency_ms,
                            )
                            chunk_saved = True
                            log_event(
                                "chunk_saved_after_error",
                                chunk_index=chunk_index,
                                chunk_dir=str(chunk_dir),
                            )
                        except Exception as save_exc:  # noqa: BLE001
                            log_event(
                                "chunk_save_failed_after_error",
                                chunk_index=chunk_index,
                                error_type=type(save_exc).__name__,
                                error=str(save_exc),
                            )
                    print(f"[runtime] ERROR: {exc}")
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
        log_event("runtime_cleanup_start", keepalive_running=keepalive.is_running())
        keepalive.stop()
        robot.hold_position()
        cameras.stop_streaming()
        cameras.disconnect_all()
        if ros_publisher is not None:
            ros_publisher.close()
        robot.disconnect()
        log_event("runtime_cleanup_end")
        runtime_event_logger.close()
    return 0


def _frame_debug_summary(frames: dict[str, object], reference_time: float) -> dict[str, object]:
    per_camera: dict[str, dict[str, object]] = {}
    frame_times: list[float] = []
    for key, frame in frames.items():
        monotonic_time = float(getattr(frame, "monotonic_time"))
        wall_time = float(getattr(frame, "wall_time"))
        frame_times.append(monotonic_time)
        per_camera[key] = {
            "frame_id": int(getattr(frame, "frame_id")),
            "source": str(getattr(frame, "source")),
            "width": int(getattr(frame, "width")),
            "height": int(getattr(frame, "height")),
            "wall_time": wall_time,
            "monotonic_time": monotonic_time,
            "age_ms": (reference_time - monotonic_time) * 1000.0,
            "delta_from_reference_ms": (monotonic_time - reference_time) * 1000.0,
        }
    if frame_times:
        span_ms: float | None = (max(frame_times) - min(frame_times)) * 1000.0
    else:
        span_ms = None
    return {
        "per_camera": per_camera,
        "inter_camera_span_ms": span_ms,
    }


def _action_debug_summary(action: DualArmHandAction) -> dict[str, object]:
    right_arm = action.right_arm_q
    right_hand = action.right_hand_q
    return {
        "right_arm": right_arm.tolist(),
        "right_hand": right_hand.tolist(),
        "right_arm_l2": float((right_arm * right_arm).sum() ** 0.5),
        "right_hand_l2": float((right_hand * right_hand).sum() ** 0.5),
    }


def _event_counts(events: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type", "unknown"))
        segment = event.get("segment")
        key = event_type if segment is None else f"{event_type}:{segment}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
