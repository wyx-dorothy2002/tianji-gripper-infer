#!/usr/bin/env python3
"""回放天机 16D 轨迹，并同步采集三路相机和机器人反馈。

主流程按交接时最容易理解的顺序设计：

1. 加载约 200 Hz 的原始示教 trace，并重采样到默认 30 Hz；
2. 连接三路相机、双臂和两个夹爪；
3. 必要时慢速移动到轨迹第一帧（该阶段不记录）；
4. 正式回放，每帧依次完成 target 下发、actual state 读取、相机快照和落盘；
5. 正常结束或异常退出时保持机器人，并写出可检查的 ``meta.json``。

接口字段遵循 TIANJI_DISPATCH_INTERFACE.md：右/B 臂 7 rad、左/A 臂
7 rad、右夹爪 motor-rad、左夹爪 motor-rad。

示例：
    # 只检查三条轨迹，不连接硬件。
    python replay_recorded_actions.py --trajectory all

    # 30 Hz target 序列按 60 Hz 执行，并同步记录。
    python replay_recorded_actions.py --trajectory episode_000011 --execute
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from PIL import Image
import yaml


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
PREINFERE_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(PREINFERE_ROOT))

from tianji_gripper_runtime.runtime.action_adapter import (  # noqa: E402
    ActionAdapter,
    RightArmGripperAction,
)
from tianji_gripper_runtime.runtime.robot_interface import (  # noqa: E402
    RobotConnectionConfig,
    make_robot,
)
from tianji_gripper_runtime.runtime.robot_state import (  # noqa: E402
    RightArmGripperState,
)
from tianji_gripper_runtime.runtime.safety import (  # noqa: E402
    SafetyConfig,
)
from tianji_gripper_runtime.runtime.slow_dispatch import (  # noqa: E402
    SlowInterpolatedDispatcher,
)
from tianji_wuji_runtime.runtime.camera_manager import (  # noqa: E402
    CameraManager,
    LatestFrame,
)


DEFAULT_DATA_ROOT = Path(
    "/home/user/workspace/"
    "TJ-gripper-codex-gripper-fixes-20260617-reclone/data/raw"
)
DEFAULT_CONFIG = RUNTIME_ROOT / "configs" / "infer.yaml"
DEFAULT_LIMITS = RUNTIME_ROOT / "configs" / "robot_limits.yaml"
DEFAULT_RECORD_ROOT = Path("data/replay_recordings")
TRAJECTORIES = ("back_to_home", "home_to_back", "episode_000011")
ACTION_COLUMNS = [
    *(f"right_joint_{index}.pos" for index in range(1, 8)),
    *(f"left_joint_{index}.pos" for index in range(1, 8)),
    "right_gripper.pos",
    "left_gripper.pos",
]
TRACE_COLUMNS = [
    "timestamp_unix",
    *(f"cmd_B_deg_{index}" for index in range(1, 8)),
    *(f"cmd_A_deg_{index}" for index in range(1, 8)),
    "gripper2_target_rad",
    "gripper_target_rad",
]

_VECTOR_CONFIG_DIMS = {
    "left_gripper_home": 1,
    "right_gripper_home": 1,
    "left_arm_freeze_q": 7,
    "left_gripper_freeze_q": 1,
    "tianji_tool_arm_a_kinematics": 6,
    "tianji_tool_arm_a_dynamics": 10,
    "tianji_tool_arm_b_kinematics": 6,
    "tianji_tool_arm_b_dynamics": 10,
    "tianji_state3_joint_k_a": 7,
    "tianji_state3_joint_d_a": 7,
    "tianji_state3_joint_k_b": 7,
    "tianji_state3_joint_d_b": 7,
    "tianji_home_joints_a": 7,
    "tianji_home_joints_b": 7,
}


@dataclass(frozen=True)
class ReplayOptions:
    """回放配置。

    ``trajectory_hz`` 只决定从源 trace 选择 target 的密度，
    ``dispatch_hz`` 决定所选 target 的实际执行频率，
    ``frame_stride`` 决定每隔多少个 target 执行一个。
    """

    trajectory: str = "all"
    data_root: Path = DEFAULT_DATA_ROOT
    source: str = "runtime-trace"
    config_path: Path = DEFAULT_CONFIG
    robot_limits_path: Path = DEFAULT_LIMITS
    record_root: Path = DEFAULT_RECORD_ROOT
    trajectory_hz: float = 30.0
    dispatch_hz: float = 60.0
    frame_stride: int = 1
    max_replay_arm_velocity: float | None = None
    start_row: int = 0
    end_row: int | None = None
    execute: bool = False
    max_start_arm_error_deg: float = 5.0
    max_start_gripper_error_rad: float = 0.3
    approach_start: bool = True
    camera_max_age_ms: float = 200.0
    camera_ready_timeout_sec: float = 10.0

    def validate(self) -> None:
        """校验调用参数；失败时不连接任何硬件。"""

        allowed_trajectories = {*TRAJECTORIES, "all"}
        if self.trajectory not in allowed_trajectories:
            raise ValueError(
                f"trajectory must be one of {sorted(allowed_trajectories)}, "
                f"got {self.trajectory!r}"
            )
        if self.source not in {"runtime-trace", "action-csv"}:
            raise ValueError(
                "source must be 'runtime-trace' or 'action-csv'"
            )
        if self.trajectory_hz <= 0 or self.dispatch_hz <= 0:
            raise ValueError(
                "trajectory_hz and dispatch_hz must be positive"
            )
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if (
            self.max_replay_arm_velocity is not None
            and self.max_replay_arm_velocity <= 0
        ):
            raise ValueError(
                "max_replay_arm_velocity must be positive"
            )
        if self.start_row < 0 or (
            self.end_row is not None and self.end_row <= self.start_row
        ):
            raise ValueError(
                "row range must satisfy 0 <= start_row < end_row"
            )
        if self.max_start_arm_error_deg < 0 or self.max_start_gripper_error_rad < 0:
            raise ValueError("start-pose tolerances must be non-negative")
        if self.camera_max_age_ms <= 0 or self.camera_ready_timeout_sec <= 0:
            raise ValueError("camera timing limits must be positive")

    @property
    def acceleration(self) -> float:
        """相对 trajectory target 时间轴的近似加速倍率。"""

        return self.dispatch_hz * self.frame_stride / self.trajectory_hz


@dataclass(frozen=True)
class TrajectoryPlan:
    """单条轨迹经过取帧、跳帧后的只读执行摘要。"""

    name: str
    frame_count: int
    duration_sec: float
    acceleration: float
    source: str


@dataclass(frozen=True)
class PreparedTrajectory:
    """已经完成取帧、跳帧和时间轴生成的轨迹。"""

    plan: TrajectoryPlan
    actions: np.ndarray
    frame_times: np.ndarray
    source_meta: dict[str, object]


def main(argv: list[str] | None = None) -> int:
    """命令行主流程；每一步都调用一个职责单一的小函数。"""

    options = _options_from_args(_build_parser().parse_args(argv))
    options.validate()

    trajectories = prepare_trajectories(options)
    if not options.execute:
        print("[replay] validation only; pass --execute to connect and send commands")
        return 0
    if len(trajectories) != 1:
        raise ValueError("--execute requires exactly one trajectory")

    safety = build_replay_safety(options)
    validate_replay_trajectory(
        trajectories[0].actions,
        frequency_hz=options.dispatch_hz,
        limits=safety,
    )
    execute_prepared_trajectory(options, trajectories[0], safety)
    return 0


# =============================================================================
# 1. YAML 配置与硬件对象构造
# =============================================================================


def _parse_vector(value: object, *, name: str, dim: int) -> tuple[float, ...] | None:
    """把 YAML 中的逗号分隔字符串或数组解析成固定长度向量。"""
    if value is None:
        return None
    if isinstance(value, str):
        array = np.fromstring(value, sep=",", dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (dim,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {dim} finite values")
    return tuple(float(item) for item in array)


def build_robot_config(config_path: Path, *, execute: bool) -> RobotConnectionConfig:
    """读取 infer.yaml，生成双臂双夹爪连接配置。"""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must contain a mapping: {config_path}")

    valid_names = {field.name for field in fields(RobotConnectionConfig)}
    values = {name: value for name, value in raw.items() if name in valid_names}
    if "robot_backend" in raw:
        values["backend"] = raw["robot_backend"]
    for name, dim in _VECTOR_CONFIG_DIMS.items():
        if name in values:
            values[name] = _parse_vector(values[name], name=name, dim=dim)

    values["command_left_side"] = True
    if not execute:
        values.update(
            backend="fake",
            gripper_backend="fake",
            left_gripper_backend="fake",
            right_gripper_backend="fake",
        )
    return RobotConnectionConfig(**values)


def build_camera_manager(config_path: Path) -> CameraManager:
    """读取 infer.yaml，创建 head/左右腕三路后台相机线程。"""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    specs = raw.get("camera")
    if not isinstance(specs, list):
        raise ValueError(f"config camera field must be a list: {config_path}")
    camera_names = ["head", "left_wrist", "right_wrist"]
    return CameraManager.from_cli_specs(
        [str(spec) for spec in specs],
        required_keys=camera_names,
        width=int(raw.get("camera_width", 424)),
        height=int(raw.get("camera_height", 240)),
        capture_fps=float(raw.get("camera_capture_fps", 30)),
        fps=float(raw.get("camera_fps", 30)),
    )


def _load_csv(path: Path, expected_columns: list[str]) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [name for name in expected_columns if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        rows = [[float(row[name]) for name in expected_columns] for row in reader]
    # Keep absolute Unix timestamps in float64 until they have been made
    # relative; float32 cannot represent 5 ms differences at epoch scale.
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(expected_columns) or len(array) == 0:
        raise ValueError(f"CSV contains no usable rows: {path}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"CSV contains NaN or Inf: {path}")
    return array


# =============================================================================
# 2. 源轨迹读取与夹爪标定转换
# =============================================================================


def _gripper_motor_radians(values: np.ndarray, calibration: dict[str, object]) -> np.ndarray:
    close = float(calibration["close_motor_rad"])
    opened = float(calibration["open_motor_rad"])
    if np.any(values < -1e-4) or np.any(values > 1.0001):
        raise ValueError("normalized gripper values must remain in [0, 1]")
    return opened + np.clip(values, 0.0, 1.0) * (close - opened)


def _gripper_normalized(value: float, calibration: dict[str, object]) -> float:
    close = float(calibration["close_motor_rad"])
    opened = float(calibration["open_motor_rad"])
    span = close - opened
    if abs(span) < 1e-8:
        raise ValueError("gripper calibration close/open values must differ")
    return float(np.clip((float(value) - opened) / span, 0.0, 1.0))


def _relative_times(timestamps: np.ndarray, *, fallback_fps: float) -> np.ndarray:
    relative_times = timestamps.astype(np.float64) - float(timestamps[0])
    if np.any(np.diff(relative_times) < 0):
        raise ValueError("timestamps are not monotonic")
    if len(timestamps) > 1 and relative_times[-1] <= 0:
        return np.arange(len(timestamps), dtype=np.float64) / fallback_fps
    return relative_times


def load_trajectory(
    episode_dir: Path,
    *,
    source: str = "runtime-trace",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """加载源轨迹并统一转换成 dispatch 接口要求的 16D 单位和顺序。"""
    meta_path = episode_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if source == "runtime-trace":
        trace = _load_csv(
            episode_dir / "_runtime" / "marvin_drag_teach_teleop_trace.csv",
            TRACE_COLUMNS,
        )
        actions = np.empty((len(trace), 16), dtype=np.float32)
        actions[:, 0:7] = np.deg2rad(trace[:, 1:8])
        actions[:, 7:14] = np.deg2rad(trace[:, 8:15])
        actions[:, 14:16] = trace[:, 15:17]
        relative_times = _relative_times(trace[:, 0], fallback_fps=200.0)
        return actions, relative_times, meta
    if source != "action-csv":
        raise ValueError(f"unknown trajectory source: {source!r}")

    actions = _load_csv(episode_dir / "arm_data" / "action.csv", ACTION_COLUMNS)

    encoding = meta.get("gripper_value_encoding")
    if encoding == "normalized_closed_1_open_0":
        calibration = meta.get("gripper_calibration")
        if not isinstance(calibration, dict):
            raise ValueError(f"missing gripper_calibration in {meta_path}")
        actions[:, 14] = _gripper_motor_radians(
            actions[:, 14], calibration["right_gripper"]
        )
        actions[:, 15] = _gripper_motor_radians(
            actions[:, 15], calibration["left_gripper"]
        )
    elif encoding not in (None, "motor_rad"):
        raise ValueError(f"unsupported gripper_value_encoding: {encoding!r}")

    timestamp_path = episode_dir / "arm_data" / "timestamp.csv"
    timestamps = _load_csv(timestamp_path, ["timestamp_unix"]).reshape(-1)
    if len(timestamps) != len(actions):
        raise ValueError("action and timestamp CSV row counts differ")
    relative_times = _relative_times(
        timestamps, fallback_fps=float(meta.get("fps", 100.0))
    )
    return actions, relative_times, meta


# =============================================================================
# 3. TIANJI_DISPATCH_INTERFACE.md 的单帧下发接口
# =============================================================================


def validate_dispatch_action(action: np.ndarray) -> np.ndarray:
    """校验接口文档规定的 16D target，并返回独立的 float32 一维数组。"""

    command = np.asarray(action, dtype=np.float32).reshape(-1)
    if command.shape != (16,) or not np.all(np.isfinite(command)):
        raise ValueError("action must be 16 finite float values")
    return command.copy()


def action_to_control_action(action: np.ndarray) -> RightArmGripperAction:
    """把 16D 接口 target 转换成机器人控制单位和左右硬件映射。

    对应 ``TIANJI_DISPATCH_INTERFACE.md``：

    - ``0:7``：右/B 臂，rad 转 deg；
    - ``7:14``：左/A 臂，rad 转 deg；
    - ``14``：右/B 夹爪，motor-rad 保持不变；
    - ``15``：左/A 夹爪，motor-rad 保持不变。
    """

    command = np.asarray(action, dtype=np.float32)
    if command.shape != (16,):
        raise ValueError(
            "action_to_control_action expects validated action with shape (16,)"
        )
    return RightArmGripperAction(
        right_arm_q=np.rad2deg(command[0:7]).astype(np.float32),
        left_arm_q=np.rad2deg(command[7:14]).astype(np.float32),
        right_gripper_q=command[14:15],
        left_gripper_q=command[15:16],
        control_left_arm=True,
        control_left_gripper=True,
    )


def dispatch_action(action: np.ndarray, robot: Any) -> None:
    """将已经完成整轨迹安全检查的一帧 target 交给统一机器人接口。"""

    command = validate_dispatch_action(action)
    control_action = action_to_control_action(command)
    robot.send_action(control_action)


# =============================================================================
# 4. 轨迹时序、跳帧与整轨迹安全检查
# =============================================================================


def scheduled_replay_frames(
    actions: np.ndarray,
    relative_times: np.ndarray,
) -> Iterator[tuple[np.ndarray, float]]:
    """按计划时间逐帧产出 ``(action, dt)``；本函数不接触硬件。"""

    start = time.monotonic()
    for index, (action, recorded_time) in enumerate(zip(actions, relative_times)):
        deadline = start + float(recorded_time)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        dt = (
            0.0
            if index == 0
            else float(relative_times[index] - relative_times[index - 1])
        )
        yield action, dt


def resample_trajectory(
    actions: np.ndarray,
    relative_times: np.ndarray,
    *,
    frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按均匀回放时钟，从原 trace 选择时间上最近的帧。"""
    if frequency_hz <= 0:
        raise ValueError("replay frequency must be positive")
    if len(actions) == 1:
        return actions.copy(), np.zeros(1, dtype=np.float64)
    sample_times = np.arange(
        0.0,
        float(relative_times[-1]) + 0.5 / frequency_hz,
        1.0 / frequency_hz,
        dtype=np.float64,
    )
    right = np.searchsorted(relative_times, sample_times, side="left")
    right = np.clip(right, 1, len(relative_times) - 1)
    left = right - 1
    choose_left = (
        sample_times - relative_times[left]
        <= relative_times[right] - sample_times
    )
    indices = np.where(choose_left, left, right)
    return actions[indices].copy(), sample_times


def apply_frame_stride(
    actions: np.ndarray,
    *,
    frame_stride: int,
    execution_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """每隔若干帧取一个 target，并仍按 execution_hz 连续执行。

    例如先得到 30 Hz 轨迹序列，执行频率为 60 Hz，再设置 ``frame_stride=3``：

    - 实际执行源序列的第 0、3、6、9 ... 帧；
    - 两个被跳过的中间帧既不下发，也不参与数采；
    - 被选中的帧每 1/60 秒执行；
    - 60/30 先产生 2 倍速，stride=3 再产生 3 倍速，合计约 6 倍速。

    最后一帧无论是否正好落在步长上都会保留，确保最终目标不会丢失。
    """

    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    indices = np.arange(0, len(actions), frame_stride, dtype=np.int64)
    if indices[-1] != len(actions) - 1:
        indices = np.append(indices, len(actions) - 1)
    selected = actions[indices].copy()
    execution_times = np.arange(len(selected), dtype=np.float64) / execution_hz
    return selected, execution_times


def validate_replay_trajectory(
    actions: np.ndarray,
    *,
    frequency_hz: float,
    limits: SafetyConfig,
) -> None:
    """实机连接前一次性检查限位、相邻步长和名义速度。"""
    control = np.concatenate(
        [np.rad2deg(actions[:, 0:14]), actions[:, 14:16]], axis=1
    )
    checks = (
        ("right arm", control[:, 0:7], limits.right_arm_joint_min, limits.right_arm_joint_max),
        ("left arm", control[:, 7:14], limits.left_arm_joint_min, limits.left_arm_joint_max),
        ("right gripper", control[:, 14:15], limits.right_gripper_min, limits.right_gripper_max),
        ("left gripper", control[:, 15:16], limits.left_gripper_min, limits.left_gripper_max),
    )
    for name, values, lower, upper in checks:
        if np.any(values < lower) or np.any(values > upper):
            raise RuntimeError(f"recorded replay exceeds configured {name} limits")
    if len(control) < 2:
        return
    arm_delta = np.abs(np.diff(control[:, 0:14], axis=0))
    gripper_delta = np.abs(np.diff(control[:, 14:16], axis=0))
    if limits.enable_arm_delta_clip and np.max(arm_delta) > limits.arm_max_step:
        raise RuntimeError(
            f"recorded arm step {np.max(arm_delta):.3f} deg exceeds "
            f"{limits.arm_max_step:.3f} deg"
        )
    if limits.enable_gripper_delta_clip and np.max(gripper_delta) > limits.gripper_max_step:
        raise RuntimeError(
            f"recorded gripper step {np.max(gripper_delta):.3f} rad exceeds "
            f"{limits.gripper_max_step:.3f} rad"
        )
    if limits.enable_arm_velocity_limit and limits.arm_max_velocity is not None:
        arm_velocity = arm_delta * frequency_hz
        velocity_limit = np.tile(limits.arm_max_velocity, 2)
        if np.any(arm_velocity > velocity_limit):
            raise RuntimeError(
                f"recorded arm velocity {np.max(arm_velocity):.3f} deg/s "
                "exceeds configured limit"
            )
    if limits.enable_gripper_velocity_limit and limits.gripper_max_velocity is not None:
        if np.any(gripper_delta * frequency_hz > limits.gripper_max_velocity):
            raise RuntimeError("recorded gripper velocity exceeds configured limit")


# =============================================================================
# 5. 数采落盘单位转换与 episode writer
# =============================================================================


def action_to_record_row(
    target: np.ndarray,
    calibration: dict[str, object],
) -> np.ndarray:
    """把实际下发 target 转成 action.csv 的训练数据单位。

    双臂在 dispatch 输入中已经是 rad，直接保留；双夹爪从 motor-rad 转回
    ``0=张开、1=闭合`` 的归一化值。
    """

    row = validate_dispatch_action(target).astype(np.float64)
    row[14] = _gripper_normalized(row[14], calibration["right_gripper"])
    row[15] = _gripper_normalized(row[15], calibration["left_gripper"])
    return row


def state_to_record_row(
    state: RightArmGripperState,
    calibration: dict[str, object],
) -> np.ndarray:
    """把机器人控制反馈转成 observation_state.csv 的训练数据单位。

    Marvin 双臂反馈为 deg，落盘前转为 rad；RS05 夹爪反馈为 motor-rad，
    落盘前转为归一化值。
    """

    return np.concatenate(
        [
            np.deg2rad(state.right_arm_q),
            np.deg2rad(state.left_arm_q),
            [
                _gripper_normalized(
                    state.right_gripper_q[0],
                    calibration["right_gripper"],
                ),
                _gripper_normalized(
                    state.left_gripper_q[0],
                    calibration["left_gripper"],
                ),
            ],
        ]
    )


class ReplayEpisodeRecorder:
    """按原数采 episode 格式写 target、actual state 和三路相机帧。"""

    def __init__(
        self,
        root: Path,
        *,
        trajectory_name: str,
        source_meta: dict[str, object],
        trajectory_hz: float,
        dispatch_hz: float,
        frame_stride: int,
        camera_names: list[str],
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.episode_dir = root / f"replay_{trajectory_name}_{stamp}"
        arm_dir = self.episode_dir / "arm_data"
        arm_dir.mkdir(parents=True, exist_ok=False)
        self._calibration = source_meta.get("gripper_calibration")
        if not isinstance(self._calibration, dict):
            raise ValueError("source episode is missing gripper_calibration")
        self._camera_names = camera_names
        self._count = 0
        self._started_at = time.time()
        self._first_sample_wall: float | None = None
        self._last_sample_wall: float | None = None
        self._action_file = (arm_dir / "action.csv").open("w", newline="", encoding="utf-8")
        self._state_file = (arm_dir / "observation_state.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._time_file = (arm_dir / "timestamp.csv").open("w", newline="", encoding="utf-8")
        self._action_writer = csv.writer(self._action_file)
        self._state_writer = csv.writer(self._state_file)
        self._time_writer = csv.writer(self._time_file)
        self._action_writer.writerow(ACTION_COLUMNS)
        self._state_writer.writerow(ACTION_COLUMNS)
        self._time_writer.writerow(["timestamp_unix"])
        self._camera_files: dict[str, object] = {}
        self._camera_writers: dict[str, object] = {}
        for name in camera_names:
            camera_dir = self.episode_dir / "camera_data" / name
            (camera_dir / "images").mkdir(parents=True, exist_ok=True)
            handle = (camera_dir / "frames.csv").open("w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame_index",
                    "wall_time_unix",
                    "sensor_timestamp_ms",
                    "image_path",
                    "source_frame_id",
                ]
            )
            self._camera_files[name] = handle
            self._camera_writers[name] = writer
        self._source_meta = source_meta
        self._trajectory_hz = trajectory_hz
        self._dispatch_hz = dispatch_hz
        self._frame_stride = frame_stride

    def write_arm_sample(
        self,
        *,
        sample_index: int,
        timestamp: float,
        target: np.ndarray,
        state: RightArmGripperState,
    ) -> None:
        """只写 timestamp/action/observation_state 三个 CSV。"""

        self._check_next_sample_index(sample_index)
        target_row = action_to_record_row(target, self._calibration)
        state_row = state_to_record_row(state, self._calibration)
        self._time_writer.writerow([f"{timestamp:.9f}"])
        self._action_writer.writerow([f"{value:.9f}" for value in target_row])
        self._state_writer.writerow([f"{value:.9f}" for value in state_row])

    def write_camera_samples(
        self,
        *,
        sample_index: int,
        frames: dict[str, LatestFrame],
    ) -> None:
        """只写三路 JPEG 和各自的 frames.csv，不读取相机。"""

        self._check_next_sample_index(sample_index)
        for name in self._camera_names:
            frame = frames[name]
            image_path = (
                self.episode_dir
                / "camera_data"
                / name
                / "images"
                / f"frame_{sample_index:06d}.jpg"
            )
            Image.fromarray(frame.image.astype(np.uint8), mode="RGB").save(
                image_path, quality=90
            )
            self._camera_writers[name].writerow(
                [
                    sample_index,
                    f"{frame.wall_time:.9f}",
                    f"{frame.wall_time * 1000.0:.6f}",
                    str(image_path.resolve()),
                    frame.frame_id,
                ]
            )

    def finish_sample(self, *, sample_index: int, timestamp: float) -> None:
        """机械臂和相机均写成功后，提交这一逻辑样本。"""

        self._check_next_sample_index(sample_index)
        if self._first_sample_wall is None:
            self._first_sample_wall = timestamp
        self._last_sample_wall = timestamp
        self._count = sample_index

    def next_sample_index(self) -> int:
        """返回下一条同步样本使用的 1-based 帧号。"""

        return self._count + 1

    def _check_next_sample_index(self, sample_index: int) -> None:
        expected = self._count + 1
        if sample_index != expected:
            raise ValueError(
                f"sample_index must be the next frame {expected}, got {sample_index}"
            )

    def close(self, *, completed: bool) -> None:
        """关闭文件并写 meta.json；异常退出时 completed=false。"""
        for handle in (
            self._action_file,
            self._state_file,
            self._time_file,
            *self._camera_files.values(),
        ):
            handle.close()
        actual_fps = 0.0
        if (
            self._count > 1
            and self._first_sample_wall is not None
            and self._last_sample_wall is not None
            and self._last_sample_wall > self._first_sample_wall
        ):
            actual_fps = (self._count - 1) / (
                self._last_sample_wall - self._first_sample_wall
            )
        meta = {
            "episode_index": 0,
            "task": f"replay_{self._source_meta.get('task', '')}",
            "fps": actual_fps or self._dispatch_hz,
            "trajectory_hz": self._trajectory_hz,
            "dispatch_hz": self._dispatch_hz,
            "frame_stride": self._frame_stride,
            "planned_acceleration": (
                self._dispatch_hz * self._frame_stride / self._trajectory_hz
            ),
            "sync_source": "replay_control_loop",
            "state_names": ACTION_COLUMNS,
            "with_gripper_data": True,
            "with_gripper2_data": True,
            "gripper_value_encoding": "normalized_closed_1_open_0",
            "gripper_calibration": self._calibration,
            "camera_names": self._camera_names,
            "num_frames": self._count,
            "completed": completed,
            "started_at_unix": self._started_at,
            "finished_at_unix": time.time(),
            "arm_data": {
                "timestamp": str((self.episode_dir / "arm_data/timestamp.csv").resolve()),
                "observation_state": str(
                    (self.episode_dir / "arm_data/observation_state.csv").resolve()
                ),
                "action": str((self.episode_dir / "arm_data/action.csv").resolve()),
            },
            "camera_data": {
                "camera_count": len(self._camera_names),
                "camera_data_dir": str((self.episode_dir / "camera_data").resolve()),
            },
        }
        (self.episode_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @property
    def sample_count(self) -> int:
        """已经成功落盘的同步样本数量。"""

        return self._count


# =============================================================================
# 6. 起点检查、实时反馈和相机读取
# =============================================================================


def _start_pose_errors(
    robot,
    first_action: np.ndarray,
) -> tuple[float, float]:
    state = robot.get_state()
    target = action_to_control_action(first_action)
    arm_error = max(
        float(np.max(np.abs(state.right_arm_q - target.right_arm_q))),
        float(np.max(np.abs(state.left_arm_q - target.left_arm_q))),
    )
    gripper_error = max(
        float(np.max(np.abs(state.right_gripper_q - target.right_gripper_q))),
        float(np.max(np.abs(state.left_gripper_q - target.left_gripper_q))),
    )
    return arm_error, gripper_error


def _check_start_pose(
    robot,
    first_action: np.ndarray,
    *,
    arm_tolerance_deg: float,
    gripper_tolerance_rad: float,
) -> None:
    arm_error, gripper_error = _start_pose_errors(robot, first_action)
    if arm_error > arm_tolerance_deg or gripper_error > gripper_tolerance_rad:
        raise RuntimeError(
            "robot is not at the recorded start pose: "
            f"max arm error={arm_error:.2f} deg (limit {arm_tolerance_deg:.2f}), "
            f"max gripper error={gripper_error:.3f} rad "
            f"(limit {gripper_tolerance_rad:.3f})"
        )


def _validate_start_target_limits(
    target: RightArmGripperAction,
    limits: SafetyConfig,
) -> None:
    checks = (
        ("right arm", target.right_arm_q, limits.right_arm_joint_min, limits.right_arm_joint_max),
        ("left arm", target.left_arm_q, limits.left_arm_joint_min, limits.left_arm_joint_max),
        (
            "right gripper",
            target.right_gripper_q,
            limits.right_gripper_min,
            limits.right_gripper_max,
        ),
        (
            "left gripper",
            target.left_gripper_q,
            limits.left_gripper_min,
            limits.left_gripper_max,
        ),
    )
    for name, values, lower, upper in checks:
        if np.any(values < lower) or np.any(values > upper):
            raise RuntimeError(f"recorded start target exceeds configured {name} limits")


def read_robot_feedback(robot: Any) -> RightArmGripperState:
    """读取双臂和两个夹爪的当前反馈。"""

    return robot.get_state()


def snapshot_camera_frames(
    cameras: CameraManager,
    *,
    max_age_ms: float,
) -> dict[str, LatestFrame]:
    """读取三路后台相机的最新缓存帧，并检查帧年龄。"""

    return cameras.snapshot_latest(max_age_ms=max_age_ms)


# =============================================================================
# 7. CLI 参数解析
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        choices=[*TRAJECTORIES, "all"],
        default="all",
        help="轨迹名称；all 仅用于一次性离线检查三条轨迹。",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--source",
        choices=["runtime-trace", "action-csv"],
        default="runtime-trace",
        help="runtime-trace 为约 200 Hz 原始双臂双夹爪日志（默认）。",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robot-limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument(
        "--record-root",
        type=Path,
        default=DEFAULT_RECORD_ROOT,
        help="新采集 replay episode 的输出根目录。",
    )
    parser.add_argument(
        "--trajectory-hz",
        type=float,
        default=30.0,
        help="从原始 trace 时间轴选取 target 的频率。",
    )
    parser.add_argument(
        "--dispatch-hz",
        type=float,
        default=60.0,
        help="所选 target 的计划实机下发频率。",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="每 N 个 trajectory target 执行一个；N 会乘到加速倍率中。",
    )
    parser.add_argument(
        "--max-replay-arm-velocity",
        type=float,
        default=None,
        help=(
            "本次加速回放的臂速度检查上限（deg/s）；"
            "默认使用 robot_limits.yaml。"
        ),
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="连接并控制实机；不加时只加载和规划轨迹。",
    )
    parser.add_argument("--max-start-arm-error-deg", type=float, default=5.0)
    parser.add_argument("--max-start-gripper-error-rad", type=float, default=0.3)
    parser.add_argument("--camera-max-age-ms", type=float, default=200.0)
    parser.add_argument("--camera-ready-timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--skip-start-approach",
        action="store_true",
        help="起点不匹配时直接拒绝，不调用慢速插值到起点。",
    )
    return parser


def _options_from_args(args: argparse.Namespace) -> ReplayOptions:
    """把命令行参数集中整理成 ReplayOptions。"""

    return ReplayOptions(
        trajectory=args.trajectory,
        data_root=args.data_root,
        source=args.source,
        config_path=args.config,
        robot_limits_path=args.robot_limits,
        record_root=args.record_root,
        trajectory_hz=args.trajectory_hz,
        dispatch_hz=args.dispatch_hz,
        frame_stride=args.frame_stride,
        max_replay_arm_velocity=args.max_replay_arm_velocity,
        start_row=args.start_row,
        end_row=args.end_row,
        execute=args.execute,
        max_start_arm_error_deg=args.max_start_arm_error_deg,
        max_start_gripper_error_rad=args.max_start_gripper_error_rad,
        approach_start=not args.skip_start_approach,
        camera_max_age_ms=args.camera_max_age_ms,
        camera_ready_timeout_sec=args.camera_ready_timeout_sec,
    )


# =============================================================================
# 8. 回放准备与实机资源生命周期
# =============================================================================


def prepare_trajectories(options: ReplayOptions) -> list[PreparedTrajectory]:
    """加载、裁剪、重采样并跳帧；本函数不连接任何硬件。"""

    names = TRAJECTORIES if options.trajectory == "all" else (options.trajectory,)
    prepared: list[PreparedTrajectory] = []
    for name in names:
        try:
            actions, frame_times, source_meta = load_trajectory(
                options.data_root / name, source=options.source
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"failed to load trajectory {name!r}: {exc}") from exc
        actions = actions[options.start_row : options.end_row]
        frame_times = frame_times[options.start_row : options.end_row]
        if len(actions) == 0:
            raise ValueError(f"{name}: selected row range is empty")
        frame_times = frame_times - frame_times[0]
        actions, frame_times = resample_trajectory(
            actions,
            frame_times,
            frequency_hz=options.trajectory_hz,
        )
        actions, frame_times = apply_frame_stride(
            actions,
            frame_stride=options.frame_stride,
            execution_hz=options.dispatch_hz,
        )
        duration = float(frame_times[-1]) if len(frame_times) > 1 else 0.0
        plan = TrajectoryPlan(
            name=name,
            frame_count=len(actions),
            duration_sec=duration,
            acceleration=options.acceleration,
            source=options.source,
        )
        print(
            f"[replay] {name}: source={options.source}, {len(actions)} frames, "
            f"{duration:.3f}s, trajectory_hz={options.trajectory_hz:g}, "
            f"dispatch_hz={options.dispatch_hz:g}, "
            f"frame_stride={options.frame_stride}, "
            f"acceleration≈{options.acceleration:g}x"
        )
        prepared.append(
            PreparedTrajectory(
                plan=plan,
                actions=actions,
                frame_times=frame_times,
                source_meta=source_meta,
            )
        )
    return prepared


def build_replay_safety(options: ReplayOptions) -> SafetyConfig:
    """读取安全 YAML，并应用本次回放显式指定的速度上限。"""

    safety_config = SafetyConfig.from_yaml(options.robot_limits_path)
    if options.max_replay_arm_velocity is not None:
        safety_config.arm_max_velocity = np.full(
            7, options.max_replay_arm_velocity, dtype=np.float32
        )
    return safety_config


def move_robot_to_start(
    robot: Any,
    first_action: np.ndarray,
    *,
    options: ReplayOptions,
    safety: SafetyConfig,
) -> None:
    """必要时慢速移动到第一帧，并用实时反馈复核起点误差。"""

    arm_error, gripper_error = _start_pose_errors(robot, first_action)
    start_is_close = (
        arm_error <= options.max_start_arm_error_deg
        and gripper_error <= options.max_start_gripper_error_rad
    )
    if not start_is_close:
        if not options.approach_start:
            _check_start_pose(
                robot,
                first_action,
                arm_tolerance_deg=options.max_start_arm_error_deg,
                gripper_tolerance_rad=options.max_start_gripper_error_rad,
            )
        start_target = action_to_control_action(first_action)
        _validate_start_target_limits(start_target, safety)
        adapter = ActionAdapter(
            policy_arm_unit="rad",
            control_arm_unit="deg",
            policy_gripper_unit="rad",
            control_gripper_unit="rad",
            control_mode="dual_arm_dual_gripper",
        )
        approach = SlowInterpolatedDispatcher.from_yaml(
            robot,
            adapter=adapter,
            config_path=options.config_path,
        )
        print(
            "[replay] current pose differs from recorded start "
            f"(arm={arm_error:.2f} deg, gripper={gripper_error:.3f} rad); "
            f"slowly approaching over {approach.duration_sec:.2f}s at "
            f"{approach.frequency_hz:.1f} Hz"
        )
        approach.dispatch_control_action(start_target)

    _check_start_pose(
        robot,
        first_action,
        arm_tolerance_deg=options.max_start_arm_error_deg,
        gripper_tolerance_rad=options.max_start_gripper_error_rad,
    )


def execute_prepared_trajectory(
    options: ReplayOptions,
    trajectory: PreparedTrajectory,
    safety: SafetyConfig,
    *,
    robot: Any | None = None,
    cameras: CameraManager | None = None,
) -> Path:
    """连接资源、执行一条已准备轨迹，并负责异常收尾。"""

    if robot is None:
        robot = make_robot(build_robot_config(options.config_path, execute=True))
    if cameras is None:
        cameras = build_camera_manager(options.config_path)

    camera_names = ["head", "left_wrist", "right_wrist"]
    recorder: ReplayEpisodeRecorder | None = None
    robot_connected = False
    cameras_connected = False
    completed = False
    try:
        # 相机先启动并等待三路都有首帧。慢速到起点期间相机继续预热，
        # 但此阶段不会写入最终 episode。
        cameras.connect_all()
        cameras_connected = True
        cameras.start_streaming()
        cameras.wait_until_ready(timeout_sec=options.camera_ready_timeout_sec)
        print("[replay] cameras ready: head, left_wrist, right_wrist")

        robot.connect()
        robot_connected = True
        move_robot_to_start(
            robot,
            trajectory.actions[0],
            options=options,
            safety=safety,
        )
        print(f"[replay] start pose accepted; executing {trajectory.plan.name}")

        recorder = ReplayEpisodeRecorder(
            options.record_root,
            trajectory_name=trajectory.plan.name,
            source_meta=trajectory.source_meta,
            trajectory_hz=options.trajectory_hz,
            dispatch_hz=options.dispatch_hz,
            frame_stride=options.frame_stride,
            camera_names=camera_names,
        )
        print(f"[replay] recording episode to {recorder.episode_dir.resolve()}")

        for action, _dt in scheduled_replay_frames(
            trajectory.actions,
            trajectory.frame_times,
        ):
            dispatch_action(action, robot)
            state = read_robot_feedback(robot)
            frames = snapshot_camera_frames(
                cameras,
                max_age_ms=options.camera_max_age_ms,
            )
            sample_index = recorder.next_sample_index()
            sample_time = time.time()
            recorder.write_camera_samples(
                sample_index=sample_index,
                frames=frames,
            )
            recorder.write_arm_sample(
                sample_index=sample_index,
                timestamp=sample_time,
                target=action,
                state=state,
            )
            recorder.finish_sample(
                sample_index=sample_index,
                timestamp=sample_time,
            )
        completed = True
        robot.hold_position()
        print(f"[replay] completed {trajectory.plan.name}; holding final pose")
    except BaseException:
        # 任意硬件、相机、写盘异常以及 Ctrl-C 都先保持当前姿态。
        if robot_connected:
            robot.hold_position()
        raise
    finally:
        if recorder is not None:
            recorder.close(completed=completed)
            print(
                f"[replay] recorded {recorder.sample_count} synchronized samples in "
                f"{recorder.episode_dir.resolve()}"
            )
        if robot_connected:
            robot.disconnect()
        if cameras_connected:
            cameras.disconnect_all()
    if recorder is None:
        raise RuntimeError("replay completed without creating a recorder")
    return recorder.episode_dir


if __name__ == "__main__":
    raise SystemExit(main())
