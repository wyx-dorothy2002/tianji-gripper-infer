# 天机双臂双夹爪轨迹回放与同步数采

## 1. 文档目的

本文是 `deployment/replay_recorded_actions.py` 的接口说明和现场交接文档。

该模块完成两项工作：

1. 将已有示教轨迹转换成天机双臂、双 RS05 夹爪的 16D target 并下发；
2. 回放期间逐个执行周期采集双臂反馈、两个夹爪反馈和三路相机图像，生成兼容原数采
   目录结构的新 episode。

模块同时提供：

- Python API：供其他控制程序 import；
- CLI：供现场人员直接运行；
- 只检查模式：不连接硬件，检查数据加载和时序规划；
- 实机模式：连接硬件、到达起点、回放、同步数采、安全收尾。

## 2. 文件位置

| 内容 | 路径 |
| --- | --- |
| 回放与数采脚本 | `deployment/replay_recorded_actions.py` |
| 推理与硬件配置 | `configs/infer.yaml` |
| 机器人安全限位 | `configs/robot_limits.yaml` |
| 单帧下发接口说明 | `TIANJI_DISPATCH_INTERFACE.md` |
| 本文档 | `REPLAY_AND_RECORD.md` |

## 3. 代码结构和主要函数

脚本按“配置 → 数据加载 → 单帧下发 → 时序处理 → 落盘 → 主流程”的顺序组织。
交接时优先阅读以下内容：

| 内容 | 作用 |
| --- | --- |
| `ReplayOptions` | 集中保存所有回放参数 |
| `main()` | 直接展示“准备→检查→执行”主流程|
| `prepare_trajectories()` | 加载、裁剪、重采样和跳帧 |
| `build_replay_safety()` | 读取并调整本次安全配置 |
| `move_robot_to_start()` | 检查起点并按需慢速到达 |
| `execute_prepared_trajectory()` | 只管理硬件连接、执行和 finally 收尾 |
| `load_trajectory()` | 读取 trace/CSV 并统一成 16D |
| `validate_dispatch_action()` | 只检查 16D 和有限值 |
| `action_to_control_action()` | 只做切片、rad/deg 和 A/B 映射 |
| `dispatch_action()` | 只调用统一机器人 `send_action()` |
| `read_robot_feedback()` | 只读取双臂和两个夹爪反馈 |
| `snapshot_camera_frames()` | 只取得三路相机缓存帧 |
| `action_to_record_row()` | 只把 target 转成 action.csv 单位 |
| `state_to_record_row()` | 只把反馈转成 observation_state.csv 单位 |
| `write_camera_samples()` | 只写三路 JPEG 和 frames.csv |
| `write_arm_sample()` | 只写 timestamp/action/state CSV |
| `finish_sample()` | 相机和机械臂都成功后提交帧计数 |
| `scheduled_replay_frames()` | 只负责等待计划时刻并产出 target |
| `resample_trajectory()` | 按 trajectory_hz 从源时间轴取帧 |
| `apply_frame_stride()` | 跳过中间 target，并生成 dispatch 时间轴 |
| `validate_replay_trajectory()` | 实机连接前检查整条轨迹 |
| `ReplayEpisodeRecorder` | 写机械臂 CSV、相机图片和 meta.json |

以下划线开头的函数是局部辅助函数，不承担隐藏接口的作用；需要排查问题时可以直接阅读。

### 3.1 `ReplayOptions`

```python
@dataclass(frozen=True)
class ReplayOptions:
    trajectory: str = "all"
    data_root: Path = DEFAULT_DATA_ROOT
    source: str = "runtime-trace"
    config_path: Path = DEFAULT_CONFIG
    robot_limits_path: Path = DEFAULT_LIMITS
    record_root: Path = Path("data/replay_recordings")
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
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `trajectory` | `str` | `"all"` | `back_to_home`、`home_to_back`、`episode_000011` 或 `all` |
| `data_root` | `Path` | 旧数采 raw 目录 | 输入轨迹根目录 |
| `source` | `str` | `"runtime-trace"` | `runtime-trace` 读取约 200 Hz 原始 trace；`action-csv` 读取同步 CSV |
| `config_path` | `Path` | `configs/infer.yaml` | 机器人、夹爪、相机和慢速到起点参数 |
| `robot_limits_path` | `Path` | `configs/robot_limits.yaml` | 关节、夹爪、单步和速度安全限制 |
| `record_root` | `Path` | `data/replay_recordings` | 新 episode 输出根目录 |
| `trajectory_hz` | `float` | `30` | 从源时间轴选取 target 的频率 |
| `dispatch_hz` | `float` | `60` | 所选 target 的计划下发频率 |
| `frame_stride` | `int` | `1` | 每隔几个 trajectory target 执行一个 |
| `max_replay_arm_velocity` | `float \| None` | `None` | 本次回放的臂速度检查上限；省略时使用安全 YAML |
| `start_row/end_row` | `int` | `0/None` | 在重采样前裁剪源 CSV 行范围 |
| `execute` | `bool` | `False` | `False` 只规划；`True` 连接并控制实机 |
| `max_start_arm_error_deg` | `float` | `5` | 慢速到起点后的最大臂误差 |
| `max_start_gripper_error_rad` | `float` | `0.3` | 慢速到起点后的最大夹爪误差 |
| `approach_start` | `bool` | `True` | 不在起点时是否调用慢速插值 |
| `camera_max_age_ms` | `float` | `200` | 相机缓存帧最大允许时间|
| `camera_ready_timeout_sec` | `float` | `10` | 等待三路相机首帧的超时 |

`ReplayOptions` 是不可变对象。创建后调用 `validate()` 可在不触碰硬件的情况下检查参数。

### 3.2 逐帧读取和落盘明确分开

正式循环没有 `send_and_record()` 或 `write_replay_sample()` 之类的组合函数。
相机读取、相机落盘和机械臂 CSV 落盘分别调用：

```python
for action, _dt in scheduled_replay_frames(actions, frame_times):
    dispatch_action(action, robot)
    state = read_robot_feedback(robot)
    frames = snapshot_camera_frames(cameras, max_age_ms=200.0)

    sample_index = recorder.next_sample_index()
    sample_time = time.time()
    recorder.write_camera_samples(sample_index=sample_index, frames=frames)
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
```

`finish_sample()` 只在相机和机械臂两部分均写成功后调用，因此 `meta.json/num_frames`
表示完整同步样本数。

### 3.3 `dispatch_action`

```python
def dispatch_action(action: np.ndarray, robot) -> None:
    ...
```

输入规范：

```text
shape: (16,)
dtype: 可转换为 float32
value: 全部有限

0:7   右/B 臂关节，rad
7:14  左/A 臂关节，rad
14    右/B 夹爪，motor-rad
15    左/A 夹爪，motor-rad
```

`dispatch_action()` 内部对应三个小函数：

```python
command = validate_dispatch_action(action)
control_action = action_to_control_action(command)
robot.send_action(control_action)
```

其中 `action_to_control_action()` 明确实现：

```text
0:7  右臂 rad -> deg -> Marvin B
7:14 左臂 rad -> deg -> Marvin A
14   右夹爪 motor-rad -> B/COM1/RS05
15   左夹爪 motor-rad -> A/COM1/RS05
```

## 4. 小功能单独调用示例

只加载和准备轨迹：

```python
from tianji_gripper_runtime.deployment.replay_recorded_actions import (
    ReplayOptions,
    prepare_trajectories,
)

options = ReplayOptions(
    trajectory="episode_000011",
    trajectory_hz=30.0,
    dispatch_hz=60.0,
    frame_stride=1,
)
trajectory = prepare_trajectories(options)[0]
print(trajectory.plan)
```

现场完整执行建议直接使用 CLI，让 `main()` 统一处理安全检查和资源收尾。

## 5. CLI

### 5.1 查看完整参数

```bash
python preinfere/tianji_gripper_runtime/deployment/replay_recorded_actions.py --help
```

### 5.2 只检查三条轨迹

```bash
python preinfere/tianji_gripper_runtime/deployment/replay_recorded_actions.py \
  --trajectory all
```

### 5.3 30 Hz target 序列，以 60 Hz 执行

```bash
python preinfere/tianji_gripper_runtime/deployment/replay_recorded_actions.py \
  --trajectory episode_000011 \
  --trajectory-hz 30 \
  --dispatch-hz 60 \
  --frame-stride 1 \
  --execute
```

## 6. 三个时序参数

三个参数必须分开理解：

```text
trajectory_hz：从原始 trace 的时间轴每秒选择多少个 target
dispatch_hz：选择后的 target 每秒计划执行多少个
frame_stride：选择后的 target 每隔多少个执行一个
```

近似加速倍率：

```text
acceleration = dispatch_hz / trajectory_hz × frame_stride
```

示例：

| trajectory_hz | dispatch_hz | stride | 近似倍率 | 含义 |
| ---: | ---: | ---: | ---: | --- |
| 30 | 30 | 1 | 1× | 30 Hz target 正常时间轴 |
| 30 | 60 | 1 | 2× | 不跳帧，以两倍频率执行 |
| 30 | 30 | 3 | 3× | 每三帧执行一帧 |
| 30 | 60 | 3 | 6× | 两倍下发频率，再叠加三帧跳采 |

`frame_stride > 1` 时最后一个 target 总是保留，防止最终姿态丢失。

注意：`dispatch_hz` 是计划频率。每个周期还包含 Marvin 下发、双臂/双夹爪反馈读取、
相机缓存快照和三张 JPEG 落盘。若单周期实际耗时超过计划周期，调度器不会补发并发
指令，而是立即执行下一帧；`timestamp.csv` 会记录真实时间。

## 7. 输入数据

默认输入根目录：

```text
/home/user/workspace/TJ-gripper-codex-gripper-fixes-20260617-reclone/data/raw
```

支持：

```text
back_to_home //从自然下垂状态抬起到初始位置
home_to_back //从抬起状态移动到自然下垂位置
episode_000011 //从初始状态模拟采集夹取闭合再回到初始位置
```

### 7.1 `runtime-trace`

默认读取：

```text
<episode>/_runtime/marvin_drag_teach_teleop_trace.csv
```

实际统计约 200 Hz。映射为：

| trace 字段 | Replay 16D 字段 |
| --- | --- |
| `cmd_B_deg_1..7` | 右臂，deg 转 rad |
| `cmd_A_deg_1..7` | 左臂，deg 转 rad |
| `gripper2_target_rad` | 右夹爪 motor-rad |
| `gripper_target_rad` | 左夹爪 motor-rad |

### 7.2 `action-csv`

读取：

```text
<episode>/arm_data/action.csv
<episode>/arm_data/timestamp.csv
```

臂字段已是 rad。若夹爪字段为 `normalized_closed_1_open_0`，脚本使用源
`meta.json/gripper_calibration` 转换为 motor-rad 后再下发。

## 8. 起点处理

连接机器人后读取实时状态，与第一条执行 target 比较。

若误差超过阈值且 `approach_start=True`：

1. 检查第一帧是否在关节和夹爪限位内；
2. 读取 `infer.yaml` 中的：
   - `slow_dispatch_duration_sec`
   - `slow_dispatch_frequency_hz`
   - `slow_dispatch_interpolation_mode`
3. 调用 `SlowInterpolatedDispatcher.dispatch_control_action()`；
4. 到位后再次读取反馈；
5. 误差合格才创建 recorder 并开始正式 episode。

慢速到起点阶段不写入 episode。

若 `approach_start=False`，起点不匹配会抛出 `RuntimeError`，且不下发正式轨迹。

## 9. 同步采集时序

三路相机在独立后台线程持续采集：

```text
head
left_wrist
right_wrist
```

正式回放的一个逻辑样本严格按以下顺序执行：

```text
等待计划时刻
  -> dispatch_action(target)
  -> robot.get_state()
  -> cameras.snapshot_latest()
  -> recorder.record(target, actual_state, camera_frames)
```

因此：

- `action.csv` 是实际送入 dispatch 接口的 target；
- `observation_state.csv` 是 target 下发后读取的 actual state；
- 每路 `frames.csv` 保存相机线程自己的采集时间和 source frame ID；
- 被 `frame_stride` 跳过的 target 不下发、不读反馈、不落盘；
- 相机帧年龄超过 `camera_max_age_ms` 时停止后续回放。

相机物理采集配置当前为 30 Hz。`dispatch_hz=60` 时，相邻两个机器人样本可能引用同一个
`source_frame_id`，这是 30 Hz 相机配合 60 Hz机器人采样的预期结果，不代表图片写入失败。

## 10. 输出 episode

```text
data/replay_recordings/
└── replay_<trajectory>_<timestamp>/
    ├── meta.json
    ├── arm_data/
    │   ├── timestamp.csv
    │   ├── action.csv
    │   └── observation_state.csv
    └── camera_data/
        ├── head/
        │   ├── frames.csv
        │   └── images/frame_000001.jpg
        ├── left_wrist/
        │   ├── frames.csv
        │   └── images/frame_000001.jpg
        └── right_wrist/
            ├── frames.csv
            └── images/frame_000001.jpg
```

### 10.1 机械臂 CSV

三个 CSV 的逻辑行数一致。

字段顺序：

```text
right_joint_1.pos ... right_joint_7.pos
left_joint_1.pos  ... left_joint_7.pos
right_gripper.pos
left_gripper.pos
```

落盘单位：

```text
双臂：rad
双夹爪：normalized，0=张开，1=闭合
```

target 和 feedback 的夹爪值均使用源 episode 的左右独立标定转换。

### 10.2 相机 `frames.csv`

| 字段 | 说明 |
| --- | --- |
| `frame_index` | 与机械臂同步样本对应的 1-based 逻辑帧号 |
| `wall_time_unix` | 相机后台线程取得该图像的 Unix 时间 |
| `sensor_timestamp_ms` | 当前适配器使用 wall time 换算的毫秒值 |
| `image_path` | JPEG 绝对路径 |
| `source_frame_id` | 相机后台线程原始递增帧号，可用于发现重复缓存帧 |

### 10.3 `meta.json`

重要字段：

| 字段 | 说明 |
| --- | --- |
| `fps` | 根据实际同步样本时间戳估算的落盘频率；不足两帧时回退到 dispatch_hz |
| `trajectory_hz` | 从源 trace 取 target 的频率 |
| `dispatch_hz` | 计划下发频率 |
| `frame_stride` | 跳帧步长 |
| `planned_acceleration` | 由三个时序参数计算的计划加速倍率 |
| `num_frames` | 成功写入的同步样本数 |
| `completed` | 是否完整执行到最后一帧 |
| `gripper_value_encoding` | `normalized_closed_1_open_0` |
| `gripper_calibration` | 本次 motor-rad/normalized 转换使用的标定 |
| `camera_names` | 三路相机名称 |

异常或 Ctrl-C 发生在 recorder 创建之后时，已有数据保留，并写：

```json
{
  "completed": false
}
```

## 11. 安全检查

正式连接硬件前检查：

1. 每帧必须是 16 个有限数值；
2. 双臂 target 转 deg 后必须在左右关节限位内；
3. 双夹爪 motor-rad 必须在左右夹爪限位内；
4. 相邻臂 target 变化不得超过 `arm_max_step`；
5. 相邻夹爪变化不得超过 `gripper_max_step`；
6. 相邻臂 target 按 `dispatch_hz` 计算的速度不得超过配置；
7. 若启用夹爪速度限制，同样按 `dispatch_hz` 检查。

`--max-replay-arm-velocity`/`max_replay_arm_velocity` 只覆盖本次离线速度检查。该参数应由
硬件负责人确认，不应为了“让程序通过”而随意增大。

例：`back_to_home` 在 `trajectory_hz=30、dispatch_hz=60、stride=1` 时峰值约为
225.8 deg/s，超过当前 YAML 的 180 deg/s。硬件负责人确认后才可显式设置：

```bash
--max-replay-arm-velocity 250
```

## 12. 异常

| 异常 | 含义 | 是否可能已下发 |
| --- | --- | --- |
| `ValueError` | 参数、行范围或输入值无效 | 否 |
| `RuntimeError` | 数据加载、安全检查或起点检查失败 | 起点复核失败时可能已执行慢速到起点 |
| `RobotError` | Marvin 或 RS05 连接/读写失败 | 可能 |
| `CameraError` | 相机连接、线程、首帧或新鲜度失败 | 正式回放中发生时可能 |
| `KeyboardInterrupt` | 操作员按 Ctrl-C | 可能 |
| `OSError` | 创建目录或写 CSV/JPEG 失败 | 可能 |

任何正式执行阶段的异常都会：

1. 停止消费后续 target；
2. 尽力调用 `robot.hold_position()`；
3. 若 recorder 已创建，关闭 CSV 并写 `completed=false`；
4. 断开机器人和相机；
5. 将异常继续抛给调用方。

## 13. 现场操作流程

1. 确认机器人运动空间无人，急停可用。
2. 确认没有其他 Marvin 控制进程：

   ```bash
   ps -ef | grep -E 'replay_recorded|infer_tianji|drag_teach'
   ```

3. 先运行只检查模式。
4. 核对输出的帧数、计划时长和 acceleration。
5. 对加速轨迹由硬件负责人确认速度上限。
6. 加 `--execute` 实机运行。
7. 完成后检查输出 episode。

Marvin SDK 本地端口只能被一个控制进程占用。出现：

```text
port bind failure, possibly occupied by another program
```

表示已有进程占用端口，不是轨迹数据错误。

## 14. 验收清单

- [ ] 控制台无 traceback，显示 `completed`。
- [ ] `meta.json/completed` 为 `true`。
- [ ] `meta.json/num_frames` 等于三个机械臂 CSV 的数据行数。
- [ ] 三路 `frames.csv` 行数均等于 `num_frames`。
- [ ] 三路首、中、末 JPEG 可打开且不是黑帧。
- [ ] `action.csv` 与 `observation_state.csv` 左右映射正确。
- [ ] 双臂字段为 rad，夹爪字段位于 `[0, 1]`。
- [ ] `frames.csv/source_frame_id` 符合相机 30 Hz 与 dispatch 频率关系。
- [ ] `timestamp.csv` 单调递增。
- [ ] 完成后机器人保持最终姿态。

## 15. 已知边界

- 本接口是单进程、同步执行接口，不支持并发调用同一个机器人。
- `dispatch_hz` 是计划频率，不保证操作系统、SDK、反馈读取和 JPEG 写盘一定满足该周期。
- 三路相机当前物理采集为 30 Hz，高于 30 Hz 的机器人样本可能复用相机帧。
- 图片当前同步编码为 JPEG；磁盘或 CPU 较慢时会降低实际循环频率。
- `sensor_timestamp_ms` 当前来自相机帧 wall time，不是 RealSense 硬件时钟。
- 本接口记录 RGB，不记录深度。
