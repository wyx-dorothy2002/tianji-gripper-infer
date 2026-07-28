# 算法动作下发至天机双臂双夹爪接口

本文交付给天机机器人控制侧，说明如何消费算法服务返回的单步 16 维动作，并下发至天机 A/B 双臂和两侧 RS05 夹爪。

> **接口边界**：当前工程没有单独暴露 HTTP/REST 的“天机下发接口”。算法运行进程在得到模型动作后，直接调用 MARVIN SDK。本说明中的 `dispatch_action` 是控制侧必须实现或复用的本地软件接口；不要把模型动作当作可直接透传给天机控制器的网络报文。

## 1. 输入契约

控制侧每个控制周期接收一行算法动作：

```text
action: float32[16]  # 绝对位置目标，全部为有限数值
```

字段顺序、硬件映射和单位固定如下：

| 下标 | 算法字段 | 天机硬件 | 算法输出单位 | SDK 下发单位 |
| --- | --- | --- | --- | --- |
| `0:7` | `right_arm_q` | B 臂右臂 7 关节 | rad | deg |
| `7:14` | `left_arm_q` | A 臂左臂 7 关节 | rad | deg |
| `14` | `right_gripper_q` | B 臂末端右夹爪 | rad | rad |
| `15` | `left_gripper_q` | A 臂末端左夹爪 | rad | rad |

**特别注意**：算法向量的顺序是“右/B 臂在前、左/A 臂在后”。天机 SDK 下发双臂时的参数名则是 `A`（左）和 `B`（右），不能按数组顺序直接传入。

当前模型返回的是**绝对目标位置**，不是增量、速度或力矩命令。控制侧应只消费动作序列的有限前缀；当前运行时默认每次取前 `16` 个动作、控制目标周期为 `0.05 s`。模型返回的序列长度不构成硬件指令数量承诺。

## 2. 推荐软件接口

控制侧对算法模块提供以下本地调用即可：

```python
def dispatch_action(action: np.ndarray) -> None:
    """下发一个经安全校验的 16D 算法绝对位置动作。"""
```

推荐参考实现：

```python
import numpy as np

from tianji_gripper_runtime.runtime.action_adapter import RightArmGripperAction


def validate_dispatch_action(action: np.ndarray) -> np.ndarray:
    command = np.asarray(action, dtype=np.float32).reshape(-1)
    if command.shape != (16,) or not np.all(np.isfinite(command)):
        raise ValueError("action must be 16 finite float values")
    return command.copy()


def action_to_control_action(command: np.ndarray) -> RightArmGripperAction:
    # 右/B、左/A 映射在这里集中完成。
    # 仅双臂 rad -> deg；两个夹爪保持 motor-rad。
    return RightArmGripperAction(
        right_arm_q=np.rad2deg(command[0:7]).astype(np.float32),
        left_arm_q=np.rad2deg(command[7:14]).astype(np.float32),
        right_gripper_q=command[14:15],
        left_gripper_q=command[15:16],
        control_left_arm=True,
        control_left_gripper=True,
    )


def dispatch_action(action: np.ndarray, robot) -> None:
    command = validate_dispatch_action(action)
    control_action = action_to_control_action(command)
    # 整轨迹限位、步长和速度检查应在进入逐帧循环前完成。
    robot.send_action(control_action)
```

生产实现必须在调用上述接口前完成第 5 节的安全处理。现有推理脚本也是调用 `RightArmGripperRobot.send_action()`：它会先暂存/批量发送双臂，再依次发送左、右夹爪。控制侧不应直接调用 `move_to_joints_direct()`，除非它正在实现或替换 `RightArmGripperRobot` 的底层天机适配器。

## 3. 天机双臂 SDK 调用

当前适配器使用 MARVIN SDK。控制命令的实际调用等价于：

```python
# A 对应左臂，B 对应右臂；两个 joints 均为长度 7 的角度（deg）列表。
sdk_robot.clear_set()
sdk_robot.set_joint_cmd_pose(arm="A", joints=left_arm_deg)
sdk_robot.set_joint_cmd_pose(arm="B", joints=right_arm_deg)
sdk_robot.send_cmd()
```

工程中由 `move_to_joints_direct(left_joints=..., right_joints=...)` 统一封装这两个写入，并通过 `_write_marvin_batch(...)` 提交。控制侧应在同一控制周期内批量写入 A、B 两臂，避免只更新一侧时另一侧使用过期目标。

下发前，天机控制器必须已连接并进入关节阻抗控制状态；当前运行时连接阶段会设置双臂状态为 `3`、关节阻抗模式，并设置速度/加速度比例。下发失败、反馈失效或急停时，不再发送新目标，而应读取当前关节位置后再次发送当前位置以保持姿态。

## 4. RS05 双夹爪下发

两侧夹爪复用对应天机臂的末端 485/CAN 通道，使用 RS05 MIT 位置控制帧。

| 夹爪 | 天机臂 | 通道 | 目标 CAN ID | 位置范围（rad） |
| --- | --- | --- | --- | --- |
| 左夹爪 | A | COM 1 | `0x08` | `[-5.283053, 0.055032]` |
| 右夹爪 | B | COM 1 | `0x08` | `[-5.283053, 0.055032]` |

每个位置目标先裁剪到上表范围，再编码为 MIT 控制数据：

```text
payload (8 bytes) = position + velocity + Kp + Kd + torque
position: command[14]（右）或 command[15]（左），单位 rad
velocity: 0 rad/s
Kp: 45
Kd: 5
torque: 0 Nm
```

当前配置使用 4 字节、小端序 CAN ID 前缀，因此完整末端通道帧为：

```text
[CAN ID 0x08 的 4 字节小端序] + [8 字节 RS05 MIT payload]
```

直接末端通道模式下，控制侧调用天机原生 API：

```python
# A 臂左夹爪
native.OnSetChDataA(frame_buffer, frame_length, 1)

# B 臂右夹爪
native.OnSetChDataB(frame_buffer, frame_length, 1)
```

请复用本工程的 `Rs05MitEndChannelGripperInterface` 编码器，而不是自行拼接位域。其 `send_position()` 已包含位置裁剪和 MIT 报文编码；`connect()` 会在配置开启时依次发送 `clear-error`、`enter-motor`，确保 RS05 进入可控状态。

## 5. 下发前强制安全处理

以下安全处理属于机器人控制侧，不能依赖算法服务完成：

1. 校验动作形状为 `(16,)`，所有值均为有限数值。
2. 将双臂 `rad` 转换为 `deg` 后，按天机关节限位裁剪。
3. 以实时反馈或上一条已执行目标为基准，裁剪单步变化：当前配置为双臂每关节最多 `2.0 deg`、每个夹爪最多 `0.2 rad`。
4. 对夹爪位置裁剪到 `[-5.283053, 0.055032] rad`。
5. 执行失败、通信超时、夹爪故障、急停或动作校验失败时，停止消费后续算法动作并保持当前位置。
6. 每步下发后读取/确认双臂和夹爪反馈；日志中应至少记录目标值、反馈值、时间戳和错误码。

原工程的 `SafetyLayer` 已实现限位、步长和速度处理。若控制侧不复用该模块，必须实现等价保护并由硬件负责人确认实际关节限位。

## 6. 控制侧验收清单

- [ ] 输入 16D 向量映射正确：`0:7 -> B`、`7:14 -> A`、`14 -> B 夹爪`、`15 -> A 夹爪`。
- [ ] 仅双臂关节从 rad 转为 deg；夹爪仍使用 rad。
- [ ] 同一个控制周期中同时提交 A、B 两臂目标。
- [ ] 左夹爪经 A/COM1，右夹爪经 B/COM1 下发。
- [ ] RS05 上电后完成清错和电机使能，且位置帧使用 CAN ID `0x08`。
- [ ] 任一异常时机械臂和夹爪保持当前位置，不继续执行动作队列。

## 7. 工程对应实现

| 职责 | 代码位置 |
| --- | --- |
| 16D 动作拆分、rad/deg 转换 | `runtime/action_adapter.py` |
| 双臂/双夹爪下发顺序 | `runtime/robot_interface.py` |
| MARVIN 双臂 SDK 批量调用 | `../tianji_wuji_runtime/runtime/tianji_arm_system.py` |
| RS05 MIT 帧编码和末端通道调用 | `runtime/gripper_interface.py` |
| 安全限位和步长裁剪 | `runtime/safety.py` |
| 模型动作慢速插值和同步下发 | `runtime/slow_dispatch.py` |
| 当前硬件通道、RS05 参数和控制周期 | `configs/infer.yaml` |

### 7.1 Replay 脚本与本文接口逐项对应

`deployment/replay_recorded_actions.py` 不把校验、转换、反馈和落盘混在一个函数中。
与本文接口的对应关系如下：

| 本文步骤 | Replay 函数 | 函数只负责 |
| --- | --- | --- |
| 16D/有限值检查 | `validate_dispatch_action()` | 返回合法的 `float32[16]` |
| 右/左切片与单位转换 | `action_to_control_action()` | 臂 rad→deg；夹爪保持 motor-rad |
| 统一硬件下发 | `dispatch_action()` | 调用 `robot.send_action()` |
| 双臂双夹爪反馈 | `read_robot_feedback()` | 调用 `robot.get_state()` |
| 三路相机缓存读取 | `snapshot_camera_frames()` | 读取最新帧并检查帧年龄 |
| target 落盘单位转换 | `action_to_record_row()` | 臂保持 rad；夹爪 motor-rad→normalized |
| feedback 落盘单位转换 | `state_to_record_row()` | 臂 deg→rad；夹爪 motor-rad→normalized |
| 相机落盘 | `ReplayEpisodeRecorder.write_camera_samples()` | JPEG 与 `frames.csv` |
| 机械臂落盘 | `ReplayEpisodeRecorder.write_arm_sample()` | timestamp/action/state CSV |
| 完整样本提交 | `ReplayEpisodeRecorder.finish_sample()` | 两侧落盘成功后增加 `num_frames` |

单帧下发的数据流为：

```text
action float32[16]
  -> validate_dispatch_action
  -> action_to_control_action
  -> dispatch_action / robot.send_action
  -> A/B 双臂批量提交
  -> 左、右 RS05 下发
```

同步数采的数据流与下发分开：

```text
read_robot_feedback
snapshot_camera_frames
write_camera_samples
write_arm_sample
finish_sample
```

## 8. 模型慢速插值下发接口

如果模型动作不应直接跳到目标位置，使用
`runtime/slow_dispatch.py` 中的 `SlowInterpolatedDispatcher`。它接收模型返回的
单个 16D 绝对动作，在每次下发前读取实时反馈，从当前反馈位置开始，对双臂和双夹爪
同时插值后再调用 `robot.send_action()`。

当前默认推理入口也支持通过 `configs/infer.yaml` 开关控制：

```yaml
slow_dispatch_enabled: false
```

保持 `false` 时，继续使用原有下发行为；改为 `true` 后，
`infer_tianji_gripper.py` 才会使用 `slow_dispatch_duration_sec`、
`slow_dispatch_frequency_hz` 和 `slow_dispatch_interpolation_mode`，并对双臂、双夹爪
同时进行慢速插值。

该接口是同步接口：一次 `dispatch()` 完成后才返回，模型调用方不能在同一个
`SlowInterpolatedDispatcher` 实例上并发发送两个动作。

### 8.1 Python 调用

```python
import numpy as np

from tianji_gripper_runtime.runtime.action_adapter import ActionAdapter
from tianji_gripper_runtime.runtime.safety import SafetyConfig, SafetyLayer
from tianji_gripper_runtime.runtime.slow_dispatch import SlowInterpolatedDispatcher


adapter = ActionAdapter(
    policy_arm_unit="rad",
    control_arm_unit="deg",
    policy_gripper_unit="rad",
    control_gripper_unit="rad",
    control_mode="dual_arm_dual_gripper",
)
safety = SafetyLayer(
    SafetyConfig.from_yaml(
        "preinfere/tianji_gripper_runtime/configs/robot_limits.yaml"
    ),
    adapter,
)
dispatcher = SlowInterpolatedDispatcher.from_yaml(
    robot,
    adapter=adapter,
    safety=safety,
    config_path="preinfere/tianji_gripper_runtime/configs/infer.yaml",
)

# 模型动作顺序：右臂 7、左臂 7、右夹爪 1、左夹爪 1；关节和夹爪均为 rad。
model_action = np.asarray(action_from_model, dtype=np.float32).reshape(16)
final_target = dispatcher.dispatch(
    model_action,
    stop_callback=lambda: operator_requested_stop,
)
```

`duration_sec` 和 `frequency_hz` 来自 `configs/infer.yaml` 中的
`slow_dispatch_duration_sec`、`slow_dispatch_frequency_hz`，分别表示一个模型动作的
总时间和插值下发频率。当前 YAML 配置会生成 `ceil(1.0 * 20.0) = 20` 个子目标；
修改 `slow_dispatch_interpolation_mode` 为 `linear` 可切换为线性插值。

模型返回动作序列时，可以按顺序逐个慢速执行：

```python
final_targets = dispatcher.dispatch_chunk(
    action_chunk_from_model,  # float32[H, 16]
)
```

### 8.2 B/b 固定复位接口

推理主循环将 `B/b` 映射为 `dispatch_control_action()`，用于直接下发已经处于机器人
控制单位的固定目标。它不经过模型动作归一化转换，也不经过模型动作的单步裁剪；仍然
使用 `infer.yaml` 中的 `slow_dispatch_duration_sec`、`slow_dispatch_frequency_hz` 和
`slow_dispatch_interpolation_mode` 做同步插值。

```python
reset_target = RightArmGripperAction(
    right_arm_q=np.asarray([-140, -90, 90, -120, 0, 0, 0], dtype=np.float32),
    left_arm_q=np.asarray([140, -90, -90, -120, 0, 0, 0], dtype=np.float32),
    right_gripper_q=np.asarray([-5.2817], dtype=np.float32),
    left_gripper_q=np.asarray([-5.0616], dtype=np.float32),
    control_left_arm=True,
    control_left_gripper=True,
)
dispatcher.dispatch_control_action(reset_target)
```

这里的“夹爪1”对应左/A 侧 `left_gripper_q`， “夹爪2”对应右/B 侧
`right_gripper_q`。复位完成后运行状态自动变为暂停，按 `R/r` 才会继续模型推理。

### 8.3 安全和停止语义

- `dispatch()` 每次以新的双臂/双夹爪实时反馈作为插值起点，不使用可能过期的模型目标。
- 传入 `SafetyLayer` 后，模型目标会先经过有限值、关节限位、夹爪限位、单步变化和速度检查，再开始插值。生产环境不要省略 `safety`。
- 插值期间 `stop_callback()` 返回 `True` 时，接口立即调用 `robot.hold_position()`，并抛出 `SlowDispatchCancelledError`；调用方应停止消费后续模型动作。
- `robot.send_action()` 仍负责同一子步内的 A/B 双臂批量提交，以及 A/B 两侧 RS05 夹爪报文发送；模型侧不应直接调用底层 SDK。
- `duration_sec=0` 只用于测试，会退化为一次目标下发；真实机器人应使用正数并先在 `dry_run` 或假设备上验证。

实现位置：`runtime/slow_dispatch.py`。当前接口默认使用控制单位 `deg` 发送双臂、
使用 `rad` 发送夹爪；模型输入仍保持本文件第 1 节定义的 16D `rad` 顺序。
