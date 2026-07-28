# Tianji Right-Arm Gripper Runtime

This runtime can talk to the original GR00T policy server or an openpi/pi0.5 websocket
policy server. The pi0.5 checkpoint in
`/mnt/efs_1/ziyi/preinfere/openpi_inference_bottle_ckpt20000.tar.gz` is documented by
`openpi/src/openpi/policies/ziyi_policy.py` as:

```text
state/action: right_arm(7) + left_arm(7) + right_gripper(1)
video: head + left_wrist + right_wrist
language: prompt
```

The robot may still be a dual-arm platform. During `connect()`, the runtime reads the left arm
and left gripper initial positions and keeps sending those freeze targets while the policy controls
only the right side in `right_arm_right_gripper` mode.

## Start pi0.5 Policy Server

Extract the openpi tarball somewhere outside this runtime, then start the websocket server:

```bash
cd /path/to/extracted/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_ziyi_dualarm \
  --policy.dir /path/to/checkpoint/20000
```

Use the actual config name and checkpoint directory from the extracted training bundle if they
differ. This runtime only needs the server endpoint.

## Control Mode

The default config uses:

```yaml
policy_type: pi
control_mode: right_arm_right_gripper
pi_action_layout: ziyi_15d_right_left_right_gripper
```

In this mode the policy still receives the full 15D training state, but the executor sends only:

```text
right_arm = actions[0:7]
right_gripper = actions[14:15]
```

`dual_arm_dual_gripper` is exposed as a command-line choice, but this checkpoint does not contain
a left-gripper action. The runtime refuses that mode for the ziyi 15D layout instead of sending an
ambiguous command.

## Start GR00T Policy Server

Use the existing GR00T server launcher with the checkpoint trained from
`demo_data/shuiping2026_6_16_lerobot/gripper_n1d7_config.py`:

```bash
cd /home/user/GR00T/tianji_wuji_runtime
CKPT=/path/to/checkpoint EMBODIMENT_TAG=NEW_EMBODIMENT CUDA_VISIBLE_DEVICES=0 \
bash deployment/start_groot_server.sh
```

## Dry Run

The default config is safe: `robot_backend: fake`, `gripper_backend: fake`, and `dry_run: true`.

```bash
cd /home/user/GR00T
uv run python tianji_gripper_runtime/deployment/infer_tianji_gripper.py \
  --config tianji_gripper_runtime/configs/infer.yaml
```

## Hardware Switches

For the Tianji arms, set:

```yaml
robot_backend: tianji_gripper_host
dry_run: false
```

For the gripper, the SDK boundary is
`tianji_gripper_runtime/runtime/gripper_interface.py`.

Until the vendor SDK shape is known, keep:

```yaml
left_gripper_backend: fake
right_gripper_backend: fake
```

Once the right gripper SDK is available, wire it through either:

```yaml
right_gripper_backend: generic_sdk
gripper_sdk_module: your_sdk_module
gripper_sdk_class: YourGripperClass
```

or implement a small `GripperInterface` subclass if the vendor API needs custom handling.

## 模型动作慢速插值下发接口

### 1. 默认推理入口开关

当前推理入口为 `preinfere/tianji_gripper_runtime/deployment/infer_tianji_gripper.py`。
慢速插值由 `configs/infer.yaml` 控制，默认关闭：

```yaml
# false：保持原有快速下发行为；true：启用双臂、双夹爪慢速插值。
slow_dispatch_enabled: false
# 每个模型动作从当前反馈位置移动到目标位置的总时间，单位：秒。
slow_dispatch_duration_sec: 1.0
# 插值子目标下发频率，单位：Hz。
slow_dispatch_frequency_hz: 20.0
# 可选：cubic（平滑）或 linear（线性）。
slow_dispatch_interpolation_mode: cubic
```

设置为慢速模式：

```yaml
slow_dispatch_enabled: true
```

修改后重启推理进程即可生效。慢速模式下，每个模型动作会生成：

```text
子目标数量 = ceil(slow_dispatch_duration_sec * slow_dispatch_frequency_hz)
```

例如 `1.0 s` 和 `20 Hz` 会下发 `20` 个子目标。动作序列中的每个模型动作会按顺序
执行，因此模型动作序列长度为 `H` 时，最少需要约 `H * duration_sec` 秒；如果模型
持续输出动作，建议先设置较小的 `execution_horizon`，并在假机器人或 `dry_run` 下验证。

当 `slow_dispatch_enabled: false` 时，以上三个慢速参数不会改变默认执行行为。

### 1.2 键盘控制和 B 复位

推理进程启动后，键盘控制如下：

- `R/r`：开始或继续模型推理。
- `E/e`：暂停模型推理；`P/p` 仍作为兼容的暂停按键。
- `B/b`：使用上面的慢速插值参数，将双臂和双夹爪移动到固定复位位置；复位完成后保持暂停，需再次按 `R/r` 才恢复模型推理。
- `Space`：停止下发并保持当前位置；`Q/q`：退出。

`B/b` 仅适用于 `--control-mode dual_arm_dual_gripper`，复位目标使用机器人控制单位：

复位角度可直接在 `configs/infer.yaml` 的 `reset_left_arm_q`、`reset_right_arm_q`、
`reset_gripper1_motor_rad`、`reset_gripper2_motor_rad` 中修改，重启推理进程后生效。

```text
A/左臂（fb，度）：[140, -90, -90, -120, 0, 0, 0]
B/右臂（fb，度）：[-140, -90, 90, -120, 0, 0, 0]
夹爪1/左夹爪：-5.0616 rad
夹爪2/右夹爪：-5.2817 rad
```

复位直接调用 `runtime/slow_dispatch.py` 的 `dispatch_control_action()`，不会把这些
控制角度再次当成模型归一化动作转换，也不会被模型动作的步长裁剪改变。

### 1.1 RS05 夹爪反馈力矩保护

`configs/infer.yaml` 中的 `gripper_torque_protection_enabled` 默认是 `false`，因此不会改变
现有夹爪下发行为。需要启用时改为：

```yaml
gripper_torque_protection_enabled: true
```

启用后，RS05 每次收到新的反馈帧都会对 `torque_nm` 做低通滤波；夹爪仍在闭合且连续
`gripper_torque_count_threshold` 帧超过 `gripper_torque_threshold_nm` 后，会判定已经夹到物体。
此时目标会限制在当前反馈位置加上 `gripper_torque_extra_tighten_rad`，并切换到
`gripper_holding_kp` / `gripper_holding_kd`，避免模型继续把夹爪向物体挤压。张开命令会立即
解除保护；未继续闭合且力矩低于 `gripper_torque_release_threshold_nm` 时也会解除。

当前标定下，增大 rad 表示闭合，所以 `gripper_closing_direction: 1.0`。如果夹爪实际方向相反，
必须改为 `-1.0` 后再启用。建议先以默认阈值 `1.0 Nm`、连续 `5` 帧在空载和常用物体上测试，
确认不会误触发后再用于正式推理。

### 1.2 夹爪归一化标定

模型部署现在与数采仓库使用相同的夹爪数据约定：模型 observation/action 中夹爪值为
`[0, 1]`，张开为 `0`、闭合为 `1`；机器人实际反馈和下发仍使用电机角度 `rad`。
转换公式为：

```text
normalized = clamp((open_motor_rad - motor_rad) / (open_motor_rad - close_motor_rad), 0, 1)
motor_rad = open_motor_rad - normalized * (open_motor_rad - close_motor_rad)
```

在 `configs/infer.yaml` 中分别填写与数采 YAML 相同的四个值：

```yaml
gripper_close_motor_rad:       # 左夹爪 A，复制 gripper.close_motor_rad
gripper_open_motor_rad:        # 左夹爪 A，复制 gripper.open_motor_rad
gripper2_close_motor_rad:      # 右夹爪 B，复制 gripper2.close_motor_rad
gripper2_open_motor_rad:       # 右夹爪 B，复制 gripper2.open_motor_rad
```

四个值为空时，使用归一化模型启动会直接报错，避免把 rad 值误当成归一化值发送。

### 1.3 state=3 后的关节阻抗 K/D

部署连接 Tianji/Marvin 机械臂时，`configs/infer.yaml` 中的
`tianji_state3_joint_k_a/d_a` 和 `tianji_state3_joint_k_b/d_b` 会在初始化阶段按以下顺序
批量下发：

```text
state=3 → impedance_type=1 → set_joint_kd_params(A/B) → set_vel_acc(A/B)
```

参数顺序均为 J1~J7。A 臂使用 `[6,6,6,4,4,3,3]` 和 `[1,0.4,0.4,0.4,0,0,0]`；
B 臂使用 `[6,6,6,4,3,3,3]` 和 `[1,0.4,0.4,0.4,0,0,0]`，与数采仓库的
`marvin_teach_runtime.py:5674` 逻辑一致。

### 2. 模型动作输入格式

双臂双夹爪模式下，接口接收模型输出的单个 `float32[16]` 绝对位置动作，机械臂单位为
`rad`、夹爪单位为 normalized（张开 `0`、闭合 `1`），顺序固定如下：

| 下标 | 字段 | 天机硬件 | 模型单位 | SDK 下发单位 |
| --- | --- | --- | --- | --- |
| `0:7` | `right_arm_q` | B 臂右臂 7 关节 | `rad` | `deg` |
| `7:14` | `left_arm_q` | A 臂左臂 7 关节 | `rad` | `deg` |
| `14` | `right_gripper_q` | B 臂右夹爪 | normalized | `rad` |
| `15` | `left_gripper_q` | A 臂左夹爪 | normalized | `rad` |

这是绝对目标位置，不是增量、速度或力矩。模型动作不能直接发送给天机 SDK，必须先
经过动作拆分、单位转换、安全检查和插值。

### 3. Python 接口

需要直接在模型控制代码中使用时，调用 `SlowInterpolatedDispatcher.from_yaml()`：

```python
from pathlib import Path

import numpy as np

from tianji_gripper_runtime.runtime.action_adapter import ActionAdapter
from tianji_gripper_runtime.runtime.safety import SafetyConfig, SafetyLayer
from tianji_gripper_runtime.runtime.slow_dispatch import (
    SlowDispatchCancelledError,
    SlowInterpolatedDispatcher,
)


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
    robot,  # 已连接的 RightArmGripperRobot
    adapter=adapter,
    safety=safety,
    config_path=Path(
        "preinfere/tianji_gripper_runtime/configs/infer.yaml"
    ),
)

model_action = np.asarray(action_from_model, dtype=np.float32).reshape(16)
try:
    final_target = dispatcher.dispatch(
        model_action,
        stop_callback=lambda: operator_requested_stop,
    )
except SlowDispatchCancelledError:
    print("slow dispatch stopped; robot is holding its position")
```

`dispatch()` 的执行过程是：

1. 读取当前双臂和双夹爪反馈，作为插值起点。
2. 校验模型动作，并通过 `SafetyLayer` 做关节限位、夹爪限位、单步变化和速度检查。
3. 按 YAML 中的时间和频率生成插值子目标。
4. 每个子目标同时批量下发 A/B 双臂，并发送两侧 RS05 夹爪位置。
5. 完成后返回最终安全目标；停止回调触发时保持当前位置并抛出
   `SlowDispatchCancelledError`。

模型返回动作序列时，可以顺序执行：

```python
final_targets = dispatcher.dispatch_chunk(action_chunk_from_model)  # float32[H, 16]
```

`dispatch_chunk()` 是同步执行的，不会并行发送动作。生产环境必须保留 `SafetyLayer`，
不要绕过 `RightArmGripperRobot.send_action()` 直接调用底层 SDK。完整的硬件映射和安全
约束见 `TIANJI_DISPATCH_INTERFACE.md`。

## Launch From This Repo

```bash
cd /home/user/workspace/TJ-gripper_infer_0727new
GIT_LFS_SKIP_SMUDGE=1 uv sync
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_ziyi_dualarm \
  --policy.dir /path/to/checkpoint/blue_bottle/30000
```

Open another terminal for right-arm right-gripper inference:

Use `uv run python` here. A plain `python` may resolve to the active Conda environment, where
the websocket client dependencies are not installed.

```bash
cd /home/user/workspace/TJ-gripper_infer_0727new
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH

uv run python preinfere/tianji_gripper_runtime/deployment/infer_tianji_gripper.py \
  --config preinfere/tianji_gripper_runtime/configs/infer.yaml \
  --policy-type pi \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --control-mode right_arm_right_gripper \
  --task "pick and place the bottle"
```

解压并启动 pi0.5 server
mkdir -p /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle
tar -xzf /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle_ckpt20000.tar.gz \
  -C /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle

cd /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_ziyi_dualarm \
  --policy.dir /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle/openpi/checkpoints/pi05_ziyi_dualarm/20000

另开一个终端启动右臂右夹爪推理
cd /mnt/efs_1/ziyi/preinfere/openpi_inference_bottle/openpi
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PYTHONPATH

uv run python /mnt/efs_1/ziyi/preinfere/tianji_gripper_runtime/deployment/infer_tianji_gripper.py \
  --config /mnt/efs_1/ziyi/preinfere/tianji_gripper_runtime/configs/infer.yaml \
  --policy-type pi \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --control-mode right_arm_right_gripper \
  --task "pick and place the bottle"
