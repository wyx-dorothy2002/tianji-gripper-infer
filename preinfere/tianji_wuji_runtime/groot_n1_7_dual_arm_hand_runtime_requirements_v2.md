# GR00T N1.7 双臂双灵巧手 Runtime Inference 工程清单

> 适用范围：本清单只覆盖 **推理 / 运行阶段**。  
> 不覆盖训练、数据采集、数据转换、fine-tune 参数搜索、Isaac Sim / Isaac Lab、RL 等训练前或训练后扩展内容。  
> 当前目标是：已有一个 GR00T N1.7 checkpoint，需要把 policy 接入你们自己的双臂双手控制端。

---

## 1. 当前任务定义

当前机器人形态为：

```text
双臂 + 双灵巧手
= 2 × (7 DoF 机械臂 + 20 DoF 灵巧手)
= 54 DoF state/action
```

Runtime 中必须统一：

```python
LEFT_ARM_DOF = 7
LEFT_HAND_DOF = 20
RIGHT_ARM_DOF = 7
RIGHT_HAND_DOF = 20

STATE_DIM = 54
ACTION_DIM = 54
```

推荐默认顺序：

```text
state/action[0:7]    = left_arm  7 joints
state/action[7:27]   = left_hand 20 joints
state/action[27:34]  = right_arm 7 joints
state/action[34:54]  = right_hand 20 joints
```

也可以采用：

```text
right_arm + right_hand + left_arm + left_hand
```

但必须满足：

```text
推理顺序 = 训练 checkpoint 的 state/action 顺序
```

这是整个系统最关键的工程约束。  
如果 54 维顺序错了，policy 输出会被错误发送到机械臂或手指关节，后续所有安全检查和控制都会失效。

---

## 2. Runtime 总体架构

推荐使用 **server-client 架构**：

```text
Terminal 1:
  GR00T policy server
  - 加载 checkpoint
  - 占用本地 4090 GPU
  - 只负责模型推理

Terminal 2:
  robot inference client
  - 连接相机
  - 读取双臂双手 state
  - 构造 GR00T observation
  - 请求 action chunk
  - 执行 action adapter
  - 执行 safety layer
  - 发送到控制端
  - 记录日志
```

数据流：

```text
Real Cameras + Robot State
        |
        v
infer_tianji_wuji.py
        |
        | observation
        v
GR00T N1.7 Policy Server on 4090
        |
        | action chunk [H, 54]
        v
Action Adapter
        |
        | split
        v
left_arm[7], left_hand[20], right_arm[7], right_hand[20]
        |
        v
Safety Layer
        |
        v
Hardware Control Backend
```

---

## 3. 推荐代码目录结构

这里保留 `deployment/` 下的四个入口脚本，不应该省略。

```text
tianji_wuji_runtime/
├── deployment/
│   ├── start_groot_server.sh
│   ├── infer_tianji_wuji.py
│   ├── dry_run_infer.py
│   └── replay_policy_check.py
│
├── runtime/
│   ├── schema.py
│   ├── robot_interface.py
│   ├── arm_interface.py
│   ├── hand_interface.py
│   ├── camera_manager.py
│   ├── observation_builder.py
│   ├── groot_policy_client.py
│   ├── action_adapter.py
│   ├── safety.py
│   ├── executor.py
│   ├── keyboard.py
│   └── recorder.py
│
├── configs/
│   ├── runtime_config.yaml
│   ├── camera_config.yaml
│   ├── robot_limits.yaml
│   └── action_schema.yaml
│
├── tools/
│   ├── inspect_checkpoint_io.py
│   ├── check_camera_slots.py
│   ├── check_robot_state.py
│   └── plot_commanded_vs_actual.py
│
├── logs/
│   └── .gitkeep
│
└── README.md
```

---

## 4. deployment 入口脚本职责

## 4.1 `start_groot_server.sh`

作用：启动 GR00T N1.7 policy server。

职责：

```text
1. 设置 Isaac-GR00T 仓库路径。
2. 设置 checkpoint 路径。
3. 设置 embodiment tag。
4. 设置 CUDA_VISIBLE_DEVICES。
5. 启动 gr00t/eval/run_gr00t_server.py。
6. 打印 server host、port、checkpoint、embodiment tag。
```

示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /path/to/Isaac-GR00T

CKPT=/path/to/your/checkpoint
EMBODIMENT_TAG=YOUR_DUAL_ARM_HAND_TAG

CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 uv run python   gr00t/eval/run_gr00t_server.py   --model-path "$CKPT"   --embodiment-tag "$EMBODIMENT_TAG"   --host 127.0.0.1   --port 5555
```

要求：

```text
1. server 端只负责模型推理。
2. server 端不连接机器人、不连接相机。
3. checkpoint 和 embodiment tag 不匹配时不得继续运行。
4. server 日志必须可追踪当前加载的 checkpoint。
```

---

## 4.2 `infer_tianji_wuji.py`

作用：真实 runtime 主入口。

职责：

```text
1. 解析 CLI 参数。
2. 连接 policy server。
3. 连接相机。
4. 连接双臂双手控制接口。
5. 构造 observation。
6. 请求 action chunk [H, 54]。
7. 调用 ActionAdapter 拆分动作。
8. 调用 SafetyLayer 做安全处理。
9. 调用 ActionExecutor 执行或 dry-run。
10. 调用 Recorder 保存日志。
11. 响应键盘急停、暂停、继续、退出。
```

推荐启动：

```bash
python deployment/infer_tianji_wuji.py   --policy-host 127.0.0.1   --policy-port 5555   --task "pick up the object"   --execution-horizon 8   --duration 0.1   --safe-mode   --record-dir ./infer_logs   --camera ego_view:0
```

必须支持：

```text
--policy-host
--policy-port
--task
--execution-horizon
--duration
--safe-mode
--dry-run
--record-dir
--camera
--left-arm-port / --left-arm-ip
--right-arm-port / --right-arm-ip
--left-hand-port / --left-hand-ip
--right-hand-port / --right-hand-ip
--max-arm-joint-step
--max-hand-joint-step
--max-arm-velocity
--max-hand-velocity
--use-filter
--no-keyboard
--save-video
```

---

## 4.3 `dry_run_infer.py`

作用：只测试 GR00T policy 输入输出，不发送任何真实控制命令。

职责：

```text
1. 连接 policy server。
2. 使用 fake robot state 或真实 state。
3. 使用本地图片、相机图像或 dummy image。
4. 构造 observation。
5. 获取 action chunk [H, 54]。
6. 检查 shape、NaN/Inf、数值范围。
7. 调用 ActionAdapter 拆分。
8. 调用 SafetyLayer 处理。
9. 保存 input/output 日志。
10. 不调用 robot.send_action。
```

推荐启动：

```bash
python deployment/dry_run_infer.py   --policy-host 127.0.0.1   --policy-port 5555   --task "pick up the object"   --state-source fake   --image-source ./assets/test_ego_view.png   --record-dir ./infer_logs/dry_run
```

验收标准：

```text
1. 能成功连上 policy server。
2. 能得到 action chunk。
3. action chunk shape == [H, 54]。
4. ActionAdapter 能拆成 left_arm[7]、left_hand[20]、right_arm[7]、right_hand[20]。
5. 全流程不触发真实机器人动作。
6. 日志可用于离线分析。
```

---

## 4.4 `replay_policy_check.py`

作用：用于检查 action schema、执行频率、控制端解释是否正确。  
它不一定调用 GR00T 模型，可以 replay 已保存的 action chunk 或一段测试轨迹。

职责：

```text
1. 加载一段 saved raw_action.npy 或 dataset action。
2. 检查 action shape == [H, 54]。
3. 调用 ActionAdapter 拆分。
4. 可选择 dry-run 打印。
5. 可选择 safe-mode 单步发送到硬件。
6. 记录 commanded vs actual。
7. 生成 plot。
```

推荐启动：

```bash
python deployment/replay_policy_check.py   --action-file ./infer_logs/run_xxx/chunks/chunk_000000/output/raw_action.npy   --duration 0.1   --safe-mode   --dry-run
```

用途：

```text
1. 排查左右臂顺序是否反了。
2. 排查左右手顺序是否反了。
3. 排查 rad / degree / encoder 单位是否错误。
4. 排查 absolute / delta action 是否解释错误。
5. 在不运行模型的情况下测试控制端。
```

---

## 5. `schema.py`：54 维 schema 定义

文件：

```text
runtime/schema.py
```

### 5.1 必须统一维护的变量

```python
LEFT_ARM_DOF = 7
LEFT_HAND_DOF = 20
RIGHT_ARM_DOF = 7
RIGHT_HAND_DOF = 20

STATE_DIM = 54
ACTION_DIM = 54

LEFT_ARM_SLICE = slice(0, 7)
LEFT_HAND_SLICE = slice(7, 27)
RIGHT_ARM_SLICE = slice(27, 34)
RIGHT_HAND_SLICE = slice(34, 54)

ACTION_SCHEMA_VERSION = "dual_arm_wuji_v1"
```

### 5.2 推荐 key 定义

```python
LEFT_ARM_KEYS = [
    "left_arm_joint_1.pos",
    "left_arm_joint_2.pos",
    "left_arm_joint_3.pos",
    "left_arm_joint_4.pos",
    "left_arm_joint_5.pos",
    "left_arm_joint_6.pos",
    "left_arm_joint_7.pos",
]

LEFT_HAND_KEYS = [
    "left_hand_joint_1.pos",
    ...
    "left_hand_joint_20.pos",
]

RIGHT_ARM_KEYS = [
    "right_arm_joint_1.pos",
    ...
    "right_arm_joint_7.pos",
]

RIGHT_HAND_KEYS = [
    "right_hand_joint_1.pos",
    ...
    "right_hand_joint_20.pos",
]

STATE_KEYS = LEFT_ARM_KEYS + LEFT_HAND_KEYS + RIGHT_ARM_KEYS + RIGHT_HAND_KEYS
ACTION_KEYS = STATE_KEYS
```

### 5.3 工程要求

```text
1. 所有模块必须从 schema.py 引入 slice 和 key。
2. 禁止在主循环中手写 action[:7]、action[7:27] 等切片。
3. 禁止在多个文件中重复定义 54 维顺序。
4. schema.py 中需要写明 action_order_version。
5. schema.py 中需要写明单位：rad / degree / encoder / normalized。
6. 若训练 checkpoint 的顺序不是上述推荐顺序，只允许修改 schema.py，不允许修改业务逻辑。
```

---

## 6. RobotInterface：双臂双手统一接口

文件：

```text
runtime/robot_interface.py
```

### 6.1 统一接口

```python
class DualArmHandRobot:
    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def get_state(self) -> np.ndarray:
        """
        Return:
            state: np.ndarray, shape=(54,)
            order follows runtime/schema.py.
        """
        ...

    def send_action(self, action: "DualArmHandAction") -> None:
        ...

    def hold_position(self) -> None:
        ...

    def go_home(self) -> None:
        ...

    def is_connected(self) -> bool:
        ...
```

### 6.2 硬性要求

```text
1. get_state() 必须返回 shape=(54,)。
2. state 顺序必须和 schema.py 一致。
3. send_action() 输入必须是 DualArmHandAction，不允许裸 list。
4. 左臂、右臂、左手、右手任意一侧发送失败，必须触发 hold_position()。
5. 主循环不得直接调用底层机械臂或灵巧手 SDK。
6. 所有 SDK 异常必须包装成统一 RobotError。
```

---

## 7. ArmInterface 与 HandInterface

## 7.1 ArmInterface

文件：

```text
runtime/arm_interface.py
```

硬件人员需要实现：

```python
class ArmInterface:
    def connect(self) -> None:
        ...

    def get_joint_state(self) -> np.ndarray:
        """Return 7-dim joint position."""
        ...

    def send_joint_position(self, q: np.ndarray) -> None:
        ...

    def hold_position(self) -> None:
        ...

    def go_home(self) -> None:
        ...
```

硬件人员需要提供：

```text
1. 7 个关节顺序。
2. 单位：rad / degree / encoder。
3. joint min / max。
4. velocity limit。
5. max single-step delta。
6. 控制频率。
7. send_joint_position 接收 absolute target 还是 delta。
```

## 7.2 HandInterface

文件：

```text
runtime/hand_interface.py
```

硬件人员需要实现：

```python
class HandInterface:
    def connect(self) -> None:
        ...

    def get_joint_state(self) -> np.ndarray:
        """Return 20-dim joint position."""
        ...

    def send_joint_position(self, q: np.ndarray) -> None:
        ...

    def hold_position(self) -> None:
        ...
```

硬件人员需要提供：

```text
1. 20 个关节顺序。
2. 每个关节含义。
3. 单位：rad / degree / encoder / normalized。
4. joint min / max。
5. 开合方向。
6. 是否存在 mimic / coupled joints。
7. 控制频率。
8. max single-step delta。
```

---

## 8. CameraManager：相机接口

文件：

```text
runtime/camera_manager.py
```

接口：

```python
class CameraManager:
    def connect_all(self) -> None:
        ...

    def read(self) -> dict[str, np.ndarray]:
        ...

    def disconnect_all(self) -> None:
        ...
```

要求：

```text
1. camera key 必须和 checkpoint 训练时 image key 一致。
2. 输出图像必须是 RGB。
3. 禁止 BGR 直接传给 policy。
4. 图像尺寸必须能被 GR00T preprocess 正确处理。
5. 相机 timeout 必须抛出 CameraError。
6. CameraError 触发 robot.hold_position()。
7. 支持 camera serial / index 配置。
8. 支持 flip / rotate。
```

示例：

```text
训练时 image key:
  observation.images.ego_view

推理时必须提供:
  images["ego_view"]
```

---

## 9. ObservationBuilder：构造 GR00T observation

文件：

```text
runtime/observation_builder.py
```

接口：

```python
class ObservationBuilder:
    def build(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
    ) -> dict:
        ...
```

必须检查：

```text
1. state.shape == (54,)
2. state 全部 finite
3. images 包含所有 required camera slots
4. image 不为 None
5. image 是 RGB
6. task 非空
7. observation 字段名和 GR00T policy server 预期一致
```

注意事项：

```text
1. ObservationBuilder 不负责机器人控制。
2. ObservationBuilder 不负责模型推理。
3. ObservationBuilder 只负责把硬件数据转为 GR00T observation。
4. 如果 server API 需要特殊字段，由 GrootPolicyClient 做最终适配。
```

---

## 10. GrootPolicyClient：GR00T client 封装

文件：

```text
runtime/groot_policy_client.py
```

接口：

```python
class GrootPolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 15000):
        ...

    def ping(self) -> bool:
        ...

    def reset(self) -> None:
        ...

    def predict_action_chunk(self, observation: dict) -> np.ndarray:
        """
        Return:
            action_chunk: np.ndarray, shape=(H, 54)
        """
        ...
```

必须实现的检查：

```text
1. 连接 server 失败时，不允许进入 RUNNING。
2. action_chunk 必须是 np.ndarray。
3. action_chunk.ndim == 2。
4. action_chunk.shape[1] == 54。
5. action_chunk 不允许包含 NaN/Inf。
6. action_chunk length 必须 >= execution_horizon。
7. 记录 inference latency。
8. 支持 server timeout。
9. server timeout 触发 robot.hold_position()。
```

---

## 11. ActionAdapter：动作格式转换

文件：

```text
runtime/action_adapter.py
```

### 11.1 数据结构

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class DualArmHandAction:
    left_arm_q: np.ndarray    # shape=(7,)
    left_hand_q: np.ndarray   # shape=(20,)
    right_arm_q: np.ndarray   # shape=(7,)
    right_hand_q: np.ndarray  # shape=(20,)
```

### 11.2 接口

```python
class ActionAdapter:
    def split_action(self, action_54: np.ndarray) -> DualArmHandAction:
        ...

    def split_chunk(self, action_chunk: np.ndarray) -> list[DualArmHandAction]:
        ...

    def merge_action(self, action: DualArmHandAction) -> np.ndarray:
        ...
```

### 11.3 强制要求

```text
1. 所有 action split 只能在 ActionAdapter 中完成。
2. 主循环中禁止直接写 action[:7] / action[7:27]。
3. 必须检查 action_54.shape == (54,)。
4. 必须检查 left_arm_q.shape == (7,)。
5. 必须检查 left_hand_q.shape == (20,)。
6. 必须检查 right_arm_q.shape == (7,)。
7. 必须检查 right_hand_q.shape == (20,)。
8. 必须支持单位转换。
9. 必须明确 action 是 absolute joint target 还是 delta joint。
10. 必须记录 action schema version。
```

### 11.4 Delta action 情况

如果 checkpoint 输出 delta action，则必须在 ActionAdapter 或 SafetyLayer 中转换：

```text
target = current_state + delta
```

禁止把 delta action 直接发送给 position controller。

---

## 12. SafetyLayer：软件安全层

文件：

```text
runtime/safety.py
```

### 12.1 SafetyConfig

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class SafetyConfig:
    left_arm_joint_min: np.ndarray
    left_arm_joint_max: np.ndarray
    left_hand_joint_min: np.ndarray
    left_hand_joint_max: np.ndarray

    right_arm_joint_min: np.ndarray
    right_arm_joint_max: np.ndarray
    right_hand_joint_min: np.ndarray
    right_hand_joint_max: np.ndarray

    arm_max_step: float
    hand_max_step: float

    arm_max_velocity: np.ndarray | None = None
    hand_max_velocity: np.ndarray | None = None

    enable_joint_limit: bool = True
    enable_delta_clip: bool = True
    enable_velocity_limit: bool = True
    enable_filter: bool = False
```

### 12.2 必须实现

```python
def check_finite(action: DualArmHandAction) -> None:
    ...

def clamp_joint_limits(action: DualArmHandAction) -> DualArmHandAction:
    ...

def clip_delta(
    current_state: np.ndarray,
    action: DualArmHandAction,
) -> DualArmHandAction:
    ...

def limit_velocity(
    previous_action: DualArmHandAction,
    action: DualArmHandAction,
    dt: float,
) -> DualArmHandAction:
    ...

def process_chunk(
    current_state: np.ndarray,
    actions: list[DualArmHandAction],
    dt: float,
) -> tuple[list[DualArmHandAction], list[dict]]:
    ...
```

### 12.3 安全逻辑要求

```text
1. NaN/Inf action：直接拒绝执行，进入 STOPPED。
2. 超过 joint limit：clamp，并记录 safety event。
3. 单步 delta 超限：clip，并记录 safety event。
4. arm 和 hand 使用不同 max_step。
5. 左侧和右侧关节限位分别配置。
6. velocity 超限：clip 或拒绝执行。
7. 连续多次触发 safety event：进入 STOPPED。
8. 相机失败：hold_position。
9. policy server 失败：hold_position。
10. robot SDK 失败：hold_position。
11. 程序退出前：hold_position。
```

---

## 13. ActionExecutor：固定周期执行 action chunk

文件：

```text
runtime/executor.py
```

接口：

```python
class ActionExecutor:
    def execute_chunk(
        self,
        actions: list[DualArmHandAction],
        dt: float,
        dry_run: bool = False,
    ) -> None:
        ...
```

执行逻辑：

```text
for each action step:
  1. 记录 step_start。
  2. 如果 dry-run，只记录，不发送硬件。
  3. 发送 left_arm command。
  4. 发送 left_hand command。
  5. 发送 right_arm command。
  6. 发送 right_hand command。
  7. 读取 actual state，可选。
  8. 记录 latency。
  9. precise_sleep(dt - elapsed)。
  10. 如果 elapsed > dt，输出 overrun warning。
```

要求：

```text
1. 左臂、左手、右臂、右手尽量同步发送。
2. 任意一侧发送失败，立即 hold_position。
3. execution_horizon 可以小于 policy chunk length。
4. 每步记录 commanded action 和 actual state。
5. 支持 dry-run。
6. 支持 safe-mode。
```

---

## 14. Keyboard State Machine：运行状态机

文件：

```text
runtime/keyboard.py
```

状态：

```text
STOPPED
RUNNING
PAUSED
ERROR
```

快捷键：

```text
Space: 急停 / hold_position
R: 开始 policy running
P: 暂停
H: 回 home
Q: 退出
N: safe-mode 下执行下一段 chunk
S: 开始记录
D: 停止记录
```

行为要求：

```text
1. 程序启动后默认 STOPPED。
2. 未按 R 前不执行任何 policy action。
3. Space 在任何状态下立即生效。
4. Q 退出前必须 hold_position。
5. ERROR 状态不允许继续执行。
6. safe-mode 下每段 chunk 执行后自动暂停，等待 N。
7. no-keyboard 模式只允许在明确配置下运行。
```

---

## 15. Recorder / Logger：运行日志

文件：

```text
runtime/recorder.py
```

日志目录结构：

```text
infer_logs/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── chunks/
    │   ├── chunk_000000/
    │   │   ├── input/
    │   │   │   ├── state.json
    │   │   │   ├── ego_view.png
    │   │   │   └── ...
    │   │   ├── output/
    │   │   │   ├── raw_action.npy
    │   │   │   ├── safe_action.npy
    │   │   │   └── safety_events.json
    │   │   └── metadata.json
    ├── trajectory.jsonl
    ├── latency.jsonl
    └── safety_events.jsonl
```

每步记录字段：

```json
{
  "timestamp": 0.0,
  "chunk_index": 0,
  "step_in_chunk": 0,
  "state_before": [],
  "raw_action": [],
  "safe_action": [],
  "executed_action": [],
  "state_after": [],
  "inference_latency_ms": 0.0,
  "control_latency_ms": 0.0,
  "safety_events": []
}
```

必须记录：

```text
1. checkpoint path
2. embodiment tag
3. action schema version
4. task instruction
5. camera names
6. raw action chunk [H, 54]
7. safe action chunk [H, 54]
8. executed action
9. state before / after
10. inference latency
11. control latency
12. safety events
```

---

## 16. 主循环设计

`infer_tianji_wuji.py` 推荐主循环：

```python
while running:
    key = keyboard.poll()
    state_machine.update(key)

    if state_machine.state != "RUNNING":
        time.sleep(0.01)
        continue

    try:
        robot_state = robot.get_state()  # shape=(54,)
        images = cameras.read()

        obs = observation_builder.build(
            state=robot_state,
            images=images,
            task=args.task,
        )

        raw_chunk = policy_client.predict_action_chunk(obs)  # shape=(H, 54)

        actions = action_adapter.split_chunk(raw_chunk)

        safe_actions, safety_events = safety.process_chunk(
            current_state=robot_state,
            actions=actions,
            dt=args.duration,
        )

        recorder.save_chunk(
            observation=obs,
            raw_chunk=raw_chunk,
            safe_actions=safe_actions,
            safety_events=safety_events,
        )

        executor.execute_chunk(
            actions=safe_actions[: args.execution_horizon],
            dt=args.duration,
            dry_run=args.dry_run,
        )

    except CameraError:
        robot.hold_position()
        state_machine.to_error()

    except PolicyServerError:
        robot.hold_position()
        state_machine.to_error()

    except RobotError:
        robot.hold_position()
        state_machine.to_error()

    except KeyboardInterrupt:
        robot.hold_position()
        break
```

要求：

```text
1. 主循环只做编排。
2. 不在主循环中写硬件 SDK 细节。
3. 不在主循环中手写 action slice。
4. 不在主循环中做复杂 safety 逻辑。
5. 所有异常都必须导向 hold_position。
```

---

## 17. Dry-run 阶段

第一版必须先跑 dry-run：

```bash
python deployment/dry_run_infer.py   --policy-host 127.0.0.1   --policy-port 5555   --task "pick up the object"   --record-dir ./infer_logs/dry_run
```

或者使用主入口：

```bash
python deployment/infer_tianji_wuji.py   --policy-host 127.0.0.1   --policy-port 5555   --task "pick up the object"   --execution-horizon 8   --duration 0.1   --safe-mode   --dry-run   --record-dir ./infer_logs   --camera ego_view:0
```

dry-run 必须完成：

```text
1. 连接 policy server。
2. 连接相机或读取测试图片。
3. 读取双臂双手 state 或 fake state。
4. 构造 observation。
5. policy 返回 action chunk。
6. 检查 action chunk shape == [H, 54]。
7. 拆分 left_arm / left_hand / right_arm / right_hand。
8. 执行 safety process。
9. 保存日志。
10. 不发送任何真机控制命令。
```

---

## 18. Replay policy check 阶段

使用 `replay_policy_check.py` 做模型无关的控制链路检查：

```bash
python deployment/replay_policy_check.py   --action-file ./infer_logs/run_xxx/chunks/chunk_000000/output/raw_action.npy   --duration 0.1   --safe-mode   --dry-run
```

检查内容：

```text
1. action 文件 shape 是否为 [H, 54]。
2. ActionAdapter 是否正确拆分。
3. 左右臂顺序是否正确。
4. 左右手顺序是否正确。
5. 单位是否正确。
6. absolute / delta action 解释是否正确。
7. commanded vs actual 是否合理。
```

---

## 19. Safe execution 阶段

safe execution 初期建议：

```text
execution_horizon = 1 或 2
safe_mode = True
duration = 0.1 或训练频率对应值
```

流程：

```text
1. 程序启动后机器人处于 STOPPED。
2. 按 R 开始。
3. policy 输出一个 chunk。
4. 只执行前 1-2 步。
5. 自动暂停。
6. 人工确认后按 N 继续。
7. 全程记录 raw/safe/executed action。
```

禁止第一版直接：

```text
execution_horizon = 8
safe_mode = False
```

---

## 20. Runtime 验收标准

### 20.1 Server 验收

```text
1. checkpoint 能被 server 正确加载。
2. server 启动日志显示 model path、embodiment tag、device。
3. client 能 ping server。
4. server 断开时 client 不会继续执行动作。
```

### 20.2 Observation 验收

```text
1. state shape == (54,)
2. state 顺序为训练时顺序。
3. camera key 与 checkpoint 期望一致。
4. image RGB 正确。
5. task string 非空。
```

### 20.3 Action 验收

```text
1. policy 输出 action chunk shape == [H, 54]。
2. action 无 NaN/Inf。
3. ActionAdapter 正确拆成四组：
   - left_arm[7]
   - left_hand[20]
   - right_arm[7]
   - right_hand[20]
4. action 单位与控制端一致。
5. absolute/delta 解释正确。
```

### 20.4 Safety 验收

```text
1. NaN/Inf 不执行。
2. 超 joint limit 被 clamp 或拒绝。
3. 单步 delta 超限被 clip。
4. arm 和 hand 分开限幅。
5. left/right 分开限位。
6. safety event 被记录。
7. 连续异常进入 STOPPED。
8. Space 急停立即 hold_position。
```

### 20.5 Execution 验收

```text
1. dry-run 可完整跑通。
2. replay_policy_check 可完整跑通。
3. safe-mode 可执行单步。
4. commanded vs actual 可记录。
5. control loop 不长期 overrun。
6. 相机失败会 hold。
7. server timeout 会 hold。
8. 程序退出会 hold。
```

---

## 21. 代码人员 P0 交付清单

```text
deployment/start_groot_server.sh
deployment/infer_tianji_wuji.py
deployment/dry_run_infer.py
deployment/replay_policy_check.py

runtime/schema.py
runtime/robot_interface.py
runtime/camera_manager.py
runtime/observation_builder.py
runtime/groot_policy_client.py
runtime/action_adapter.py
runtime/safety.py
runtime/executor.py
runtime/keyboard.py
runtime/recorder.py
```

P0 最小功能：

```text
1. server 启动。
2. client 连接 server。
3. dry_run_infer 能输出 [H, 54]。
4. replay_policy_check 能读取并检查 [H, 54] action 文件。
5. client 读取 state 和 image。
6. state shape == 54。
7. client 构造 observation。
8. client 得到 [H, 54] action chunk。
9. ActionAdapter 拆分 left_arm / left_hand / right_arm / right_hand。
10. safety layer 处理。
11. dry-run 保存完整日志。
12. safe-mode 执行 1 step。
13. Space 急停。
```

---

## 22. 可交给硬件人员补充的部分

```text
1. 左臂 SDK 接入。
2. 右臂 SDK 接入。
3. 左手 SDK 接入。
4. 右手 SDK 接入。
5. 相机驱动接入。
6. 关节限位参数。
7. 最大速度参数。
8. 最大单步变化参数。
9. hold_position。
10. go_home。
11. 底层急停。
12. 通信断连保护。
```

硬件人员最终需要提供：

```python
class DualArmHandRobot:
    def connect(self) -> None:
        ...

    def get_state(self) -> np.ndarray:
        """Return shape=(54,), order follows schema.py."""
        ...

    def send_action(self, action: DualArmHandAction) -> None:
        ...

    def hold_position(self) -> None:
        ...

    def go_home(self) -> None:
        ...
```

---

## 23. 关键风险清单

```text
1. checkpoint 的 embodiment tag 和 runtime tag 不一致。
2. state/action 54 维顺序不一致。
3. 左右臂顺序反了。
4. 左右手顺序反了。
5. 机械臂和灵巧手单位不一致：rad / degree / encoder。
6. policy 输出是 delta，但控制端当 absolute 执行。
7. policy 输出已反归一化，client 又重复反归一化。
8. image key 和训练时不一致。
9. BGR/RGB 搞反。
10. control frequency 和训练数据频率不一致。
11. action chunk 执行过长，closed-loop 反馈太慢。
12. 没有保存 raw_action，失败后无法分析。
13. safety layer 对灵巧手过度平滑，导致抓取闭合失败。
14. server timeout 后机器人仍继续执行旧 action。
15. 左右侧其中一侧通信失败后，另一侧仍继续运动。
```

---

## 24. 最终工程原则

当前阶段只做运行，不做训练。  
核心目标是：

```text
已有 checkpoint
→ 启动 policy server
→ 控制端构造 observation
→ policy 输出 [H, 54]
→ 适配为双臂双手动作
→ 安全检查
→ dry-run
→ replay check
→ safe-mode 单步执行
→ 完整日志
→ 再进入真实闭环
```

第一版优先保证：

```text
schema 严格
安全可控
日志完整
dry-run 可复现
replay 可排查
急停可靠
```

不要第一版就追求复杂功能。  
最重要的工程原则是：

```text
训练 checkpoint 如何定义 54 维 state/action，runtime 就必须完全按同一顺序构造 observation 和执行 action。
```
