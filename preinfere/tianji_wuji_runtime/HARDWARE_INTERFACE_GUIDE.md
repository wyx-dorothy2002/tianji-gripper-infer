# Tianji Wuji 硬件接口对接说明

这份文档给硬件同学使用，用于把真实双臂双灵巧手和三路相机接入：

```bash
/mnt/data/qdhe/workspace/GR00T/tianji_wuji_runtime
```

当前已经写好的 runtime 主流程包括：

```text
deployment/infer_tianji_wuji.py
runtime/observation_builder.py
runtime/action_adapter.py
runtime/safety.py
runtime/executor.py
runtime/recorder.py
```

硬件同学主要需要补齐：

```text
runtime/arm_interface.py
runtime/hand_interface.py
runtime/robot_interface.py
runtime/camera_manager.py  # 只有 OpenCV 无法打开相机时才需要改
configs/robot_limits.yaml
```

## 1. 总体数据约定

### 1.1 状态和动作维度

整个系统使用 54 维 state/action：

```text
0:7    left_arm_joint   左臂 7 维
7:14   right_arm_joint  右臂 7 维
14:34  left_hand        左手 20 维
34:54  right_hand       右手 20 维
```

这个顺序和训练数据集一致：

```bash
/mnt/data/qdhe/workspace/datasets/local_lerobot_dataset/meta/modality.json
```

所有硬件接口对 runtime 暴露的数据都必须保持这个顺序。硬件 SDK 内部如果顺序不同，应在对应接口类内部做映射，不要改主推理循环。

### 1.2 相机 key

policy 需要三路 RGB 图像：

```text
head
left_wrist
right_wrist
```

每张图对 runtime 的输出格式必须是：

```text
np.ndarray
dtype = uint8
shape = H x W x 3
color = RGB
```

OpenCV 默认读出来是 BGR，当前 `camera_manager.py` 已经会转成 RGB。

### 1.3 单位

policy 输入 state 和 policy 输出 action 必须使用训练数据单位。

当前 checkpoint 的统计显示：

```text
机械臂关节数值更像 degree，例如 90, -90, -70 ...
灵巧手是另一套手指关节单位，数值大多在 -0.5 到 1.5 附近
```

所以不要使用一个全局 rad/degree 转换一刀切。

建议原则：

```text
ArmInterface 对 runtime 暴露训练数据里的机械臂单位
HandInterface 对 runtime 暴露训练数据里的手指单位
硬件 SDK 需要什么单位，就在对应 Interface 内部转换
```

例如：

```text
policy/action 是 degree，机械臂 SDK 要 rad：
  send_joint_position() 内部 degree -> rad

机械臂 SDK 返回 rad，policy/state 要 degree：
  get_joint_state() 内部 rad -> degree
```

## 2. 机械臂接口

文件：

```text
runtime/arm_interface.py
```

### 2.1 作用

`ArmInterface` 是单只机械臂的 SDK 适配边界。runtime 不直接调用硬件 SDK，只调用这个接口。

每只机械臂必须对外表现为：

```text
7 维关节位置读取
7 维关节位置目标下发
hold 当前姿态
go_home 安全回 home
connect / disconnect
```

### 2.2 需要实现的类

建议新增：

```python
class TianjiArmInterface(ArmInterface):
    ...
```

可以直接写在 `runtime/arm_interface.py`，也可以新建 `runtime/tianji_arm_interface.py` 后在 `robot_interface.py` 里导入。

### 2.3 必须实现的方法

#### connect()

功能：

```text
建立和单只机械臂的连接
初始化控制模式
清除错误
必要时使能电机
```

要求：

```text
失败时抛 ArmError 或普通 Exception
不要静默失败
connect 成功后 get_joint_state() 必须可用
```

#### disconnect()

功能：

```text
释放 SDK 连接
关闭 socket / serial / ROS client / CAN 连接
```

要求：

```text
尽量不要抛异常
退出程序时会被 finally 调用
```

#### get_joint_state()

功能：

```text
读取当前 7 维机械臂关节位置
```

最终输出格式：

```python
np.ndarray
shape == (7,)
dtype == np.float32
finite，无 NaN/Inf
```

左臂顺序必须对应：

```text
left_joint_1.pos
left_joint_2.pos
left_joint_3.pos
left_joint_4.pos
left_joint_5.pos
left_joint_6.pos
left_joint_7.pos
```

右臂顺序必须对应：

```text
right_joint_1.pos
right_joint_2.pos
right_joint_3.pos
right_joint_4.pos
right_joint_5.pos
right_joint_6.pos
right_joint_7.pos
```

如果硬件 SDK 的关节顺序不同，必须在这里重排。

#### send_joint_position(q)

功能：

```text
下发单只机械臂 7 维关节位置目标
```

输入格式：

```python
q: np.ndarray
q.shape == (7,)
q.dtype 可转为 np.float32
```

注意：

```text
runtime 传入的是 policy/safety 处理后的目标位置
当前默认 action mode 是 absolute
不要把 absolute target 当 delta 发给底层
```

如果硬件底层只支持 delta command，应在这里做明确转换，或者先不要接入真机执行。

#### hold_position()

功能：

```text
让机械臂保持当前位置
```

推荐实现：

```text
读取当前关节位置
下发当前位置作为 position target
或调用 SDK 自带 hold/brake/servo hold 接口
```

要求：

```text
异常、退出、用户按 Q/Space 时都会依赖这个函数
必须尽量可靠
```

#### go_home()

功能：

```text
让机械臂回 home 姿态
```

要求：

```text
必须是安全轨迹
不能直接给一个远距离 home target 导致突然大幅运动
需要限速、插值、可中断
```

真机早期如果 home 轨迹还没验证，建议先实现为：

```text
hold_position()
```

并在日志中提示未实现安全 home。

### 2.4 代码骨架

```python
class TianjiArmInterface(ArmInterface):
    def __init__(self, config: ArmConnectionConfig) -> None:
        self.config = config
        self.client = None
        self.connected = False
        self._last_q = np.zeros(schema.LEFT_ARM_DOF, dtype=np.float32)

    def connect(self) -> None:
        # TODO: 建立硬件 SDK 连接
        # self.client = ArmSDK(ip=self.config.ip, port=self.config.port, side=self.config.side)
        # self.client.connect()
        # self.client.enable()
        self.connected = True

    def disconnect(self) -> None:
        if self.client is not None:
            # self.client.disconnect()
            pass
        self.connected = False

    def get_joint_state(self) -> np.ndarray:
        # raw = self.client.get_joint_positions()
        raw = ...
        q = np.asarray(raw, dtype=np.float32)

        # TODO: 如果 SDK 返回 rad，而训练数据是 degree：
        # q = np.rad2deg(q)

        # TODO: 如果 SDK 顺序和训练数据不同，在这里重排：
        # q = q[SDK_TO_DATASET_INDEX]

        if q.shape != (schema.LEFT_ARM_DOF,):
            raise ArmError(f"{self.config.side} arm state must be shape (7,), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ArmError(f"{self.config.side} arm state contains NaN/Inf")

        self._last_q = q.copy()
        return q

    def send_joint_position(self, q: np.ndarray) -> None:
        target = np.asarray(q, dtype=np.float32)
        if target.shape != (schema.LEFT_ARM_DOF,):
            raise ArmError(f"{self.config.side} arm command must be shape (7,), got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ArmError(f"{self.config.side} arm command contains NaN/Inf")

        # TODO: 如果训练数据是 degree，而 SDK 要 rad：
        # command = np.deg2rad(target)
        command = target

        # TODO: 如果 SDK 顺序和训练数据不同，在这里逆向重排：
        # command = command[DATASET_TO_SDK_INDEX]

        # self.client.send_joint_position(command.tolist())
        self._last_q = target.copy()

    def hold_position(self) -> None:
        try:
            q = self.get_joint_state()
        except Exception:
            q = self._last_q
        self.send_joint_position(q)

    def go_home(self) -> None:
        # TODO: 实现安全插值轨迹
        self.hold_position()
```

## 3. 灵巧手接口

文件：

```text
runtime/hand_interface.py
```

### 3.1 作用

`HandInterface` 是单只灵巧手的 SDK 适配边界。

每只手必须对外表现为：

```text
20 维手指关节状态读取
20 维手指关节目标下发
hold 当前手指姿态
go_home / open hand
connect / disconnect
```

### 3.2 需要实现的类

建议新增：

```python
class TianjiHandInterface(HandInterface):
    ...
```

### 3.3 必须实现的方法

#### connect()

功能：

```text
建立和单只灵巧手的连接
初始化控制模式
必要时清错、使能
```

#### disconnect()

功能：

```text
释放 SDK 连接
```

#### get_joint_state()

功能：

```text
读取当前 20 维手指关节位置
```

最终输出格式：

```python
np.ndarray
shape == (20,)
dtype == np.float32
finite，无 NaN/Inf
```

左手顺序必须对应：

```text
left_finger1_joint1
left_finger1_joint2
left_finger1_joint3
left_finger1_joint4
left_finger2_joint1
...
left_finger5_joint4
```

右手顺序必须对应：

```text
right_finger1_joint1
right_finger1_joint2
right_finger1_joint3
right_finger1_joint4
right_finger2_joint1
...
right_finger5_joint4
```

也就是：

```text
finger1 joint1-4
finger2 joint1-4
finger3 joint1-4
finger4 joint1-4
finger5 joint1-4
```

如果硬件 SDK 是另一种手指顺序，必须在这里重排。

#### send_joint_position(q)

功能：

```text
下发单只灵巧手 20 维手指目标
```

输入格式：

```python
q: np.ndarray
q.shape == (20,)
q.dtype 可转为 np.float32
```

注意：

```text
当前 checkpoint 里 hand action 是 ABSOLUTE
不要当 delta 执行
如果 SDK 使用 0-1000、0-1、degree、rad，需要在这里转换
```

#### hold_position()

功能：

```text
保持当前手指姿态
```

推荐实现：

```text
读取当前 20 维手指状态
下发当前值作为目标
或调用 SDK 自带 hold 接口
```

#### go_home()

功能：

```text
回到安全手部 home 姿态，通常是张开手或预设安全手型
```

要求：

```text
必须限速
必须确认不会夹伤物体/人
未验证前建议实现为 hold_position()
```

### 3.4 代码骨架

```python
class TianjiHandInterface(HandInterface):
    def __init__(self, config: HandConnectionConfig) -> None:
        self.config = config
        self.client = None
        self.connected = False
        self._last_q = np.zeros(schema.LEFT_HAND_DOF, dtype=np.float32)

    def connect(self) -> None:
        # TODO: 建立硬件 SDK 连接
        self.connected = True

    def disconnect(self) -> None:
        if self.client is not None:
            # self.client.disconnect()
            pass
        self.connected = False

    def get_joint_state(self) -> np.ndarray:
        # raw = self.client.get_joint_positions()
        raw = ...
        q = np.asarray(raw, dtype=np.float32)

        # TODO: 单位转换和顺序重排

        if q.shape != (schema.LEFT_HAND_DOF,):
            raise HandError(f"{self.config.side} hand state must be shape (20,), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise HandError(f"{self.config.side} hand state contains NaN/Inf")

        self._last_q = q.copy()
        return q

    def send_joint_position(self, q: np.ndarray) -> None:
        target = np.asarray(q, dtype=np.float32)
        if target.shape != (schema.LEFT_HAND_DOF,):
            raise HandError(f"{self.config.side} hand command must be shape (20,), got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise HandError(f"{self.config.side} hand command contains NaN/Inf")

        # TODO: 转成 SDK 需要的单位和顺序
        command = target
        # self.client.send_joint_position(command.tolist())
        self._last_q = target.copy()

    def hold_position(self) -> None:
        try:
            q = self.get_joint_state()
        except Exception:
            q = self._last_q
        self.send_joint_position(q)

    def go_home(self) -> None:
        # TODO: 实现安全张手/回 home
        self.hold_position()
```

## 4. 机器人组合接口

文件：

```text
runtime/robot_interface.py
```

### 4.1 作用

`DualArmHandRobot` 把四个硬件对象组合成一个机器人：

```text
left_arm
right_arm
left_hand
right_hand
```

主推理脚本只和 `DualArmHandRobot` 交互。

### 4.2 当前已经实现的功能

已经实现：

```text
connect()
disconnect()
get_state()
send_action()
hold_position()
go_home()
```

其中 `get_state()` 会返回结构化状态：

```text
DualArmHandState(
  left_arm_q=(7,),
  right_arm_q=(7,),
  left_hand_q=(20,),
  right_hand_q=(20,),
)
```

需要和数据集或日志对齐时，再调用：

```python
state.as_flat()
```

得到 canonical `(54,)`：

```text
left_arm + right_arm + left_hand + right_hand
```

### 4.3 硬件同学需要补什么

当前 `make_robot()` 只支持：

```text
backend = fake
```

需要新增一个真实 backend，例如：

```text
backend = tianji
```

示例：

```python
def make_robot(config: RobotConnectionConfig) -> DualArmHandRobot:
    if config.backend == "fake":
        ...

    if config.backend == "tianji":
        return DualArmHandRobot(
            left_arm=TianjiArmInterface(
                ArmConnectionConfig("left", ip=config.left_arm_ip, port=config.left_arm_port)
            ),
            left_hand=TianjiHandInterface(
                HandConnectionConfig("left", ip=config.left_hand_ip, port=config.left_hand_port)
            ),
            right_arm=TianjiArmInterface(
                ArmConnectionConfig("right", ip=config.right_arm_ip, port=config.right_arm_port)
            ),
            right_hand=TianjiHandInterface(
                HandConnectionConfig("right", ip=config.right_hand_ip, port=config.right_hand_port)
            ),
        )

    raise RobotError(f"Unsupported robot backend: {config.backend}")
```

如果硬件不是 IP/port，而是串口、CAN、ROS topic，需要扩展：

```python
ArmConnectionConfig
HandConnectionConfig
RobotConnectionConfig
```

例如：

```python
device: str | None = None
baudrate: int | None = None
can_id: int | None = None
ros_namespace: str | None = None
```

### 4.4 send_action 的最终输入

`send_action()` 接收的是：

```python
DualArmHandAction(
    left_arm_q: np.ndarray shape (7,),
    right_arm_q: np.ndarray shape (7,),
    left_hand_q: np.ndarray shape (20,),
    right_hand_q: np.ndarray shape (20,),
)
```

这些动作已经经过：

```text
policy 反归一化
ActionAdapter 拆分
SafetyLayer 限幅
```

硬件接口不需要再做 policy 相关处理，只需要：

```text
检查 shape
检查有限值
按 SDK 单位/顺序转换
下发给硬件
```

## 5. 相机接口

文件：

```text
runtime/camera_manager.py
```

### 5.1 当前已支持

当前支持：

```text
OpenCV camera index，例如 --camera head:0
图片文件，例如 --camera head:/tmp/head.png
dummy/zeros，用于 dry-run
```

如果真实相机能用 OpenCV 打开，不需要改代码：

```bash
--camera head:0 \
--camera left_wrist:1 \
--camera right_wrist:2
```

### 5.2 需要改的情况

如果相机必须使用：

```text
RealSense SDK
厂商 SDK
ROS topic
共享内存
网络流
```

则需要在 `camera_manager.py` 里新增 source 类型。

最终 `CameraManager.read()` 必须返回：

```python
{
    "head": np.ndarray,        # H x W x 3, uint8, RGB
    "left_wrist": np.ndarray,  # H x W x 3, uint8, RGB
    "right_wrist": np.ndarray, # H x W x 3, uint8, RGB
}
```

### 5.3 对齐要求

必须确认：

```text
head 画面确实是头部视角
left_wrist 画面确实是左腕视角
right_wrist 画面确实是右腕视角
图像不是上下颠倒
图像不是左右镜像
颜色是 RGB，不是 BGR
时间延迟可接受
```

## 6. 安全限位配置

文件：

```text
configs/robot_limits.yaml
```

硬件同学需要提供：

```text
左臂 7 维 min/max
右臂 7 维 min/max
左手 20 维 min/max
右手 20 维 min/max
每个关节最大速度
每个控制周期最大允许变化量
home 姿态
```

注意单位必须和 runtime/policy 交互单位一致。

如果机械臂使用 degree，则机械臂限位也应填 degree。

## 7. 验收命令

### 7.1 检查 state 输出

真实 backend 接好后，先运行：

```bash
cd /mnt/data/qdhe/workspace/GR00T

python tianji_wuji_runtime/tools/check_robot_state.py \
  --robot-backend tianji
```

期望：

```text
structured segments: left_arm/right_arm/left_hand/right_hand
flat state shape: (54,)
无 NaN/Inf
数值范围合理
```

重点检查：

```text
左臂动时只改变 0:7
右臂动时只改变 7:14
左手动时只改变 14:34
右手动时只改变 34:54
```

### 7.2 检查相机

```bash
python tianji_wuji_runtime/tools/check_camera_slots.py \
  --camera head:0 \
  --camera left_wrist:1 \
  --camera right_wrist:2
```

期望：

```text
每个 key 都有图
shape 是 H x W x 3
dtype 是 uint8
画面方向、左右、颜色正确
```

### 7.3 检查 policy 输入输出

先启动 policy server：

```bash
cd /mnt/data/qdhe/workspace/GR00T/tianji_wuji_runtime

CKPT=/mnt/data/qdhe/workspace/checkpoints/gr00t/checkpoint-40000 \
EMBODIMENT_TAG=NEW_EMBODIMENT \
CUDA_VISIBLE_DEVICES=0 \
bash deployment/start_groot_server.sh
```

再 dry-run：

```bash
python deployment/infer_tianji_wuji.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "pick up bottle" \
  --execution-horizon 1 \
  --duration 0.05 \
  --safe-mode \
  --dry-run \
  --camera head:0 \
  --camera left_wrist:1 \
  --camera right_wrist:2
```

期望：

```text
raw_action.npy shape = (16, 54)
safe_action.npy shape = (16, 54)
不下发真机动作
日志正常保存
```

### 7.4 第一次真机执行

必须使用：

```bash
--safe-mode
--execution-horizon 1
--duration 0.05
```

启动示例：

```bash
python deployment/infer_tianji_wuji.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "pick up bottle" \
  --robot-backend tianji \
  --execution-horizon 1 \
  --duration 0.05 \
  --safe-mode \
  --camera head:0 \
  --camera left_wrist:1 \
  --camera right_wrist:2
```

启动后默认不动，需要按：

```text
N 或 R
```

每次只放行一步。若方向、单位、左右顺序正确，再逐步提高：

```text
execution_horizon = 2
execution_horizon = 4
```

## 8. 硬件同学需要给出的信息

请硬件同学提供：

```text
1. 左臂 SDK 文档和示例代码
2. 右臂 SDK 文档和示例代码
3. 左手 SDK 文档和示例代码
4. 右手 SDK 文档和示例代码
5. 每个设备的连接方式：IP/port、串口、CAN、ROS topic 等
6. 机械臂 7 维关节顺序和正方向
7. 灵巧手 20 维关节顺序和正方向
8. 读状态的单位
9. 发控制的单位
10. position command 是 absolute 还是 delta
11. 控制频率上限
12. 底层是否支持轨迹插补
13. 底层是否支持速度/加速度限制
14. 急停、清错、使能、掉线保护接口
15. 安全 home 姿态和安全回 home 方法
16. 关节限位和速度限位
```

## 9. 最容易出错的地方

重点排查：

```text
1. 左右臂顺序反了
2. 左右手顺序反了
3. 机械臂单位 degree/rad 搞错
4. 手指关节顺序和训练数据不一致
5. policy absolute action 被当成 delta 执行
6. OpenCV BGR 没转 RGB
7. wrist 相机左右接反
8. go_home 直接发远距离目标
9. hold_position 实际没有 hold
10. 硬件限位单位和 policy 单位不一致
```

## 10. 最终接口验收标准

硬件接口完成后，应满足：

```text
robot.connect() 成功连接四个设备
robot.get_state() 返回 DualArmHandState
robot.get_state().as_flat() 返回 (54,) float32
as_flat() 顺序为 left_arm + right_arm + left_hand + right_hand
camera.read() 返回 head/left_wrist/right_wrist 三张 RGB uint8 图
robot.send_action() 能接收 7/7/20/20 分段动作
hold_position() 在异常和退出时可靠生效
go_home() 是安全、限速、可中断的轨迹
infer_tianji_wuji.py --dry-run 能跑通
infer_tianji_wuji.py --safe-mode --execution-horizon 1 能单步安全执行
```
