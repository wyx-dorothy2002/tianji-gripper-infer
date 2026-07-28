# Tianji Wuji GR00T 推理操作指南

这份文档用于指导天机无际双臂双灵巧手真机的 GR00T 推理操作。当前 runtime 目录是：

```bash
/mnt/data/qdhe/workspace/GR00T/tianji_wuji_runtime
```

主推理脚本是：

```bash
deployment/infer_tianji_wuji.py
```

整体链路是：

```text
真机状态 + 三路相机 + 语言任务
  -> 组装 GR00T observation
  -> 请求 GR00T policy server
  -> 得到 action chunk [H, 54]
  -> 拆成左臂/右臂/左手/右手
  -> 安全层限幅/限速
  -> dry-run 或下发真机
  -> 保存日志
```

当前先统一采用同步推理闭环：

```text
采集当前 state/image/task
  -> policy 推理一次
  -> 按固定 duration 执行 action chunk 的前 N 步
  -> 再回到下一轮采集和推理
```

暂不使用异步 policy thread、异步控制线程或 action buffer。若训练数据采样频率按 20Hz 处理，部署侧默认 action step 使用：

```bash
--duration 0.05
```

## 1. 数据和维度约定

当前 runtime 已对齐数据集：

```bash
/mnt/data/qdhe/workspace/datasets/local_lerobot_dataset
```

54 维状态和动作顺序固定为：

```text
0:7    left_arm_joint   左臂 7 维
7:14   right_arm_joint  右臂 7 维
14:34  left_hand        左手 20 维
34:54  right_hand       右手 20 维
```

相机 key 固定为：

```text
head
left_wrist
right_wrist
```

语言任务来自启动参数：

```bash
--task "pick up bottle"
```

真机接口最终必须保证：

```text
robot.get_state() -> DualArmHandState(left_arm_q, right_arm_q, left_hand_q, right_hand_q)
robot.get_state().as_flat() -> shape (54,)
policy raw action -> shape [H, 54]
robot.send_action() -> 分别下发 7/7/20/20 维动作
```

## 2. 真机前必须确认的接口

当前 `fake` backend 只能用于链路验证。真实上机前，需要硬件同学补齐：

```text
runtime/arm_interface.py
runtime/hand_interface.py
runtime/robot_interface.py
```

必须确认：

```text
左臂/右臂 get_joint_state 是否返回 7 维
左手/右手 get_joint_state 是否返回 20 维
关节顺序是否和数据集一致
单位是 rad、degree、encoder 还是 normalized
send_joint_position 接收 absolute target 还是 delta
go_home 是否是安全轨迹
hold_position 是否能可靠保持当前位置
通信断开或异常时是否会进入安全状态
```

建议 runtime 和 policy 交互时统一使用训练数据单位：

```text
单位：和训练数据 state/action 保持一致
动作模式：absolute joint target
```

例如当前 checkpoint 里机械臂数值更像 degree，灵巧手是另一套手指关节单位；不要用一个全局 rad/degree 转换一刀切。硬件 SDK 如果使用不同单位，需要在 `ArmInterface` / `HandInterface` 的真实实现里按 arm/hand 分段转换，不要在主循环里临时切片或转换。

## 3. 环境检查

进入 runtime 目录：

```bash
cd /mnt/data/qdhe/workspace/GR00T/tianji_wuji_runtime
```

检查数据集和 runtime schema 是否一致：

```bash
python tools/check_dataset_alignment.py \
  --dataset-path /mnt/data/qdhe/workspace/datasets/local_lerobot_dataset
```

检查相机 key 是否能被 runtime 读取：

```bash
python tools/check_camera_slots.py \
  --camera head:0 \
  --camera left_wrist:1 \
  --camera right_wrist:2
```

如果相机编号不对，调整 `0/1/2`。如果真实相机不能用 OpenCV index 打开，需要在 `runtime/camera_manager.py` 里接相机 SDK。

## 4. 启动 GR00T policy server

先启动模型服务。这个进程负责加载 checkpoint：

```bash
cd /mnt/data/qdhe/workspace/GR00T/tianji_wuji_runtime

CKPT=/path/to/your/checkpoint \
EMBODIMENT_TAG=NEW_EMBODIMENT \
CUDA_VISIBLE_DEVICES=0 \
bash deployment/start_groot_server.sh
```

默认 endpoint 是：

```text
127.0.0.1:5555
```

如果 checkpoint 使用了其他 embodiment tag，需要和训练/数据同学确认：

```text
checkpoint 的 embodiment tag
video modality keys
state modality keys
action modality keys
policy 输出是否已反归一化
policy 输出是 absolute 还是 delta
```

server 启动后，可以检查 server 返回的 modality config：

```bash
python tools/inspect_checkpoint_io.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555
```

重点看：

```text
video keys  是否是 head / left_wrist / right_wrist
state keys  是否是 left_arm_joint / right_arm_joint / left_hand / right_hand
action keys 是否是 left_arm_joint / right_arm_joint / left_hand / right_hand
```

## 5. Dry-run 验证

dry-run 不会给真机发控制命令，只检查：

```text
policy server 能否访问
observation 能否构造
action chunk 是否是 [H, 54]
动作能否按 7/7/20/20 拆分
安全层能否处理
日志能否保存
```

推荐先跑：

```bash
python deployment/dry_run_infer.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "pick up bottle" \
  --execution-horizon 1 \
  --duration 0.05 \
  --record-dir ./infer_logs/dry_run
```

如果要用真实相机但仍不发真机：

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

启动后默认是 `STOPPED`，需要按 `R` 或 safe-mode 下按 `N` 才会执行一段。

## 5.1 数据集帧 Policy 对比

如果想确认“数据集某一帧的相机 + state + prompt -> policy 动作”是否和 GT action 大致对得上，可以使用：

```bash
python tools/dataset_policy_check.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --dataset-path /mnt/data/qdhe/workspace/datasets/local_lerobot_dataset \
  --episodes 0,8,11 \
  --fractions 0.15,0.55 \
  --save-images
```

这个工具会：

```text
1. 从 LeRobot 数据集中读取指定 episode/step 的三路图像、state、task。
2. 请求已经启动的 GR00T policy server。
3. 得到 pred action chunk [16, 54]。
4. 从数据集中读取同一时刻的 GT future action chunk [16, 54]。
5. 按 left_arm / right_arm / left_hand / right_hand 计算 MAE/RMSE。
6. 保存 pred_action.npy、gt_action.npy、input_state.npy 和 metrics.json。
```

也可以精确指定帧：

```bash
python tools/dataset_policy_check.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --dataset-path /mnt/data/qdhe/workspace/datasets/local_lerobot_dataset \
  --sample 0:38 \
  --sample 8:206 \
  --save-images
```

输出目录默认是：

```text
infer_logs/dataset_policy_check/run_*/
```

解读时优先看：

```text
overall_first_step_mae
right_arm_first_step_mae
right_hand_first_step_mae
```

因为真机初期通常使用 `--execution-horizon 1`，第一步动作是否对齐比完整 16 步 future chunk 更直接。完整 horizon 的误差随未来步增大是正常现象，尤其是演示轨迹有多种可能走法时。

## 6. 真机安全启动建议

第一次真机上电测试，不建议连续运行。建议使用：

```text
--safe-mode
--execution-horizon 1
--duration 0.05
```

含义：

```text
safe-mode             每执行完一段自动暂停
execution-horizon 1   每次只执行 policy 输出的第 1 帧动作
duration 0.05         每帧动作间隔 0.05 秒，约 20Hz
```

示例命令：

```bash
python deployment/infer_tianji_wuji.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "pick up bottle" \
  --robot-backend tianji \
  --left-arm-ip <LEFT_ARM_IP> \
  --right-arm-ip <RIGHT_ARM_IP> \
  --left-hand-ip <LEFT_HAND_IP> \
  --right-hand-ip <RIGHT_HAND_IP> \
  --execution-horizon 1 \
  --duration 0.05 \
  --safe-mode \
  --camera head:0 \
  --camera left_wrist:1 \
  --camera right_wrist:2 \
  --record-dir ./infer_logs
```

如果只是想验证一轮后自动退出：

```bash
python deployment/infer_tianji_wuji.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "pick up bottle" \
  --robot-backend tianji \
  --execution-horizon 1 \
  --duration 0.05 \
  --safe-mode \
  --auto-start \
  --no-keyboard \
  --max-chunks 1 \
  --camera head:0 \
  --camera left_wrist:1 \
  --camera right_wrist:2
```

无键盘模式必须同时使用：

```text
--no-keyboard --auto-start
```

否则程序没有键盘，也不会自动进入 `RUNNING`。

## 7. 键盘控制说明

runtime 启动后，如果没有 `--auto-start`，默认状态是：

```text
STOPPED
```

此时不会采集、不会请求 policy、不会发动作。需要按键进入运行。

按键含义：

```text
R      开始/继续
P      暂停，但程序不退出
Space  停止当前运行状态
H      命令机器人回 home
Q      退出整个 runtime，并断开资源
N      safe-mode 下放行下一段，执行完又暂停
```

### R

`R` 会让状态进入：

```text
RUNNING
```

不开 safe-mode 时，按 `R` 后会连续循环：

```text
采集 -> 推理 -> 执行 -> 再采集 -> 再推理 -> 再执行
```

开 safe-mode 时，按 `R` 后只执行一段，然后自动暂停。

### P

`P` 会让状态进入：

```text
PAUSED
```

暂停后不会继续采集、推理和执行。程序还在，可以按 `R` 继续。

如果正在执行一个 action chunk，执行器会通过 `stop_requested()` 检查到状态不是 `RUNNING`，从而停止后续动作。

### Space

`Space` 会让状态进入：

```text
STOPPED
```

它更像软件层的“停住当前运行”。当前代码里，Space 本身只改状态；实际 `hold_position()` 会在执行器停止、退出、异常等路径里触发。

注意：Space 不能替代硬件急停。真机调试时，硬件急停必须随时可用。

### H

`H` 会设置：

```text
home_requested = True
```

随后 runtime 会调用：

```text
robot.go_home()
```

这个是具体运动命令，不只是状态切换。真机前必须确认 `go_home()` 是安全轨迹，不能简单地直接发送一个离当前位置很远的 home 关节目标。

### Q

`Q` 会设置：

```text
quit_requested = True
```

随后 runtime 会：

```text
robot.hold_position()
跳出主循环
disconnect cameras
disconnect robot
```

所以 `Q` 是推荐的正常退出方式。

### N

`N` 只在 `--safe-mode` 下生效。

safe-mode 的行为是：

```text
执行一段 chunk -> 自动 PAUSED -> 等待用户确认
```

按 `N` 表示：

```text
确认当前状态没问题，放行下一段
```

推荐真机初期使用方式：

```text
按 N
执行 1 帧或一小段
自动暂停
观察机器人
再按 N
```

## 8. safe-mode 推荐流程

真机第一次跑时，建议按下面节奏：

```text
1. 启动 policy server
2. 启动 infer_tianji_wuji.py，带 --safe-mode --execution-horizon 1
3. 确认机器人没有立刻运动
4. 按 N 或 R 放行第一步
5. 观察动作方向、幅度、左右手是否正确
6. 正常则继续按 N
7. 有异常按 Space 或 P
8. 实验结束按 Q
```

如果出现以下情况，立即停止：

```text
左右臂动作反了
左右手动作反了
动作方向和预期相反
手指关节顺序明显不对
动作幅度过大
安全层频繁触发 delta_clip
相机画面 key 对不上
policy 输出含 NaN/Inf
```

## 9. 日志位置和复盘

每次运行会创建：

```text
infer_logs/run_YYYYMMDD_HHMMSS_xxxxxx/
```

关键文件：

```text
config.json
chunks/chunk_000000/input/state.json
chunks/chunk_000000/input/head.png
chunks/chunk_000000/input/left_wrist.png
chunks/chunk_000000/input/right_wrist.png
chunks/chunk_000000/output/raw_action.npy
chunks/chunk_000000/output/safe_action.npy
chunks/chunk_000000/output/safety_events.json
trajectory.jsonl
latency.jsonl
safety_events.jsonl
```

复盘时重点看：

```text
input 图片是否来自正确相机
state.json 的状态顺序是否正确
raw_action.npy 是否 [H, 54]
safe_action.npy 是否被安全层大量裁剪
trajectory.jsonl 中 state_before / state_after 是否合理
```

如果想画 commanded vs actual，可以使用：

```bash
python tools/plot_commanded_vs_actual.py \
  --run-dir ./infer_logs/run_YYYYMMDD_HHMMSS_xxxxxx
```

## 10. 常见问题

### 程序启动后不动

如果没有 `--auto-start`，这是正常的。默认状态是 `STOPPED`。

按：

```text
R
```

或 safe-mode 下按：

```text
N
```

### safe-mode 下执行完一段就停了

这是正常行为。safe-mode 会自动暂停，等待人工确认。

继续下一段：

```text
N
```

### dry-run 没有真实动作

这是正常的。`--dry-run` 会完整跑 policy 和 safety，但不会调用真实 `robot.send_action()`。

### Q 和 Space 的区别

```text
Space  停止当前运行状态，但程序还在
Q      退出整个 runtime，并断开资源
```

### H 是否可以随便按

不建议。`H` 会执行 `robot.go_home()`，它必须由硬件接口实现成安全轨迹后再使用。

### policy 一次执行完后会自动退出吗

默认不会。它会继续下一轮推理。

如果想执行一轮后退出，使用：

```bash
--max-chunks 1
```

## 11. 推荐的阶段性验证顺序

建议按这个顺序推进：

```text
1. check_dataset_alignment.py
2. check_camera_slots.py
3. 启动 policy server
4. inspect_checkpoint_io.py
5. dry_run_infer.py
6. infer_tianji_wuji.py --dry-run --safe-mode
7. fake backend replay_policy_check.py
8. 接真实硬件 backend，但先 --dry-run
9. 真实硬件 --safe-mode --execution-horizon 1
10. 确认稳定后再考虑提高 execution_horizon；duration 默认保持 0.05，除非训练频率不同
```

真机稳定前，不建议直接连续运行，也不建议一开始就执行完整 action horizon。
