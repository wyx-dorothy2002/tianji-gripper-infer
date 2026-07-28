# Tianji Wuji GR00T N1.7 Runtime

这个目录是在不改动上游 `gr00t/` 代码的前提下，为天机无际双臂双灵巧手真机推理搭的本地 runtime。当前版本优先完成 P0 核心闭环：

```text
robot state/image/task
  -> GR00T policy server
  -> raw action chunk [H, 54]
  -> ActionAdapter split
  -> SafetyLayer
  -> dry-run / executor
  -> recorder logs
```

当前 runtime 采用同步推理闭环：

```text
采集 state/image/task -> policy 推理 -> 执行前 N 步 action chunk -> 再采集
```

暂不引入异步 policy thread 或 action buffer。若训练数据采样频率按 20Hz 处理，部署侧 action step 默认使用 `--duration 0.05` 对齐。

## 54 维顺序

当前 schema 固定在 `runtime/schema.py`，已对齐
`/mnt/data/qdhe/workspace/datasets/local_lerobot_dataset/meta/modality.json`：

```text
0:7    left_arm_joint 7 DoF
7:14   right_arm_joint 7 DoF
14:34  left_hand 20 DoF
34:54  right_hand 20 DoF
```

这必须和训练 checkpoint 的 `state/action` 顺序完全一致。如果训练顺序不同，优先只改 `runtime/schema.py` 和 `configs/action_schema.yaml`，不要在主循环里写临时切片。

## 目录

```text
deployment/
  start_groot_server.sh      # 启动 GR00T server
  infer_tianji_wuji.py       # 真机/安全模式主入口
  dry_run_infer.py           # 只测 policy 输入输出，不发真机
  replay_policy_check.py     # 回放 saved action chunk，模型无关检查控制链路

runtime/
  schema.py                  # 54 DoF 唯一 schema
  robot_interface.py         # 双臂双手统一接口
  arm_interface.py           # 机械臂 SDK 接入边界
  hand_interface.py          # 灵巧手 SDK 接入边界
  camera_manager.py          # RGB 相机读取
  observation_builder.py     # 构造 GR00T observation
  groot_policy_client.py     # GR00T PolicyClient wrapper
  action_adapter.py          # [54] <-> 四段动作
  safety.py                  # joint limit / delta / velocity safety
  executor.py                # 固定周期执行 chunk
  keyboard.py                # STOPPED/RUNNING/PAUSED/ERROR 状态机
  recorder.py                # raw/safe/action 日志
```

## 启动 server

```bash
cd /mnt/data/qdhe/workspace/GR00T/tianji_wuji_runtime

CKPT=/path/to/checkpoint \
EMBODIMENT_TAG=NEW_EMBODIMENT \
CUDA_VISIBLE_DEVICES=0 \
bash deployment/start_groot_server.sh
```

注意：当前上游 `EmbodimentTag` 已支持 `NEW_EMBODIMENT -> new_embodiment`。如果你们 checkpoint 使用了别的自定义 tag，需要确认 `gr00t/data/embodiment_tags.py` 和 checkpoint processor 里的 tag 是否都能解析；这个可能需要改原仓库或重新导出 checkpoint。

## Dry Run

先跑 dry-run，不会发送真机控制命令：

```bash
python deployment/dry_run_infer.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "pick up bottle" \
  --image-source ./assets/test_head.png \
  --record-dir ./infer_logs/dry_run
```

如果没有测试图，脚本会用 dummy RGB 图像补齐 server modality config 里的 camera key。对当前数据集，server 应该要求三个 camera key：`head`、`left_wrist`、`right_wrist`。成功后会打印：

```text
action chunk shape: (H, 54)
left_arm=(7,), right_arm=(7,), left_hand=(20,), right_hand=(20,)
logs: infer_logs/dry_run/run_...
```

## 主入口

安全起步建议：

```bash
python deployment/infer_tianji_wuji.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "pick up bottle" \
  --execution-horizon 1 \
  --duration 0.05 \
  --safe-mode \
  --dry-run \
  --record-dir ./infer_logs \
  --camera head:0 \
  --camera left_wrist:1 \
  --camera right_wrist:2
```

默认启动后是 `STOPPED`，需要按 `R` 才会请求 policy 并执行。快捷键：

```text
R      start / resume
P      pause
Space  hold_position
H      go_home
N      safe-mode 下执行下一段 chunk
Q      hold_position 并退出
```

无键盘环境必须显式加 `--no-keyboard --auto-start`，建议同时加 `--max-chunks 1`。

## Replay Check

用已保存的 `raw_action.npy` 检查 action schema 和 safety，不依赖模型：

```bash
python deployment/replay_policy_check.py \
  --action-file ./infer_logs/run_xxx/chunks/chunk_000000/output/raw_action.npy \
  --duration 0.05 \
  --safe-mode \
  --dry-run
```

## 日志

每次运行会创建：

```text
infer_logs/run_YYYYMMDD_HHMMSS_xxxxxx/
  config.json
  chunks/chunk_000000/input/state.json
  chunks/chunk_000000/input/<camera>.png
  chunks/chunk_000000/output/raw_action.npy
  chunks/chunk_000000/output/safe_action.npy
  chunks/chunk_000000/output/safety_events.json
  trajectory.jsonl
  latency.jsonl
  safety_events.jsonl
```

## 需要硬件同学对接

当前 `ArmInterface` / `HandInterface` 是 fake 实现，只用于 dry-run 和链路检查。真机执行前需要硬件同学补：

- 左臂/右臂 SDK：`connect/get_joint_state/send_joint_position/hold_position/go_home`
- 左手/右手 SDK：`connect/get_joint_state/send_joint_position/hold_position/go_home`
- 7 个臂关节顺序、20 个手关节顺序、左右侧定义
- 真机 `get_state()` 返回结构化 `DualArmHandState`，需要 flat 日志/数据集视图时用 `state.as_flat()` 得到 `(54,)`
- 单位：rad / degree / encoder / normalized
- joint min/max、max velocity、max single-step delta
- `send_joint_position` 接收 absolute target 还是 delta
- 底层急停、通信断连保护、一侧失败时四侧 hold 的真实实现
- 控制频率和 `duration` 是否匹配训练数据频率

建议接入点：只实现 `runtime/arm_interface.py` 和 `runtime/hand_interface.py` 的 SDK 子类，然后在 `runtime/robot_interface.py::make_robot()` 增加 backend 分支。

## 需要数据/训练同学确认

- checkpoint 的 `embodiment tag`
- checkpoint 的 video modality keys 当前应为 `head` / `left_wrist` / `right_wrist`
- checkpoint 的 state modality keys 当前应为 `left_arm_joint` / `right_arm_joint` / `left_hand` / `right_hand`
- checkpoint 的 action modality keys 及拼接顺序当前应为 `left_arm_joint` / `right_arm_joint` / `left_hand` / `right_hand`
- policy 输出是否已经由 GR00T processor 反归一化
- policy 输出是 absolute joint target 还是 delta action
- state/action 单位是否和真机控制端一致
- 训练时是否用了历史帧/历史 state，`delta_indices` 是否不只是 `[0]`

可以用：

```bash
python tools/inspect_checkpoint_io.py --policy-host 127.0.0.1 --policy-port 5555
```

先把 server 返回的 modality config 打出来，再对齐相机和 state/action 构造。

也可以直接检查本地数据集和 runtime schema 是否一致：

```bash
python tools/check_dataset_alignment.py \
  --dataset-path /mnt/data/qdhe/workspace/datasets/local_lerobot_dataset
```

## 可能需要改原仓库的地方

目前没有修改上游 `gr00t/`。只有这些情况可能需要你判断是否改原仓库：

- 自定义 `embodiment tag` 不在 `EmbodimentTag`，server 在启动阶段无法 resolve。
- checkpoint processor 里的 modality key 或 tag 与训练数据不一致，需要重新注册或重新导出。
- 需要把天机无际的 modality config 注册到 `gr00t/configs/data/embodiment_configs.py`，供 ReplayPolicy 或训练链路复用。

在确认这些之前，runtime 侧保持独立。
