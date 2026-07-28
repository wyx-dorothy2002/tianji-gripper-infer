# 三路相机 + 双手双臂观测采样实现方案

本文档只覆盖“读取三路相机 + Wuji 双手 + Tianji 双臂，并保存带时间戳的观测数据，后续可直接接入 GR00T policy 输入接口”的实现细节。

不讨论机械臂本体接管、抱闸、上电流程、急停流程。这些由硬件侧或现有天机流程保证。runtime 只负责读取状态、采图、组 observation、下发模型动作。

## 1. 目标

当前目标分两步：

1. 先实现一个独立的观测采样能力。
   采样时读取：
   - `head`
   - `left_wrist`
   - `right_wrist`
   - `left_arm`
   - `right_arm`
   - `left_hand`
   - `right_hand`

   并保存：
   - 三张 RGB 图片
   - 54 维 robot state
   - 分段 state
   - 每个读取动作的 wall timestamp 和 monotonic timestamp
   - 每路相机 frame timestamp / frame id / age

2. 后续把同一套读取接口接入 `infer_tianji_wuji.py`。
   在每个 action chunk 刚执行完之后，读取当前 robot state，并从相机缓存取最新帧，构造 GR00T policy observation。

## 2. 当前模型输入契约

GR00T policy server 的 modality config 当前需要：

```text
video:
  head
  left_wrist
  right_wrist

state:
  left_arm_joint
  right_arm_joint
  left_hand
  right_hand

language:
  annotation.human.action.task_description
```

当前 `delta_indices=[0]`，所以每次只需要当前一帧历史。最终进入 policy 的形状应为：

```text
video["head"]        -> uint8, shape (1, 1, H, W, 3), RGB
video["left_wrist"]  -> uint8, shape (1, 1, H, W, 3), RGB
video["right_wrist"] -> uint8, shape (1, 1, H, W, 3), RGB

state["left_arm_joint"]  -> float32, shape (1, 1, 7)
state["right_arm_joint"] -> float32, shape (1, 1, 7)
state["left_hand"]       -> float32, shape (1, 1, 20)
state["right_hand"]      -> float32, shape (1, 1, 20)

language["annotation.human.action.task_description"] -> [[task]]
```

runtime 内部统一 54 维顺序必须保持：

```text
[0:7]   left_arm
[7:14]  right_arm
[14:34] left_hand
[34:54] right_hand
```

单位当前按 `deg` 处理，action/state 都是 absolute。

## 3. 图像采样分辨率

采样时相机目标分辨率设为：

```text
width: 424
height: 240
fps: 20
```

选择 `424x240` 的原因：

- 属于“四百多 x 二百多”的低延迟分辨率。
- 接近 16:9，和常见 USB/RealSense 模式比较贴合。
- 比 `320x240` 保留更多视觉细节，但仍适合 20Hz 实时采集。
- GR00T N1.7 不要求原始相机图像必须是 `224x224`；processor 会在 policy server 侧按 checkpoint 配置继续 resize/crop。

如果某个相机实际不支持 `424x240`，实现上仍然应强制在 runtime 输出前 resize 到 `424x240`，保证三路图像 shape 一致。

## 4. CameraManager 改造细节

当前 `camera_manager.py` 是同步 `read()`，后续建议改成“3 路相机各自后台采集 + 最新帧快照”。

### 4.0 三线程采集模型

三路相机必须各自使用一个独立采集线程：

```text
CameraThread(head)        -> 持续读取 head 最新帧
CameraThread(left_wrist)  -> 持续读取 left_wrist 最新帧
CameraThread(right_wrist) -> 持续读取 right_wrist 最新帧
```

这样做的原因：

- 任意一路 OpenCV `read()` 阻塞时，不会拖住另外两路。
- 三路相机可以并行解码和更新缓存，chunk 边界只做一次快照读取。
- policy 推理和 action 执行期间，相机线程仍然继续更新 latest frame。
- 保存和推理都能拿到“当前最新可用”的三路图像，而不是在主线程里串行等待三次 camera read。

每个线程只负责自己的 camera slot：

```python
def _camera_worker(slot: CameraSlotConfig) -> None:
    while not stop_event.is_set():
        image = read_one_frame(slot)
        frame = LatestFrame(...)
        with slot_lock:
            latest_frames[slot.key] = frame
```

线程共享数据结构建议：

```python
self._threads: dict[str, threading.Thread]
self._stop_events: dict[str, threading.Event]
self._frame_locks: dict[str, threading.Lock]
self._latest_frames: dict[str, LatestFrame]
```

锁粒度保持在“每路一个 lock”，不要用一个全局大锁包住三路相机。`snapshot_latest()` 只需要很短时间拿锁复制三路 `LatestFrame` 引用，不在锁里做图像编码、保存 PNG 或长时间处理。

线程失败策略：

- worker 内部捕获相机读取异常，保存到 `self._thread_errors[key]`。
- `snapshot_latest()` 如果发现某路线程有错误，抛 `CameraError`。
- `stop_streaming()` 必须 set event、join 三个线程，然后释放 capture。
- 第一版不做自动重连；如果某路相机断开，直接报错停推理。

### 4.1 CameraSlotConfig 字段

建议扩展：

```python
@dataclass
class CameraSlotConfig:
    key: str
    source: str = "dummy"
    width: int = 424
    height: int = 240
    fps: float = 20.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotate_degrees: int = 0
    stereo_crop: str | None = None  # None, "left", "right"
```

`stereo_crop` 用于 head 相机如果是 side-by-side 双目图时裁半：

- `None`：不裁剪
- `"left"`：取左半边
- `"right"`：取右半边

裁剪之后再 resize 到 `424x240`。

### 4.2 LatestFrame 数据结构

每路相机后台线程只保留自己的最新帧：

```python
@dataclass
class LatestFrame:
    key: str
    image: np.ndarray
    frame_id: int
    wall_time: float
    monotonic_time: float
    source: str
    width: int
    height: int
```

时间戳含义：

- `wall_time`: `time.time()`，用于和日志/文件时间对齐。
- `monotonic_time`: `time.perf_counter()`，用于计算相对延迟和同步 age。

### 4.3 支持的 camera source

建议支持：

```text
dummy / zeros
图片路径 .png/.jpg/.jpeg/.bmp/.webp
数字 index，例如 0
/dev/videoX
```

OpenCV 打开设备后设置：

```python
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capture.set(cv2.CAP_PROP_FPS, fps)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

如果需要 MJPEG 以降低 USB 带宽，可后续加：

```python
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
```

### 4.4 后台采集接口

建议新增接口：

```python
cameras.connect_all()
cameras.start_streaming()
cameras.snapshot_latest(reference_time: float | None = None) -> dict[str, LatestFrame]
cameras.stop_streaming()
cameras.disconnect_all()
```

`start_streaming()` 要为每个 slot 启动一个线程。当前 checkpoint 需要三路相机，所以正常真机运行时应该启动 3 个线程：

```python
for slot in self.slots:
    thread = threading.Thread(
        target=self._camera_worker,
        args=(slot,),
        name=f"camera-{slot.key}",
        daemon=True,
    )
    thread.start()
```

`snapshot_latest()` 行为：

- 一次性取出 `head / left_wrist / right_wrist` 最新帧。
- 从三个线程维护的 latest frame 缓存读取，不直接调用 `capture.read()`。
- 如果某一路还没有帧，抛 `CameraError`。
- 如果 `reference_time` 不为空，计算每路 `age_ms = (reference_time - frame.monotonic_time) * 1000`。
- 如果某路图像太旧，例如超过 `max_camera_age_ms=150`，第一版直接抛 `CameraError`。

保留原有 `read()` 作为兼容接口：

- 如果 streaming 已启动，`read()` 可以返回 `snapshot_latest()` 的 image dict。
- 如果 streaming 未启动，保持同步读取行为。

## 5. Robot State 读取细节

状态读取走当前统一接口：

```python
robot_state = robot.get_state()
```

它内部读取：

```text
left_arm.get_joint_state()  -> shape (7,)
right_arm.get_joint_state() -> shape (7,)
left_hand.get_joint_state() -> shape (20,)
right_hand.get_joint_state()-> shape (20,)
```

最终得到：

```python
DualArmHandState(
    left_arm_q=(7,),
    right_arm_q=(7,),
    left_hand_q=(20,),
    right_hand_q=(20,),
)
```

保存时同时保存：

```json
{
  "flat": [54 values],
  "segments": {
    "left_arm": [7 values],
    "right_arm": [7 values],
    "left_hand": [20 values],
    "right_hand": [20 values]
  },
  "schema": {
    "state_unit": "deg",
    "action_unit": "deg",
    "order": {
      "left_arm": [0, 7],
      "right_arm": [7, 14],
      "left_hand": [14, 34],
      "right_hand": [34, 54]
    }
  }
}
```

状态读取也要带时间：

```python
state_read_start = time.perf_counter()
state_wall_start = time.time()
robot_state = robot.get_state()
state_read_end = time.perf_counter()
state_wall_end = time.time()
reference_time = 0.5 * (state_read_start + state_read_end)
```

`reference_time` 用于从相机缓存计算每路图像相对状态读取中心点的 age。

## 6. 采样保存格式

建议新增采样脚本：

```text
tianji_wuji_runtime/tools/capture_observation_samples.py
```

默认输出目录：

```text
tianji_wuji_runtime/test/observation_capture/run_YYYYMMDD_HHMMSS_xxxxxx/
```

目录结构：

```text
run_xxx/
  metadata.json
  samples.jsonl
  samples/
    sample_000000/
      state.json
      head.png
      left_wrist.png
      right_wrist.png
      observation.npz
      sample_metadata.json
    sample_000001/
      ...
```

### 6.1 metadata.json

记录本次采样配置：

```json
{
  "created_wall_time": 0.0,
  "created_iso": "2026-06-04T...",
  "robot_backend": "tianji_wuji_host",
  "robot_ip": "192.168.8.166",
  "left_hand_serial": "385E367D3533",
  "right_hand_serial": "3860364D3533",
  "camera": [
    "head:/dev/video0",
    "left_wrist:/dev/video2",
    "right_wrist:/dev/video4"
  ],
  "camera_width": 424,
  "camera_height": 240,
  "camera_fps": 20,
  "camera_keys": ["head", "left_wrist", "right_wrist"],
  "state_unit": "deg",
  "schema_order": "left_arm,right_arm,left_hand,right_hand"
}
```

### 6.2 samples.jsonl

每一行记录一个 sample：

```json
{
  "sample_index": 0,
  "wall_time": 0.0,
  "monotonic_time": 0.0,
  "state_read_start_monotonic": 0.0,
  "state_read_end_monotonic": 0.0,
  "state_read_latency_ms": 1.2,
  "reference_monotonic": 0.0,
  "paths": {
    "state": "samples/sample_000000/state.json",
    "head": "samples/sample_000000/head.png",
    "left_wrist": "samples/sample_000000/left_wrist.png",
    "right_wrist": "samples/sample_000000/right_wrist.png",
    "observation": "samples/sample_000000/observation.npz",
    "metadata": "samples/sample_000000/sample_metadata.json"
  },
  "camera": {
    "head": {
      "frame_id": 123,
      "frame_wall_time": 0.0,
      "frame_monotonic_time": 0.0,
      "age_ms": 12.3,
      "shape": [240, 424, 3],
      "dtype": "uint8"
    }
  }
}
```

### 6.3 observation.npz

为了后续接模型接口，建议直接保存一份 policy-ready observation 数组：

```text
video.head
video.left_wrist
video.right_wrist
state.left_arm_joint
state.right_arm_joint
state.left_hand
state.right_hand
```

对应 shape：

```text
video.* -> (1, 1, 240, 424, 3), uint8
state.left_arm_joint -> (1, 1, 7), float32
state.right_arm_joint -> (1, 1, 7), float32
state.left_hand -> (1, 1, 20), float32
state.right_hand -> (1, 1, 20), float32
```

language 不建议放进 npz，放在 `sample_metadata.json` 即可。

## 7. 采样脚本 CLI

建议命令：

```bash
conda run -n dexproj python tianji_wuji_runtime/tools/capture_observation_samples.py \
  --robot-backend tianji_wuji_host \
  --robot-ip 192.168.8.166 \
  --left-hand-serial 385E367D3533 \
  --right-hand-serial 3860364D3533 \
  --tianji-sdk-root /home/user/workspace/DexProj_back_up_0602/wuji-hand-teleop/src/output_devices/tianji_output \
  --tianji-config-path /home/user/workspace/DexProj_back_up_0602/wuji-hand-teleop/src/output_devices/tianji_output/tianji_output/config/ccs_m6.MvKDCfg \
  --camera head:/dev/video0 \
  --camera left_wrist:/dev/video2 \
  --camera right_wrist:/dev/video4 \
  --camera-fps 20 \
  --camera-width 424 \
  --camera-height 240 \
  --samples 20 \
  --sample-hz 1 \
  --task "pick up the object"
```

如果只调相机和 fake robot：

```bash
conda run -n dexproj python tianji_wuji_runtime/tools/capture_observation_samples.py \
  --robot-backend fake \
  --camera head:0 \
  --camera left_wrist:2 \
  --camera right_wrist:4 \
  --camera-fps 20 \
  --camera-width 424 \
  --camera-height 240 \
  --samples 5 \
  --sample-hz 1 \
  --task "debug observation"
```

## 8. 采样时序

单次 sample 的精确流程：

```text
state_t0 = perf_counter()
state_wall_t0 = time()
robot_state = robot.get_state()
state_t1 = perf_counter()
state_wall_t1 = time()

reference_time = 0.5 * (state_t0 + state_t1)
frames = cameras.snapshot_latest(reference_time=reference_time)

images = {key: frame.image for key, frame in frames.items()}
observation = obs_builder.build(robot_state, images, task)

保存 state.json
保存三张 PNG
保存 observation.npz
追加 samples.jsonl
```

这样 state 是“采样瞬间现读”，相机是“离 state 读取中心时间最近的最新缓存帧”。实际相机不是严格硬同步，但在 20Hz 和 `max_camera_age_ms=150` 下，足够作为实时 VLA 输入的第一版方案。

## 9. 接入 infer_tianji_wuji.py

后续把当前：

```python
robot_state = robot.get_state()
images = cameras.read()
observation = obs_builder.build(robot_state, images, args.task)
```

替换成：

```python
state_t0 = time.perf_counter()
robot_state = robot.get_state()
state_t1 = time.perf_counter()
reference_time = 0.5 * (state_t0 + state_t1)

frames = cameras.snapshot_latest(reference_time=reference_time)
images = {key: frame.image for key, frame in frames.items()}
observation = obs_builder.build(robot_state, images, args.task)
```

启动时增加：

```python
cameras.connect_all()
cameras.start_streaming()
time.sleep(args.camera_warmup_sec)
```

退出时增加：

```python
cameras.stop_streaming()
cameras.disconnect_all()
```

命令行增加：

```text
--camera-fps
--camera-width
--camera-height
--max-camera-age-ms
--camera-warmup-sec
```

推荐默认值：

```text
camera_fps: 20
camera_width: 424
camera_height: 240
max_camera_age_ms: 150
camera_warmup_sec: 1.0
```

## 10. 与 action chunk 的关系

实时推理闭环应保持：

```text
采样 observation
policy 输出 action chunk
executor 按 20Hz 执行 execution_horizon 步
chunk 执行结束
立刻采样下一次 observation
```

`executor.execute_chunk(...)` 是阻塞的，所以 while loop 下一轮开头天然就是 chunk 边界。采样逻辑放在 policy 调用前即可。

注意：

- 相机后台一直采。
- robot state 不需要后台一直采，chunk 边界现读即可。
- 如果需要调试每步跟踪误差，再单独打开 per-step state logging。
- 生产实时推理建议不要强制每步读手臂状态，避免给 SDK 增加额外负担。

## 11. 校验标准

第一阶段采样工具完成后，应满足：

```text
每个 sample 都有 3 张 PNG
PNG shape 全部是 240x424x3
PNG 是 RGB uint8
state.json 中 flat 长度为 54
state.json 中四段长度为 7/7/20/20
samples.jsonl 中每路 camera 都有 frame_id / frame timestamp / age_ms
observation.npz 中 video/state shape 与 policy 输入契约一致
```

建议增加测试：

- dummy camera 采样保存测试
- 图片源 resize 到 `424x240` 测试
- streaming 模式下三路相机分别启动 `head / left_wrist / right_wrist` 三个线程
- `snapshot_latest()` 无帧时报错测试
- stale frame 超过 `max_camera_age_ms` 报错测试
- fake robot 采样得到 54 维 state 测试

## 12. 风险点

1. 左右腕相机可能物理接反。
   采样工具保存 PNG 后，第一件事就是人工看 `left_wrist.png` 和 `right_wrist.png` 是否对应真实左右。

2. head 如果是 side-by-side stereo，必须明确裁哪一半。
   第一版建议参数化 `--head-stereo-crop left/right`。

3. `/dev/videoX` 实际分辨率可能不是请求值。
   所以输出前必须 resize 到 `424x240`。

4. Wuji 左手 SN 之前出现过找不到。
   运行 `tianji_wuji_host` 前要确认两个 SN 都能被 `wujihandpy` 枚举到。

5. 相机和机器人不是硬件同步。
   第一版用 monotonic timestamp + frame age 控制误差；后续如果需要更严，可以改为每路保留 ring buffer，并选离 `reference_time` 最近的一帧。

## 13. 推荐实施顺序

1. 扩展 `CameraManager`：
   `fps/width/height`、`/dev/videoX`、三路相机三线程后台 streaming、`LatestFrame`、`snapshot_latest()`。

2. 新增 `capture_observation_samples.py`：
   读取 robot state + 三路 camera snapshot，保存 state、图片、metadata、observation.npz。

3. 用 fake robot + dummy/image camera 跑通保存格式。

4. 用真实三路相机 + fake robot 验证图像 shape、左右对应和 timestamp。

5. 用真实 robot + dummy camera 验证手臂状态读取。

6. 用真实 robot + 真实三路相机采样 20 条，检查所有 sample。

7. 最后把同一套 snapshot 逻辑接入 `infer_tianji_wuji.py`。

## 14. 最终运行心智模型

三路相机是三个后台连续流，机器人状态是 chunk 边界现读。保存和推理使用同一份 observation 组织逻辑：

```text
state + latest camera frames + task
-> ObservationBuilder
-> GR00T policy-ready dict
```

第一阶段先把这份输入完整落盘，确认时间戳、shape、左右相机、状态单位都对。第二阶段只需要把“保存到磁盘”换成“送给 policy server”，主链路就能自然接上实时推理。
