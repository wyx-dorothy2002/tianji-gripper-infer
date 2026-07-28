# 天机双臂双夹爪算法推理接口

本文描述当前 `storage` 算法服务对外暴露的**推理接口**。调用方负责采集机器人状态和相机图像、调用算法服务、对返回动作做安全校验并下发给机器人；算法服务本身不直接连接或控制天机机器人。

## 1. 服务信息

| 项目 | 约定 |
| --- | --- |
| 协议 | WebSocket（二进制帧） |
| 地址 | `ws://<算法服务器IP>:8000` |
| 序列化 | MessagePack，NumPy 数组使用本仓库的 `openpi_client.msgpack_numpy` 编码 |
| 健康检查 | `GET http://<算法服务器IP>:8000/healthz`，成功返回 HTTP 200 和 `OK` |
| 鉴权 | 当前服务未实现鉴权；必须部署在受信任的内网并由网络策略限制访问。 |

服务端启动示例（在本仓库根目录执行）：

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config storage \
  --policy.dir /path/to/checkpoint/50000
```

服务监听 `0.0.0.0:8000`。将 `<算法服务器IP>` 替换为实际服务器的内网 IP；不要让机器人端继续使用 `127.0.0.1`，除非算法服务与机器人客户端在同一台机器上。

## 2. 交互流程

一条 WebSocket 连接可连续执行多轮推理，且每次只能有一个未完成的请求。

1. 客户端连接 `ws://<host>:8000`。
2. 服务端先发送一条 MessagePack 二进制帧，内容是模型元数据 `dict`。客户端必须先读取该帧；目前客户端可以只记录或忽略其内容。
3. 客户端发送一条观测请求（二进制 MessagePack 帧）。
4. 服务端返回一条推理结果（二进制 MessagePack 帧）。
5. 重复步骤 3–4；结束时关闭连接。

如果服务端推理发生异常，它会先返回文本格式的 Python 错误栈，再以 WebSocket 错误状态关闭连接。收到文本帧、连接关闭、超时或返回非法动作时，调用方应停止机器人并重新建立连接。

## 3. 请求数据

请求是以下 Python 字典的 MessagePack 编码：

```python
{
    "state": np.ndarray,   # float32，形状 (16,)
    "images": {
        "head": np.ndarray,         # uint8，H x W x 3，RGB
        "left_wrist": np.ndarray,   # uint8，H x W x 3，RGB
        "right_wrist": np.ndarray,  # uint8，H x W x 3，RGB
    },
    "prompt": str,  # 非空任务文本
}
```

### `state`：16 维本体状态

`state` 必须为有限 `float32` 数值，单位全部为**弧度（rad）**，顺序固定如下：

| 下标 | 字段 | 数据来源 | 单位 |
| --- | --- | --- | --- |
| `0:7` | 右臂 7 个关节 | 天机 B 臂实时关节反馈 | rad |
| `7:14` | 左臂 7 个关节 | 天机 A 臂实时关节反馈 | rad |
| `14` | 右夹爪位置 | 天机 B 臂末端夹爪反馈 | rad |
| `15` | 左夹爪位置 | 天机 A 臂末端夹爪反馈 | rad |

接口顺序是“右臂、左臂、右夹爪、左夹爪”，而不是按天机 A/B 的顺序。天机 SDK 的关节反馈若为角度，调用方必须先转换：`state[0:14] = np.deg2rad(joints_deg)`；两侧夹爪保持其 RS05 标定后的弧度值。采集的三路图像和这 16 维状态应尽量来自同一控制时刻。

### `images`：三路 RGB 图像

三个键均为必填。每张图像必须是连续或可序列化的 `uint8`、`H x W x 3`、RGB 数组；不得传 BGR、灰度图、浮点归一化图像或 JPEG 字节串。当前机器人运行时采集分辨率为 `424 x 240`，服务端会在模型预处理阶段缩放到模型输入尺寸，因此接口不要求调用方先缩放。

### `prompt`：任务文本

传入非空的 UTF-8 字符串，例如：

```text
Put all the objects on the desk into the storage box
```

建议在一个任务执行期间保持文本不变；切换任务时更新 `prompt`。

## 4. 响应数据

响应是以下结构的 MessagePack 二进制帧：

```python
{
    "actions": np.ndarray,       # float32，形状 (H, 16)
    "server_timing": {           # 可选诊断字段，单位 ms
        "infer_ms": float,
        "prev_total_ms": float,  # 从第二次响应开始可能出现
    },
    # 可能还有模型侧诊断字段；调用方应忽略未知字段。
}
```

`H` 为模型返回的动作序列长度，不应由客户端假定为固定值。调用方应校验 `H >= 1`、第二维严格等于 `16`、所有元素均为有限数值。当前机器人运行时每次只取前 `16` 步执行；这是客户端执行策略，不是 WebSocket 服务端的限制。

每一行 `actions[i]` 的字段顺序与状态完全一致：

| 下标 | 字段 | 算法输出单位 | 下发前处理 |
| --- | --- | --- | --- |
| `0:7` | 右臂（B 臂）目标关节 | rad | 转换为 deg 后下发给天机 B 臂 |
| `7:14` | 左臂（A 臂）目标关节 | rad | 转换为 deg 后下发给天机 A 臂 |
| `14` | 右夹爪（B 臂）目标位置 | rad | 直接作为 RS05 位置目标下发 |
| `15` | 左夹爪（A 臂）目标位置 | rad | 直接作为 RS05 位置目标下发 |

在当前部署配置中，动作被解释为**绝对位置目标**，而不是相对增量。调用方在任何下发前必须按机器人标定的关节/夹爪限位、单步变化和速度限制进行裁剪；不能把网络返回值未经校验直接发送给硬件。

## 5. Python 调用示例

推荐直接复用仓库内的 NumPy MessagePack 编码器，避免自行实现跨语言数组编码细节。

```python
import numpy as np
from websockets.sync.client import connect

from openpi_client import msgpack_numpy


def infer_once(host: str, state_rad: np.ndarray, images_rgb: dict[str, np.ndarray], prompt: str):
    state = np.asarray(state_rad, dtype=np.float32).reshape(-1)
    if state.shape != (16,):
        raise ValueError(f"state must have shape (16,), got {state.shape}")

    request = {
        "state": state,
        "images": {
            name: np.asarray(images_rgb[name], dtype=np.uint8)
            for name in ("head", "left_wrist", "right_wrist")
        },
        "prompt": prompt.strip(),
    }
    if not request["prompt"]:
        raise ValueError("prompt must not be empty")

    with connect(
        f"ws://{host}:8000",
        compression=None,
        max_size=None,
        open_timeout=15,
        ping_interval=None,
    ) as websocket:
        metadata_frame = websocket.recv()  # 服务端握手元数据，必须先读取。
        if isinstance(metadata_frame, str):
            raise RuntimeError(f"unexpected handshake error: {metadata_frame}")
        metadata = msgpack_numpy.unpackb(metadata_frame)
        print("server metadata:", metadata)

        websocket.send(msgpack_numpy.packb(request))
        response_frame = websocket.recv()
        if isinstance(response_frame, str):
            raise RuntimeError(f"server inference error:\n{response_frame}")
        response = msgpack_numpy.unpackb(response_frame)

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16 or actions.shape[0] < 1:
        raise RuntimeError(f"invalid actions shape: {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("actions contain NaN or Inf")
    return actions, response.get("server_timing", {})


# state_rad 和 images_rgb 由机器人客户端在同一时刻采集。
# actions, timing = infer_once("192.168.x.x", state_rad, images_rgb, task_text)
```

若调用方使用 C++、Java 或其他语言，应按 `packages/openpi-client/src/openpi_client/msgpack_numpy.py` 复现数组编码：每个 NumPy 数组编码为包含 `__ndarray__`、`data`、`dtype`、`shape` 的 MessagePack map。其中 `data` 是按 C 顺序排列的原始字节，`dtype` 对本接口分别为 `<f4`（状态/动作）和 `|u1`（图像）。建议优先提供 Python 适配进程或复用该编码器，以避免字节序和形状错误。

## 6. 机器人侧执行要求

1. 每轮请求前读取双臂和双夹爪的实时反馈，并同步采集三路图像。
2. 校验返回动作的形状、有限性、关节/夹爪限位、单步变化和速度。
3. 将关节目标从 rad 转为天机 SDK 所需的 deg；夹爪保持 rad。
4. 按控制周期执行有限个动作，例如当前运行时的前 16 步、每步 `0.05 s`。
5. 通信异常、超时、安全校验失败或急停时，立即停止取新动作并保持机器人当前位置。

本仓库已有可参考的完整客户端实现：请求构造见 `runtime/observation_builder.py`，WebSocket 收发见 `runtime/pi_policy_client.py`，动作拆分、单位转换和下发见 `runtime/action_adapter.py` 与 `runtime/robot_interface.py`。
