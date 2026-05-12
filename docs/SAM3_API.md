# SAM3 HTTP API

本服务由 [`servers.py`](../servers.py) 提供，是一个 FastAPI + uvicorn 进程，负责加载并对外暴露 SAM3 的三套模型（image / tracker / video）。所有项目内的调用方（`app.py`、`scripts/blade/predictors.py`、`scripts/batch_detect_persons.py`、`scripts/classify_persons.py`、`scripts/generate_report.py`）都通过 HTTP 调用本服务，进程内**没有**重复加载 SAM3。

---

## 1. 概览

| 项 | 值 |
|---|---|
| 进程 | `python servers.py`（FastAPI / uvicorn） |
| 绑定 | `${SAM3_HOST:-0.0.0.0}:${SAM3_PORT:-5050}` |
| 宿主端口（compose） | `${SAM3_HOST_PORT:-5050}` → 容器 5050 |
| 客户端基础 URL | `${SAM3_SERVER_URL:-http://127.0.0.1:5050}` |
| 模型 | `Sam3Model` + `Sam3TrackerModel` + `Sam3VideoModel` |
| 冷启动 | ~30–45s（首次 `/health` 三项全 true 即就绪） |
| 显存 | 三个模型常驻 ~7-9 GB |
| 离线 | `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，权重从 `./models/facebook/sam3` 读 |

启动时通过 `Sam3VideoModel.from_pretrained` 一次性加载权重，`Sam3Model` / `Sam3TrackerModel` / `Sam3VideoModel` 三个对象**共享同一份 vision_encoder backbone**（GPU 内存里只有 1 份骨干副本）。冷启动 ≈ 30-45s，显存常驻 ~7-9GB。

启动方式：

- 本地裸机：`make serve`（前台）或 `make dev`（后台 server + 前台 UI）
- 容器：`make deploy`（build + up + 健康校验，详见 [Makefile](../Makefile)）

---

## 2. 端点目录

| Method | Path | 用途 | 后端模型 |
|---|---|---|---|
| GET | `/health` | 健康检查 / 模型加载状态 | — |
| POST | `/v1/sam3/detect` | 图像 + 文本 prompt → bbox + mask | `Sam3Model`（实例 = `VID_MODEL.detector_model`） |
| POST | `/v1/sam3/track` | 图像 + 点击点 → 渲染好的 overlay 图像 | `Sam3TrackerModel`（vision_encoder 共享 detector backbone） |
| POST | `/v1/sam3/video` | 视频路径 + 文本 prompt → 渲染好的 mp4 | `Sam3VideoModel` |

---

## 3. `GET /health`

无请求体。响应（[servers.py:352-362](../servers.py#L352-L362)）：

```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": {
    "sam3_image": true,
    "sam3_tracker": true,
    "sam3_video": true
  }
}
```

**就绪判定**：Makefile / entrypoint / docker healthcheck 全部用同一条 grep —— `'"sam3_video":true'`。视频模型是最后加载的，它 true 则其它两项必然 true。

---

## 4. `POST /v1/sam3/detect`

**请求**（[servers.py:294-299](../servers.py#L294-L299) `DetectRequest`）：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image_b64` | string | — | base64 编码的 PNG/JPG（不带 `data:` 前缀） |
| `prompt` | string | — | 文本 prompt，必填非空 |
| `conf_threshold` | float | `0.45` | 候选 region 的置信度下限 |
| `mask_threshold` | float | `0.5` | mask 二值化阈值 |
| `return_masks` | bool | `true` | 是否在响应里返回 `mask_b64` |

**响应**（[servers.py:302-312](../servers.py#L302-L312) `DetectResponse` / `Region`）：

```json
{
  "width": 1920,
  "height": 1080,
  "regions": [
    {
      "region_index": 0,
      "bbox": [x1, y1, x2, y2],
      "score": 0.83,
      "mask_b64": "<base64 PNG L-mode 或 null>"
    }
  ]
}
```

- `bbox` 是 xyxy 像素整数，已 clamp 到 `[0, width-1] × [0, height-1]`。
- `mask_b64` 为单通道 PNG（L mode），尺寸 = 原图。`return_masks=false` 时为 `null`。
- 没有 region 匹配时返回 `regions: []`，仍 200。

**错误码**：

| 状态 | 原因 |
|---|---|
| 400 | `prompt is required` |
| 503 | `SAM3 image model not loaded`（启动尚未完成 / 加载失败） |

---

## 5. `POST /v1/sam3/track`

**请求**（[servers.py:315-319](../servers.py#L315-L319) `TrackRequest`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image_b64` | string | base64 编码图像 |
| `points` | `int[][]` | `[[x,y], ...]` 累积点击坐标 |
| `labels` | `int[]` | 每点的前/背景标签（1=foreground、0=background），长度需 = `points` |

**响应**（[servers.py:321-323](../servers.py#L321-L323) `TrackResponse`）：

```json
{
  "overlay_image_b64": "<base64 PNG>",
  "has_mask": true
}
```

服务端会直接把 mask + 点位绘制到图像上返回——客户端不需要自己合成。`has_mask=false` 表示当前点位组合没产生有效 mask（仍会返回画了点位的原图）。

**错误码**：400（points 为空 / 长度不匹配）、503（tracker 未加载）。

---

## 6. `POST /v1/sam3/video`

**请求**（[servers.py:326-331](../servers.py#L326-L331) `VideoRequest`）：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `video_path` | string | — | **服务进程能读到的本地路径**（见下方注意） |
| `prompt` | string | — | 文本 prompt |
| `frame_limit` | int | `60` | 处理帧数上限 |
| `time_limit` | int | `60` | 预留字段，当前不强制 |
| `render_mode` | `"annotated"` \| `"mask"` | `"annotated"` | annotated = mask + 轮廓 + bbox + label；mask = 半透明彩色覆盖 |

**视频路径要求**：当前单容器部署下，调用方与服务在同一进程空间，路径可以是临时文件。若以后拆成"两容器"，需把 `SAM3_VIDEO_OUT_DIR` 挂为共享 volume，让两端都能 read/write（详见 [servers.py:58-63 注释](../servers.py#L58-L63)）。`servers_client.sam3_video` 已经做了"按需 stage 到 `SAM3_VIDEO_OUT_DIR/inputs/`" 的兜底。

**响应**（[servers.py:334-338](../servers.py#L334-L338) `VideoResponse`）：

```json
{
  "output_video_path": "/var/sam3_data/sam3_video_xxx.mp4",
  "processed_frames": 60,
  "masked_frames": 41,
  "status": "render_mode=annotated; processed 60, masked 41"
}
```

**错误码**：400（路径不可读 / prompt 空 / render_mode 非法 / 视频无可读帧）、503（video 模型未加载）。

---

## 7. Python 客户端

项目内推荐统一通过 [`servers_client.py`](../servers_client.py) 调用，它已经封装了 base64 编解码、httpx 超时、错误抽象（`ServerError`）、以及视频文件的 stage 逻辑。

```python
import servers_client

# 健康
servers_client.health()

# 检测
regions = servers_client.sam3_detect(pil_image, "blade defect", conf_threshold=0.45)
# regions: list[DetectedRegion(region_index, bbox, score, mask_np)]

# 追踪
overlay_pil, has_mask = servers_client.sam3_track(pil_image, points=[[100,200]], labels=[1])

# 视频
out_path, processed, masked = servers_client.sam3_video(
    video_path="/abs/path.mp4", prompt="blade defect", frame_limit=60
)
```

环境变量：`SAM3_SERVER_URL`（默认 `http://127.0.0.1:5050`）。

---

## 8. cURL 速查

```bash
# 健康
curl -fs http://localhost:5050/health | jq

# 检测
b64=$(base64 -w0 sample.jpg)
curl -s -X POST http://localhost:5050/v1/sam3/detect \
  -H 'Content-Type: application/json' \
  -d "{\"image_b64\":\"$b64\",\"prompt\":\"blade defect\"}" \
  | jq '{w:.width,h:.height,n:(.regions|length),scores:[.regions[].score]}'

# 视频（路径必须服务侧可读）
curl -s -X POST http://localhost:5050/v1/sam3/video \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"/var/sam3_data/inputs/x.mp4","prompt":"blade defect","frame_limit":60,"render_mode":"annotated"}' \
  | jq
```

---

## 9. 部署一览

| 场景 | API 地址 | UI 地址 |
|---|---|---|
| 本地 `make dev` | `http://127.0.0.1:5050` | `http://127.0.0.1:7860` |
| 容器 `make deploy`（默认 host port） | `http://localhost:5050` | `http://localhost:17860` |
| 容器（多实例） | `http://localhost:${SAM3_HOST_PORT}` | `http://localhost:${GRADIO_HOST_PORT}` |

容器内部 entrypoint 始终先起 server（5050），健康检查通过后再 exec UI（7860）—— Gradio UI 进程会通过 `127.0.0.1:5050` 调用 server，**和外部调用方走同一份 API**。

---

## 10. 并发模型

服务进程持有一把全局推理锁（`threading.Lock`）。三个 endpoint 的 model forward
阶段是**串行**的——并发请求会在锁上 FIFO 排队，**任一时刻最多一个推理在跑**。

- 视频 endpoint 锁住整个帧 propagate 循环（视频 session 跨帧有状态）。
- 视频解码/编码 I/O 在锁外，多请求的 I/O 可并行。
- detect/track 单次推理通常几百毫秒，排队对前端影响小。
- 这是为了配合"vision_encoder 单份显存"的策略，避免并发推理把 backbone activations 撑爆 VRAM。

如果要提高并发，唯一安全的方式是再起一个容器实例（用 `GRADIO_HOST_PORT` / `SAM3_HOST_PORT` 多实例部署，见 docker-compose.yml 顶部注释）。
