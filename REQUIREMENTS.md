# RTSP 人员检测流处理系统 — 需求文档

## 1. 项目背景

在现有 YOLO11n RKNN 图片检测方案基础上，扩展为实时视频流检测系统：
从多路 RTSP 摄像头流中读取视频帧，通过 YOLO 进行目标检测，将标注结果合成一路画面推送到 go2rtc，
供下游消费者订阅。

硬件平台：Orange Pi 5 Max（RK3588），使用 RKNNLite 推理。

---

## 2. 功能需求

### 2.1 输入

| 项目 | 描述 |
|------|------|
| 协议 | RTSP（H.264 / H.265） |
| 来源 | go2rtc 已接入的小米摄像头流（`rtsp://localhost:8554/micam_*`） |
| 路数 | 可配置，当前 3 路 |

当前配置的摄像头流：
- `micam_bedroom`
- `micam_kitchen`
- `micam_gate`

### 2.2 检测处理

**各路独立推理**

每路摄像头帧独立经过 letterbox 缩放至 640×640 后送入 YOLO，各自完成推理与坐标反算，保证每路均以完整分辨率检测。

```
Camera 0 ──► letterbox(640×640) ──► YOLO ──► 标注帧 0 ──┐
Camera 1 ──► letterbox(640×640) ──► YOLO ──► 标注帧 1 ──┤──► 2-col Grid ──► /grid
Camera 2 ──► letterbox(640×640) ──► YOLO ──► 标注帧 2 ──┘
```

- 每路 3 次独立 NPU 推理，检测精度与单路相同
- 推理完成后，3 路标注帧通过 `grid.compose` 拼成 2 列网格图输出

### 2.3 检测目标

- 默认只检测 `person`（COCO class_id = 0）
- 目标类别通过配置文件指定，支持多类别，如 `["person", "car"]`
- 置信度阈值（`conf_thresh`）和 NMS 阈值（`nms_thresh`）可配置

### 2.4 帧率控制

- 输入流持续读取（防止缓冲区堆积）
- 推理帧率（`process_fps`）可配置，默认 10 FPS
- 超过目标帧率的帧直接丢弃
- 输出流帧率与推理帧率一致

### 2.5 延迟控制

- 端到端延迟目标可配置，默认 ≤ 1 秒
- OpenCV 读取时通过 FFmpeg 参数降低缓冲：
  `-rtsp_transport tcp -fflags nobuffer -flags low_delay -max_delay <ms>`

### 2.6 多路同步

- 多路流**不强制同步**，每路取当前最新帧
- 若某路流帧尚未到达，使用该路上一帧

### 2.7 输出

| 项目 | 描述 |
|------|------|
| 协议 | HTTP MJPEG（`multipart/x-mixed-replace`） |
| 服务地址 | `http://0.0.0.0:8090/grid` |
| 画面内容 | 3 路标注帧拼合的 2 列网格图 |
| go2rtc 流名 | `cam_dashboard` |
| go2rtc 消费地址 | `rtsp://localhost:8554/cam_dashboard` |
| JPEG 质量 | 可配置，默认 85 |

输出帧叠加内容：
- 检测框（绿色矩形）+ 标签（类别名 + 置信度）
- 左上角 OSD：当前推理 FPS、本帧检测目标总数

---

## 3. 配置文件规范

路径通过环境变量 `CONFIG_PATH` 指定，默认 `/app/config.yaml`。

```yaml
model:
  path: /app/models/yolo11n.rknn
  input_size: 640        # YOLO 输入边长（正方形）
  conf_thresh: 0.25
  nms_thresh: 0.45
  target_classes:
    - person

processing:
  process_fps: 10         # 推理目标帧率
  max_delay_ms: 1000      # RTSP 读取最大缓冲延迟（毫秒）

output:
  mjpeg_port: 8090
  jpeg_quality: 85

streams:
  - name: bedroom
    input: rtsp://localhost:8554/micam_bedroom
  - name: kitchen
    input: rtsp://localhost:8554/micam_kitchen
  - name: gate
    input: rtsp://localhost:8554/micam_gate
```

---

## 4. 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  Docker Container: yolo-detector                                     │
│                                                                      │
│  ReaderThread × 3                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  bedroom    │  │  kitchen    │  │  gate       │                  │
│  │  cap.read() │  │  cap.read() │  │  cap.read() │                  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         └────────────────┼────────────────┘                         │
│                          ▼                                           │
│              Main Loop (10 FPS)                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  for each frame:                                              │   │
│  │    letterbox → 640×640 → RKNNLite → postprocess → draw       │   │
│  │  compose(annotated × 3) → grid_frame                         │   │
│  │  OSD overlay (FPS + count)                                   │   │
│  └───────────────────────────┬───────────────────────────────────┘   │
│                              ▼                                       │
│              MJPEGServer :8090/grid                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP MJPEG pull
┌──────────────────────────────▼───────────────────────────────────────┐
│  go2rtc                                                              │
│  cam_dashboard ←── http://localhost:8090/grid                        │
│  rtsp://localhost:8554/cam_dashboard  (消费者订阅)                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. 文件结构

```
orange-pi-rknn/
├── detect.py                    # 图片单张检测（独立工具）
├── convert_rknn.py              # ONNX → RKNN 模型转换
├── yolo11n.rknn                 # 模型文件（挂载入容器）
├── stream_detector/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py                  # 入口：加载配置，启动处理循环
│   ├── config.py                # YAML 配置加载与 dataclass 校验
│   ├── reader.py                # RTSP 读取线程（断流自动重连）
│   ├── grid.py                  # 2-col 网格拼合（仅用于输出合成）
│   ├── detector.py              # 各路独立 RKNNLite 推理封装
│   └── writer.py                # HTTP MJPEG 服务器
├── config.yaml                  # 运行配置
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
└── rknn_compatible              # RK3588 平台标识文件
```

---

## 6. 关键模块说明

### 6.1 `reader.py` — RTSP 读取线程

- 每路流独立线程，循环 `cap.read()` 保持最新帧
- 断流后指数退避重连（1→2→4→…→30 秒）
- OpenCV 打开参数：`-rtsp_transport tcp -fflags nobuffer -flags low_delay`

### 6.2 `detector.py` — 推理封装

- 对每帧独立执行：`letterbox → RKNNLite.inference → NMS → draw`
- 按 `target_classes` 过滤结果
- 返回标注后的原始尺寸帧

### 6.3 `grid.py` — 网格合成（输出用）

- `compose(frames)`：将标注帧按 2 列排列拼成网格图，供 MJPEG 输出
- `decompose` 保留但不参与当前推理流程

### 6.4 `writer.py` — HTTP MJPEG 服务器

- `ThreadingHTTPServer` 监听 `0.0.0.0:8090`
- 每个已注册路径维护独立帧队列，支持多客户端并发拉取
- `push(name, frame)` 编码 JPEG 并广播给所有连接的客户端

---

## 7. go2rtc 配置

在 go2rtc.yaml 的 `streams:` 中添加：

```yaml
streams:
  cam_dashboard:
    - http://localhost:8090/grid
```

消费地址：`rtsp://localhost:8554/cam_dashboard`

---

## 8. Docker 部署

### 开发/测试直接运行

```bash
cd /home/orangepi/orange-pi-rknn
source .venv/bin/activate
CONFIG_PATH=config.yaml python3 -m stream_detector
```

### 生产部署

```bash
docker compose up --build -d
docker logs -f orange-pi-rknn-yolo-detector-1
```

rknnlite 与 librknnrt 通过 volume 从宿主机挂载，无需在容器内安装。

---

## 9. 硬件 & 技术选型说明

| 项目 | 状态 | 说明 |
|------|------|------|
| HTTP MJPEG 输出 | ✅ 选用 | 纯 Python，无额外依赖，go2rtc 原生支持拉取 |
| go2rtc RTSP push | ❌ 不可用 | 需预声明 stream，无法动态推流 |
| RKNNLite 推理 | ✅ 可用 | 挂载自 `~/.local/lib/python3.10/site-packages/rknnlite` |
| 网格单次推理 | ❌ 放弃 | 各路分辨率被稀释，检测效果差 |
| 各路独立推理 | ✅ 选用 | 每路 640×640 完整精度，3 次 NPU 调用/帧 |

---

## 10. 性能参考（RK3588，yolo11n）

| 配置 | 每帧 NPU 调用 | 每路 YOLO 输入分辨率 |
|------|-------------|-------------------|
| 3 路独立推理 | 3 次 | 640×640（完整） |
| 3 路 grid 单次推理（已弃用） | 1 次 | ~320×320（降级） |

小米 069a01 摄像头实际分辨率：848×480（SD 子码流）。

---

## 11. 已确认事项

| 事项 | 结论 |
|------|------|
| go2rtc 输出方式 | HTTP MJPEG，go2rtc 主动拉取 `/grid` |
| go2rtc 流名 | `cam_dashboard` |
| RKNN 模型路径 | `/home/orangepi/orange-pi-rknn/yolo11n.rknn` |
| rknnlite 安装位置 | `~/.local/lib/python3.10/site-packages/rknnlite` |
| 摄像头实际分辨率 | 848×480（SD 子码流） |
| 推理策略 | 各路独立 640×640 推理，标注后合成 grid 输出 |
| 帧同步策略 | 各路取最新帧，不等待 |
