# RTSP 人员检测流处理系统 — 需求文档

## 1. 项目背景

在现有 YOLO11n RKNN 图片检测方案基础上，扩展为实时视频流检测系统：
从一路或多路 RTSP 摄像头流中读取视频帧，通过 YOLO 进行目标检测，将标注结果推送到 go2rtc，
供下游消费者订阅。

硬件平台：Orange Pi 5 Max（RK3588），使用 RKNNLite 推理。

---

## 2. 功能需求

### 2.1 输入

| 项目 | 描述 |
|------|------|
| 协议 | RTSP（H.264 / H.265） |
| 来源 | go2rtc 已接入的小米摄像头流（`rtsp://localhost:8554/micam_*`） |
| 示例 URL | `rtsp://localhost:8554/micam_bedroom` |
| 鉴权 | 支持 URL 内嵌用户名/密码 |
| 路数 | 可配置，初期 1 路，架构支持多路 |

go2rtc 当前已配置的摄像头流：
- `micam_kitchen`
- `micam_gate`
- `micam_bedroom`

### 2.2 检测处理

**N×M 网格合并推理（Grid Inference）**

多路摄像头帧以原始分辨率拼合成 2 列 × ⌈N/2⌉ 行的网格图，整体 letterbox 缩放至 640×640
后送入 YOLO，单次推理覆盖全部画面，再逐级逆映射回各路原始帧坐标。

```
Camera 0 ──┐
Camera 1 ──┤──► [2-col Grid @ orig res] ──► letterbox(640×640) ──► YOLO
Camera 2 ──┤
Camera N ──┘
                    ──► [逆 letterbox] ──► [逆 grid cell] ──► 各路标注帧
```

- **固定 2 列**，行数 = ⌈流路数 / 2⌉，单路时退化为 1×1
- 每个 Cell 尺寸 = 原始分辨率（不预缩放），Grid 图整体做 letterbox → 640×640
- 坐标链：YOLO 空间 → 去 letterbox → 网格图空间 → 去 Cell 偏移 → 原始帧空间

### 2.3 检测目标

- 默认只检测 `person`（COCO class_id = 0）
- 目标类别通过配置文件指定，支持多类别，如 `["person", "car"]`
- 置信度阈值（`conf_thresh`）和 NMS 阈值（`nms_thresh`）可配置

### 2.4 帧率控制

- 输入流按原始帧率持续 `grab()`（防止缓冲区堆积）
- 推理帧率（`process_fps`）可配置，默认 10 FPS
- 超过目标帧率的帧直接丢弃
- 输出流帧率与推理帧率一致

### 2.5 延迟控制

- 端到端延迟目标可配置，默认 ≤ 1 秒
- OpenCV 读取时通过 FFmpeg 参数降低缓冲：
  `-rtsp_transport tcp -fflags nobuffer -flags low_delay -max_delay <ms>`
- 延迟参数通过 `max_delay_ms` 配置项控制

### 2.6 多路同步

- 多路流**不强制同步**，每路取当前最新帧直接拼合
- 若某路流帧尚未到达，使用该路上一帧

### 2.7 输出

| 项目 | 描述 |
|------|------|
| 协议 | HTTP MJPEG（`multipart/x-mixed-replace`） |
| 服务方式 | Python 内置 HTTP 服务器监听 `0.0.0.0:8090` |
| go2rtc 集成 | go2rtc 主动从 `http://localhost:8090/<name>` 拉取 MJPEG，再转发给消费者 |
| 分辨率 | 与输入分辨率保持一致 |
| 输出路数 | 每路输入对应一个 MJPEG 路径 |
| 示例 MJPEG URL | `http://localhost:8090/bedroom` |
| 示例 go2rtc URL | `rtsp://localhost:8554/detected_bedroom`（经 go2rtc 转发） |

go2rtc.yaml 需一次性添加 detected_* 流声明（见第 7 节），
go2rtc 重启后自动开始从 MJPEG 服务器拉流。

> **方案选型原因**：go2rtc RTSP 服务器不接受无预声明的推流；WHIP(WebRTC) 编码复杂；
> HTTP MJPEG 为纯 Python 实现，无额外依赖，go2rtc 原生支持作为 source。

输出帧叠加内容：
- 检测框（绿色矩形）+ 标签（类别名 + 置信度）
- 左上角 OSD：当前推理 FPS、本帧检测目标数

---

## 3. 配置文件规范

路径通过环境变量 `CONFIG_PATH` 指定，默认 `/app/config.yaml`。

```yaml
# config.yaml 示例

model:
  path: /app/models/yolo11n.rknn
  input_size: 640        # YOLO 输入边长（正方形）
  conf_thresh: 0.25
  nms_thresh: 0.45
  target_classes:
    - person              # 可添加其他 COCO 类别，如 car、bicycle

processing:
  process_fps: 10         # 推理目标帧率
  max_delay_ms: 1000      # RTSP 读取最大缓冲延迟（毫秒）

output:
  encoder: libx264        # 当前平台唯一可用编码器
  encoder_preset: ultrafast

streams:
  - name: bedroom
    input:  rtsp://localhost:8554/micam_bedroom
    output: rtsp://localhost:8554/detected_bedroom
  # 多路示例（自动组成 2 列 × 2 行 grid）
  # - name: kitchen
  #   input:  rtsp://localhost:8554/micam_kitchen
  #   output: rtsp://localhost:8554/detected_kitchen
  # - name: gate
  #   input:  rtsp://localhost:8554/micam_gate
  #   output: rtsp://localhost:8554/detected_gate
```

---

## 4. 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│  宿主机 / Docker Container: yolo-rtsp-detector                           │
│                                                                          │
│  ┌─────────────────┐   ┌───────────────────────────────────────────────┐ │
│  │  ReaderThread   │   │             Main Loop (单线程推理)             │ │
│  │  (per stream)   │   │                                               │ │
│  │                 │   │  1. 从各 Reader 取最新帧                      │ │
│  │  cap.grab() ──► │──►│  2. 2-col Grid 拼合（原始分辨率）             │ │
│  │  latest_frame   │   │  3. letterbox → 640×640                      │ │
│  └─────────────────┘   │  4. RKNNLite 推理 (YOLO11n)                  │ │
│                        │  5. 坐标逆映射链                              │ │
│                        │  6. 类别过滤 + 绘制标注框                     │ │
│                        └───────────────────┬───────────────────────────┘ │
│                                            │ BGR frames                   │
│                        ┌───────────────────▼───────────────────────────┐ │
│                        │   FFmpeg Subprocess (per stream)               │ │
│                        │   rawvideo pipe → libx264 → -f rtsp            │ │
│                        └───────────────────┬───────────────────────────┘ │
└────────────────────────────────────────────┼─────────────────────────────┘
                                             │ RTSP push (localhost:8554)
┌────────────────────────────────────────────▼─────────────────────────────┐
│  Docker Container: go2rtc  (host network)                                │
│  ├─ micam_bedroom  ←── xiaomi 摄像头（已有）                             │
│  ├─ micam_kitchen  ←── xiaomi 摄像头（已有）                             │
│  ├─ micam_gate     ←── xiaomi 摄像头（已有）                             │
│  ├─ detected_bedroom  ←── yolo-detector push（新增）                     │
│  └─ detected_*        ←── yolo-detector push（新增）                     │
│  Config: /mnt/shared/docker-container/go2rtc/go2rtc.yaml                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 文件结构

```
orange-pi-rknn/
├── detect.py                    # 现有：图片检测（保留不动）
├── convert_rknn.py              # 现有：模型转换
├── yolo11n.rknn                 # 模型文件（挂载入容器）
├── stream_detector/
│   ├── __init__.py
│   ├── main.py                  # 入口：加载配置，启动处理循环
│   ├── config.py                # YAML 配置加载与 dataclass 校验
│   ├── reader.py                # RTSP 读取线程（grab 循环 + 断流重连）
│   ├── grid.py                  # 2-col 网格拼合 & 坐标逆映射链
│   ├── detector.py              # RKNNLite 推理封装（复用 detect.py 逻辑）
│   └── writer.py                # FFmpeg pipe RTSP 推流（进程守护）
├── config.yaml                  # 用户运行配置
├── Dockerfile
└── docker-compose.yml
```

---

## 6. 关键模块说明

### 6.1 `reader.py` — RTSP 读取线程

- 每路流独立 `threading.Thread`，循环调用 `cap.grab()` 消耗缓冲区
- `retrieve()` 仅在主循环请求时调用，保证取到最新帧
- 断流后指数退避重连（1→2→4→…→30 秒）
- OpenCV 打开参数：`-rtsp_transport tcp -fflags nobuffer -flags low_delay -max_delay <max_delay_ms * 1000>`

### 6.2 `grid.py` — 网格合并与坐标映射

**拼合（compose）**
```
frames = [f0(W×H), f1(W×H), f2(W×H)]   # 原始分辨率，不缩放
cols = 2, rows = ceil(N/2)
grid_img = 2W × rows*H                  # 直接拼合，不预缩放
grid_letterboxed, r, pad = letterbox(grid_img, 640)
```

**逆映射（decompose）**
```
# YOLO空间 → 网格图空间
x_grid = (x_yolo - pad_x) / r
y_grid = (y_yolo - pad_y) / r

# 网格图空间 → 原始帧空间
col_idx = x_grid // W
row_idx = y_grid // H
stream_idx = row_idx * cols + col_idx
x_orig = x_grid - col_idx * W
y_orig = y_grid - row_idx * H
```

过滤掉框中心不在目标 Cell 范围内的溢出检测。

### 6.3 `detector.py` — 推理封装

- 复用现有 `letterbox`、`postprocess`、`draw` 函数
- 在网格图上做整体 letterbox → RKNN 推理 → NMS
- postprocess 返回坐标为网格图空间，交由 `grid.py` 继续逆映射
- 按 `target_classes` 过滤结果

### 6.4 `writer.py` — FFmpeg 推流

```
Python (BGR numpy) ──stdin pipe──►
  ffmpeg -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i pipe:0
         -c:v libx264 -preset ultrafast -tune zerolatency
         -f rtsp rtsp://localhost:8554/<name>
```

- 每路流独立 `subprocess.Popen`
- `stdin.write(frame.tobytes())` 写帧
- 写入异常时自动重启 FFmpeg 进程

---

## 7. go2rtc 配置（一次性手动操作）

在 `/mnt/shared/docker-container/go2rtc/go2rtc/go2rtc.yaml` 的 `streams:` 末尾追加：

```yaml
  detected_bedroom:
    - http://localhost:8090/bedroom

  # 若启用多路，对应添加：
  # detected_kitchen:
  #   - http://localhost:8090/kitchen
  # detected_gate:
  #   - http://localhost:8090/gate
```

添加后重启 go2rtc 容器：`docker restart go2rtc`

---

## 8. Docker 部署

### 8.1 直接运行（开发/测试）

```bash
cd /home/orangepi/orange-pi-rknn
CONFIG_PATH=config.yaml python3 -m stream_detector
```

### 8.2 Docker 构建运行

```bash
docker compose up --build -d
```

rknnlite 通过 volume 挂载宿主机已安装的包：
- `/home/orangepi/.local/lib/python3.10/site-packages/rknnlite`
- 宿主机共享库 `/usr/lib/aarch64-linux-gnu`（rknnlite `.so` 依赖）

---

## 9. 硬件 & 技术选型说明（已验证）

| 项目 | 状态 | 说明 |
|------|------|------|
| `h264_rkmpp` 解码 | ✅ 可用 | OpenCV + FFmpeg 硬件解码 RTSP 输入 |
| `h264_v4l2m2m` 编码 | ❌ 不可用 | RK3588 无标准 V4L2 M2M 节点，非权限问题 |
| HTTP MJPEG 输出 | ✅ 选用 | 纯 Python，无额外依赖，go2rtc 原生支持拉取 |
| go2rtc RTSP push | ❌ 不可用 | 需预声明 stream，API 无法创建空 listener |
| RKNNLite 推理 | ✅ 可用 | 安装于 `~/.local/lib/python3.10/site-packages` |

---

## 10. 性能预估（RK3588）

| 配置 | 网格尺寸（YOLO输入视角） | 预估推理 FPS |
|------|------------------------|-------------|
| 1 路，1×1 | 640×360（848×480 letterbox） | ~15–20 FPS |
| 2 路，2×1 grid | 320×360 per Cell | ~12–15 FPS |
| 4 路，2×2 grid | 320×180 per Cell | ~10–12 FPS |

小米 069a01 摄像头实际分辨率：848×480（SD 子码流）。

---

## 11. 已确认事项

| 事项 | 结论 |
|------|------|
| go2rtc 输出方式 | HTTP MJPEG，go2rtc 主动拉取 |
| go2rtc config 路径（宿主机） | `/mnt/shared/docker-container/go2rtc/go2rtc/go2rtc.yaml` |
| go2rtc RTSP push | ❌ 不支持无预声明推流（已实测） |
| RKNN 模型路径 | `/home/orangepi/orange-pi-rknn/yolo11n.rknn` |
| rknnlite 安装位置 | `~/.local/lib/python3.10/site-packages/rknnlite` |
| 摄像头实际分辨率 | 848×480（SD 子码流） |
| 网格布局 | 最多 2 列，原始分辨率拼合后整体 letterbox 至 640×640 |
| 帧同步策略 | 各路取最新帧，不等待 |
| 延迟配置 | 可配置，默认 1000ms |
