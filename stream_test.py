import os
import sys
import time
import cv2
import numpy as np
from rknnlite.api import RKNNLite

RKNN_MODEL = 'yolo11n.rknn'
RTSP_URL = sys.argv[1] if len(sys.argv) > 1 else 'rtsp://localhost:8554/micam_bedroom'
OUT_PATH = 'result/latest.jpg'
PROCESS_FPS = 5
CONF_THRESH = 0.25
NMS_THRESH = 0.45
TARGET_CLASSES = {'person'}

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = int(round((size - nh) / 2 - 0.1))
    left = int(round((size - nw) / 2 - 0.1))
    bottom = size - nh - top
    right = size - nw - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def postprocess(output, orig_shape, r, pad):
    pred = output[0].T
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), class_ids]

    mask = confs > CONF_THRESH
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confs, class_ids = confs[mask], class_ids[mask]

    x1 = np.clip(((cx - bw / 2) - pad[0]) / r, 0, orig_shape[1])
    y1 = np.clip(((cy - bh / 2) - pad[1]) / r, 0, orig_shape[0])
    x2 = np.clip(((cx + bw / 2) - pad[0]) / r, 0, orig_shape[1])
    y2 = np.clip(((cy + bh / 2) - pad[1]) / r, 0, orig_shape[0])

    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(boxes, confs.tolist(), CONF_THRESH, NMS_THRESH)

    dets = []
    for i in (idxs.flatten() if len(idxs) else []):
        label = CLASSES[int(class_ids[i])]
        if label in TARGET_CLASSES:
            dets.append({
                'box': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                'conf': float(confs[i]),
                'label': label,
            })
    return dets


def draw(img, dets, fps):
    for d in dets:
        x1, y1, x2, y2 = d['box']
        text = f"{d['label']} {d['conf']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    osd = f'FPS:{fps:.1f}  person:{len(dets)}'
    cv2.putText(img, osd, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(img, osd, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


os.makedirs('result', exist_ok=True)

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
    'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;1000000'
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print(f'无法打开流: {RTSP_URL}')
    sys.exit(1)
print(f'已连接: {RTSP_URL}')

rknn = RKNNLite(verbose=False)
rknn.load_rknn(RKNN_MODEL)
rknn.init_runtime()
print('RKNN 模型已加载')

interval = 1.0 / PROCESS_FPS
frame_count = 0
t_last = time.monotonic()

print(f'开始处理，目标 {PROCESS_FPS} FPS，结果写入 {OUT_PATH}  (Ctrl+C 退出)')

try:
    while True:
        ret = cap.grab()
        if not ret:
            print('流中断，退出')
            break

        if time.monotonic() - t_last < interval:
            continue

        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue

        t0 = time.monotonic()
        lb, r, pad = letterbox(frame)
        inp = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, 0)
        outputs = rknn.inference(inputs=[inp])
        dets = postprocess(outputs[0], frame.shape, r, pad)

        elapsed = time.monotonic() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0

        result = draw(frame.copy(), dets, fps)
        cv2.imwrite(OUT_PATH, result)

        frame_count += 1
        t_last = time.monotonic()
        print(f'帧 {frame_count:4d}  FPS:{fps:.1f}  person:{len(dets)}')

except KeyboardInterrupt:
    print(f'\n已停止，共处理 {frame_count} 帧')
finally:
    cap.release()
    rknn.release()
