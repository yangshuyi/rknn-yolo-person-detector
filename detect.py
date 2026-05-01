import sys
import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

RKNN_MODEL = 'yolo11n.rknn'
IMG_SIZE = 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45

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


def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = (new_shape[0] - nh) / 2, (new_shape[1] - nw) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def postprocess(output, orig_shape, r, pad):
    # output: [1, 84, 8400] → [8400, 84]
    pred = output[0].T
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    class_scores = pred[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confidences > CONF_THRESH
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # xywh (letterbox space) → xyxy (original image space)
    x1 = ((cx - bw / 2) - pad[0]) / r
    y1 = ((cy - bh / 2) - pad[1]) / r
    x2 = ((cx + bw / 2) - pad[0]) / r
    y2 = ((cy + bh / 2) - pad[1]) / r
    x1 = np.clip(x1, 0, orig_shape[1])
    y1 = np.clip(y1, 0, orig_shape[0])
    x2 = np.clip(x2, 0, orig_shape[1])
    y2 = np.clip(y2, 0, orig_shape[0])

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), CONF_THRESH, NMS_THRESH)

    detections = []
    for i in (indices.flatten() if len(indices) else []):
        detections.append({
            'box': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
            'confidence': float(confidences[i]),
            'class_id': int(class_ids[i]),
            'label': CLASSES[int(class_ids[i])],
        })
    return detections


def draw(img, detections):
    for d in detections:
        x1, y1, x2, y2 = d['box']
        label = f"{d['label']} {d['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'image.jpg'
    filename = os.path.basename(img_path)
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'result'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    img = cv2.imread(img_path)
    if img is None:
        print(f'无法读取图片: {img_path}')
        sys.exit(1)

    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(RKNN_MODEL)
    rknn.init_runtime()

    # 预处理：letterbox + BGR→RGB
    img_lb, r, pad = letterbox(img)
    inp = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(inp, 0)  # [1, 640, 640, 3]

    outputs = rknn.inference(inputs=[inp])

    detections = postprocess(outputs[0], img.shape, r, pad)
    result = draw(img.copy(), detections)
    cv2.imwrite(out_path, result)

    print(f'检测到 {len(detections)} 个目标:')
    for d in detections:
        print(f"  {d['label']:15s}  置信度: {d['confidence']:.3f}  框: {d['box']}")
    print(f'结果已保存: {out_path}')

    rknn.release()


if __name__ == '__main__':
    main()
