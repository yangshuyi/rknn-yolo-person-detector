from typing import List, Tuple

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from .config import ModelConfig

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


def _letterbox(img, new_shape=(640, 640)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = (new_shape[0] - nh) / 2, (new_shape[1] - nw) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_lb = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_lb, r, (left, top)


def _postprocess(output, orig_shape, r, pad, conf_thresh, nms_thresh):
    pred = output[0].T
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confidences > conf_thresh
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confidences, class_ids = confidences[mask], class_ids[mask]

    x1 = np.clip(((cx - bw / 2) - pad[0]) / r, 0, orig_shape[1])
    y1 = np.clip(((cy - bh / 2) - pad[1]) / r, 0, orig_shape[0])
    x2 = np.clip(((cx + bw / 2) - pad[0]) / r, 0, orig_shape[1])
    y2 = np.clip(((cy + bh / 2) - pad[1]) / r, 0, orig_shape[0])

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), conf_thresh, nms_thresh)

    detections = []
    for i in (indices.flatten() if len(indices) else []):
        detections.append({
            'box': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
            'confidence': float(confidences[i]),
            'class_id': int(class_ids[i]),
            'label': CLASSES[int(class_ids[i])],
        })
    return detections


def _draw(img, detections):
    for d in detections:
        x1, y1, x2, y2 = d['box']
        label = f"{d['label']} {d['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img


class Detector:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.target_classes = set(cfg.target_classes)
        self._rknn = RKNNLite(verbose=False)
        self._rknn.load_rknn(cfg.path)
        self._rknn.init_runtime()

    def detect_frames(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, list]]:
        """Run independent YOLO inference on each frame at full 640x640 resolution."""
        results = []
        for frame in frames:
            lb_img, r, pad = _letterbox(frame, (self.cfg.input_size, self.cfg.input_size))
            inp = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(inp, 0)

            outputs = self._rknn.inference(inputs=[inp])

            dets = _postprocess(outputs[0], frame.shape, r, pad,
                                self.cfg.conf_thresh, self.cfg.nms_thresh)
            dets = [d for d in dets if d['label'] in self.target_classes]

            annotated = frame.copy()
            _draw(annotated, dets)
            results.append((annotated, dets))
        return results

    def release(self):
        self._rknn.release()
