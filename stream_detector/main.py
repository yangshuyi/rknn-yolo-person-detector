import logging
import os
import signal
import sys
import time

import cv2
import numpy as np

from .config import load_config
from .detector import Detector
from .grid import compose
from .reader import ReaderThread
from .writer import MJPEGServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def _osd(frame, text: str):
    cv2.putText(frame, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def _no_signal_frame(shape):
    h, w = shape[:2]
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    text = 'NO SIGNAL'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    return frame


def _placeholder_shape(last_frames, default=(480, 848, 3)):
    for f in last_frames:
        if f is not None:
            return f.shape
    return default


MAX_FPS = 30  # 输出帧率上限


# 可用布局 → 最大列数（环境变量 LAYOUTS，逗号分隔，默认 grid）
LAYOUT_COLS = {'grid': 2, 'flex-column': 1}


def _parse_layouts():
    raw = os.environ.get('LAYOUTS', 'grid')
    names = [n.strip() for n in raw.split(',') if n.strip()]
    unknown = [n for n in names if n not in LAYOUT_COLS]
    if unknown or not names:
        raise SystemExit(f'Invalid LAYOUTS={raw!r}; valid: {", ".join(LAYOUT_COLS)}')
    return names


def run():
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    cfg = load_config(config_path)
    layouts = _parse_layouts()

    logger.info('Streams: %s', [s.name for s in cfg.streams])
    detect_enabled = cfg.detection.enabled
    if detect_enabled:
        logger.info('Target classes: %s', cfg.model.target_classes)
    else:
        logger.info('Detection disabled: grid only')
    process_fps = max(1, min(cfg.processing.process_fps, MAX_FPS))
    if process_fps != cfg.processing.process_fps:
        logger.warning('process_fps=%s clamped to %d (max %d)', cfg.processing.process_fps, process_fps, MAX_FPS)
    logger.info('Process FPS: %d', process_fps)

    readers = [ReaderThread(s.input, cfg.processing.max_delay_ms) for s in cfg.streams]
    for r in readers:
        r.start()

    mjpeg = MJPEGServer(cfg.output.mjpeg_port, layouts, cfg.output.jpeg_quality)
    mjpeg.start()
    logger.info('MJPEG server listening on port %d  →  %s',
                cfg.output.mjpeg_port, ', '.join('/' + n for n in layouts))

    logger.info('Waiting for first frames...')
    for r in readers:
        for _ in range(60):
            if r.get_frame() is not None:
                break
            time.sleep(0.5)
        else:
            logger.error('Timed out waiting for stream: %s', r.url)

    detector = None
    if detect_enabled:
        detector = Detector(cfg.model)
        logger.info('RKNN model loaded: %s', cfg.model.path)

    frame_interval = 1.0 / process_fps
    last_frames = [None] * len(cfg.streams)
    fps_samples: list = []
    prev_t0 = None

    def _shutdown(sig, _frame):
        logger.info('Shutting down...')
        for r in readers:
            r.stop()
        if detector:
            detector.release()
        mjpeg.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        t0 = time.monotonic()

        # Collect latest frames; fall back to last known frame if stream is stalled
        frames = []
        for i, reader in enumerate(readers):
            f = reader.get_frame()
            if f is not None:
                last_frames[i] = f
            frames.append(last_frames[i])

        # Streams that never delivered a frame get a "no signal" placeholder instead
        # of blocking the whole pipeline (e.g. a camera that's offline shouldn't stop
        # detection on the streams that are working).
        live_indices = [i for i, f in enumerate(frames) if f is not None]

        if not live_indices:
            results = []
        elif detect_enabled:
            results = detector.detect_frames([frames[i] for i in live_indices])
        else:
            results = [(frames[i], []) for i in live_indices]

        # 真实输出帧率：相邻两次循环起点的间隔（含限速等待）
        if prev_t0 is not None:
            fps_samples.append(t0 - prev_t0)
            if len(fps_samples) > 20:
                fps_samples.pop(0)
        prev_t0 = t0
        fps = len(fps_samples) / sum(fps_samples) if fps_samples else 0.0

        annotated_frames = [None] * len(frames)
        for i, (annotated, _) in zip(live_indices, results):
            annotated_frames[i] = annotated
        placeholder = _no_signal_frame(_placeholder_shape(frames))
        for i, f in enumerate(annotated_frames):
            if f is None:
                annotated_frames[i] = placeholder

        total_dets = sum(len(dets) for _, dets in results)
        label = '/'.join(cfg.model.target_classes)

        osd_text = f'FPS:{fps:.1f}  {label}:{total_dets}' if detect_enabled else f'FPS:{fps:.1f}'
        for name in layouts:
            layout_frame, _ = compose(annotated_frames, LAYOUT_COLS[name])
            _osd(layout_frame, osd_text)
            mjpeg.push(name, layout_frame)

        to_sleep = frame_interval - (time.monotonic() - t0)
        if to_sleep > 0:
            time.sleep(to_sleep)
