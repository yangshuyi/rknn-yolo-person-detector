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


def run():
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    cfg = load_config(config_path)

    logger.info('Streams: %s', [s.name for s in cfg.streams])
    detect_enabled = cfg.detection.enabled
    if detect_enabled:
        logger.info('Target classes: %s', cfg.model.target_classes)
    else:
        logger.info('Detection disabled: grid only')
    logger.info('Process FPS: %d', cfg.processing.process_fps)

    readers = [ReaderThread(s.input, cfg.processing.max_delay_ms) for s in cfg.streams]
    for r in readers:
        r.start()

    mjpeg = MJPEGServer(cfg.output.mjpeg_port, ['grid'], cfg.output.jpeg_quality)
    mjpeg.start()
    logger.info('MJPEG server listening on port %d  →  http://localhost:%d/grid',
                cfg.output.mjpeg_port, cfg.output.mjpeg_port)

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

    frame_interval = 1.0 / cfg.processing.process_fps
    last_frames = [None] * len(cfg.streams)
    fps_samples: list = []

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

        elapsed = time.monotonic() - t0
        fps_samples.append(elapsed)
        if len(fps_samples) > 20:
            fps_samples.pop(0)
        fps = 1.0 / (sum(fps_samples) / len(fps_samples))

        annotated_frames = [None] * len(frames)
        for i, (annotated, _) in zip(live_indices, results):
            annotated_frames[i] = annotated
        placeholder = _no_signal_frame(_placeholder_shape(frames))
        for i, f in enumerate(annotated_frames):
            if f is None:
                annotated_frames[i] = placeholder

        total_dets = sum(len(dets) for _, dets in results)
        label = '/'.join(cfg.model.target_classes)

        grid_frame, _ = compose(annotated_frames)
        if detect_enabled:
            _osd(grid_frame, f'FPS:{fps:.1f}  {label}:{total_dets}')
        else:
            _osd(grid_frame, f'FPS:{fps:.1f}')
        mjpeg.push('grid', grid_frame)

        to_sleep = frame_interval - (time.monotonic() - t0)
        if to_sleep > 0:
            time.sleep(to_sleep)
