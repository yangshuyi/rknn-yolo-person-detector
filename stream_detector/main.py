import logging
import os
import signal
import sys
import time

import cv2

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


def run():
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    cfg = load_config(config_path)

    logger.info('Streams: %s', [s.name for s in cfg.streams])
    logger.info('Target classes: %s', cfg.model.target_classes)
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

    detector = Detector(cfg.model)
    logger.info('RKNN model loaded: %s', cfg.model.path)

    frame_interval = 1.0 / cfg.processing.process_fps
    last_frames = [None] * len(cfg.streams)
    fps_samples: list = []

    def _shutdown(sig, _frame):
        logger.info('Shutting down...')
        for r in readers:
            r.stop()
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

        if any(f is None for f in frames):
            time.sleep(0.1)
            continue

        results = detector.detect_frames(frames)

        elapsed = time.monotonic() - t0
        fps_samples.append(elapsed)
        if len(fps_samples) > 20:
            fps_samples.pop(0)
        fps = 1.0 / (sum(fps_samples) / len(fps_samples))

        annotated_frames = []
        total_dets = sum(len(dets) for _, dets in results)
        label = '/'.join(cfg.model.target_classes)
        for annotated, _ in results:
            annotated_frames.append(annotated)

        grid_frame, _ = compose(annotated_frames)
        _osd(grid_frame, f'FPS:{fps:.1f}  {label}:{total_dets}')
        mjpeg.push('grid', grid_frame)

        to_sleep = frame_interval - (time.monotonic() - t0)
        if to_sleep > 0:
            time.sleep(to_sleep)
