import os
import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)


class ReaderThread:
    def __init__(self, url: str, max_delay_ms: int = 1000):
        self.url = url
        self.max_delay_ms = max_delay_ms
        self._frame = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f'reader-{url}')
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _open(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            f'rtsp_transport;tcp|'
            f'fflags;nobuffer|'
            f'flags;low_delay|'
            f'max_delay;{self.max_delay_ms * 1000}'
        )
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _run(self):
        retry_delay = 1.0
        while self._running:
            cap = self._open()
            if not cap.isOpened():
                logger.warning('Cannot open %s, retry in %.0fs', self.url, retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            retry_delay = 1.0
            logger.info('Stream opened: %s', self.url)

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning('Stream lost: %s', self.url)
                    break
                with self._lock:
                    self._frame = frame

            cap.release()
            if self._running:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
