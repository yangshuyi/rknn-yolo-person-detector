import json
import logging
import subprocess
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

FFMPEG_BIN = '/usr/lib/jellyfin-ffmpeg/ffmpeg'
FFPROBE_BIN = '/usr/lib/jellyfin-ffmpeg/ffprobe'


class ReaderThread:
    def __init__(self, url: str, max_delay_ms: int = 1000):
        self.url = url
        self.max_delay_ms = max_delay_ms
        self._frame = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f'reader-{url}')
        self._running = False
        self._proc = None

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        if self._proc is not None:
            self._proc.terminate()

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _probe_size(self):
        """Ask ffprobe for the stream's resolution so the raw pipe can be sliced correctly."""
        out = subprocess.run(
            [FFPROBE_BIN, '-v', 'error', '-rtsp_transport', 'tcp',
             '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
             '-of', 'json', self.url],
            capture_output=True, text=True, timeout=10,
        )
        info = json.loads(out.stdout)['streams'][0]
        return info['width'], info['height']

    def _open(self, width, height):
        cmd = [
            FFMPEG_BIN, '-loglevel', 'warning',
            '-rtsp_transport', 'tcp', '-fflags', 'nobuffer', '-flags', 'low_delay',
            '-max_delay', str(self.max_delay_ms * 1000),
            '-hwaccel', 'rkmpp', '-hwaccel_output_format', 'drm_prime',
            '-i', self.url,
            '-an', '-vsync', '0',
            '-vf', 'hwdownload,format=nv12,format=bgr24',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1',
        ]
        frame_bytes = width * height * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 bufsize=frame_bytes)
        return proc, frame_bytes

    @staticmethod
    def _read_exact(stream, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _run(self):
        retry_delay = 1.0
        while self._running:
            try:
                width, height = self._probe_size()
            except Exception as e:
                logger.warning('Cannot probe %s (%s), retry in %.0fs', self.url, e, retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
                continue

            self._proc, frame_bytes = self._open(width, height)
            logger.info('Stream opened (hw decode): %s (%dx%d)', self.url, width, height)
            retry_delay = 1.0

            while self._running:
                buf = self._read_exact(self._proc.stdout, frame_bytes)
                if buf is None:
                    logger.warning('Stream lost: %s', self.url)
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
                with self._lock:
                    self._frame = frame

            self._proc.terminate()
            self._proc.wait()
            if self._running:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
