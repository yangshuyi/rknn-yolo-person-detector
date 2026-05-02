import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List

import cv2


class _FrameStore:
    def __init__(self):
        self._data: bytes = None
        self._frame_id = 0
        self._cond = threading.Condition()

    def put(self, jpeg_bytes: bytes):
        with self._cond:
            self._data = jpeg_bytes
            self._frame_id += 1
            self._cond.notify_all()

    def get_next(self, last_id: int, timeout: float = 2.0):
        """Block until a new frame arrives (frame_id > last_id), then return (data, new_id)."""
        with self._cond:
            deadline = time.monotonic() + timeout
            while self._frame_id == last_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._data, self._frame_id
                self._cond.wait(remaining)
            return self._data, self._frame_id


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.lstrip('/')
        stores = self.server.stores  # type: ignore[attr-defined]

        if path not in stores:
            self.send_error(404, f'Stream "{path}" not found')
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=mjpegframe')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        store = stores[path]
        last_id = 0
        try:
            while True:
                data, last_id = store.get_next(last_id)
                if data is None:
                    continue
                self.wfile.write(
                    b'--mjpegframe\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n' +
                    data + b'\r\n'
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class MJPEGServer:
    def __init__(self, port: int, names: List[str], jpeg_quality: int = 85):
        self.port = port
        self.quality = jpeg_quality
        self.stores = {name: _FrameStore() for name in names}
        self._server = ThreadingHTTPServer(('0.0.0.0', port), _MJPEGHandler)
        self._server.stores = self.stores  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name='mjpeg-server')

    def start(self):
        self._thread.start()

    def push(self, name: str, frame_bgr):
        _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        self.stores[name].put(bytes(buf))

    def stop(self):
        self._server.shutdown()
