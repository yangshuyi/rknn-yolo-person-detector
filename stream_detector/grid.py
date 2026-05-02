import math
from typing import List, Tuple

import numpy as np

# (x_offset, y_offset, cell_w, cell_h, stream_idx)
CellInfo = Tuple[int, int, int, int, int]


def compose(frames: List[np.ndarray]) -> Tuple[np.ndarray, List[CellInfo]]:
    """Arrange frames into a 2-column grid at original resolution."""
    n = len(frames)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    h, w = frames[0].shape[:2]

    grid = np.full((rows * h, cols * w, 3), 114, dtype=np.uint8)
    cells: List[CellInfo] = []

    for i, frame in enumerate(frames):
        col = i % cols
        row = i // cols
        x_off, y_off = col * w, row * h
        fh, fw = frame.shape[:2]
        grid[y_off:y_off + fh, x_off:x_off + fw] = frame
        cells.append((x_off, y_off, fw, fh, i))

    return grid, cells


def decompose(detections: list, cells: List[CellInfo]) -> List[list]:
    """Map detections from grid-image space back to per-stream original frame space."""
    stream_dets: List[list] = [[] for _ in cells]

    for det in detections:
        x1, y1, x2, y2 = det['box']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        for x_off, y_off, cw, ch, stream_idx in cells:
            if x_off <= cx < x_off + cw and y_off <= cy < y_off + ch:
                stream_dets[stream_idx].append({
                    **det,
                    'box': [
                        int(np.clip(x1 - x_off, 0, cw)),
                        int(np.clip(y1 - y_off, 0, ch)),
                        int(np.clip(x2 - x_off, 0, cw)),
                        int(np.clip(y2 - y_off, 0, ch)),
                    ],
                })
                break

    return stream_dets
