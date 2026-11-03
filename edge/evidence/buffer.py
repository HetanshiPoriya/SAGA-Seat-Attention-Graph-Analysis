"""Rolling evidence buffer (spec §13 "raw video is held in a rolling RAM buffer (~30 s) only to cut
evidence clips", §5 clip cutter).

Nothing is written to disk until an alert asks for a clip, and what is written has already been
through the privacy filter (edge/evidence/privacy.py). The buffer keeps the last `seconds` of frames
in memory as JPEG bytes — a 30 s window of raw 1080p BGR arrays would be ~1.8 GB, the same window
JPEG-encoded at 1280 px wide is ~20 MB.

Keypoints are kept beside each frame because the blur needs them: SAGA never runs face detection,
so the face region comes from the pose model's nose/eye/ear points (and, when the head is not
visible at all, from the top of the person box).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BufferedFrame:
    t: float                 # seconds since session start
    jpeg: bytes              # encoded frame, possibly downscaled
    scale: float             # stored size / original size (keypoints are in ORIGINAL coordinates)
    people: list             # Person objects captured with this frame

    def image(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)


class ClipBuffer:
    def __init__(self, seconds: float = 30.0, max_width: int = 1280, quality: int = 80):
        self.seconds, self.max_width, self.quality = seconds, max_width, quality
        self.frames: deque[BufferedFrame] = deque()
        self.bytes = 0

    def add(self, t: float, frame: np.ndarray, people: list) -> None:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_width / w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        self.frames.append(BufferedFrame(t, buf.tobytes(), scale, list(people)))
        self.bytes += buf.size
        while self.frames and self.frames[0].t < t - self.seconds:
            self.bytes -= len(self.frames.popleft().jpeg)

    def window(self, t0: float, t1: float) -> list[BufferedFrame]:
        return [f for f in self.frames if t0 <= f.t <= t1]

    @property
    def covered(self) -> tuple[float, float]:
        return (self.frames[0].t, self.frames[-1].t) if self.frames else (0.0, 0.0)

    @property
    def memory_mb(self) -> float:
        return self.bytes / 1e6

    def clear(self) -> None:
        """Drop every buffered frame (end of session, or when evidence capture is switched off)."""
        self.frames.clear()
        self.bytes = 0
