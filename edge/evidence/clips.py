"""Evidence clip cutter (spec §19 P6, §12.2 "8 s blurred clip with skeleton overlay").

An alert fires in the middle of the behaviour, so a clip needs frames from *before* it (the rolling
buffer has them) and from *after* it (they have not happened yet). `ClipRecorder` therefore records
a request and cuts the clip a few seconds later, once the post-roll has arrived — `pump()` is called
every frame and returns the clips that became ready.

Every frame written to disk goes through the privacy filter first; the raw frames never leave RAM.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from edge.evidence.buffer import ClipBuffer
from edge.evidence.privacy import blur_faces

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


@dataclass
class Clip:
    id: str
    path: Path
    alert_id: str | None
    seat: str
    start_t: float
    end_t: float
    frames: int
    faces_masked: int
    created_at: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t


def frame_rate(frames, fallback: float = 10.0) -> float:
    """Real capture rate of these frames. A live webcam run buffers at whatever FPS the pipeline
    managed (~5), a video file buffers at the file's own rate (30) — writing both at a fixed 10 would
    play one in fast motion and the other in slow motion, which misrepresents how long a behaviour
    lasted. That matters: the clip is evidence."""
    if len(frames) < 2:
        return fallback
    span = frames[-1].t - frames[0].t
    return max(1.0, min((len(frames) - 1) / span, 60.0)) if span > 0 else fallback


def cut(frames, out_path: Path, fps: float = 10.0, blur: bool = True, skeleton: bool = True,
        caption: str = "", highlight: str = "") -> tuple[int, int]:
    """Write buffered frames to `out_path` as mp4. Returns (frames written, faces masked).

    blur=False is only for a local debugging session — never for anything that leaves the machine.
    """
    from edge.perception.pose_tracker import id_color
    from edge.viz import draw_skeleton, label as draw_label

    if not frames:
        return 0, 0
    fps = frame_rate(frames, fps)   # play back at the rate the frames were captured, not a guess
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0].image()
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), FOURCC, fps, (w, h))
    masked = 0
    try:
        for bf in frames:
            img = bf.image()
            if img is None:
                continue
            if blur:
                masked += blur_faces(img, bf.people, scale=bf.scale)
            if skeleton:
                for p in bf.people:
                    scaled = _scaled(p, bf.scale)
                    draw_skeleton(img, scaled, id_color(p.track_id))
            if caption:
                draw_label(img, caption, (8, 26), (0, 0, 255), 0.6)
            if highlight:
                draw_label(img, f"seat {highlight}", (8, int(h - 10)), (0, 165, 255), 0.6)
            writer.write(img)
    finally:
        writer.release()
    return len(frames), masked


def _scaled(person, scale: float):
    """A copy of the person with keypoints in the stored frame's resolution."""
    if scale == 1.0:
        return person
    from copy import copy
    p = copy(person)
    p.kpts = person.kpts * scale
    p.box = np.asarray(person.box, float) * scale
    return p


@dataclass
class _Pending:
    alert_id: str | None
    seat: str
    t0: float
    t1: float
    caption: str


class ClipRecorder:
    """Cuts one clip per alert, a few seconds after the alert, from the rolling buffer."""

    def __init__(self, buffer: ClipBuffer, out_dir: Path, pre_s: float = 5.0, post_s: float = 3.0,
                 fps: float = 10.0, blur: bool = True, skeleton: bool = True,
                 cooldown_s: float = 30.0, max_clips: int | None = None):
        self.buffer, self.out_dir = buffer, Path(out_dir)
        self.pre_s, self.post_s, self.fps = pre_s, post_s, fps
        self.blur, self.skeleton = blur, skeleton
        self.cooldown_s = cooldown_s          # one clip per seat per cooldown: alerts can repeat
        self.max_clips = max_clips            # safety valve for a long unattended run
        self.pending: list[_Pending] = []
        self.last_cut: dict[str, float] = {}
        self.clips: list[Clip] = []

    def request(self, t: float, seat: str, alert_id: str | None = None, caption: str = "") -> bool:
        if self.max_clips is not None and len(self.clips) + len(self.pending) >= self.max_clips:
            return False
        if t - self.last_cut.get(seat, -1e9) < self.cooldown_s:
            return False
        self.last_cut[seat] = t
        self.pending.append(_Pending(alert_id, seat, t - self.pre_s, t + self.post_s, caption))
        return True

    def pump(self, t_now: float) -> list[Clip]:
        """Cut every request whose post-roll is now in the buffer. Call once per frame."""
        ready = [p for p in self.pending if t_now >= p.t1]
        self.pending = [p for p in self.pending if t_now < p.t1]
        out = []
        for p in ready:
            frames = self.buffer.window(p.t0, p.t1)
            if not frames:
                continue
            cid = uuid.uuid4().hex[:12]
            path = self.out_dir / f"clip_{p.seat}_{cid}.mp4"
            n, masked = cut(frames, path, self.fps, self.blur, self.skeleton, p.caption, p.seat)
            if not n:
                continue
            clip = Clip(cid, path, p.alert_id, p.seat, frames[0].t, frames[-1].t, n, masked)
            self.clips.append(clip)
            out.append(clip)
        return out

    def flush(self, t_now: float) -> list[Clip]:
        """Cut whatever is still pending (end of the video / session), with a shorter post-roll."""
        for p in self.pending:
            p.t1 = min(p.t1, t_now)
        return self.pump(t_now)


def purge(out_dir: Path, retention_days: float, now: float | None = None) -> list[Path]:
    """Delete clips older than the retention window (spec §13: auto-delete). Returns what went."""
    now = time.time() if now is None else now
    cutoff = now - retention_days * 86400
    gone = []
    for p in sorted(Path(out_dir).glob("clip_*.mp4")):
        if p.stat().st_mtime < cutoff:
            p.unlink()
            gone.append(p)
    return gone
