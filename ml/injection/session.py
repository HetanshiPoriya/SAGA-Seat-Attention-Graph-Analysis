"""Per-seat feature time series ("sessions") for the injection simulator (spec §16.3).

A session comes either from a real features CSV written by run_seats.py, or from the synthetic
normal-class generator below. Synthetic sessions are for testing the pipeline only — headline
numbers must come from real tracks (webcam recordings now, SAV classroom videos later).
"""
from __future__ import annotations

import csv
import math
import string
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SERIES = ["yaw", "pitch", "lean_x", "shoulder_tilt", "wrist_drop", "wrists_vis", "hand_face", "head_px", "kp_conf"]


@dataclass
class SeatSeries:
    t: np.ndarray
    data: dict[str, np.ndarray]          # SERIES -> float arrays (NaN = unknown)
    facing: np.ndarray                   # str array


@dataclass
class Session:
    source: str
    seats: dict[str, SeatSeries]
    pos: dict[str, tuple[float, float]]  # seat -> (x, y); x grows toward image-right
    natural: list[dict] = field(default_factory=list)  # distractors added by the generator (not misconduct)

    @property
    def duration(self) -> float:
        return max(float(s.t[-1]) for s in self.seats.values())

    def same_row_neighbours(self, seat: str) -> list[str]:
        x, y = self.pos[seat]
        row = [s for s, (xx, yy) in self.pos.items() if s != seat and abs(yy - y) < 0.5]
        row.sort(key=lambda s: abs(self.pos[s][0] - x))
        nearest = abs(self.pos[row[0]][0] - x) if row else 0
        return [s for s in row if abs(self.pos[s][0] - x) <= nearest * 1.5]

    def to_frames(self) -> list[tuple[float, list[dict]]]:
        frames: dict[float, list[dict]] = {}
        for seat, s in self.seats.items():
            for i, t in enumerate(s.t):
                row = {"seat": seat, "facing": str(s.facing[i])}
                for k in SERIES:
                    row[k] = float(s.data[k][i])
                frames.setdefault(round(float(t), 3), []).append(row)
        return sorted(frames.items())

    def copy(self) -> "Session":
        return Session(self.source, {k: SeatSeries(v.t.copy(), {kk: vv.copy() for kk, vv in v.data.items()},
                                                   v.facing.copy()) for k, v in self.seats.items()},
                       dict(self.pos), list(self.natural))


# ------------------------------------------------------------------ real data
def from_features_csv(path: str | Path, seatmap_json: str | Path | None = None) -> Session:
    rows: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["seat"], []).append(r)
    centres = {}
    if seatmap_json:
        import json
        centres = {s["label"]: tuple(s["center"]) for s in json.loads(Path(seatmap_json).read_text())["seats"]}
    seats, pos = {}, {}
    for seat, rs in rows.items():
        def col(k):
            alt = {"yaw": "yaw_s", "pitch": "pitch_s"}.get(k)
            out = []
            for r in rs:
                v = r.get(alt) if alt and r.get(alt) not in (None, "") else r.get(k, "")
                out.append(float(v) if v not in ("", None) else math.nan)
            return np.array(out)
        seats[seat] = SeatSeries(np.array([float(r["t"]) for r in rs]), {k: col(k) for k in SERIES},
                                 np.array([r.get("facing", "unknown") for r in rs], dtype=object))
        pos[seat] = centres.get(seat) or _label_pos(seat)
    return Session(f"real:{Path(path).name}", seats, pos)


def _label_pos(label: str) -> tuple[float, float]:
    """'B4' -> (4, 1): column grows image-right, row letter from the front."""
    return float(int("".join(c for c in label if c.isdigit()) or 0)), float(string.ascii_uppercase.find(label[0]))


# ------------------------------------------------------------------ synthetic normal class
def synthetic_session(rows=3, cols=4, duration_s=480.0, fps=10.0, seed=0, distractors=True) -> Session:
    """Normal exam behaviour: reading/writing, slow head drift, looking up at the paper, missed
    detections — plus natural distractors (single glance at a clock/door, stretching, touching the
    face, an occasional pair of glances) that a good system must NOT flag."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    seats, pos, natural = {}, {}, []
    for r in range(rows):
        for c in range(cols):
            label = f"{string.ascii_uppercase[r]}{c + 1}"
            yaw = _ar1(rng, n, sd=3.0, a=0.97) + rng.normal(0, 12) + _steps(rng, t, every=20, sd=5)
            pitch = _ar1(rng, n, sd=3.0, a=0.97) + rng.normal(-25, 5)
            wrist = 1.3 + _ar1(rng, n, sd=0.06, a=0.95)
            hand_face = 1.6 + _ar1(rng, n, sd=0.08, a=0.95)
            data = {"yaw": yaw, "pitch": pitch, "lean_x": _ar1(rng, n, sd=0.03, a=0.98),
                    "shoulder_tilt": rng.normal(0, 3) + _ar1(rng, n, sd=1.5, a=0.95), "wrist_drop": wrist,
                    "wrists_vis": np.where(rng.random(n) < 0.03, 1.0, 2.0), "hand_face": hand_face,
                    "head_px": np.full(n, 55.0 - 8 * r), "kp_conf": np.clip(rng.normal(0.75, 0.05, n), 0, 1)}
            for _ in range(rng.poisson(duration_s / 40)):        # look up at the question paper
                s = rng.uniform(0, duration_s - 6)
                _add(data["pitch"], t, s, rng.uniform(2, 5), 18)
            if distractors:
                for kind, rate, fn in (("single_glance", 180, _glance), ("stretch", 600, _stretch),
                                       ("touch_face", 300, _touch), ("glance_pair", 600, _pair)):
                    for _ in range(rng.poisson(duration_s / rate)):
                        s = rng.uniform(0, duration_s - 10)
                        fn(rng, data, t, s)
                        natural.append({"seat": label, "kind": kind, "start": s})
            keep = rng.random(n) > 0.02                           # 2% missed detections
            seats[label] = SeatSeries(t[keep], {k: v[keep] for k, v in data.items()},
                                      np.full(int(keep.sum()), "camera", dtype=object))
            pos[label] = (float(c + 1), float(r))
    return Session(f"synthetic(seed={seed})", seats, pos, natural)


def profile(t, start, hold, ramp=0.3):
    """0 → 1 over `ramp` s, hold, back to 0 over `ramp` s (spec §16.3 glance shape)."""
    up = np.clip((t - start) / ramp, 0, 1)
    down = np.clip((start + ramp + hold + ramp - t) / ramp, 0, 1)
    return np.minimum(up, down)


def _add(arr, t, start, hold, amount, ramp=0.3):
    arr += amount * profile(t, start, hold, ramp)


def _ar1(rng, n, sd, a):
    e = rng.normal(0, sd * math.sqrt(1 - a * a), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def _steps(rng, t, every, sd):
    """Piecewise-constant shifts (moving between question paper and answer sheet), smoothed."""
    out = np.zeros_like(t)
    level, nxt = 0.0, 0.0
    for i, ti in enumerate(t):
        if ti >= nxt:
            level, nxt = rng.normal(0, sd), ti + rng.exponential(every)
        out[i] = level
    k = np.ones(15) / 15
    return np.convolve(out, k, mode="same")


# ASSUMPTION (stated in reports): natural glances (clock, door, window, invigilator) are level or
# upward (pitch −5…+25° vs the writing posture); copying glances stay down at desk level.
def _glance(rng, d, t, s):
    hold = rng.uniform(0.5, 1.5)
    _add(d["yaw"], t, s, hold, rng.choice([-1, 1]) * rng.uniform(20, 45))
    _add(d["pitch"], t, s, hold, rng.uniform(-5, 25))


def _pair(rng, d, t, s):
    sign = rng.choice([-1, 1])
    for k in range(2):
        g, hold = s + k * rng.uniform(8, 25), rng.uniform(0.6, 1.2)
        _add(d["yaw"], t, g, hold, sign * rng.uniform(20, 40))
        _add(d["pitch"], t, g, hold, rng.uniform(-5, 25))


def _stretch(rng, d, t, s):
    _add(d["pitch"], t, s, rng.uniform(1.5, 3), 40)
    _add(d["wrist_drop"], t, s, rng.uniform(1.5, 3), -1.0)


def _touch(rng, d, t, s):
    _add(d["hand_face"], t, s, rng.uniform(2, 4), -1.1)
