"""Feature windows for the temporal model (spec §9.3: "2 s windows at 10 Hz → 20 × ~15 features",
baseline-normalised).

The model must see what the rules see, not raw angles: a 30° yaw means nothing on its own, but
`yaw_z = 4` means "far outside what THIS student does". So the window is built from the same
baseline context the rule engine evaluates, which also means the model inherits the per-seat
calibration for free and never has to learn who sits where.

Missing values are real information here (a hidden wrist, an unmeasurable head angle), so every
feature is paired with a 0/1 "present" flag instead of being silently filled with a number the model
would read as a measurement.
"""
from __future__ import annotations

import math

import numpy as np

from reasoner.baseline import BaselineModel

# Baseline-normalised where it makes sense; raw where the scale is already meaningful (counts,
# confidences, flags). Order is frozen — the trained weights depend on it.
FEATURES = [
    "yaw_z", "pitch_z", "lean_x_z", "shoulder_tilt_z", "wrist_drop_z", "hand_face_z",
    "yaw_dev", "pitch_dev", "yaw_dev_long",
    "wrists_vis", "kp_conf", "obj_interact", "obj_below",
    "facing_camera", "facing_away", "facing_profile",
]
SCALE = {"yaw_dev": 30.0, "pitch_dev": 30.0, "yaw_dev_long": 30.0, "wrists_vis": 2.0}
CLIP_Z = 8.0          # z-scores are heavy-tailed on real keypoints; a spike must not dominate
WINDOW = 20           # 2 s at 10 Hz
N_FEATURES = len(FEATURES) * 2   # value + "present" flag


def vector(ctx: dict) -> np.ndarray:
    """One frame of baseline context -> fixed-length vector (values followed by presence flags)."""
    vals = np.zeros(len(FEATURES), np.float32)
    present = np.zeros(len(FEATURES), np.float32)
    facing = ctx.get("facing", "unknown")
    for i, name in enumerate(FEATURES):
        if name.startswith("facing_"):
            vals[i] = float(facing == name.split("_", 1)[1])
            present[i] = float(facing != "unknown")
            continue
        v = ctx.get(name, math.nan)
        if v is None or v != v:          # NaN: leave the value at 0 and flag it as missing
            continue
        v = v / SCALE.get(name, 1.0)
        vals[i] = float(np.clip(v, -CLIP_Z, CLIP_Z))
        present[i] = 1.0
    return np.concatenate([vals, present])


def session_sequences(session, calib_s: float, min_frames: int = WINDOW + 1) -> dict[str, np.ndarray]:
    """Run a Session through the baseline model and return {seat: (T, N_FEATURES) array}.

    Only frames after the baseline has frozen are returned — before that the z-scores do not exist.
    """
    model = BaselineModel(calib_s)
    rows: dict[str, list[np.ndarray]] = {}
    times: dict[str, list[float]] = {}
    for t, frame in session.to_frames():
        for row in frame:
            b = model.update(t, row)
            if not b.frozen:
                continue
            ctx = model.context(b, row)
            ctx["facing"] = row.get("facing", "unknown")
            rows.setdefault(row["seat"], []).append(vector(ctx))
            times.setdefault(row["seat"], []).append(t)
    return {s: np.stack(v) for s, v in rows.items() if len(v) >= min_frames}


def windows(seq: np.ndarray, window: int = WINDOW, stride: int = 1) -> np.ndarray:
    """(T, F) -> (N, window, F) sliding windows."""
    if len(seq) <= window:
        return np.empty((0, window, seq.shape[1]), np.float32)
    idx = np.arange(0, len(seq) - window, stride)
    return np.stack([seq[i:i + window] for i in idx]).astype(np.float32)
