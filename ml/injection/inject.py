"""Inject controlled, labelled misconduct into normal seat tracks (spec §16.3 step 3).

Every function edits the session in place and returns ground-truth records:
  {"seat", "type", "start", "end", "expected": [rule ids], "target"?, params...}
"""
from __future__ import annotations

import numpy as np

from ml.injection.session import Session, profile


def _series(sess: Session, seat: str):
    s = sess.seats[seat]
    return s, s.t


def _glance_times(rng, start, window, k, dur):
    """k glance start times inside [start, start+window], at least dur+1.5 s apart."""
    for _ in range(200):
        ts = np.sort(rng.uniform(start, start + max(window - dur, dur), k))
        if k == 1 or np.min(np.diff(ts)) > dur + 1.5:
            return ts
    return start + np.arange(k) * (dur + 2.0)


def copying(sess: Session, rng, seat: str, target: str, start: float, k=4, amplitude=35.0, hold=1.5,
            window=120.0, ramp=0.3, torso=True) -> list[dict]:
    """One-way copying i -> j: k glances toward j's bearing within `window` s."""
    s, t = _series(sess, seat)
    sign = np.sign(sess.pos[target][0] - sess.pos[seat][0]) or 1.0
    starts = _glance_times(rng, start, window, k, hold + 2 * ramp)
    for g in starts:
        p = profile(t, g, hold, ramp)
        s.data["yaw"] += sign * amplitude * p
        if torso:
            s.data["shoulder_tilt"] += sign * 0.15 * amplitude * p   # small body rotation
    expected = ["SIDE_GLANCE_REPEAT", "COPYING_ONE_WAY"] + (["LONG_SIDE_LOOK"] if hold >= 5 else [])
    return [{"seat": seat, "type": "copying", "target": target, "start": float(starts[0]),
             "end": float(starts[-1] + hold + 2 * ramp), "expected": expected, "relational": "COPYING_ONE_WAY",
             "k": k, "amplitude": amplitude, "hold": hold, "glances": [float(g) for g in starts]}]


def mutual(sess: Session, rng, a: str, b: str, start: float, k=3, amplitude=35.0, hold=1.5, lag_max=5.0,
           window=120.0) -> list[dict]:
    """Collusion i <-> j: both glance toward each other, j within 0–lag_max s after i."""
    gt = copying(sess, rng, a, b, start, k, amplitude, hold, window)
    s, t = _series(sess, b)
    sign = np.sign(sess.pos[a][0] - sess.pos[b][0]) or 1.0
    starts = [g + rng.uniform(0, lag_max) for g in gt[0]["glances"]]
    for g in starts:
        s.data["yaw"] += sign * amplitude * profile(t, g, hold)
    gt[0].update(type="mutual", expected=["SIDE_GLANCE_REPEAT", "MUTUAL_ORIENTATION"], relational="MUTUAL_ORIENTATION")
    gt.append({"seat": b, "type": "mutual", "target": a, "start": float(starts[0]),
               "end": float(starts[-1] + hold + 0.6), "expected": ["SIDE_GLANCE_REPEAT", "MUTUAL_ORIENTATION"],
               "relational": "MUTUAL_ORIENTATION", "k": k,
               "amplitude": amplitude, "hold": hold, "glances": starts})
    return gt


def hands_under_desk(sess: Session, rng, seat: str, start: float, hold=10.0, pitch_drop=25.0,
                     hide_wrists=True) -> list[dict]:
    s, t = _series(sess, seat)
    p = profile(t, start, hold, ramp=1.0)
    s.data["pitch"] -= pitch_drop * p
    s.data["wrist_drop"] += 1.2 * p
    if hide_wrists:
        s.data["wrists_vis"][p > 0.5] = 0.0
        s.data["wrist_drop"][p > 0.5] = np.nan
    return [{"seat": seat, "type": "hands_under_desk", "start": start, "end": start + hold + 2,
             "expected": ["HANDS_BELOW_DESK"], "hold": hold}]


def turn_around(sess: Session, rng, seat: str, start: float, k=2, hold=2.5, window=60.0) -> list[dict]:
    s, t = _series(sess, seat)
    starts = _glance_times(rng, start, window, k, hold)
    for g in starts:
        m = (t >= g) & (t <= g + hold)
        s.facing[m] = "away"
        s.data["yaw"][m] = np.nan
        s.data["pitch"][m] = np.nan
    return [{"seat": seat, "type": "turn_around", "start": float(starts[0]), "end": float(starts[-1] + hold),
             "expected": ["TURNED_AROUND"], "k": k}]


def hand_near_face(sess: Session, rng, seat: str, start: float, hold=8.0) -> list[dict]:
    s, t = _series(sess, seat)
    p = profile(t, start, hold, ramp=0.5)
    on = p > 0
    before = np.nan_to_num(s.data["hand_face"], nan=1.5)  # no raised hand before = far from the face
    s.data["hand_face"] = np.where(on, before * (1 - p) + 0.35 * p, s.data["hand_face"])
    s.data["wrist_drop"] = np.where(p > 0.5, -0.2, s.data["wrist_drop"])  # the hand is raised to the head
    return [{"seat": seat, "type": "hand_near_face", "start": start, "end": start + hold + 1,
             "expected": ["HAND_NEAR_FACE"], "hold": hold}]
