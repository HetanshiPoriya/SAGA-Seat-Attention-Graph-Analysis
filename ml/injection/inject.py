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


def _hold_object(sess: Session, seat: str, start: float, hold: float, cls: str, obj_id: int,
                 below: bool, dist=0.45, conf=0.6) -> np.ndarray:
    """Mark one object as held by `seat` for `hold` s. Returns the time mask it was written on."""
    s, t = _series(sess, seat)
    m = (t >= start) & (t <= start + hold)
    s.objects()[m] = cls
    s.data["obj_id"][m] = float(obj_id)
    s.data["obj_dist"][m] = dist
    s.data["obj_interact"][m] = 1.0
    s.data["obj_below"][m] = 1.0 if below else 0.0
    s.data["obj_conf"][m] = conf
    return m


def phone_in_lap(sess: Session, rng, seat: str, start: float, hold=12.0, cls="phone",
                 hide_wrists=True, pitch_drop=25.0) -> list[dict]:
    """A phone used below the desk: object in the lap, head down, hands usually out of sight
    (spec P5 'phone-in-lap'). Normally also looks like HANDS_BELOW_DESK — that rule is allowed to
    fire too, but the object rule is the one that must."""
    s, t = _series(sess, seat)
    m = _hold_object(sess, seat, start, hold, cls, int(rng.integers(1, 10 ** 6)), below=True,
                     dist=rng.uniform(0.3, 0.8), conf=rng.uniform(0.4, 0.8))
    p = profile(t, start, hold, ramp=1.0)
    s.data["pitch"] -= pitch_drop * p
    s.data["wrist_drop"] += 1.0 * p
    if hide_wrists:
        s.data["wrists_vis"][m] = 0.0
    return [{"seat": seat, "type": "phone_in_lap", "start": start, "end": start + hold + 2,
             "expected": ["PHONE_IN_LAP"], "hold": hold, "obj_class": cls}]


def object_in_hand(sess: Session, rng, seat: str, start: float, hold=6.0, cls="phone") -> list[dict]:
    """A prohibited object held at desk level (phone on the desk, earpiece in the hand)."""
    _hold_object(sess, seat, start, hold, cls, int(rng.integers(1, 10 ** 6)), below=False,
                 dist=rng.uniform(0.2, 0.7), conf=rng.uniform(0.4, 0.8))
    return [{"seat": seat, "type": "object_in_hand", "start": start, "end": start + hold + 2,
             "expected": ["OBJECT_IN_USE"], "hold": hold, "obj_class": cls}]


def chit_pass(sess: Session, rng, src: str, dst: str, start: float, hold=2.0, gap=1.0, cls="chit",
              glance=True) -> list[dict]:
    """A chit passed from `src` to `dst`: the SAME object id is held by the sender, then by the
    receiver a moment later (spec §8.2 'Chit passing'). Both students usually glance at each other
    while doing it, which the SAG may also pick up."""
    oid = int(rng.integers(1, 10 ** 6))
    _hold_object(sess, src, start, hold, cls, oid, below=False, dist=rng.uniform(0.2, 0.6))
    _hold_object(sess, dst, start + hold + gap, hold, cls, oid, below=False, dist=rng.uniform(0.2, 0.6))
    if glance:
        for a, b in ((src, dst), (dst, src)):
            s, t = _series(sess, a)
            sign = np.sign(sess.pos[b][0] - sess.pos[a][0]) or 1.0
            s.data["yaw"] += sign * 30.0 * profile(t, start + rng.uniform(0, hold), 1.0)
    end = start + 2 * hold + gap
    gt = {"seat": src, "type": "chit_pass", "target": dst, "start": start, "end": end,
          "expected": ["CHIT_TRANSFER"], "relational": "CHIT_TRANSFER", "obj_class": cls}
    return [gt, {**gt, "seat": dst, "target": src}]


def hand_near_face(sess: Session, rng, seat: str, start: float, hold=8.0) -> list[dict]:
    s, t = _series(sess, seat)
    p = profile(t, start, hold, ramp=0.5)
    on = p > 0
    before = np.nan_to_num(s.data["hand_face"], nan=1.5)  # no raised hand before = far from the face
    s.data["hand_face"] = np.where(on, before * (1 - p) + 0.35 * p, s.data["hand_face"])
    s.data["wrist_drop"] = np.where(p > 0.5, -0.2, s.data["wrist_drop"])  # the hand is raised to the head
    return [{"seat": seat, "type": "hand_near_face", "start": start, "end": start + hold + 1,
             "expected": ["HAND_NEAR_FACE"], "hold": hold}]
