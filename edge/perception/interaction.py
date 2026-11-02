"""Hand-object interaction features (spec §7: `obj_interact`, §3 gap G3).

Object *presence* is not misconduct — a phone lying on a desk at the far end of the hall is not
evidence, and a water bottle near a hand is not either. What matters is

  * which student's hand the object is in       -> owner seat (attribution, and transfers)
  * how close it is to that student's wrist     -> `obj_dist` in shoulder widths
  * whether it is down in the lap / below desk  -> `obj_below`
  * whether its class is prohibited at all      -> `obj_class`

The per-seat row these produce is what the YAML rules (OBJECT_IN_USE, PHONE_IN_LAP) and the transfer
detector read, exactly like the head-pose features.
"""
from __future__ import annotations

import math

import numpy as np

from edge.perception.head_pose import LHIP, LSH, LWRIST, RHIP, RSH, RWRIST
from edge.perception.objects import ObjectDet, iou, lap_roi

NAN = float("nan")
# Row keys this module produces. obj_class is a string; the rest are floats (NaN = not measurable).
OBJ_FIELDS = ("obj_class", "obj_id", "obj_dist", "obj_interact", "obj_below", "obj_conf")
EMPTY = {"obj_class": "", "obj_id": NAN, "obj_dist": NAN, "obj_interact": 0.0, "obj_below": 0.0,
         "obj_conf": NAN}

INTERACT_K = 1.0        # wrist-to-object distance (shoulder widths) that counts as "in the hand"
BELOW_K = 1.1           # object this far below the shoulder line = lap / below desk level


def _scale(person, thr: float) -> float:
    v = person.kpt_conf >= thr
    if v[LSH] and v[RSH]:
        return float(np.hypot(*(person.kpts[RSH] - person.kpts[LSH])))
    return NAN


def object_features(person, dets: list[ObjectDet], thr: float = 0.4) -> dict:
    """Features for ONE student from the objects already assigned to them (see assign_objects)."""
    f = dict(EMPTY)
    if not dets:
        return f
    sw = _scale(person, thr)
    v = person.kpt_conf >= thr
    wrists = [person.kpts[w] for w in (LWRIST, RWRIST) if v[w]]
    lap = lap_roi(person, (10 ** 6, 10 ** 6), thr)
    sh_y = float((person.kpts[LSH, 1] + person.kpts[RSH, 1]) / 2) if (v[LSH] and v[RSH]) else NAN
    hip_y = float((person.kpts[LHIP, 1] + person.kpts[RHIP, 1]) / 2) if (v[LHIP] and v[RHIP]) else NAN

    best = None
    for d in dets:
        c = d.center
        dist = min((float(np.hypot(*(c - w))) / sw for w in wrists), default=NAN) if sw == sw else NAN
        in_lap = lap is not None and iou(d.box, lap) > 0.05
        # "in the hand": close to a wrist, or inside the lap/desk ROI when the wrists are hidden
        # (hands in the lap are usually occluded by the desk — the reason we have a lap ROI at all)
        interact = (dist == dist and dist <= INTERACT_K) or (in_lap and not wrists)
        below = False
        if hip_y == hip_y:
            below = c[1] > hip_y
        elif sh_y == sh_y and sw == sw:
            below = c[1] > sh_y + BELOW_K * sw
        below = below or (in_lap and d.roi_kind == "lap" and not wrists)
        # prohibited first, then actually interacting, then nearest, then most confident
        key = (not d.prohibited, not interact, dist if dist == dist else 9e9, -d.conf)
        if best is None or key < best[0]:
            best = (key, d, dist, interact, below)
    _, d, dist, interact, below = best
    f.update({"obj_class": d.cls, "obj_id": float(d.track_id) if d.track_id else NAN,
              "obj_dist": dist, "obj_interact": float(bool(interact)), "obj_below": float(bool(below)),
              "obj_conf": d.conf})
    return f


def assign_objects(people: list, dets: list[ObjectDet], thr: float = 0.4,
                   max_dist: float = 2.5) -> dict[int, list[ObjectDet]]:
    """Attribute each detection to the nearest student (by wrist, else by lap ROI), within
    `max_dist` shoulder widths. Returns {id(person): [dets]}.

    Nearest *wrist* rather than nearest body deliberately: while a chit is being passed it should
    belong to whoever's hand holds it, so the owner seat flips from the sender to the receiver and
    the transfer becomes visible (reasoner/transfer.py)."""
    out: dict[int, list[ObjectDet]] = {}
    cache = []
    for p in people:
        sw = _scale(p, thr)
        v = p.kpt_conf >= thr
        cache.append((p, sw, [p.kpts[w] for w in (LWRIST, RWRIST) if v[w]], lap_roi(p, (10 ** 6, 10 ** 6), thr)))
    for d in dets:
        best, best_key = None, None
        for p, sw, wrists, lap in cache:
            if sw != sw or sw < 1:
                continue
            dist = min((float(np.hypot(*(d.center - w))) / sw for w in wrists), default=math.inf)
            if lap is not None and iou(d.box, lap) > 0.05:
                dist = min(dist, 1.0)   # inside the lap ROI counts as close even with hidden hands
            if dist <= max_dist and (best_key is None or dist < best_key):
                best, best_key = p, dist
        if best is not None:
            out.setdefault(id(best), []).append(d)
    return out


def seat_object_rows(seats: dict, dets: list[ObjectDet], thr: float = 0.4) -> dict[str, dict]:
    """{seat label -> object feature dict} for the seats that currently hold a person.
    `seats` is SeatLocker.update()'s output (label -> state with .person)."""
    people = [st.person for st in seats.values() if getattr(st, "person", None) is not None]
    by_person = assign_objects(people, dets, thr)
    rows = {}
    for label, st in seats.items():
        p = getattr(st, "person", None)
        rows[label] = object_features(p, by_person.get(id(p), []), thr) if p is not None else dict(EMPTY)
    return rows
