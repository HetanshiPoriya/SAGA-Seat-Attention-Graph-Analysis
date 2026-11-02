"""Reasoning-layer evaluation by behaviour injection (spec §16.3 steps 4–5, §17.1).

Metrics
  recall           injected behaviours detected by their expected rule (event within the behaviour
                   window + TOL_S)
  alert recall     injected behaviours whose seat reached amber/red within the window + 60 s
  precision        events that overlap an injected behaviour on the same seat / all events
  FA/student-hour  events on seats/times with no injected behaviour, per monitored seat-hour
                   (after calibration)
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from ml.injection import inject
from ml.injection.session import Session, synthetic_session
from reasoner.reasoner import Reasoner

TOL_S = 10.0


@dataclass
class RunResult:
    gt: list[dict]
    events: list
    alerts: list
    calib_s: float
    duration: float
    n_seats: int
    detected: list[bool] = field(default_factory=list)
    alerted: list[bool] = field(default_factory=list)
    false_events: list = field(default_factory=list)
    false_alerts: list = field(default_factory=list)
    latency: list[float] = field(default_factory=list)
    relational: list[dict] = field(default_factory=list)
    clean_events: list = field(default_factory=list)   # on seats with NO injected behaviour at all
    clean_alerts: list = field(default_factory=list)
    clean_seats: int = 0


def run(sess: Session, gt: list[dict], calib_s: float, rules_file=None, sag=True) -> RunResult:
    positions = {s: sess.pos[s] for s in sess.seats} if sag else None
    r = Reasoner(calib_s=calib_s, rules_file=rules_file, seat_positions=positions)
    events, alerts = [], []
    for t, rows in sess.to_frames():
        out = r.step(t, rows)
        events += out.events
        alerts += out.alerts
    res = RunResult(gt, events, alerts, calib_s, sess.duration, len(sess.seats))
    for g in gt:
        def in_win(e):
            return g["start"] <= e.t <= g["end"] + TOL_S
        hits = [e for e in events if e.seat == g["seat"] and e.rule_id in g["expected"] and in_win(e)]
        res.detected.append(bool(hits))
        if hits:
            res.latency.append(min(e.t for e in hits) - g["end"])
        res.alerted.append(any(a.seat == g["seat"] and g["start"] <= a.t <= g["end"] + 60 for a in alerts))
        if g.get("relational"):  # spec §16.3 step 5: right neighbour, right direction
            rel = [e for e in events if e.seat == g["seat"] and e.rule_id == g["relational"] and in_win(e)]
            res.relational.append({
                "type": g["type"], "detected": any(g["target"] in e.related for e in rel),
                "wrong_target": any(g["target"] not in e.related for e in rel),
                "reversed": any(e.rule_id == "COPYING_ONE_WAY" and e.seat == g["target"] and g["seat"] in e.related
                                and in_win(e) for e in events)})
    for e in events:
        if e.weight == 0:  # informational (attention hub) — not a flag
            continue
        if not any(g["seat"] == e.seat and g["start"] - TOL_S <= e.t <= g["end"] + TOL_S for g in gt):
            res.false_events.append(e)
    for a in alerts:
        if not any(g["seat"] == a.seat and g["start"] - TOL_S <= a.t <= g["end"] + 60 for g in gt):
            res.false_alerts.append(a)
    # A seat that was given a behaviour keeps residual risk long after its scoring window, so a later
    # ordinary glance there is counted "false" although most of the risk is real. The honest
    # "normal student is left alone" number therefore uses only seats that were never touched.
    touched = {g["seat"] for g in gt} | {g["target"] for g in gt if g.get("target")}
    res.clean_seats = sum(s not in touched for s in sess.seats)
    res.clean_events = [e for e in res.false_events if e.seat not in touched]
    res.clean_alerts = [a for a in res.false_alerts if a.seat not in touched]
    return res


def summarize(results: list[RunResult]) -> dict:
    det = [d for r in results for d in r.detected]
    alr = [a for r in results for a in r.alerted]
    n_ev = sum(len(r.events) for r in results)
    n_fa = sum(len(r.false_events) for r in results)
    hours = sum(r.n_seats * max(r.duration - r.calib_s, 0) for r in results) / 3600
    clean_hours = sum(r.clean_seats * max(r.duration - r.calib_s, 0) for r in results) / 3600
    lat = [x for r in results for x in r.latency]
    rel = [x for r in results for x in r.relational]
    det_rel = [x for x in rel if x["detected"] or x["wrong_target"]]
    return {"behaviours": len(det), "recall": np.mean(det) if det else float("nan"),
            "pair_recall": np.mean([x["detected"] for x in rel]) if rel else float("nan"),
            "attribution": np.mean([x["detected"] and not x["wrong_target"] for x in det_rel])
            if det_rel else float("nan"),
            "reversed": np.mean([x["reversed"] for x in rel if x["type"] == "copying"])
            if any(x["type"] == "copying" for x in rel) else float("nan"),
            "alert_recall": np.mean(alr) if alr else float("nan"),
            "precision": (n_ev - n_fa) / n_ev if n_ev else float("nan"),
            "events": n_ev, "false_events": n_fa, "student_hours": hours,
            "clean_alerts_per_student_hour": (sum(len(r.clean_alerts) for r in results) / clean_hours
                                              if clean_hours else float("nan")),
            "clean_events_per_student_hour": (sum(len(r.clean_events) for r in results) / clean_hours
                                              if clean_hours else float("nan")),
            "clean_student_hours": clean_hours,
            "fa_per_student_hour": n_fa / hours if hours else float("nan"),
            "false_alerts_per_student_hour": (sum(len(r.false_alerts) for r in results) / hours
                                              if hours else float("nan")),
            "median_latency_s": float(np.median(lat)) if lat else float("nan")}


# ------------------------------------------------------------------ scenarios
def scenario(sess: Session, rng, calib_s: float) -> list[dict]:
    """One of each behaviour on different seats, after calibration; the remaining seats stay untouched."""
    seats = list(sess.seats)
    rng.shuffle(seats)
    t0 = calib_s + 30
    span = max(sess.duration - t0 - 130, 1)
    gt = []
    used = set()

    def free():
        s = next(s for s in seats if s not in used)
        used.add(s)
        return s

    def at():
        return t0 + rng.uniform(0, span)

    # behaviours between two seats: copying (the source stays untouched), collusion, chit passing
    for fn in (inject.copying, inject.mutual, inject.chit_pass):
        for s in seats:
            nb = [n for n in sess.same_row_neighbours(s) if n not in used]
            if s not in used and nb:
                used.update({s, nb[0]})
                gt += fn(sess, rng, s, nb[0], at())
                break
    for fn in (inject.hands_under_desk, inject.turn_around, inject.hand_near_face,
               inject.phone_in_lap, inject.object_in_hand):
        if len(used) < len(seats):
            gt += fn(sess, rng, free(), at())
    return gt


def copy_target(sess: Session, seat: str, rng) -> str:
    """A same-row neighbour to copy from; for a lone seat (webcam) a virtual one that only sets direction."""
    nb = sess.same_row_neighbours(seat)
    if nb:
        return nb[0]
    x, y = sess.pos[seat]
    side = float(rng.choice([-1, 1]))
    sess.pos[f"virtual{'R' if side > 0 else 'L'}"] = (x + side, y)
    return f"virtual{'R' if side > 0 else 'L'}"


def neighbour_seat(sess: Session, seat: str) -> str | None:
    """A real same-row neighbour (chit passing needs a seat that is actually monitored)."""
    nb = sess.same_row_neighbours(seat)
    return nb[0] if nb else None


BEHAVIOURS = {
    "copying": lambda s, rng, seat, t0: inject.copying(s, rng, seat, copy_target(s, seat, rng), t0),
    "hands_under_desk": lambda s, rng, seat, t0: inject.hands_under_desk(s, rng, seat, t0),
    "turn_around": lambda s, rng, seat, t0: inject.turn_around(s, rng, seat, t0),
    "hand_near_face": lambda s, rng, seat, t0: inject.hand_near_face(s, rng, seat, t0),
    "phone_in_lap": lambda s, rng, seat, t0: inject.phone_in_lap(s, rng, seat, t0),
    "object_in_hand": lambda s, rng, seat, t0: inject.object_in_hand(s, rng, seat, t0),
    # needs a second monitored seat: skipped for single-seat (webcam) recordings
    "chit_pass": lambda s, rng, seat, t0: (inject.chit_pass(s, rng, seat, neighbour_seat(s, seat), t0)
                                           if neighbour_seat(s, seat) else []),
}


def per_type(base: Session, rng, calib_s: float, repeats: int = 3) -> dict[str, list[RunResult]]:
    """Each behaviour injected alone into a copy of the session (works with a single real seat)."""
    out = {}
    for name, fn in BEHAVIOURS.items():
        results = []
        for _ in range(repeats):
            sess = base.copy()
            seat = str(rng.choice(list(sess.seats)))
            t0 = calib_s + 20 + rng.uniform(0, max(sess.duration - calib_s - 150, 1))
            gt = fn(sess, rng, seat, t0)
            if not gt:      # behaviour not applicable to this session (e.g. chit pass, one seat only)
                break
            results.append(run(sess, gt, calib_s))
        if results:
            out[name] = results
    return out


def sweep(param: str, values, trials=4, calib_s=180.0, duration_s=480.0, base=None, seed0=0,
          session_factory=None):
    """Detection rate of one-way copying vs subtlety (spec §16.3 step 4). Each trial: a fresh
    session, copying injected on up to 3 seats; the rest untouched (for false alarms)."""
    base = {"amplitude": 35.0, "hold": 1.5, "k": 4, **(base or {})}
    factory = session_factory or (lambda tr: synthetic_session(duration_s=duration_s, seed=seed0 + tr))
    rows = []
    for v in values:
        params = {**base, param: v}
        results = []
        for tr in range(trials):
            rng = np.random.default_rng(seed0 + 1000 * tr + zlib.crc32(param.encode()) % 997)
            sess = factory(tr)
            gt, used = [], set()
            for s in rng.permutation(list(sess.seats)):
                s = str(s)
                if s in used or len(gt) >= 3:
                    continue
                target = copy_target(sess, s, rng)
                used.update({s, target})
                start = calib_s + 20 + rng.uniform(0, max(sess.duration - calib_s - 150, 1))
                gt += inject.copying(sess, rng, s, target, start,
                                     k=int(params["k"]), amplitude=params["amplitude"], hold=params["hold"])
            results.append(run(sess, gt, calib_s))
        rows.append({param: v, **summarize(results)})
    return rows
