"""Object transfer between seats — chit passing (spec §8.2 "Chit passing", §9.2 CHIT_TRANSFER).

The interaction layer attributes every detected object to the student whose hand holds it, and the
object tracker keeps one id across frames. A transfer is then simply: *the same object id was held
by seat i, and a moment later it is held by seat j*. No desk polygons, no extra calibration.

Guards, in the order they matter:
  * the object must have been held by the source for `min_hold_s` (one bad frame is not a pass)
  * the handover must complete within `max_gap_s` (otherwise it is two unrelated sightings)
  * the two seats must be neighbours (within `reach` seat spacings) when positions are known
  * prohibited classes only — passing a pen is not misconduct unless the hall says so

The event names both seats (`related`), so the dashboard can draw the pair like a SAG edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from reasoner.rules.engine import Event

DEFAULTS = {"classes": ["chit", "phone", "calculator"], "min_hold_s": 0.5, "max_gap_s": 4.0,
            "reach": 1.6, "cooldown_s": 120.0, "weight": 0.70, "min_conf": 0.0}


@dataclass
class _Held:
    seat: str
    cls: str
    since: float
    last: float


class TransferDetector:
    def __init__(self, positions: dict[str, tuple[float, float]] | None = None, cfg: dict | None = None):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.pos = {k: (float(v[0]), float(v[1])) for k, v in (positions or {}).items()}
        pts = np.array(list(self.pos.values())) if self.pos else np.zeros((0, 2))
        if len(pts) > 1:
            d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
            np.fill_diagonal(d, np.inf)
            self.spacing = float(np.median(d.min(axis=1)))
        else:
            self.spacing = 1.0
        self.held: dict[int, _Held] = {}         # object track id -> who is holding it
        self.last_fire: dict[tuple, float] = {}
        self.transfers: list[dict] = []          # for the dashboard / reports

    def neighbours(self, a: str, b: str) -> bool:
        if a not in self.pos or b not in self.pos:
            return True                           # unknown geometry: do not block the event
        (ax, ay), (bx, by) = self.pos[a], self.pos[b]
        return math.hypot(bx - ax, by - ay) <= self.c["reach"] * self.spacing

    def observe(self, t: float, rows: list[dict]) -> list[tuple[str, Event]]:
        """Feed every seat row of one frame. Returns (seat, event) pairs, one per seat involved."""
        out = []
        for row in rows:
            oid, cls = row.get("obj_id", math.nan), row.get("obj_class", "")
            if oid != oid or not cls or not row.get("obj_interact"):
                continue
            if cls not in self.c["classes"] or (row.get("obj_conf") or 0) < self.c["min_conf"]:
                continue
            oid, seat = int(oid), row["seat"]
            prev = self.held.get(oid)
            if prev is None or prev.cls != cls:
                self.held[oid] = _Held(seat, cls, t, t)
                continue
            if prev.seat == seat:
                prev.last = t
                continue
            handover = t - prev.last
            if (prev.last - prev.since >= self.c["min_hold_s"] and handover <= self.c["max_gap_s"]
                    and self.neighbours(prev.seat, seat) and self._ready((prev.seat, seat), t)):
                out += self._events(t, prev.seat, seat, cls, prev.since)
            self.held[oid] = _Held(seat, cls, t, t)
        return out

    def _events(self, t: float, src: str, dst: str, cls: str, start: float) -> list[tuple[str, Event]]:
        self.transfers.append({"t": t, "src": src, "dst": dst, "cls": cls})
        msg = f"{cls.capitalize()}-like object moved from {src} to {dst}"
        w = self.c["weight"]
        return [(s, Event("CHIT_TRANSFER", s, t, start, t, w, 1.0, msg, 1, related=[other]))
                for s, other in ((src, dst), (dst, src))]

    def _ready(self, key: tuple, t: float) -> bool:
        if t - self.last_fire.get(key, -math.inf) < self.c["cooldown_s"]:
            return False
        self.last_fire[key] = t
        return True
