"""Seat Attention Graph (spec §8) — who looks toward whose desk, how often, and whether it is returned.

Every desk-level sideward glance becomes a directed edge `seat_i -> seat_j`, where j is the nearest
neighbour on the side the head turned to (same row first, then diagonal, within ~1.6 seat spacings).
Over sliding windows the graph yields relational events:

  COPYING_ONE_WAY      i -> j repeatedly while j hardly looks back      (copier i, source j)
  MUTUAL_ORIENTATION   i -> j and j -> i within a few seconds, repeatedly (collusion / talking)
  ATTENTION_HUB        several different seats repeatedly oriented toward one seat (possible source;
                       informational, adds no risk)

Geometry v1 works on seat positions (plan coordinates for a clicked grid, image centres for auto
seats) and the image-space head yaw sign. Full ray projection through the homography is a refinement
for halls where rows are strongly foreshortened.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from reasoner.rules.engine import Event


@dataclass
class Edge:
    src: str
    dst: str
    start: float
    end: float
    obs: float

    @property
    def dwell(self) -> float:
        return self.end - self.start


@dataclass
class _Glance:
    start: float | None = None
    last: float = -math.inf
    devs: list[float] = field(default_factory=list)
    obs: list[float] = field(default_factory=list)
    peak: float = 0.0


class SeatAttentionGraph:
    def __init__(self, positions: dict[str, tuple[float, float]], cfg: dict | None = None):
        c = {"start_z": 2.5, "start_deg": 12.0, "end_z": 1.2, "end_frac": 0.4, "max_pitch_up": 10.0, "max_profile_base": 0.2, "min_s": 0.6, "max_s": 4.0,
             "gap_s": 0.4, "reach": 1.6, "window_s": 120.0, "copy_min": 3, "copy_max_back": 1,
             "mutual_lag_s": 5.0, "mutual_min": 2, "hub_window_s": 600.0, "hub_sources": 3,
             "cooldown_s": 60.0, "copy_weight": 0.35, "mutual_weight": 0.45, **(cfg or {})}
        self.c = c
        self.pos = {k: (float(v[0]), float(v[1])) for k, v in positions.items()}
        pts = np.array(list(self.pos.values())) if self.pos else np.zeros((0, 2))
        if len(pts) > 1:
            d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
            np.fill_diagonal(d, np.inf)
            self.spacing = float(np.median(d.min(axis=1)))
        else:
            self.spacing = 1.0
        self.glance: dict[str, _Glance] = defaultdict(_Glance)
        self.edges: deque[Edge] = deque()
        self.last_fire: dict[tuple, float] = {}

    # ---------------------------------------------------------------- geometry
    def target(self, seat: str, side: float) -> str | None:
        """Nearest seat on `side` (+1 image-right / −1 image-left): same row first, then diagonal."""
        if seat not in self.pos:
            return None
        x, y = self.pos[seat]
        s = self.spacing
        best, best_key = None, None
        for other, (ox, oy) in self.pos.items():
            dx, dy = ox - x, oy - y
            if other == seat or dx * side < 0.3 * s or math.hypot(dx, dy) > self.c["reach"] * s:
                continue
            key = (abs(dy) > 0.5 * s, math.hypot(dx, dy))  # same row first, then distance
            if best_key is None or key < best_key:
                best, best_key = other, key
        return best

    # ---------------------------------------------------------------- updates
    def observe(self, t: float, seat: str, ctx: dict, obs: float) -> list[tuple[str, Event]]:
        """Feed one seat-frame (baseline context). Returns (seat, event) pairs — events may concern
        the partner seat too (mutual orientation)."""
        c, g = self.c, self.glance[seat]
        dev, z, pdev = ctx.get("yaw_dev", math.nan), ctx.get("yaw_dev_z", math.nan), ctx.get("pitch_dev", math.nan)
        if g.start is not None and t - g.last > c["gap_s"]:
            out = self._close(seat, g)
            if out:
                return out
        if g.start is None:
            on = (abs(z) > c["start_z"] and abs(dev) > c["start_deg"] and not (pdev > c["max_pitch_up"])
                  and ctx.get("profile_base", 0.0) < c["max_profile_base"])
        else:
            # hysteresis: back within end_z, or most of the way back from the peak; NaN keeps it open
            on = not (abs(z) < c["end_z"] or abs(dev) < c["end_frac"] * g.peak)
        if on:
            if g.start is None:
                g.start, g.devs, g.obs, g.peak = t, [], [], 0.0
            g.last = t
            if dev == dev:
                g.devs.append(dev)
                g.peak = max(g.peak, abs(dev))
            g.obs.append(obs)
            return []
        if g.start is not None and t - g.last > c["gap_s"]:
            return self._close(seat, g)  # same closing gap as the rule engine -> events land together
        return []

    def _close(self, seat: str, g: _Glance) -> list[tuple[str, Event]]:
        start, end = g.start, g.last
        g.start = None
        if not g.devs or not (self.c["min_s"] <= end - start <= self.c["max_s"]):
            return []
        dst = self.target(seat, math.copysign(1.0, float(np.mean(g.devs))))
        if dst is None:
            return []
        edge = Edge(seat, dst, start, end, float(np.mean(g.obs)) if g.obs else 1.0)
        self.edges.append(edge)
        while self.edges and self.edges[0].end < end - self.c["hub_window_s"]:
            self.edges.popleft()
        return self._patterns(edge)

    # ---------------------------------------------------------------- queries
    def recent(self, t: float, window: float) -> list[Edge]:
        return [e for e in self.edges if e.end >= t - window]

    def counts(self, t: float, window: float | None = None) -> dict[tuple[str, str], tuple[int, float]]:
        """(src, dst) -> (number of glances, total dwell s) in the window."""
        out: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0.0])
        for e in self.recent(t, window or self.c["window_s"]):
            out[(e.src, e.dst)][0] += 1
            out[(e.src, e.dst)][1] += e.dwell
        return {k: (v[0], v[1]) for k, v in out.items()}

    def reciprocal_pairs(self, a: str, b: str, t: float, window: float) -> int:
        ab = [e for e in self.recent(t, window) if e.src == a and e.dst == b]
        ba = [e for e in self.recent(t, window) if e.src == b and e.dst == a]
        lag = self.c["mutual_lag_s"]
        return sum(any(abs(x.start - y.start) <= lag for y in ba) for x in ab)

    # ---------------------------------------------------------------- patterns
    def _patterns(self, e: Edge) -> list[tuple[str, Event]]:
        c, t = self.c, e.end
        w = c["window_s"]
        out = []
        pair = tuple(sorted((e.src, e.dst)))
        mutual = self.reciprocal_pairs(e.src, e.dst, t, w)
        if mutual >= c["mutual_min"] and self._ready(("MUTUAL",) + pair, t):
            msg = "Mutual orientation between {a} and {b} ({n}x within {lag:.0f} s of each other in {w} s)"
            for s, other in ((e.src, e.dst), (e.dst, e.src)):
                out.append((s, Event("MUTUAL_ORIENTATION", s, t, t - w, t, c["mutual_weight"], 1.0,
                                     msg.format(a=s, b=other, n=mutual, lag=c["mutual_lag_s"], w=int(w)),
                                     mutual, related=[other])))
            return out
        fwd = [x for x in self.recent(t, w) if x.src == e.src and x.dst == e.dst]
        back = [x for x in self.recent(t, w) if x.src == e.dst and x.dst == e.src]
        if len(fwd) >= c["copy_min"] and len(back) <= c["copy_max_back"] and self._ready(("COPY", e.src, e.dst), t):
            avg = sum(x.dwell for x in fwd) / len(fwd)
            msg = (f"Looked toward {e.dst}'s desk {len(fwd)}x (avg {avg:.1f} s) in {int(w)} s; "
                   f"{e.dst} {'not reciprocating' if not back else 'looked back once'}")
            out.append((e.src, Event("COPYING_ONE_WAY", e.src, t, fwd[0].start, t, c["copy_weight"], 1.0, msg,
                                     len(fwd), related=[e.dst])))
        sources = {x.src for x in self.recent(t, c["hub_window_s"]) if x.dst == e.dst}
        if len(sources) >= c["hub_sources"] and self._ready(("HUB", e.dst), t, c["hub_window_s"]):
            msg = f"Several seats ({', '.join(sorted(sources))}) repeatedly oriented toward {e.dst}"
            out.append((e.dst, Event("ATTENTION_HUB", e.dst, t, t - c["hub_window_s"], t, 0.0, 1.0, msg,
                                     len(sources), related=sorted(sources))))
        return out

    def _ready(self, key: tuple, t: float, cooldown: float | None = None) -> bool:
        if t - self.last_fire.get(key, -math.inf) < (cooldown or self.c["cooldown_s"]):
            return False
        self.last_fire[key] = t
        return True
