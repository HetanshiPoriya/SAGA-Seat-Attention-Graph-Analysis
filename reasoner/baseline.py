"""Self-calibrating per-seat baseline (spec §9.1).

During the first `calib_s` seconds every seat collects its own normal: robust median and spread
(MAD × 1.4826) of each feature. Then the baseline freezes, so a student cannot "train" the system
by misbehaving later. Spreads are floored (nobody is perfectly still) and capped at the hall's
75th percentile (a student who fidgets a lot during calibration cannot widen their own tolerance
beyond what is typical for the hall). Seats that appear late keep collecting until they have enough
frames (rolling baseline for late joiners).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

Z_FEATURES = ["yaw", "pitch", "lean_x", "shoulder_tilt", "wrist_drop", "hand_face"]


@dataclass
class SeatBaseline:
    first_t: float
    samples: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in Z_FEATURES})
    facing: list[str] = field(default_factory=list)
    wrists_vis: list[float] = field(default_factory=list)
    med: dict[str, float] = field(default_factory=dict)
    sigma: dict[str, float] = field(default_factory=dict)
    away_base: float = 0.0
    profile_base: float = 0.0
    wrists_vis_base: float = 2.0
    frozen: bool = False

    def add(self, row: dict):
        for k in Z_FEATURES:
            v = row.get(k, math.nan)
            if v == v:  # not NaN
                self.samples[k].append(v)
        self.facing.append(row.get("facing", "unknown"))
        if row.get("wrists_vis") == row.get("wrists_vis"):
            self.wrists_vis.append(row["wrists_vis"])

    @property
    def n(self) -> int:
        return len(self.facing)

    def raw_stats(self) -> dict[str, tuple[float, float]]:
        out = {}
        for k, xs in self.samples.items():
            if len(xs) >= 5:
                a = np.asarray(xs)
                med = float(np.median(a))
                out[k] = (med, float(1.4826 * np.median(np.abs(a - med))))
        return out


class BaselineModel:
    def __init__(self, calib_s: float = 180, min_samples: int = 30, sigma_floor: dict | None = None,
                 sigma_cap_percentile: float = 75, settle_s: float = 10.0):
        self.calib_s, self.min_samples = calib_s, min_samples
        self.settle_s = settle_s  # ignore each seat's first seconds: people are still settling in
        self.floor = sigma_floor or {}
        self.cap_pct = sigma_cap_percentile
        self.seats: dict[str, SeatBaseline] = {}
        self.start_t: float | None = None
        self.hall_cap: dict[str, float] = {}
        self.recent: dict[str, RecentMedian] = {}
        self.recent_long: dict[str, RecentMedian] = {}

    def update(self, t: float, row: dict) -> SeatBaseline:
        """Feed one seat-frame. Returns that seat's baseline (check .frozen before using z-scores)."""
        if self.start_t is None:
            self.start_t = t
        b = self.seats.setdefault(row["seat"], SeatBaseline(first_t=t))
        self.recent.setdefault(row["seat"], RecentMedian()).add(t, row)
        self.recent_long.setdefault(row["seat"], RecentMedian(LONG_WINDOW_S)).add(t, row)
        if not b.frozen:
            if t - b.first_t >= min(self.settle_s, self.calib_s / 3):
                b.add(row)
            if t - self.start_t >= self.calib_s:
                self._freeze_ready(t)
        return b

    def _freeze_ready(self, t: float):
        ready = [b for b in self.seats.values()
                 if not b.frozen and b.n >= self.min_samples and t - b.first_t >= min(self.calib_s, 30)]
        if not ready:
            return
        if not self.hall_cap:  # hall-level cap from every seat that has data at the first freeze
            spreads: dict[str, list[float]] = {k: [] for k in Z_FEATURES}
            for b in self.seats.values():
                for k, (_, s) in b.raw_stats().items():
                    spreads[k].append(s)
            self.hall_cap = {k: float(np.percentile(v, self.cap_pct)) for k, v in spreads.items() if v}
        for b in ready:
            for k, (med, s) in b.raw_stats().items():
                s = max(s, self.floor.get(k, 1e-6))
                if k in self.hall_cap:
                    s = min(s, max(self.hall_cap[k], self.floor.get(k, 1e-6)))
                b.med[k], b.sigma[k] = med, s
            b.away_base = sum(f == "away" for f in b.facing) / max(b.n, 1)
            b.profile_base = sum(f == "profile" for f in b.facing) / max(b.n, 1)
            b.wrists_vis_base = float(np.median(b.wrists_vis)) if b.wrists_vis else 0.0
            b.frozen = True
            b.samples = {k: [] for k in Z_FEATURES}  # free memory; stats are kept
            b.facing, b.wrists_vis = [], []

    def context(self, b: SeatBaseline, row: dict) -> dict:
        """Variables available to rule conditions."""
        ctx = {k: row.get(k, math.nan) for k in
               ("yaw", "pitch", "lean_x", "shoulder_tilt", "wrist_drop", "wrists_vis", "hand_face",
                "head_px", "kp_conf")}
        ctx["facing"] = row.get("facing", "unknown")
        # hand-object interaction (P5): absolute, not baseline-relative — a phone is a phone whatever
        # this student's normal looks like. obj_class is "" when nothing was detected for this seat.
        ctx["obj_class"] = row.get("obj_class") or ""
        ctx["obj_dist"], ctx["obj_conf"] = row.get("obj_dist", math.nan), row.get("obj_conf", math.nan)
        ctx["obj_interact"] = row.get("obj_interact", 0.0) or 0.0
        ctx["obj_below"] = row.get("obj_below", 0.0) or 0.0
        for k in Z_FEATURES:
            med, s = b.med.get(k, math.nan), b.sigma.get(k, math.nan)
            ctx[f"{k}_med"] = med
            ctx[f"{k}_z"] = (ctx[k] - med) / s if s == s and s > 0 else math.nan
        ctx["away_base"], ctx["wrists_vis_base"] = b.away_base, b.wrists_vis_base
        ctx["profile_base"] = b.profile_base  # seat seen side-on: head yaw is not measurable here
        for suffix, recent in (("dev", self.recent), ("dev_long", self.recent_long)):
            # deviation from the student's RECENT direction: 20 s for quick glances, 2 min for long
            # looks (absorbs slow posture changes that a frozen baseline would misread)
            rec = recent.get(row["seat"])
            for k in RecentMedian.KEYS:
                m = rec.median(k) if rec else math.nan
                ctx[f"{k}_{suffix}"] = ctx[k] - m
                s = b.sigma.get(k, math.nan)
                ctx[f"{k}_{suffix}_z"] = ctx[f"{k}_{suffix}"] / s if s == s and s > 0 else math.nan
        return ctx


LONG_WINDOW_S = 120.0


class RecentMedian:
    """Rolling median of head angles over the last `window_s` seconds. Slow drift (shifting between
    question paper and answer sheet) moves it; a 1–3 s glance barely does."""
    KEYS = ("yaw", "pitch")

    REFRESH_S = 0.5  # a 20 s median barely moves in half a second; recomputing per frame is wasteful

    def __init__(self, window_s: float = 20.0):
        self.window_s = window_s
        self.buf: dict[str, deque] = {k: deque() for k in self.KEYS}
        self._cache: dict[str, float] = {}
        self._cache_t = -math.inf

    def add(self, t: float, row: dict):
        if t - self._cache_t >= self.REFRESH_S:
            self._cache.clear()
            self._cache_t = t
        for k in self.KEYS:
            v = row.get(k, math.nan)
            if v == v:
                self.buf[k].append((t, v))
            while self.buf[k] and self.buf[k][0][0] < t - self.window_s:
                self.buf[k].popleft()

    def median(self, k: str) -> float:
        if k not in self._cache:
            b = self.buf[k]
            if len(b) < 5:
                return math.nan  # not cached: keep trying while the buffer fills
            vals = sorted(v for _, v in b)
            n = len(vals)
            self._cache[k] = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        return self._cache[k]


CONTEXT_NAMES = ({"yaw", "pitch", "lean_x", "shoulder_tilt", "wrist_drop", "wrists_vis", "hand_face",
                  "head_px", "kp_conf", "facing", "away_base", "profile_base", "wrists_vis_base",
                  "obj_class", "obj_dist", "obj_interact", "obj_below", "obj_conf"}
                 | {f"{k}_z" for k in Z_FEATURES} | {f"{k}_med" for k in Z_FEATURES}
                 | {f"{k}_{sfx}" for k in RecentMedian.KEYS for sfx in ("dev", "dev_z", "dev_long", "dev_long_z")}
                 | {"yaw_dev_peak", "yaw_dev_ref"})  # set by the rule engine inside an episode (end_when)
