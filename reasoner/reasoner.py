"""Reasoning service: seat features -> baseline z-scores -> rules -> risk -> events/alerts."""
from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from reasoner.baseline import BaselineModel
from reasoner.risk import Alert, RiskScorer, observability
from reasoner.rules.engine import Event, RuleEngine, load_rules
from reasoner.sag import SeatAttentionGraph

HERE = Path(__file__).resolve().parent
RELATIONAL_GLANCE = {"COPYING_ONE_WAY", "MUTUAL_ORIENTATION"}
NUMERIC = ["yaw", "pitch", "yaw_s", "pitch_s", "lean_x", "shoulder_tilt", "shoulder_ratio", "wrist_drop",
           "wrists_vis", "hand_face", "head_px", "kp_conf"]


@dataclass
class SeatView:
    """What the UI needs per seat."""
    calibrating: bool
    level: str
    risk: float
    obs: float


@dataclass
class StepResult:
    events: list[Event] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)


class Reasoner:
    def __init__(self, config: str | Path = HERE / "config.yaml", calib_s: float | None = None,
                 rules_file: str | Path | None = None, seat_positions: dict | None = None):
        """seat_positions: seat -> (x, y) for the Seat Attention Graph (plan coords for a clicked grid,
        image centres for auto seats). Without it the relational (SAG) layer is off."""
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        b = cfg["baseline"]
        if calib_s is not None:
            b["calib_s"] = calib_s
        self.config = cfg
        self.baseline = BaselineModel(b["calib_s"], b["min_samples"], b["sigma_floor"], b["sigma_cap_percentile"],
                                      b.get("settle_s", 10.0))
        self.rules = RuleEngine(load_rules(rules_file or (Path(config).parent / cfg["rules_file"])))
        self.risk = RiskScorer(**cfg["risk"])
        self.sag = SeatAttentionGraph(seat_positions, cfg.get("sag")) if seat_positions else None
        self.pending: dict[str, list[Event]] = defaultdict(list)  # SAG events waiting for that seat's row
        self.despike_s = cfg.get("despike_s", 1.0)
        self.recent_angles: dict[str, deque] = defaultdict(deque)

    def step(self, t: float, rows: list[dict]) -> StepResult:
        """Process all seat rows observed at time t (seconds since session start).
        A row: {'seat', 'facing', 'yaw', 'pitch', ...}; smoothed yaw_s/pitch_s are preferred if present."""
        out = StepResult()
        ready = []
        for row in rows:
            row = self._despike(t, _prefer_smoothed(row))
            b = self.baseline.update(t, row)
            if not b.frozen:
                continue
            ctx = self.baseline.context(b, row)
            self.pending[row["seat"]] += self.rules.update(t, row["seat"], ctx)
            if self.sag:
                for seat, e in self.sag.observe(t, row["seat"], ctx, observability(row)):
                    self.pending[seat].append(e)
            ready.append(row)
        for row in ready:  # second pass: relational events may concern any seat in this frame
            events = self.pending.pop(row["seat"], [])
            if any(e.rule_id in RELATIONAL_GLANCE for e in events):
                # the SAG event explains the same glances, with the neighbour named: keep only it
                events = [e for e in events if e.rule_id != "SIDE_GLANCE_REPEAT"]
            alert = self.risk.update(t, row["seat"], row, events)
            out.events += events
            if alert:
                out.alerts.append(alert)
        return out

    def _despike(self, t: float, row: dict) -> dict:
        """Causal median of yaw/pitch over the last `despike_s` seconds. Real keypoint glitches are
        0.1–0.3 s spikes of 17–50°; a median removes them, while real glances (≥ 0.6 s) survive with
        ~despike_s/2 delay."""
        if self.despike_s <= 0:
            return row
        buf = self.recent_angles[row["seat"]]
        buf.append((t, row.get("yaw", math.nan), row.get("pitch", math.nan)))
        while buf and buf[0][0] < t - self.despike_s:
            buf.popleft()
        if len(buf) < 3:
            return row
        out = dict(row)
        for i, k in ((1, "yaw"), (2, "pitch")):
            vals = sorted(v[i] for v in buf if v[i] == v[i])
            if len(vals) * 2 > len(buf):  # mostly valid: median of valid values
                out[k] = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
        return out

    def edges(self, t: float, window: float = 120.0) -> dict[tuple[str, str], tuple[int, float]]:
        """Seat Attention Graph edges in the window: (src, dst) -> (glances, dwell s)."""
        return self.sag.counts(t, window) if self.sag else {}

    def seat_view(self, seat: str) -> SeatView:
        b = self.baseline.seats.get(seat)
        r = self.risk.seats.get(seat)
        return SeatView(calibrating=not (b and b.frozen), level=r.level if r else "green",
                        risk=r.risk if r else 0.0, obs=r.obs if r else 0.0)

    @property
    def calibrated(self) -> bool:
        return bool(self.baseline.seats) and all(b.frozen for b in self.baseline.seats.values())


def _prefer_smoothed(row: dict) -> dict:
    out = dict(row)
    for raw, sm in (("yaw", "yaw_s"), ("pitch", "pitch_s")):
        v = row.get(sm, math.nan)
        if v == v:
            out[raw] = v
    return out


def read_features_csv(path: str | Path) -> list[tuple[float, list[dict]]]:
    """Features CSV (from run_seats.py) -> [(t, [row per seat]), ...] in time order."""
    frames: dict[float, list[dict]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {"seat": r["seat"], "facing": r.get("facing", "unknown")}
            for k in NUMERIC:
                v = r.get(k, "")
                row[k] = float(v) if v not in ("", None) else math.nan
            frames.setdefault(float(r["t"]), []).append(row)
    return sorted(frames.items())
