"""Per-seat risk score with decay, observability weighting and alert levels (spec §10).

    R(t) = R(t−Δt)·e^(−λΔt) + Σ_k w_k · c_k · obs
Levels: green < amber (0.4) < red (0.75); grey = coverage gap (observability low for a while).
Language stays factual — "observed behaviour", never "cheating".
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

from reasoner.rules.engine import Event

LEVELS = ["green", "amber", "red"]


def observability(row: dict) -> float:
    """Spec §6.7, the parts available per seat now: keypoint confidence × head-size factor."""
    kp = row.get("kp_conf", math.nan)
    head = row.get("head_px", math.nan)
    if kp != kp:
        return 0.0
    size = min(max(head / 40.0, 0.0), 1.0) if head == head and head > 0 else 0.5
    return float(kp * size)


@dataclass
class Alert:
    seat: str
    level: str
    risk: float
    observability: float
    t: float
    reasons: list[str]
    related: list[str] = field(default_factory=list)  # other seats named in the reasons (SAG)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class SeatRisk:
    risk: float = 0.0
    t: float = 0.0
    obs: float = 1.0
    low_obs_since: float | None = None
    level: str = "green"
    events: list[Event] = field(default_factory=list)


class RiskScorer:
    def __init__(self, half_life_s=180, amber=0.4, red=0.75, red_on_weight=0.6, grey_obs=0.3,
                 grey_after_s=60, reasons_window_s=180):
        self.lam = math.log(2) / half_life_s
        self.amber, self.red, self.red_on_weight = amber, red, red_on_weight
        self.grey_obs, self.grey_after_s, self.reasons_window_s = grey_obs, grey_after_s, reasons_window_s
        self.seats: dict[str, SeatRisk] = {}

    def update(self, t: float, seat: str, row: dict, events: list[Event]) -> Alert | None:
        s = self.seats.setdefault(seat, SeatRisk(t=t))
        s.risk *= math.exp(-self.lam * max(t - s.t, 0.0))
        s.t = t
        s.obs = 0.8 * s.obs + 0.2 * observability(row)
        s.low_obs_since = (s.low_obs_since or t) if s.obs < self.grey_obs else None

        forced_red = False
        for e in events:
            s.risk = min(1.0, s.risk + e.weight * e.confidence * s.obs)  # 0 = nothing observed, 1 = maximum
            s.events.append(e)
            forced_red |= e.weight >= self.red_on_weight and e.confidence > 0.8
        s.events = [e for e in s.events if t - e.t <= self.reasons_window_s]

        if s.low_obs_since is not None and t - s.low_obs_since >= self.grey_after_s:
            level = "grey"
        elif forced_red or s.risk >= self.red:
            level = "red"
        elif s.risk >= self.amber:
            level = "amber"
        else:
            level = "green"

        prev, s.level = s.level, level
        rising = level in ("amber", "red") and (prev not in LEVELS or LEVELS.index(level) > LEVELS.index(prev))
        if rising:
            related = sorted({r for e in s.events for r in e.related})
            return Alert(seat, level, round(s.risk, 3), round(s.obs, 3), t, [e.message for e in s.events], related)
        return None
