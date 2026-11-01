"""Explainable rule engine (spec §9.2): per-seat episode detection, counting and cooldowns."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from reasoner.baseline import CONTEXT_NAMES
from reasoner.rules.safe_expr import compile_condition

GAP_TOLERANCE_S = 0.4  # bridge short dropouts (missed detections, blinks of the condition)


@dataclass
class Rule:
    id: str
    when: str
    weight: float
    message: str
    min_duration_s: float = 0.0
    max_duration_s: float | None = None  # if set, episodes count when they END and only if this short
    count: int = 1
    window_s: float = 60.0
    cooldown_s: float | None = None
    end_when: str | None = None          # optional hysteresis: episode lasts until this is true
    same_direction: bool = False         # count only episodes toward the same side (image-left/right)
    cond: object = None
    end_cond: object = None

    def __post_init__(self):
        self.cond = compile_condition(self.when, CONTEXT_NAMES)
        self.end_cond = compile_condition(self.end_when, CONTEXT_NAMES) if self.end_when else None
        self.count = int(str(self.count).replace(">=", "").strip())
        if self.cooldown_s is None:
            self.cooldown_s = self.window_s


@dataclass
class Event:
    rule_id: str
    seat: str
    t: float              # when the rule fired
    start_t: float        # start of the first episode that contributed
    end_t: float
    weight: float
    confidence: float
    message: str
    count: int = 1
    related: list[str] = field(default_factory=list)  # other seats involved (SAG events)


@dataclass
class _State:
    active_since: float | None = None
    last_true: float = -math.inf
    counted: bool = False
    yaw_dev: list[float] = field(default_factory=list)
    peak: float = 0.0
    ref: float = math.nan   # head direction just before the episode (2-min median)
    episodes: deque = field(default_factory=deque)  # (start, end, mean yaw deviation)
    last_fire: float = -math.inf


def load_rules(path: str | Path) -> list[Rule]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rules, seen = [], set()
    for r in raw:
        if r["id"] in seen:
            raise ValueError(f"duplicate rule id {r['id']}")
        seen.add(r["id"])
        rules.append(Rule(**r))
    return rules


class RuleEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self.state: dict[tuple[str, str], _State] = {}

    def update(self, t: float, seat: str, ctx: dict) -> list[Event]:
        events = []
        for rule in self.rules:
            st = self.state.setdefault((seat, rule.id), _State())
            if st.active_since is not None and rule.end_cond is not None:
                ctx["yaw_dev_peak"] = st.peak  # largest |yaw_dev| so far in this episode
                ctx["yaw_dev_ref"] = ctx["yaw"] - st.ref  # vs where the head was before the episode
                on = not rule.end_cond(ctx)  # hysteresis: stay in the episode until clearly back
            else:
                on = rule.cond(ctx)
            if on:
                if st.active_since is not None and t - st.last_true > GAP_TOLERANCE_S:
                    self._close(rule, st)
                if st.active_since is None:
                    st.active_since, st.counted, st.yaw_dev, st.peak = t, False, [], 0.0
                    st.ref = ctx["yaw"] - ctx.get("yaw_dev_long", math.nan)
                st.last_true = t
                dev = ctx.get("yaw_dev", math.nan)
                if dev == dev:
                    st.yaw_dev.append(dev)
                    st.peak = max(st.peak, abs(dev))
                if (rule.max_duration_s is None and not st.counted
                        and t - st.active_since >= rule.min_duration_s):
                    st.counted = True  # counted as soon as long enough -> low latency
                    st.episodes.append((st.active_since, t, sum(st.yaw_dev) / max(len(st.yaw_dev), 1)))
            elif st.active_since is not None and t - st.last_true > GAP_TOLERANCE_S:
                self._close(rule, st)

            while st.episodes and st.episodes[0][0] < t - rule.window_s:
                st.episodes.popleft()
            eps = self._matching(rule, st)
            if len(eps) >= rule.count and t - st.last_fire >= rule.cooldown_s:
                events.append(self._fire(rule, st, eps, seat, t))
        return events

    @staticmethod
    def _matching(rule: Rule, st: _State) -> list:
        """Episodes that count together; with same_direction only those on the latest episode's side."""
        if not rule.same_direction or not st.episodes:
            return list(st.episodes)
        side = math.copysign(1, st.episodes[-1][2])
        return [e for e in st.episodes if math.copysign(1, e[2]) == side]

    @staticmethod
    def _close(rule: Rule, st: _State):
        """Episode ended. Bounded rules (glances) count it here, only if it was short enough."""
        dur = st.last_true - st.active_since
        if rule.max_duration_s is not None and rule.min_duration_s <= dur <= rule.max_duration_s:
            st.episodes.append((st.active_since, st.last_true, sum(st.yaw_dev) / max(len(st.yaw_dev), 1)))
        st.active_since = None

    def _fire(self, rule: Rule, st: _State, eps: list, seat: str, t: float) -> Event:
        start = eps[0][0]
        dev = sum(e[2] for e in eps) / len(eps)
        direction = "toward the right (camera view)" if dev > 0 else "toward the left (camera view)"
        duration = (st.last_true - st.active_since) if st.active_since is not None else (eps[-1][1] - eps[-1][0])
        msg = rule.message.format(count=len(eps), window_s=int(rule.window_s), duration=f"{duration:.0f}",
                                  direction=direction, seat=seat)
        st.last_fire = t
        st.episodes.clear()
        return Event(rule.id, seat, t, start, t, rule.weight, 1.0, msg, len(eps))
