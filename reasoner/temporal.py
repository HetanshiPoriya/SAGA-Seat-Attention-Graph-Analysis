"""Fusing the temporal model into the rule layer (spec §9.3, last bullet).

The rule is explicit in the spec and enforced here, not left to configuration:

    The temporal model **never alerts alone**; it contributes to the risk score and can only
    escalate an alert that has at least one rule-level explanation, or produce a low-priority
    "unusual pattern, review" note.

So this class never creates risk by itself. It adjusts the weight of events the rules already
produced, in both directions:

* a glance burst that *also* looks unlike anything this student has done keeps its full weight;
* a glance burst that looks like their ordinary fidgeting is damped — which is where the
  false-alarm reduction comes from.

A seat with no model score (too few frames, or the model is off) is left exactly as the rules had
it: `factor()` returns 1.0, so switching the model off can never change a rule decision.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from ml.temporal.features import WINDOW, vector
from ml.temporal.model import TemporalScorer
from reasoner.rules.engine import Event

DEFAULTS = {
    "enabled": False,         # opt-in until it is proven on real recordings
    "damp": 0.55,             # weight multiplier when the window looks completely ordinary
    "boost": 1.25,            # ... and when it is more surprising than 99% of normal behaviour
    "low": 0.35,              # surprise at or below this = "ordinary"
    "high": 1.0,              # surprise at or above this = "unusual" (1.0 = normal p99)
    "note_surprise": 1.6,     # emit the informational "unusual pattern" note above this
    "note_cooldown_s": 300.0,
    "min_frames": WINDOW + 1,
}


class TemporalLayer:
    def __init__(self, model_path: str | None, cfg: dict | None = None):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.scorer = TemporalScorer(model_path) if (model_path and self.c["enabled"]) else None
        self.buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.surprise: dict[str, float] = {}
        self.history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.last_note: dict[str, float] = {}
        self.counts = {"damped": 0, "boosted": 0, "unchanged": 0}

    @property
    def active(self) -> bool:
        return self.scorer is not None

    def observe(self, t: float, seat: str, ctx: dict) -> None:
        """Feed one seat-frame; keeps a rolling 2 s window and scores it when full."""
        if not self.active:
            return
        self.buf[seat].append(vector(ctx))
        if len(self.buf[seat]) == WINDOW:
            s = float(self.scorer.score(np.stack(self.buf[seat])[None])[0])
            self.surprise[seat] = s
            self.history[seat].append((t, s))

    def factor(self, seat: str) -> float:
        """Weight multiplier for this seat's rule events right now (1.0 = leave the rules alone)."""
        s = self.surprise.get(seat)
        if not self.active or s is None:
            return 1.0
        lo, hi = self.c["low"], self.c["high"]
        if s <= lo:
            return self.c["damp"]
        if s >= hi:
            return self.c["boost"]
        # between the two: interpolate, so a small change in surprise cannot flip the verdict
        f = (s - lo) / max(hi - lo, 1e-6)
        return self.c["damp"] + f * (self.c["boost"] - self.c["damp"])

    def apply(self, seat: str, events: list[Event]) -> list[Event]:
        """Scale the weights of events the RULES produced. Never adds a risk-bearing event."""
        if not self.active or not events:
            return events
        f = self.factor(seat)
        if f == 1.0:
            return events
        for e in events:
            if e.weight > 0:
                e.weight = round(e.weight * f, 4)
                e.message += f" [pattern {'unusual' if f > 1 else 'typical'} for this seat]"
        self.counts["boosted" if f > 1 else "damped"] += len(events)
        return events

    def note(self, t: float, seat: str) -> Event | None:
        """The low-priority "unusual pattern, review" note: weight 0, so it cannot raise risk."""
        s = self.surprise.get(seat)
        if not self.active or s is None or s < self.c["note_surprise"]:
            return None
        if t - self.last_note.get(seat, -1e9) < self.c["note_cooldown_s"]:
            return None
        self.last_note[seat] = t
        return Event("UNUSUAL_PATTERN", seat, t, t, t, 0.0, 1.0,
                     f"Unusual movement pattern for this seat (surprise {s:.1f}x) — review, no rule matched", 1)
