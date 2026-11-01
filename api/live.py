"""Live hall state shared between the pipeline (writer) and the dashboard server (readers).

Holds only what the invigilator UI needs: seat labels and layout, risk levels, Seat Attention Graph
edges, alerts with reasons, and camera health. No frames, no faces, no identities.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque


def _num(v, nd=2):
    return None if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else round(float(v), nd)


class LiveState:
    def __init__(self, name: str, source: str, mode: str, store=None, session_id: str | None = None):
        self.lock = threading.Lock()
        self.meta = {"name": name, "source": source, "mode": mode, "started": time.time()}
        self.store, self.session_id = store, session_id
        self.layout: list[dict] = []
        self.seats: dict[str, dict] = {}
        self.timeline: dict[str, deque] = defaultdict(lambda: deque(maxlen=900))
        self.alerts: deque[dict] = deque(maxlen=300)
        self.events: deque[dict] = deque(maxlen=100)
        self.edges: list[dict] = []
        self.health = {"fps": None, "frames": 0, "people": 0, "t": 0.0, "updated": None}
        self.status = {"phase": "starting", "calib_left_s": None}
        self.version = 0
        self.reasoner = None

    # ------------------------------------------------------------- writer side
    def set_layout(self, seats: list[dict]):
        """seats: [{label, row, col, x, y}] with x, y in 0..1 (front row at the top)."""
        with self.lock:
            self.layout = seats
            self.seats = {s["label"]: {"level": "calibrating", "risk": 0.0, "obs": None, "present": False,
                                       "yaw": None, "pitch": None, "facing": None} for s in seats}
            self.version += 1

    def update(self, t: float, reasoner, rows: list[dict], out=None, fps: float | None = None, people: int = 0):
        with self.lock:
            self.reasoner = reasoner
            seen = {r["seat"]: r for r in rows}
            for label, s in self.seats.items():
                v = reasoner.seat_view(label)
                row = seen.get(label)
                s.update(level="calibrating" if v.calibrating else v.level, risk=round(v.risk, 3),
                         obs=_num(v.obs), present=row is not None)
                if row is not None:
                    yaw = row.get("yaw_s", row.get("yaw"))
                    yaw = row.get("yaw") if yaw is None or yaw != yaw else yaw
                    s.update(yaw=_num(yaw, 1), pitch=_num(row.get("pitch_s", row.get("pitch")), 1),
                             facing=row.get("facing"))
                    tl = self.timeline[label]
                    if not tl or t - tl[-1][0] >= 0.2:
                        tl.append((round(t, 2), s["yaw"], s["pitch"], s["risk"]))
            self.edges = [{"src": a, "dst": b, "n": n, "dwell": round(d, 1)}
                          for (a, b), (n, d) in reasoner.edges(t, 120).items()]
            bl = reasoner.baseline
            left = None if bl.start_t is None else max(0.0, bl.calib_s - (t - bl.start_t))
            self.status = {"phase": "calibrating" if left is None or left > 0 else "monitoring",
                           "calib_left_s": None if left is None else round(left, 1)}
            if out is not None:
                for e in out.events:
                    self.events.appendleft({"t": round(e.t, 1), "seat": e.seat, "rule": e.rule_id,
                                            "message": e.message, "related": list(e.related)})
                for a in out.alerts:
                    self.alerts.appendleft({"id": a.id, "seat": a.seat, "level": a.level, "risk": a.risk,
                                            "obs": a.observability, "t": round(a.t, 1), "reasons": a.reasons,
                                            "related": a.related, "verdict": None, "wall": time.time()})
            self.health = {"fps": _num(fps, 1), "frames": self.health["frames"] + 1, "people": people,
                           "t": round(t, 1), "updated": time.time()}
            self.version += 1

    # ------------------------------------------------------------- reader side
    def snapshot(self) -> dict:
        with self.lock:
            return {"meta": self.meta, "status": self.status, "health": self.health, "version": self.version,
                    "layout": self.layout, "seats": self.seats, "edges": self.edges,
                    "alerts": list(self.alerts)[:50], "events": list(self.events)[:20]}

    def seat_timeline(self, seat: str) -> dict:
        with self.lock:
            tl = list(self.timeline.get(seat, []))
            band = None
            if self.reasoner is not None:
                b = self.reasoner.baseline.seats.get(seat)
                if b is not None and b.frozen and "yaw" in b.med:
                    band = {"yaw_med": _num(b.med["yaw"], 1), "yaw_sigma": _num(b.sigma["yaw"], 1),
                            "pitch_med": _num(b.med.get("pitch"), 1), "pitch_sigma": _num(b.sigma.get("pitch"), 1)}
            events = [e for e in self.events if e["seat"] == seat or seat in e["related"]]
        return {"seat": seat, "t": [x[0] for x in tl], "yaw": [x[1] for x in tl], "pitch": [x[2] for x in tl],
                "risk": [x[3] for x in tl], "baseline": band, "events": events}

    def feedback(self, alert_id: str, verdict: str, note: str = "") -> bool:
        with self.lock:
            hit = next((a for a in self.alerts if a["id"] == alert_id), None)
            if hit is None:
                return False
            hit["verdict"] = verdict
            self.version += 1
        if self.store is not None:
            self.store.add_feedback(alert_id, verdict, note)
        return True


def layout_from_seatmap(seatmap) -> list[dict]:
    """Hall map as seen from the camera, front row (A) at the top."""
    if seatmap.mode == "grid":
        return [{"label": s.label, "row": s.row, "col": s.col, "x": (s.col + 0.5) / seatmap.cols,
                 "y": (s.row + 0.5) / seatmap.rows} for s in seatmap.seats]
    w, h = seatmap.image_size
    return [{"label": s.label, "row": s.row, "col": s.col, "x": s.center[0] / w, "y": 1 - s.center[1] / h}
            for s in seatmap.seats]


def layout_from_positions(pos: dict[str, tuple[float, float]]) -> list[dict]:
    """Grid positions (col 1.., row 0..) as used by synthetic sessions."""
    cols = max(p[0] for p in pos.values())
    rows = max(p[1] for p in pos.values()) + 1
    return [{"label": k, "row": int(p[1]), "col": int(p[0]) - 1, "x": (p[0] - 0.5) / cols, "y": (p[1] + 0.5) / rows}
            for k, p in pos.items()]
