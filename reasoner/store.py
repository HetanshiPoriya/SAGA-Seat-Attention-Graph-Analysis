"""Event/alert/feedback storage (spec §11.2). SQLite for now — the edge-side local buffer the spec
allows; tables mirror the Postgres schema so a server backend can replace it without changing callers.
Only seat labels, rule IDs, numbers and text reasons are stored: no images, no identities.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS exam_session (
    id TEXT PRIMARY KEY, name TEXT, source TEXT, started_at REAL, config TEXT);
CREATE TABLE IF NOT EXISTS behaviour_event (
    id TEXT PRIMARY KEY, session_id TEXT REFERENCES exam_session(id), seat TEXT, rule_id TEXT,
    confidence REAL, weight REAL, start_t REAL, end_t REAL, fired_t REAL, count INTEGER, message TEXT,
    related TEXT);
CREATE TABLE IF NOT EXISTS alert (
    id TEXT PRIMARY KEY, session_id TEXT REFERENCES exam_session(id), seat TEXT, level TEXT,
    risk REAL, observability REAL, t REAL, reasons TEXT, related TEXT);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY, alert_id TEXT REFERENCES alert(id), verdict TEXT, note TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS ix_event_session ON behaviour_event(session_id, seat);
CREATE INDEX IF NOT EXISTS ix_alert_session ON alert(session_id, seat);
CREATE INDEX IF NOT EXISTS ix_feedback_alert ON feedback(alert_id);
"""
VERDICTS = {"confirm", "dismiss", "unsure"}


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)  # shared with the dashboard thread
        self.lock = threading.Lock()
        self.db.executescript(SCHEMA)
        for table in ("behaviour_event", "alert"):  # migrate databases created before these columns
            cols = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if "related" not in cols:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN related TEXT")

    def new_session(self, name: str, source: str, config: dict | None = None) -> str:
        sid = uuid.uuid4().hex
        with self.lock:
            self.db.execute("INSERT INTO exam_session VALUES (?,?,?,?,?)",
                            (sid, name, source, time.time(), json.dumps(config or {})))
            self.db.commit()
        return sid

    def add_event(self, sid: str, e):
        with self.lock:
            self.db.execute("INSERT INTO behaviour_event (id, session_id, seat, rule_id, confidence, weight, "
                            "start_t, end_t, fired_t, count, message, related) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (uuid.uuid4().hex, sid, e.seat, e.rule_id, e.confidence, e.weight, e.start_t,
                             e.end_t, e.t, e.count, e.message, json.dumps(getattr(e, "related", []))))

    def add_alert(self, sid: str, a):
        with self.lock:
            self.db.execute("INSERT INTO alert (id, session_id, seat, level, risk, observability, t, reasons, "
                            "related) VALUES (?,?,?,?,?,?,?,?,?)",
                            (a.id, sid, a.seat, a.level, a.risk, a.observability, a.t, json.dumps(a.reasons),
                             json.dumps(getattr(a, "related", []))))

    def add_feedback(self, alert_id: str, verdict: str, note: str = "") -> str:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VERDICTS)}")
        fid = uuid.uuid4().hex
        with self.lock:
            self.db.execute("INSERT INTO feedback VALUES (?,?,?,?,?)", (fid, alert_id, verdict, note[:500], time.time()))
            self.db.commit()
        return fid

    def feedback_for(self, alert_ids: list[str]) -> dict[str, str]:
        """Latest verdict per alert."""
        if not alert_ids:
            return {}
        q = ",".join("?" * len(alert_ids))
        with self.lock:
            rows = self.db.execute(f"SELECT alert_id, verdict FROM feedback WHERE alert_id IN ({q}) "
                                   "ORDER BY created_at", alert_ids).fetchall()
        return dict(rows)

    def commit(self):
        with self.lock:
            self.db.commit()

    def close(self):
        with self.lock:
            self.db.commit()
            self.db.close()
