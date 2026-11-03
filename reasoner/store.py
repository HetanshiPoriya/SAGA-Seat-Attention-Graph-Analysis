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
CREATE TABLE IF NOT EXISTS evidence_clip (
    id TEXT PRIMARY KEY, session_id TEXT REFERENCES exam_session(id), alert_id TEXT REFERENCES alert(id),
    seat TEXT, path TEXT, start_t REAL, end_t REAL, frames INTEGER, faces_masked INTEGER,
    created_at REAL, deleted_at REAL);
CREATE TABLE IF NOT EXISTS clip_view (
    id TEXT PRIMARY KEY, clip_id TEXT REFERENCES evidence_clip(id), viewer TEXT, viewed_at REAL);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY, alert_id TEXT REFERENCES alert(id), verdict TEXT, note TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS ix_event_session ON behaviour_event(session_id, seat);
CREATE INDEX IF NOT EXISTS ix_alert_session ON alert(session_id, seat);
CREATE INDEX IF NOT EXISTS ix_feedback_alert ON feedback(alert_id);
CREATE INDEX IF NOT EXISTS ix_clip_alert ON evidence_clip(alert_id);
CREATE INDEX IF NOT EXISTS ix_clipview_clip ON clip_view(clip_id);
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

    def add_clip(self, sid: str, clip) -> str:
        """Record an evidence clip (spec §11.2 EVIDENCE_CLIP). Only the path is stored — the file
        itself lives under outputs/evidence/ and is deleted by the retention job."""
        with self.lock:
            self.db.execute("INSERT INTO evidence_clip (id, session_id, alert_id, seat, path, start_t, "
                            "end_t, frames, faces_masked, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (clip.id, sid, clip.alert_id, clip.seat, str(clip.path), clip.start_t,
                             clip.end_t, clip.frames, clip.faces_masked, clip.created_at))
            self.db.commit()
        return clip.id

    def clip_for_alert(self, alert_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT id, path, seat, start_t, end_t, frames FROM evidence_clip "
                                  "WHERE alert_id = ? AND deleted_at IS NULL ORDER BY created_at DESC "
                                  "LIMIT 1", (alert_id,)).fetchone()
        keys = ("id", "path", "seat", "start_t", "end_t", "frames")
        return dict(zip(keys, row)) if row else None

    def clip_by_id(self, clip_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT id, path, seat, alert_id, start_t, end_t FROM evidence_clip "
                                  "WHERE id = ? AND deleted_at IS NULL", (clip_id,)).fetchone()
        keys = ("id", "path", "seat", "alert_id", "start_t", "end_t")
        return dict(zip(keys, row)) if row else None

    def clip_views(self, clip_id: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT viewer, viewed_at FROM clip_view WHERE clip_id = ? "
                                   "ORDER BY viewed_at", (clip_id,)).fetchall()
        return [{"viewer": v, "at": t} for v, t in rows]

    def log_clip_view(self, clip_id: str, viewer: str = "invigilator") -> str:
        """Audit log on every clip view (spec §13 access control)."""
        vid = uuid.uuid4().hex
        with self.lock:
            self.db.execute("INSERT INTO clip_view VALUES (?,?,?,?)", (vid, clip_id, viewer[:100], time.time()))
            self.db.commit()
        return vid

    def mark_clips_deleted(self, paths: list[str]) -> int:
        if not paths:
            return 0
        q = ",".join("?" * len(paths))
        with self.lock:
            cur = self.db.execute(f"UPDATE evidence_clip SET deleted_at = ? WHERE path IN ({q})",
                                  [time.time(), *paths])
            self.db.commit()
        return cur.rowcount

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
