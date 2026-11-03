"""Invigilator dashboard server (spec §11.3, §12): REST + WebSocket over the live hall state.

Binds to 127.0.0.1 by default — the dashboard is for the invigilator's own machine. Expose it on a
network only behind proper authentication (planned with the full backend).
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.live import LiveState

STATIC = Path(__file__).resolve().parent / "static"


class FeedbackIn(BaseModel):
    verdict: str = Field(pattern="^(confirm|dismiss|unsure)$")
    note: str = Field(default="", max_length=500)


def create_app(state: LiveState) -> FastAPI:
    app = FastAPI(title="SAGA dashboard", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def get_state():
        return state.snapshot()

    @app.get("/api/alerts")
    def get_alerts(level: str | None = None):
        alerts = state.snapshot()["alerts"]
        return [a for a in alerts if level is None or a["level"] == level]

    @app.post("/api/alerts/{alert_id}/feedback")
    def post_feedback(alert_id: str, body: FeedbackIn):
        if not state.feedback(alert_id, body.verdict, body.note):
            raise HTTPException(404, "unknown alert")
        return {"ok": True}

    @app.get("/api/clips/{clip_id}")
    def get_clip(clip_id: str, request: Request):
        """Serve an evidence clip. Every view is written to the audit log (spec §13)."""
        if state.store is None:
            raise HTTPException(404, "no clip store")
        row = state.store.clip_by_id(clip_id)
        if row is None or not Path(row["path"]).exists():
            raise HTTPException(404, "unknown clip")
        state.store.log_clip_view(clip_id, viewer=request.client.host if request.client else "unknown")
        return FileResponse(row["path"], media_type="video/mp4", filename=Path(row["path"]).name)

    @app.get("/api/clips/{clip_id}/views")
    def get_clip_views(clip_id: str):
        if state.store is None:
            raise HTTPException(404, "no clip store")
        return {"clip": clip_id, "views": state.store.clip_views(clip_id)}

    @app.get("/api/seats/{seat}/timeline")
    def get_timeline(seat: str):
        if seat not in state.seats:
            raise HTTPException(404, "unknown seat")
        return state.seat_timeline(seat)

    @app.get("/api/health")
    def get_health():
        snap = state.snapshot()
        return {"health": snap["health"], "status": snap["status"]}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        last = -1
        try:
            while True:
                if state.version != last:  # push on change, at most 10x per second
                    snap = state.snapshot()
                    last = snap["version"]
                    await socket.send_json(snap)
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def serve_in_background(state: LiveState, host: str = "127.0.0.1", port: int = 8000) -> threading.Thread:
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(create_app(state), host=host, port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True, name="dashboard")
    th.start()
    return th
