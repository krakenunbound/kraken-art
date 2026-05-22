"""WebSocket progress stream — subscribe to a job by id and receive real-time events."""
from __future__ import annotations
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from jobs import manager

router = APIRouter()


@router.websocket("/ws/jobs/{job_id}")
async def job_progress(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    sub = manager.subscribe(job_id)
    if sub is None:
        await ws.send_json({"type": "error", "error": "job_not_found"})
        await ws.close()
        return
    queue, unsubscribe = sub
    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
            if event.get("type") == "status" and event.get("status") in ("succeeded", "failed", "cancelled"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
        try:
            await ws.close()
        except Exception:
            pass
