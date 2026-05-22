"""Log read/clear endpoints. Plain GET — frontend polls or fetches on demand."""
from __future__ import annotations
from fastapi import APIRouter

from log_buffer import ring

router = APIRouter()


@router.get("/logs")
def get_logs(limit: int = 500, since_id: int | None = None) -> dict:
    items = ring.snapshot(limit=limit, since_id=since_id)
    last_id = items[-1]["id"] if items else (since_id if since_id is not None else -1)
    return {"items": items, "last_id": last_id}


@router.post("/logs/clear")
def clear_logs() -> dict:
    ring.clear()
    return {"cleared": True}
