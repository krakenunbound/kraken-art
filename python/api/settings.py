"""Settings GET/PATCH endpoints."""
from __future__ import annotations
from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

import config_store

router = APIRouter()


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return config_store.public_view()


class SettingsPatch(BaseModel):
    civitai: dict[str, Any] | None = None
    downloads: dict[str, Any] | None = None
    performance: dict[str, Any] | None = None
    lastGenerate: dict[str, Any] | None = None


@router.patch("/settings")
def patch_settings(patch: SettingsPatch) -> dict[str, Any]:
    payload = patch.model_dump(exclude_none=True)
    config_store.save(payload)
    return config_store.public_view()
