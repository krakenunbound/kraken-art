"""
Platform bootstrap (PR 1 skeleton).

Called early in sidecar startup to register built-in capabilities
and wire platform services.

PR 1: Does nothing visible yet. Will grow as we add real adapters.
"""

from __future__ import annotations

import logging

log = logging.getLogger("kraken.platform.bootstrap")


def bootstrap_platform() -> None:
    """
    Idempotent bootstrap of the Kraken platform layer.

    Currently (PR 1): No-op. Exists so we can incrementally add
    registry population, VRAM wiring, etc. without touching main.py logic yet.
    """
    log.info("platform bootstrap (PR 1 skeleton) — nothing to do yet")
    # Future:
    # from .registry import registry
    # from .capabilities.image import ImageCapability
    # registry.register(ImageCapability().info)
    # etc.