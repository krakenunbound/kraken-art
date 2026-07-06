"""
Standardized Job Events + Legacy Compatibility Layer (PR 1 skeleton).

Goal: Move toward consistent event shapes across all modalities while
preserving 100% compatibility with existing consumers (src/Music.tsx,
existing WS subscribers, etc.).

PR 1: Base helpers + explicit compat contract. No behavior change yet.
"""

from __future__ import annotations

from typing import Any, Dict


def canonical_event(event_type: str, **payload: Any) -> Dict[str, Any]:
    """Create a future canonical event shape."""
    return {
        "type": event_type,
        "version": 2,  # v1 = legacy ad-hoc events
        **payload,
    }


def legacy_event(event_type: str, **payload: Any) -> Dict[str, Any]:
    """Emit an event using the exact legacy shape for current consumers."""
    # Today many places emit raw dicts like {"type": "audio_progress", ...}
    # We will keep emitting those shapes from adapters for now.
    return {"type": event_type, **payload}


# Known legacy event names (documented here for the compat contract)
LEGACY_EVENTS = {
    "audio_progress",
    "audio_cover_progress",
    "ace_submitted",
    "bulk_song_done",
    "bulk_song_failed",
    # ... add others as discovered during PR 3
}