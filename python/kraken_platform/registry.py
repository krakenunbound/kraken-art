"""
Capability / Modality Registry (PR 1 skeleton).

This is the single source of truth that both the React GUI and future MCP
server will use to discover available capabilities, their schemas, health,
and supported operations.

Current state (PR 1): Pure skeleton. In-memory only. No real capabilities
registered yet. No behavior change to any existing endpoints.

See the approved design document for the full target shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CapabilityInfo:
    """Minimal capability descriptor (will grow in later PRs)."""
    id: str
    name: str
    kind: str  # "image" | "music" | "video" | ...
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=dict)


class CapabilityRegistry:
    """
    Central registry.

    PR 1: Skeleton implementation only.
    - Stores registered capabilities in memory.
    - list() / get() work.
    - health() is a stub (will add caching + background refresh in PR 1+).
    """

    _instance: Optional["CapabilityRegistry"] = None

    def __init__(self) -> None:
        self._capabilities: Dict[str, CapabilityInfo] = {}

    @classmethod
    def instance(cls) -> "CapabilityRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, cap: CapabilityInfo) -> None:
        """Register a capability (additive — called at bootstrap)."""
        self._capabilities[cap.id] = cap

    def list(self, *, include_health: bool = True) -> List[Dict[str, Any]]:
        """
        Return all registered capabilities.

        PR 1: Health is stubbed. Real health probes + caching come later.
        """
        result = []
        for cap in self._capabilities.values():
            entry: Dict[str, Any] = {
                "id": cap.id,
                "name": cap.name,
                "kind": cap.kind,
                "description": cap.description,
                "input_schema": cap.input_schema,
                "output_schema": cap.output_schema,
            }
            if include_health:
                entry["health"] = cap.health  # will be real in future
            result.append(entry)
        return result

    def get(self, cap_id: str) -> Optional[CapabilityInfo]:
        return self._capabilities.get(cap_id)


# Convenience singleton for early bootstrap code
registry = CapabilityRegistry.instance()