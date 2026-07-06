"""
Kraken Platform — Core abstractions for the unified ecosystem.

This package is the foundation of the Ecosystem Unification effort (see
Documentation/design-runs/kraken-unified-ecosystem-architecture-6f9f838e.md).

Phase: PR 1 (Platform Core Abstractions)
Status: Skeleton only — no behavior change to existing image or audio flows.

All capabilities (Image, Music, future modalities) will register here.
Both the GUI and MCP agents will discover capabilities through this layer.
"""

from __future__ import annotations

__version__ = "0.1.0-pr1-skeleton"