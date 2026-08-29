"""Run: end-to-end experiment entrypoints."""

from __future__ import annotations


def phase2_main() -> None:
    """Lazily import and run the phase 2 experiment (avoids eager HF imports)."""
    from .phase2 import main

    raise SystemExit(main())


__all__ = ["phase2_main"]