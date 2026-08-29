"""Losses: fused spoken-language objective."""
from __future__ import annotations

from .fused import cross_entropy, fused_loss

__all__ = ["cross_entropy", "fused_loss"]