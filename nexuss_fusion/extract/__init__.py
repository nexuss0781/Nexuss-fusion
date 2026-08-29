"""Extract: frozen modality encoders producing cached features."""

from __future__ import annotations

from .text import TextEmbedder
from .vision import VisionExtractor, validate_patch_states

__all__ = ["TextEmbedder", "VisionExtractor", "validate_patch_states"]
