"""Math: representation alignment, whitening, and sequence-length unification."""
from __future__ import annotations

from .normalize import Normalizer, canonicalize_whiten
from .procrustes import affine_map_lsq, procrustes_map, ridge_least_squares
from .resample import AttentionPooling, budget_for_audio, budget_for_image

__all__ = [
    "Normalizer",
    "canonicalize_whiten",
    "procrustes_map",
    "ridge_least_squares",
    "affine_map_lsq",
    "AttentionPooling",
    "budget_for_audio",
    "budget_for_image",
]