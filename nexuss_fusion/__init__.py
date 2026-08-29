"""Nexuss-Fusion: fuse models of different modalities/architectures into one native model."""
from __future__ import annotations

__version__ = "0.2.0"

from .backend import Backend, eigen_available, eigen_parity_ok, get_backend
from .data import FeaturesCache
from .math import (
    AttentionPooling,
    Normalizer,
    affine_map_lsq,
    budget_for_audio,
    budget_for_image,
    canonicalize_whiten,
    procrustes_map,
    ridge_least_squares,
)
from .sequence import Interleaver, TypedBlock, build_sequence

__all__ = [
    "Backend",
    "eigen_available",
    "eigen_parity_ok",
    "get_backend",
    "FeaturesCache",
    "Interleaver",
    "TypedBlock",
    "AttentionPooling",
    "Normalizer",
    "budget_for_audio",
    "budget_for_image",
    "canonicalize_whiten",
    "procrustes_map",
    "ridge_least_squares",
    "affine_map_lsq",
    "build_sequence",
]