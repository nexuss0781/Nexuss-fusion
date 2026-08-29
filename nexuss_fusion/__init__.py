"""Nexuss-Fusion: fuse models of different modalities/architectures into one native model."""

__version__ = "0.1.0"

from .math.normalize import canonicalize_whiten, Normalizer
from .math.procrustes import procrustes_map, ridge_least_squares, affine_map_lsq
from .math.resample import AttentionPooling, budget_for_audio, budget_for_image
from .sequence.interleave import Interleaver, TypedBlock

__all__ = [
    "canonicalize_whiten",
    "Normalizer",
    "procrustes_map",
    "ridge_least_squares",
    "affine_map_lsq",
    "AttentionPooling",
    "budget_for_audio",
    "budget_for_image",
    "Interleaver",
    "TypedBlock",
]