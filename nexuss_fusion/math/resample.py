"""Sequence-length unification via learned query resamplers.

Compress arbitrary-length modality states to a bounded soft-token budget so the
fused sequence stays text-like. Perceiver/Q-Former style cross-attention.
"""
from __future__ import annotations

import numpy as np


class AttentionPooling:
    """Q-Former / Perceiver-style resampler: (b_m x d_src) queries attend to source states.

    state: (T x d_src) source features, already projected into d_in
    queries: (b_m x d_q) learned query vectors
    """

    def __init__(self, d_in: int, d_q: int, budget: int, scale: float | None = None) -> None:
        self.d_in = d_in
        self.d_q = d_q
        self.budget = budget
        self.scale = 1.0 / np.sqrt(d_q) if scale is None else scale

    def forward(self, keys_queries: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
        """queries: (b_m x d_q), K (b_m x T) via scaled dot product, softmax over T, mix values."""
        T = keys_queries.shape[0]
        attn = (queries @ keys_queries.T) * self.scale
        attn = np.exp(attn - attn.max(axis=1, keepdims=True))
        attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-9)
        return attn @ values

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        return self.forward(*args)


def budget_for_image(patches: int, budget: int | None = None) -> int:
    return min(budget or 64, patches)


def budget_for_audio(duration_s: float, rate: float = 12.5, ceiling: int = 256) -> int:
    return max(1, min(ceiling, int(np.ceil(duration_s * rate))))