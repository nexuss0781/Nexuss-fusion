"""Sequence-length unification via learned query resamplers (torch nn.Module)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Q-Former / Perceiver-style resampler: (b_m x d_q) queries attend to states.

    Maps variable-length modality states (T x d_in) onto a bounded number of
    soft tokens (b_m x d_out) via scaled dot-product cross-attention.
    """

    def __init__(self, d_in: int, d_q: int, d_out: int, budget: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_q = d_q
        self.d_out = d_out
        self.budget = budget
        self.query = nn.Parameter(torch.randn(budget, d_q) * 0.02)
        self.wk = nn.Linear(d_in, d_q, bias=False)
        self.wv = nn.Linear(d_in, d_out, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batches = states.shape[:-2]
        if not batches:
            raise ValueError("states must have a batch dimension")
        flat_batch = int(math.prod(batches))
        x = states.reshape(flat_batch, states.shape[-2], self.d_in)
        k = self.wk(x).transpose(1, 2)  # (B, d_q, T)
        v = self.wv(x)  # (B, T, d_out)
        scores = torch.matmul(self.query, k) / math.sqrt(self.d_q)  # (B, budget, T)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, budget, d_out)
        return out.reshape(*batches, self.budget, self.d_out)


def budget_for_image(patches: int, budget: int | None = None) -> int:
    return min(budget or 64, patches)


def budget_for_audio(duration_s: float, rate: float = 12.5, ceiling: int = 256) -> int:
    return max(1, min(ceiling, int(math.ceil(duration_s * rate))))