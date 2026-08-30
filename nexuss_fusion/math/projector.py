"""Vision projector: resampler + MLP that maps patch states to decoder hidden dim."""

from __future__ import annotations

import torch
import torch.nn as nn

from .resample import AttentionPooling


class VisionProjector(nn.Module):
    """Resample + project vision patch states into the decoder's hidden space.

    Architecture: AttentionPooling (cross-attention resampler) → 2-layer MLP.
    Input:  (B, T, d_in)  — variable-length patch states (e.g. 1024 × 768)
    Output: (B, budget, d_out) — bounded soft tokens (e.g. 64 × 960)
    """

    def __init__(
        self,
        d_in: int = 768,
        d_hidden: int = 1024,
        d_out: int = 960,
        budget: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.resample = AttentionPooling(d_in=d_in, d_q=d_in, d_out=d_hidden, budget=budget)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )

    def forward(self, patch_states: torch.Tensor) -> torch.Tensor:
        """Resample patch states into bounded soft tokens.

        Args:
            patch_states: (B, T, d_in) or (T, d_in) — patch-level hidden states.
        Returns:
            (B, budget, d_out) — projected soft tokens.
        """
        if patch_states.dim() == 2:
            patch_states = patch_states.unsqueeze(0)
        resampled = self.resample(patch_states)  # (B, budget, d_hidden)
        return self.proj(resampled)  # (B, budget, d_out)
