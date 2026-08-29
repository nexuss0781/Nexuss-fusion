"""Held-out alignment metrics vs zero/random/whitening baselines."""

from __future__ import annotations

import torch

from ..calibration.bridge import CalibrationBridge


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    num = (pred * target).sum(dim=dim)
    den = pred.norm(dim=dim) * target.norm(dim=dim) + 1e-12
    return (num / den).mean()


def rel_fro_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).norm() / (target.norm() + 1e-12)


def alignment_report(source: torch.Tensor, target: torch.Tensor, bridge: dict, seed: int = 42) -> dict:
    rng = torch.Generator().manual_seed(seed)
    ridge_pred = CalibrationBridge.apply(source, bridge)
    ortho_pred = CalibrationBridge.apply_orthogonal(source, bridge)
    random_a = torch.randn_like(bridge["A"], generator=rng)
    norm_src = bridge["normalizer"].transform(source.double())
    zero_pred = norm_src.new_zeros_like(target)

    return {
        "cosine_ridge": float(cosine_similarity(ridge_pred, target).item()),
        "cosine_orthogonal": float(cosine_similarity(ortho_pred, target).item()),
        "cosine_random": float(cosine_similarity(norm_src @ random_a, target).item()),
        "cosine_zero": float(cosine_similarity(zero_pred, target).item()),
        "rel_fro_ridge": float(rel_fro_error(ridge_pred, target).item()),
        "rel_fro_orthogonal": float(rel_fro_error(ortho_pred, target).item()),
        "rel_fro_random": float(rel_fro_error(norm_src @ random_a, target).item()),
    }
