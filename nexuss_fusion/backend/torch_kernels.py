"""Procrustes / ridge alignment kernels over torch tensors."""

from __future__ import annotations

import torch


def procrustes(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Orthogonal Procrustes: R = argmin||target - scale * source @ R||_F.

    Let C = target^T @ source; SVD(C) = U S V^T; R = V U^T, scale = sum(S)/||source||_F^2.
    """
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError("procrustes expects 2-D tensors")
    if source.shape[0] != target.shape[0]:
        raise ValueError("procrustes expects paired rows (shape[0] mismatch)")
    corr = target.T @ source  # (d_tgt, d_src)
    U, _, Vt = torch.linalg.svd(corr)
    R = Vt.T @ U.T  # (d_src, d_tgt)
    scale = torch.sum(corr * R.mT) / torch.sum(source * source)
    return R, scale


def ridge_least_squares(source: torch.Tensor, target: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    r"""A = argmin||target - source A||_F^2 + lam||A||_F^2 = (H_m^T H_m + lam I)^{-1} H_m^T H*."""
    gs = source.T @ source
    dim = gs.shape[0]
    regularized = gs + lam * torch.eye(dim, device=gs.device, dtype=gs.dtype)
    return torch.linalg.solve(regularized, source.T @ target)


def affine_map_lsq(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares affine fit y ~= x @ W + b. Returns (W, b)."""
    ones = torch.ones((source.shape[0], 1), device=source.device, dtype=source.dtype)
    X = torch.cat([source, ones], dim=1)
    coefs = torch.linalg.lstsq(X, target).solution
    return coefs[:-1], coefs[-1:]


def whiten(h: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (h - mean) / std * scale
