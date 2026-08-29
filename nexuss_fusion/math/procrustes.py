"""Optimal linear alignment between representation spaces (torch primary)."""

from __future__ import annotations

import torch

from ..backend import Backend, get_backend


def procrustes_map(
    source: torch.Tensor,
    target: torch.Tensor,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal+scale Procrustes init: minimize ||target - s*source@R||_F.

    Returns (R, scale) via the resolved backend kernel.
    """
    return _resolve_backend(backend).procrustes(source, target)


def ridge_least_squares(
    source: torch.Tensor,
    target: torch.Tensor,
    lam: float = 1e-3,
    backend: str | None = None,
) -> torch.Tensor:
    """Best linear map minimizing ||target - source A||_F^2 + lam ||A||_F^2."""
    return _resolve_backend(backend).ridge_least_squares(source, target, lam)


def affine_map_lsq(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares affine fit y ~= x @ W + b. Returns (W, b)."""
    source = source.double()
    target = target.double()
    ones = torch.ones((source.shape[0], 1), device=source.device, dtype=source.dtype)
    X = torch.cat([source, ones], dim=1)
    coefs = torch.linalg.lstsq(X, target).solution
    return coefs[:-1], coefs[-1:]


def _resolve_backend(backend: str | None) -> Backend:
    return get_backend(backend) if backend else get_backend()
