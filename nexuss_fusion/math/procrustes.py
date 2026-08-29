"""Optimal linear alignment between representation spaces.

Closed-form maps that make any (source width, target width) pair compatible:

  procrustes_map   — best orthogonal+scale map via thin SVD (rigid-motion init)
  ridge_least_squares — general best linear map with ridge regularization
  affine_map_lsq   — least-squares affine fit (with bias), gradient-usable init
"""
from __future__ import annotations

import numpy as np


def procrustes_map(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (R, s, U, ..., ) components of the orthogonal Procrustes problem.

    R, s minimize ||target - s * source @ R||_F over orthogonal R, scalar s>=0.
    H*^T H = U Sigma V^T  ->  R = U V^T,  s = trace(Sigma)/||source||_F^2.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape[0] != target.shape[0]:
        raise ValueError("paired matrices must share the number of rows")
    corr = target.T @ source
    U, sigma, Vt = np.linalg.svd(corr, full_matrices=False)
    R = Vt.T @ U.T
    denom = np.sum(source ** 2) + 1e-12
    s = float(np.trace(np.diag(sigma))) / denom
    scale = max(float(s), 0.0)
    return R, scale, U, sigma, Vt


def ridge_least_squares(source: np.ndarray, target: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Best linear map A minimizing ||target - source A||_F^2 + lam ||A||_F^2.

    A = (source^T source + lam I)^-1 source^T target   (d_src x d_tgt)
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    design = source.T @ source + lam * np.eye(source.shape[1])
    return np.linalg.solve(design, source.T @ target)


def affine_map_lsq(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine fit y ~= x @ W + b. Returns (W, b)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape[0] != target.shape[0]:
        raise ValueError("paired matrices must share the number of rows")
    X = np.hstack([source, np.ones((source.shape[0], 1))])
    coefs, *_ = np.linalg.lstsq(X, target, rcond=None)
    W, b = coefs[:-1], coefs[-1]
    return W, b.reshape(1, -1) if b.ndim else b.reshape(1, 1)