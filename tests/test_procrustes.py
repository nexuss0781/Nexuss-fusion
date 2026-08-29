from nexuss_fusion.math.procrustes import (
    affine_map_lsq,
    procrustes_map,
    ridge_least_squares,
)

import numpy as np
import pytest


def _make_pair(d_src: int, d_tgt: int, n: int, T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n * T, d_src))
    true_A = rng.normal(size=(d_src, d_tgt))
    target = source @ true_A + 0.02 * rng.normal(size=(n * T, d_tgt))
    return source, target


def test_procrustes_recovers_orthogonal_map():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(200, 10))
    R_true = rng.normal(size=(10, 10))
    R_true, _ = np.linalg.qr(R_true)
    target = source @ R_true
    R, scale, *_ = procrustes_map(source, target)
    assert np.allclose(R @ R.T, np.eye(10), atol=1e-8)
    assert np.allclose(source @ R, target, atol=1e-4)
    assert scale == pytest.approx(1.0, abs=1e-4)


def test_ridge_least_squares_fits_affine_target():
    source, target = _make_pair(d_src=8, d_tgt=5, n=30, T=10, seed=1)
    A = ridge_least_squares(source, target, lam=0.01)
    pred = source @ A
    resid = np.mean((pred - target) ** 2)
    assert resid < 0.05


def test_affine_map_matches_target_with_bias():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(150, 6))
    W_true = rng.normal(size=(6, 4))
    bias = rng.normal(size=4)
    target = source @ W_true + bias
    W, b = affine_map_lsq(source, target)
    assert np.allclose(W, W_true, atol=1e-8)
    assert np.allclose(b, bias, atol=1e-8)