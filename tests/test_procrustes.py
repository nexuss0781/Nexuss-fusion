import pytest
import torch

from nexuss_fusion.math.procrustes import affine_map_lsq, procrustes_map, ridge_least_squares

DT = torch.float64


def _make_pair(d_src: int, d_tgt: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    source = torch.randn(n, d_src, dtype=DT)
    true_A = torch.randn(d_src, d_tgt, dtype=DT)
    target = source @ true_A + 0.02 * torch.randn(n, d_tgt, dtype=DT)
    return source, target


def test_procrustes_recovers_orthogonal_map():
    torch.manual_seed(0)
    source = torch.randn(200, 10, dtype=DT)
    R_true = torch.randn(10, 10, dtype=DT)
    q, _ = torch.linalg.qr(R_true)
    target = source @ q
    R, scale = procrustes_map(source, target)
    assert torch.allclose(R @ R.mT, torch.eye(10, dtype=DT), atol=1e-8)
    assert torch.allclose(source @ R, target, atol=1e-4)
    assert scale == pytest.approx(1.0, abs=1e-4)


def test_procrustes_rectangular_target_recovers_orthonormal_columns():
    torch.manual_seed(5)
    source = torch.randn(120, 10, dtype=DT)
    q, _ = torch.linalg.qr(torch.randn(10, 6, dtype=DT))
    target = source @ q
    R, scale = procrustes_map(source, target)
    assert R.shape == (10, 6)
    aligned = source @ R * scale
    assert torch.allclose(aligned, target, atol=1e-4)
    assert scale == pytest.approx(1.0, abs=1e-4)


def test_ridge_least_squares_fits_affine_target():
    source, target = _make_pair(d_src=8, d_tgt=5, n=300, seed=1)
    A = ridge_least_squares(source, target, lam=0.01)
    pred = source @ A
    resid = ((pred - target) ** 2).mean()
    assert resid < 0.05


def test_affine_map_matches_target_with_bias():
    torch.manual_seed(2)
    source = torch.randn(150, 6, dtype=DT)
    W_true = torch.randn(6, 4, dtype=DT)
    bias = torch.randn(4, dtype=DT)
    target = source @ W_true + bias
    W, b = affine_map_lsq(source, target)
    assert torch.allclose(W, W_true, atol=1e-8)
    assert torch.allclose(b, bias, atol=1e-8)