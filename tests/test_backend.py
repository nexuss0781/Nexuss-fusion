import torch

from nexuss_fusion.backend import get_backend
from nexuss_fusion.backend import torch_kernels as tk
from nexuss_fusion.math.procrustes import procrustes_map


def test_get_backend_defaults_to_torch():
    assert get_backend().name == "torch"


def test_get_backend_torch_explicit():
    assert get_backend("torch").name == "torch"


def test_torch_backend_kernels_family_consistency():
    b = get_backend("torch")
    torch.manual_seed(0)
    src = torch.randn(60, 12).double()
    tgt = torch.randn(60, 8).double()
    R, scale = b.procrustes(src, tgt)
    assert R.shape == (12, 8)
    assert scale > 0
    A = b.ridge_least_squares(src, tgt, lam=0.01)
    assert A.shape == (12, 8)
    norm = b.whiten(src, torch.zeros(12, dtype=torch.float64), torch.ones(12, dtype=torch.float64), torch.ones(12, dtype=torch.float64))
    assert torch.equal(norm, src)


def test_backend_dispatch_env_auto(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUSS_FUSION_BACKEND", "torch")
    assert get_backend().name == "torch"


def test_procrustes_map_uses_backend_kernel():
    torch.manual_seed(3)
    src = torch.randn(30, 5)
    q, _ = torch.linalg.qr(torch.randn(5, 5))
    tgt = src @ q
    R, scale = procrustes_map(src, tgt, backend="torch")
    assert torch.allclose(src @ R, tgt, atol=1e-4)