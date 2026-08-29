import pytest
import torch

from nexuss_fusion.calibration.bridge import CalibrationBridge
from nexuss_fusion.calibration.split import split_by_key


def _make_paired(n: int, d_src: int, d_tgt: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    source = torch.randn(n, d_src)
    true_A = torch.randn(d_src, d_tgt) * 0.5
    target = source @ true_A + 0.05 * torch.randn(n, d_tgt)
    return source, target


def test_bridge_reconstruction_on_in_distribution():
    src, tgt = _make_paired(80, 10, 7, seed=0)
    bridge = CalibrationBridge.fit(src, tgt, lam=0.1)
    pred = CalibrationBridge.apply(src, bridge)
    assert torch.nn.functional.cosine_similarity(pred, tgt, dim=0).mean().abs() > 0.9


def test_split_is_deterministic_and_disjoint():
    keys = [f"k{i}" for i in range(10)]
    a_tr, a_te = split_by_key(keys, te=0.3, seed=0)
    b_tr, b_te = split_by_key(keys, te=0.3, seed=0)
    assert a_tr == b_tr and a_te == b_te
    assert set(a_tr).isdisjoint(set(a_te))
    assert len(a_te) == 3


def test_bridge_save_load_manifest(tmp_path):
    src, tgt = _make_paired(40, 8, 6, seed=1)
    bridge = CalibrationBridge.fit(src, tgt)
    manifest_path = CalibrationBridge.save(bridge, tmp_path, meta={"seed": 0})
    assert manifest_path.name == "bridge_manifest.json"
    assert (tmp_path / "bridge.pt").exists()


def test_bridge_requires_matching_row_counts():
    with pytest.raises(ValueError):
        CalibrationBridge.assemble([torch.randn(3, 4)], [torch.randn(2, 4)])
