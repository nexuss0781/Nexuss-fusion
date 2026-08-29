import torch

from nexuss_fusion.math.normalize import Normalizer, canonicalize_whiten


def test_canonicalize_zero_mean_unit_std():
    torch.manual_seed(0)
    stack = torch.randn(50, 12) * 3 + 5
    out, normalizer = canonicalize_whiten(stack)
    assert torch.allclose(out.mean(dim=0), torch.zeros(12), atol=1e-5)
    assert torch.allclose(out.std(dim=0), torch.ones(12), atol=1e-5)
    assert normalizer.mean is not None


def test_normalizer_transform_reuses_stats():
    torch.manual_seed(1)
    stack = torch.randn(64, 8)
    normalizer = Normalizer((64, 8)).fit(stack)
    out = normalizer.transform(stack)
    assert out.shape == (64, 8)
    assert torch.allclose(normalizer.fit_transform(stack), out)


def test_normalizer_roundtrip_state_dict():
    torch.manual_seed(2)
    stack = torch.randn(40, 6)
    normalizer = Normalizer((40, 6)).fit(stack)
    state = normalizer.state_dict()
    restored = Normalizer.from_state((40, 6), state)
    assert torch.allclose(normalizer.transform(stack), restored.transform(stack))
