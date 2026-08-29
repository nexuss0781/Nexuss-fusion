import pytest
import torch

from nexuss_fusion.losses.fused import cross_entropy, fused_loss


def test_cross_entropy_greater_for_wrong_answer():
    logits_good = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    logits_bad = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0])
    assert cross_entropy(logits_bad, labels) > cross_entropy(logits_good, labels)


def test_fused_loss_combination():
    torch.manual_seed(0)
    logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))
    teacher = torch.randn(3, 960)
    student = torch.randn(3, 960)
    total, parts = fused_loss(
        logits,
        labels,
        feature_terms=[(0.1, student, teacher)],
        logit_term=(2.0, 0.5, torch.log_softmax(torch.randn(8, 10), dim=-1), torch.log_softmax(torch.randn(8, 10), dim=-1)),
    )
    assert total > 0
    assert set(parts) == {"answer", "feat", "logit_kl", "contrastive", "replay", "missing"}
    assert parts["feat"] >= 0


def test_fused_loss_requires_2d_logits():
    with pytest.raises(ValueError):
        fused_loss(torch.randn(8), torch.zeros(8), [])