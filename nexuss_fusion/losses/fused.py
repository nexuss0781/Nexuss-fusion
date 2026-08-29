"""Fused training objective: answer + feature + logit + contrastive + replay + missing (torch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 2 or labels.dim() != 1:
        raise ValueError("logits must be (N, V) and labels (N,)")
    return F.cross_entropy(logits, labels)


def fused_loss(
    answer_logits: torch.Tensor,
    answer_labels: torch.Tensor,
    feature_terms: list[tuple[float, torch.Tensor, torch.Tensor]],
    logit_term: tuple[float, float, torch.Tensor, torch.Tensor] | None = None,
    contrastive: float = 0.0,
    replay_ce: float = 0.0,
    missing_ce: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined objective matching docs/MATHS.md section 6.

    feature_terms: (lambda, student_norm, teacher_norm) via MSE.
    logit_term: (T, lambda, teacher_logp, student_logp) -> lambda * T^2 * KL.
    """
    l_answer = cross_entropy(answer_logits, answer_labels)
    l_feat = torch.zeros((), dtype=answer_logits.dtype, device=answer_logits.device)
    for lam, student, teacher in feature_terms:
        l_feat = l_feat + lam * F.mse_loss(student, teacher)
    l_logit = torch.zeros_like(l_feat)
    if logit_term is not None:
        temperature, lam, teacher_logp, student_logp = logit_term
        teacher_p = teacher_logp.exp()
        kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1).mean()
        l_logit = lam * (temperature**2) * kl
    total = l_answer + l_feat + l_logit + contrastive + replay_ce + missing_ce
    parts = {
        "answer": float(l_answer.item()),
        "feat": float(l_feat.item()),
        "logit_kl": float(l_logit.item()),
        "contrastive": float(contrastive),
        "replay": float(replay_ce),
        "missing": float(missing_ce),
    }
    return total, parts
