"""Fused training objective: answer + feature + logit + contrastive + replay + missing."""
from __future__ import annotations

import numpy as np


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must share the number of positions")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    rows = np.arange(labels.shape[0])
    return float(-np.mean(log_probs[rows, labels]))


def fused_loss(
    answer_logits: np.ndarray,
    answer_labels: np.ndarray,
    feature_terms: list[tuple[float, np.ndarray, np.ndarray]],
    logit_term: tuple[float, float, np.ndarray, np.ndarray] | None,
    contrastive: float = 0.0,
    replay_ce: float = 0.0,
    missing_ce: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Signature matching docs/MATHS.md section 6.

    feature terms: list of (lambda, student_norm, teacher_norm)
    logit_term: (T, lambda, teacher_logp, student_logp) -> lambda * T^2 * KL.
    Returns (loss, components).
    """
    l_answer = cross_entropy(answer_logits, answer_labels)
    l_feat = 0.0
    for lam, student, teacher in feature_terms:
        l_feat += float(lam * np.mean((student - teacher) ** 2))
    l_logit = 0.0
    if logit_term is not None:
        temperature, lam, teacher_logp, student_logp = logit_term
        l_logit = float(
            lam * (temperature**2) * np.mean(np.exp(teacher_logp) * (teacher_logp - student_logp))
        )
    total = (
        l_answer
        + l_feat
        + l_logit
        + contrastive
        + replay_ce
        + missing_ce
    )
    parts = {
        "answer": l_answer,
        "feat": l_feat,
        "logit_kl": l_logit,
        "contrastive": float(contrastive),
        "replay": float(replay_ce),
        "missing": float(missing_ce),
    }
    return total, parts