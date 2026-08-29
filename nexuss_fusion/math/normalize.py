"""Whitening / scale-harmonization canonicalizer.

Makes features from arbitrary expert spaces comparable before cross-space
mapping. Diagonal reparameterization: preserves information along every
principal direction while removing global scale/mean/variance differences.
"""
from __future__ import annotations

import numpy as np


class Normalizer:
    def __init__(self, shape: tuple[int, ...], eps: float = 1e-6) -> None:
        self.shape = shape
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, stack: np.ndarray) -> "Normalizer":
        flat = stack.reshape(-1, stack.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + self.eps
        self.scale = np.ones_like(self.std)
        return self

    def transform(self, h: np.ndarray) -> np.ndarray:
        return (h - self.mean) / self.std * self.scale

    def fit_transform(self, stack: np.ndarray) -> np.ndarray:
        return self.fit(stack).transform(stack)


def canonicalize_whiten(stack: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, Normalizer]:
    normalizer = Normalizer(stack.shape, eps=eps)
    return normalizer.fit_transform(stack), normalizer