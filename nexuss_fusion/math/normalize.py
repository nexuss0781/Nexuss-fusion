"""Whitening / scale-harmonization canonicalizer (torch, CPU/GPU-capable)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Normalizer:
    shape: tuple[int, ...]
    eps: float = 1e-6
    _dims: int = field(init=False, default=0)
    mean: torch.Tensor | None = field(init=False, default=None)
    std: torch.Tensor | None = field(init=False, default=None)
    scale: torch.Tensor | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._dims = self.shape[-1]

    def fit(self, stack: torch.Tensor) -> "Normalizer":
        flat = stack.reshape(-1, stack.shape[-1])
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0) + self.eps
        self.scale = torch.ones_like(self.std)
        return self

    def transform(self, h: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None or self.scale is None:
            raise RuntimeError("Normalizer not fitted")
        return self._whiten(h)

    def fit_transform(self, stack: torch.Tensor) -> torch.Tensor:
        return self.fit(stack).transform(stack)

    def _whiten(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.mean) / self.std * self.scale

    def state_dict(self) -> dict[str, list[float]]:
        if self.mean is None or self.std is None or self.scale is None:
            raise RuntimeError("Normalizer not fitted")
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_state(cls, shape: tuple[int, ...], state: dict[str, list[float]], eps: float = 1e-6) -> "Normalizer":
        obj = cls(shape, eps)
        obj.mean = torch.tensor(state["mean"], dtype=torch.float32)
        obj.std = torch.tensor(state["std"], dtype=torch.float32)
        obj.scale = torch.tensor(state["scale"], dtype=torch.float32)
        return obj


def canonicalize_whiten(stack: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, Normalizer]:
    normalizer = Normalizer(stack.shape, eps=eps)
    return normalizer.fit_transform(stack), normalizer