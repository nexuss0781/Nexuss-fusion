"""Compute-once feature cache keyed by content sha256 (the CPU-training enabler)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from .manifest import content_hash

log = logging.getLogger(__name__)


@dataclass
class FeaturesCache:
    root: str | Path

    def __post_init__(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def key(self, extra: dict) -> str:
        return content_hash(extra)

    def path_for(self, feature_key: str) -> Path:
        return Path(self.root) / f"{feature_key}.pt"

    def has(self, feature_key: str) -> bool:
        return self.path_for(feature_key).exists()

    def get(self, feature_key: str) -> torch.Tensor | None:
        path = self.path_for(feature_key)
        if not path.exists():
            return None
        return torch.load(path, weights_only=True)

    def put(self, feature_key: str, tensor: torch.Tensor) -> Path:
        tensor = tensor.detach().float().cpu()
        path = self.path_for(feature_key)
        torch.save(tensor, path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        sidecar.write_text(json.dumps({"feature_key": feature_key, "entries": tensor.shape[0]}) + "\n")
        return path

    def compute_or_get(self, feature_key: str, compute: callable) -> tuple[torch.Tensor, bool]:
        cached = self.get(feature_key)
        if cached is not None:
            return cached, False
        tensor = compute()
        self.put(feature_key, tensor)
        log.info("cached %s -> %s", feature_key[:12], self.path_for(feature_key))
        return tensor, True
