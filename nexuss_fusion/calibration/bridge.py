"""Calibration bridge: fit Procrustes/Ridge maps from paired cached features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.manifest import write_manifest
from ..math import normalize, procrustes

log = logging.getLogger(__name__)


@dataclass
class CalibrationBridge:
    @staticmethod
    def assemble(
        source_rows: list[torch.Tensor], target_rows: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(source_rows) != len(target_rows):
            raise ValueError("source and target row counts must match")
        if not source_rows:
            raise ValueError("no paired rows to calibrate")
        if source_rows[0].shape[0] != target_rows[0].shape[0]:
            raise ValueError("paired rows must have matching examples")
        src = torch.cat([r.double() for r in source_rows], dim=0)
        tgt = torch.cat([r.double() for r in target_rows], dim=0)
        return src, tgt

    @classmethod
    def fit(
        cls, source: torch.Tensor, target: torch.Tensor, lam: float = 1e-3, backend: str | None = None
    ) -> dict:
        """Return fitted bridge: whitening (train stats) + procrustes R, scale + ridge A."""
        normalizer = normalize.Normalizer(source.shape)
        norm_src = normalizer.fit_transform(source)
        R, scale = procrustes.procrustes_map(norm_src, target, backend=backend)
        A = procrustes.ridge_least_squares(norm_src, target, lam=lam, backend=backend)
        return {"normalizer": normalizer, "R": R, "scale": scale, "A": A, "lam": lam}

    @staticmethod
    def apply(source: torch.Tensor, bridge: dict) -> torch.Tensor:
        norm = bridge["normalizer"].transform(source.double())
        return norm @ bridge["A"]

    @staticmethod
    def apply_orthogonal(source: torch.Tensor, bridge: dict) -> torch.Tensor:
        norm = bridge["normalizer"].transform(source.double())
        return norm @ bridge["R"] * bridge["scale"]

    @classmethod
    def save(cls, bridge: dict, out_dir: str | Path, meta: dict) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"A": bridge["A"], "R": bridge["R"], "scale": bridge["scale"], "lam": bridge["lam"]},
            out_dir / "bridge.pt",
        )
        manifest = {
            "meta": meta,
            "normalizer": bridge["normalizer"].state_dict(),
            "bridge_file": "bridge.pt",
        }
        write_manifest(manifest, out_dir / "bridge_manifest.json")
        return out_dir / "bridge_manifest.json"
