"""Merge trained model chunks into one final model.

Usage:
    python -m nexuss_fusion.run.merge_chunks --chunks-dir chunks --out final_model

Loads all chunk model files and averages their weights element-wise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from ..math.projector import VisionProjector

log = logging.getLogger("nexuss_fusion.merge_chunks")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge model chunks")
    parser.add_argument("--chunks-dir", default="chunks", help="Directory containing chunk_XX/ subdirs")
    parser.add_argument("--out", default="final_model", help="Output directory")
    parser.add_argument("--budget", type=int, default=64)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    chunks_dir = Path(args.chunks_dir)
    chunk_dirs = sorted(chunks_dir.glob("chunk_*"))
    log.info("found %d chunks", len(chunk_dirs))

    all_states: list[dict[str, torch.Tensor]] = []
    for chunk_dir in chunk_dirs:
        final_path = chunk_dir / "final.pt"
        if not final_path.exists():
            log.warning("skipping %s: no final.pt", chunk_dir)
            continue
        checkpoint = torch.load(final_path, weights_only=True)
        if isinstance(checkpoint, dict) and "projector" in checkpoint:
            state = checkpoint["projector"]
        else:
            state = checkpoint
        all_states.append(state)
        log.info("loaded %s", chunk_dir.name)

    if not all_states:
        log.error("no chunks found")
        return 1

    merged: dict[str, torch.Tensor] = {}
    for key in all_states[0]:
        stacked = torch.stack([s[key].double() for s in all_states])
        merged[key] = stacked.mean(dim=0).float()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    projector = VisionProjector(d_in=768, d_out=960, budget=args.budget)
    projector.load_state_dict(merged)
    torch.save(merged, out_dir / "nexuss_fusion_vision.pt")

    (out_dir / "merge_info.json").write_text(
        json.dumps(
            {
                "n_chunks": len(all_states),
                "budget": args.budget,
                "chunk_dirs": [str(d) for d in chunk_dirs],
            },
            indent=2,
        )
    )

    log.info("merged %d chunks → %s/nexuss_fusion_vision.pt", len(all_states), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
