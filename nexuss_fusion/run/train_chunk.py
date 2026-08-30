"""Train vision projection on one chunk of Flickr30k data.

Usage:
    python -m nexuss_fusion.run.train_chunk --chunk-dir chunk_01 --out chunk_01_model

This script trains a VisionProjector on one chunk of image-caption pairs.
It saves checkpoints every 100 epochs and outputs a final model file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..calibration.bridge import CalibrationBridge
from ..data import FeaturesCache
from ..eval.alignment_metrics import cosine_similarity
from ..extract import TextEmbedder, VisionExtractor
from ..extract.text import caption_cache_key
from ..math.projector import VisionProjector

log = logging.getLogger("nexuss_fusion.train_chunk")


def load_chunk(
    chunk_dir: Path, cache: FeaturesCache, captions_file: Path | None = None
) -> tuple[list[torch.Tensor], list[str]]:
    captions_raw: dict[str, list[str]] = {}
    if captions_file is None:
        captions_file = chunk_dir.parent / "captions.json"
    if captions_file.exists():
        captions_raw = json.loads(captions_file.read_text())

    vision_model, vision_rev = VisionExtractor().model_id, VisionExtractor().revision
    text_model, text_rev = TextEmbedder().model_id, TextEmbedder().revision

    patches: list[torch.Tensor] = []
    captions: list[str] = []

    for image_path in sorted(chunk_dir.glob("*.jpg")):
        fname = image_path.name
        cap_list = captions_raw.get(fname, [])
        if not cap_list:
            continue
        caption = cap_list[0]

        img_key = cache.key(
            {"kind": "image", "model": vision_model, "revision": vision_rev, "image": str(image_path)}
        )
        txt_key = caption_cache_key(cache, text_model, text_rev, caption)
        cached_img = cache.get(img_key)
        cached_txt = cache.get(txt_key)
        if cached_img is None or cached_txt is None:
            continue
        patches.append(cached_img)
        captions.append(caption)

    log.info("loaded %d pairs from %s", len(patches), chunk_dir)
    return patches, captions


def train(
    patches: list[torch.Tensor],
    captions: list[str],
    budget: int = 64,
    lr: float = 3e-4,
    epochs: int = 200,
    checkpoint_every: int = 50,
    out_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)

    projector = VisionProjector(d_in=768, d_out=960, budget=budget)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_patches = torch.cat([p.double() for p in patches], dim=0).float()
    bridge = CalibrationBridge.fit(all_patches.double(), all_patches.double(), lam=1e-3)
    ridge_A = bridge["A"].float()
    ridge_normalizer = bridge["normalizer"]
    with torch.no_grad():
        src_norm = ridge_normalizer.transform(all_patches.double()).float()
        baseline_ridge = src_norm @ ridge_A
        tgt = torch.cat([p.double() for p in patches], dim=0).float()
        baseline_cos = cosine_similarity(baseline_ridge, tgt).item()

    log.info(
        "training: %d pairs, budget=%d, epochs=%d, baseline_cos=%.4f",
        len(patches),
        budget,
        epochs,
        baseline_cos,
    )

    projector.train()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        projected = projector(all_patches.unsqueeze(1))
        projected_pooled = projected.mean(dim=1)
        loss = nn.functional.mse_loss(projected_pooled, tgt)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % checkpoint_every == 0 or epoch == epochs:
            projector.eval()
            with torch.no_grad():
                proj_pooled = projector(all_patches.unsqueeze(1)).mean(dim=1)
                cos = cosine_similarity(proj_pooled, tgt).item()
                retention = cos / max(baseline_cos, 1e-8)
            projector.train()
            best_loss = min(best_loss, loss.item())

            log.info(
                "epoch %d | loss %.4f | cosine %.4f | retention %.1f%%",
                epoch,
                loss.item(),
                cos,
                retention * 100,
            )

            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(projector.state_dict(), out_dir / f"checkpoint_epoch{epoch}.pt")

    if out_dir:
        torch.save(projector.state_dict(), out_dir / "final.pt")

    return {
        "n_pairs": len(patches),
        "budget": budget,
        "epochs": epochs,
        "final_loss": loss.item(),
        "best_loss": best_loss,
        "final_cosine": cos,
        "retention": retention,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train vision projection on one chunk")
    parser.add_argument("--chunk-dir", required=True, help="Directory with images for this chunk")
    parser.add_argument("--captions-file", default=None, help="Path to captions.json")
    parser.add_argument("--cache", default="features-cache")
    parser.add_argument("--out", required=True, help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    chunk_dir = Path(args.chunk_dir)
    cache = FeaturesCache(args.cache)
    captions_file = Path(args.captions_file) if args.captions_file else None
    patches, captions = load_chunk(chunk_dir, cache, captions_file)

    if not patches:
        log.error("no pairs loaded from %s", chunk_dir)
        return 1

    result = train(
        patches,
        captions,
        budget=args.budget,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        out_dir=Path(args.out),
        seed=args.seed,
    )

    (Path(args.out) / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
