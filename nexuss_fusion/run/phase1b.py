"""Stage 1b: train vision projector + query resampler on SPACE benchmark pairs.

Usage:
    python -m nexuss_fusion.run.phase1b --images-dir bench/space --cache features-cache

Gate: ≥95% specialist unimodal retention (cosine similarity between projected
vision and caption embeddings must be ≥95% of the baseline cosine).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..data import FeaturesCache
from ..eval.alignment_metrics import cosine_similarity
from ..extract import TextEmbedder, VisionExtractor
from ..extract.text import caption_cache_key
from ..math.projector import VisionProjector

log = logging.getLogger("nexuss_fusion.phase1b")

RETENTION_GATE = 0.95
EMBUDGET = 64


def load_cached_pairs(
    images_dir: Path, cache: FeaturesCache
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    """Load cached (patch_states, caption_embedding) pairs from disk."""
    captions: dict[str, str] = {}
    captions_file = images_dir / "captions.json"
    if captions_file.exists():
        captions = {str(k): str(v) for k, v in json.loads(captions_file.read_text()).items()}

    vision_model, vision_rev = VisionExtractor().model_id, VisionExtractor().revision
    text_model, text_rev = TextEmbedder().model_id, TextEmbedder().revision

    patch_states_list: list[torch.Tensor] = []
    caption_embs_list: list[torch.Tensor] = []
    keys: list[str] = []

    for image_path in sorted(images_dir.rglob("*.jpg")):
        caption = captions.get(image_path.name)
        if not caption:
            continue
        img_key = cache.key(
            {"kind": "image", "model": vision_model, "revision": vision_rev, "image": str(image_path)}
        )
        txt_key = caption_cache_key(cache, text_model, text_rev, caption)
        cached_img = cache.get(img_key)
        cached_txt = cache.get(txt_key)
        if cached_img is None or cached_txt is None:
            log.warning("skipping %s: missing cache entries", image_path)
            continue
        patch_states_list.append(cached_img)  # (1, 768) — mean-pooled row
        caption_embs_list.append(cached_txt)  # (1, 960)
        keys.append(str(image_path))

    if not patch_states_list:
        raise SystemExit("no cached pairs found; run with --extract first")

    log.info("loaded %d pairs from %s", len(keys), images_dir)
    return patch_states_list, caption_embs_list, keys


def train_projector(
    patch_states_list: list[torch.Tensor],
    caption_embs_list: list[torch.Tensor],
    d_in: int = 768,
    d_out: int = 960,
    budget: int = EMBUDGET,
    lr: float = 3e-4,
    epochs: int = 200,
    seed: int = 42,
) -> tuple[VisionProjector, dict]:
    """Train the vision projector on cached pairs. Returns (model, training_log)."""
    torch.manual_seed(seed)
    projector = VisionProjector(d_in=d_in, d_out=d_out, budget=budget)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Stack all pairs into a single batch
    src = torch.cat([p.double() for p in patch_states_list], dim=0).float()  # (N, 768)
    tgt = torch.cat([p.double() for p in caption_embs_list], dim=0).float()  # (N, 960)

    # Compute baseline cosine (mean-pooled vision vs caption)
    with torch.no_grad():
        baseline_cos = cosine_similarity(src, tgt).item()

    log.info("baseline cosine (mean-pooled): %.4f", baseline_cos)
    log.info("training projector: %d params", sum(p.numel() for p in projector.parameters()))

    projector.train()
    log_entries: list[dict] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        projected = projector(src)  # (N, budget, 960)
        # Mean-pool over budget dimension → (N, 960)
        projected_pooled = projected.mean(dim=1)
        loss = nn.functional.mse_loss(projected_pooled, tgt)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            projector.eval()
            with torch.no_grad():
                proj_pooled = projector(src).mean(dim=1)
                proj_cos = cosine_similarity(proj_pooled, tgt).item()
                retention = proj_cos / max(baseline_cos, 1e-8)
            projector.train()
            log_entries.append(
                {"epoch": epoch, "loss": loss.item(), "cosine": proj_cos, "retention": retention}
            )
            log.info(
                "epoch %d | loss %.4f | cosine %.4f | retention %.2f%%",
                epoch,
                loss.item(),
                proj_cos,
                retention * 100,
            )

    # Final evaluation
    projector.eval()
    with torch.no_grad():
        proj_pooled = projector(src).mean(dim=1)
        final_cos = cosine_similarity(proj_pooled, tgt).item()
        final_retention = final_cos / max(baseline_cos, 1e-8)

    return projector, {
        "baseline_cosine": baseline_cos,
        "final_cosine": final_cos,
        "retention": final_retention,
        "retention_gate": RETENTION_GATE,
        "retention_pass": final_retention >= RETENTION_GATE,
        "epochs": epochs,
        "lr": lr,
        "budget": budget,
        "n_pairs": len(patch_states_list),
        "training_log": log_entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1b: train vision projector")
    parser.add_argument("--images-dir", default="bench/space")
    parser.add_argument("--cache", default="features-cache")
    parser.add_argument("--out", default="phase1b-results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--budget", type=int, default=EMBUDGET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    images_dir = Path(args.images_dir)
    cache = FeaturesCache(args.cache)
    patch_states_list, caption_embs_list, keys = load_cached_pairs(images_dir, cache)

    projector, result = train_projector(
        patch_states_list,
        caption_embs_list,
        budget=args.budget,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), out_dir / "vision_projector.pt")
    (out_dir / "phase1b.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log.info("results written to %s", out_dir)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["retention_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
