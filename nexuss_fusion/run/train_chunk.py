"""Train vision projection on one chunk of Flickr30k data.

Usage:
    python -m nexuss_fusion.run.train_chunk --chunk-dir chunks/chunk_01/images \
        --captions-file chunks/chunk_01/captions.json \
        --cache features-cache --out chunk_01_model

This script trains a VisionProjector + unfrozen decoder layers on one chunk
of image-caption pairs using LM cross-entropy loss (same as stage 2b).
It saves checkpoints and outputs a final model file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import FeaturesCache
from ..eval.alignment_metrics import cosine_similarity
from ..extract import TextEmbedder, VisionExtractor
from ..extract.text import normalize_caption
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

    patches: list[torch.Tensor] = []
    captions: list[str] = []

    for image_path in sorted([*chunk_dir.glob("*.jpg"), *chunk_dir.glob("*.png")]):
        fname = image_path.name
        cap_list = captions_raw.get(fname, [])
        if not cap_list:
            continue
        caption = cap_list[0]

        img_key = cache.key(
            {"kind": "image", "model": vision_model, "revision": vision_rev, "image": str(image_path)}
        )
        cached_img = cache.get(img_key)
        if cached_img is None:
            continue
        patches.append(cached_img)
        captions.append(caption)

    log.info("loaded %d pairs from %s", len(patches), chunk_dir)
    return patches, captions


def train(
    patches: list[torch.Tensor],
    captions: list[str],
    budget: int = 64,
    lr: float = 5e-6,
    epochs: int = 50,
    unfreeze_last_n: int = 4,
    checkpoint_every: int = 10,
    out_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)

    text_embedder = TextEmbedder()
    tokenizer = AutoTokenizer.from_pretrained(
        text_embedder.model_id, revision=text_embedder.revision
    )
    decoder = AutoModelForCausalLM.from_pretrained(
        text_embedder.model_id, revision=text_embedder.revision, torch_dtype=torch.float32
    )

    for p in decoder.parameters():
        p.requires_grad = False

    layers = decoder.model.layers
    for layer in layers[-unfreeze_last_n:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in decoder.lm_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in decoder.parameters())
    log.info("unfroze last %d layers: %d / %d params trainable", unfreeze_last_n, trainable, total)

    projector = VisionProjector(d_in=768, d_out=960, budget=budget)

    all_patches = torch.cat([p.double() for p in patches], dim=0).float()

    encodings = tokenizer(
        [normalize_caption(c) for c in captions],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    params = list(projector.parameters()) + [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info(
        "training chunk: %d pairs, budget=%d, epochs=%d",
        len(captions),
        budget,
        epochs,
    )

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        vision_prefix = projector(all_patches.unsqueeze(1))

        with torch.no_grad():
            text_embeds = decoder.get_input_embeddings()(input_ids)

        combined = torch.cat([vision_prefix, text_embeds], dim=1)
        vision_mask = torch.ones(
            vision_prefix.shape[:2], dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        position_ids = torch.arange(combined.shape[1]).unsqueeze(0).expand(
            combined.shape[0], -1
        )

        out = decoder(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            position_ids=position_ids,
        )
        token_logits = out.logits[:, budget:, :]

        shift_logits = token_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_labels[shift_labels == tokenizer.pad_token_id] = -100
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % checkpoint_every == 0 or epoch == epochs:
            with torch.no_grad():
                vision_pooled = vision_prefix.mean(dim=1)
                caption_pooled = text_embedder.embed_batch_cached(
                    [normalize_caption(c) for c in captions],
                    FeaturesCache("features-cache"),
                ).float()
                cos = cosine_similarity(vision_pooled, caption_pooled).item()

            log.info("epoch %d | loss %.4f | cosine %.4f", epoch, loss.item(), cos)
            best_loss = min(best_loss, loss.item())

            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "projector": projector.state_dict(),
                        "decoder_frozen": True,
                        "unfreeze_last_n": unfreeze_last_n,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "cosine": cos,
                    },
                    out_dir / f"checkpoint_epoch{epoch}.pt",
                )

    if out_dir:
        torch.save(
            {
                "projector": projector.state_dict(),
                "decoder_frozen": True,
                "unfreeze_last_n": unfreeze_last_n,
                "epoch": epochs,
                "loss": loss.item(),
                "cosine": cos,
            },
            out_dir / "final.pt",
        )

    return {
        "n_pairs": len(captions),
        "budget": budget,
        "epochs": epochs,
        "final_loss": loss.item(),
        "best_loss": best_loss,
        "final_cosine": cos,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train vision projection on one chunk"
    )
    parser.add_argument(
        "--chunk-dir", required=True, help="Directory with images for this chunk"
    )
    parser.add_argument(
        "--captions-file", default=None, help="Path to captions.json"
    )
    parser.add_argument("--cache", default="features-cache")
    parser.add_argument(
        "--out", required=True, help="Output directory for model checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
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
        unfreeze_last_n=args.unfreeze_last_n,
        checkpoint_every=args.checkpoint_every,
        out_dir=Path(args.out),
        seed=args.seed,
    )

    (Path(args.out) / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
