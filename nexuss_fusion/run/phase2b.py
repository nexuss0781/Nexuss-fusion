"""Stage 2b: unfreeze decoder + train end-to-end on vision-conditioned captioning.

Usage:
    python -m nexuss_fusion.run.phase2b --images-dir bench/space --cache features-cache

Architecture: same as 2a, but unfreezes the last N decoder layers + the
VisionProjector. Uses gradient checkpointing to fit on CPU.
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
from ..extract.text import caption_cache_key, normalize_caption
from ..math.projector import VisionProjector

log = logging.getLogger("nexuss_fusion.phase2b")


def load_pairs(images_dir: Path, cache: FeaturesCache) -> tuple[list[torch.Tensor], list[str]]:
    captions: dict[str, str] = {}
    captions_file = images_dir / "captions.json"
    if captions_file.exists():
        captions = {str(k): str(v) for k, v in json.loads(captions_file.read_text()).items()}

    vision_model, vision_rev = VisionExtractor().model_id, VisionExtractor().revision
    text_model, text_rev = TextEmbedder().model_id, TextEmbedder().revision

    patch_states_list: list[torch.Tensor] = []
    caption_strs: list[str] = []

    for image_path in sorted([*images_dir.rglob("*.jpg"), *images_dir.rglob("*.png")]):
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
        patch_states_list.append(cached_img)
        caption_strs.append(caption)

    if not patch_states_list:
        raise SystemExit("no cached pairs found; run with --extract first")
    log.info("loaded %d pairs from %s", len(caption_strs), images_dir)
    return patch_states_list, caption_strs


def train_e2e(
    patch_states_list: list[torch.Tensor],
    captions: list[str],
    budget: int = 64,
    lr: float = 5e-6,
    epochs: int = 50,
    unfreeze_last_n: int = 4,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)

    text_embedder = TextEmbedder()
    tokenizer = AutoTokenizer.from_pretrained(text_embedder.model_id, revision=text_embedder.revision)
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
    stage1b_path = Path("phase1b-results/vision_projector.pt")
    if stage1b_path.exists():
        projector.load_state_dict(torch.load(stage1b_path, weights_only=True))
        log.info("loaded stage 1b projector weights")

    all_patches = torch.cat([p.double() for p in patch_states_list], dim=0).float()

    encodings = tokenizer(
        [normalize_caption(c) for c in captions],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    L = input_ids.shape[1]

    params = list(projector.parameters()) + [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info("training e2e: %d pairs, budget=%d, L=%d, epochs=%d", len(captions), budget, L, epochs)

    log_entries: list[dict] = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        vision_prefix = projector(all_patches.unsqueeze(1))

        with torch.no_grad():
            text_embeds = decoder.get_input_embeddings()(input_ids)

        combined = torch.cat([vision_prefix, text_embeds], dim=1)
        vision_mask = torch.ones(vision_prefix.shape[:2], dtype=attention_mask.dtype)
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        position_ids = torch.arange(combined.shape[1]).unsqueeze(0).expand(combined.shape[0], -1)

        out = decoder(inputs_embeds=combined, attention_mask=combined_mask, position_ids=position_ids)
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

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                vision_pooled = vision_prefix.mean(dim=1)
                caption_pooled = text_embedder.embed_batch_cached(
                    [normalize_caption(c) for c in captions], FeaturesCache("features-cache")
                ).float()
                cos = cosine_similarity(vision_pooled, caption_pooled).item()

            log_entries.append({"epoch": epoch, "loss": loss.item(), "cosine": cos})
            log.info("epoch %d | loss %.4f | cosine %.4f", epoch, loss.item(), cos)
            best_loss = min(best_loss, loss.item())

    return {
        "n_pairs": len(captions),
        "budget": budget,
        "epochs": epochs,
        "lr": lr,
        "unfreeze_last_n": unfreeze_last_n,
        "trainable_params": trainable,
        "total_params": total,
        "final_loss": loss.item(),
        "best_loss": best_loss,
        "final_cosine": cos,
        "training_log": log_entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2b: e2e training")
    parser.add_argument("--images-dir", default="bench/space")
    parser.add_argument("--cache", default="features-cache")
    parser.add_argument("--out", default="phase2b-results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    images_dir = Path(args.images_dir)
    cache = FeaturesCache(args.cache)
    patch_states_list, caption_strs = load_pairs(images_dir, cache)

    result = train_e2e(
        patch_states_list,
        caption_strs,
        budget=args.budget,
        lr=args.lr,
        epochs=args.epochs,
        unfreeze_last_n=args.unfreeze_last_n,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase2b.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log.info("results written to %s", out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
