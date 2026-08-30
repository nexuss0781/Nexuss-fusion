"""Stage 2a: train SmolLM2 to consume projected vision tokens as prefix.

Usage:
    python -m nexuss_fusion.run.phase2a --images-dir bench/space --cache features-cache

Architecture:
    Frozen SigLIP → VisionProjector → soft tokens (B, 64, 960) →
    prepended to SmolLM2 token embeddings → frozen SmolLM2 decoder →
    LM loss on caption tokens only (vision prefix positions masked with -100).
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
from ..extract.text import caption_cache_key, normalize_caption
from ..math.projector import VisionProjector

log = logging.getLogger("nexuss_fusion.phase2a")


def load_pairs(
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
        caption_embs_list.append(cached_txt)
        caption_strs.append(caption)

    if not patch_states_list:
        raise SystemExit("no cached pairs found; run with --extract first")

    log.info("loaded %d pairs from %s", len(caption_strs), images_dir)
    return patch_states_list, caption_embs_list, caption_strs


def train_decoder(
    patch_states_list: list[torch.Tensor],
    captions: list[str],
    budget: int = 64,
    lr: float = 2e-5,
    epochs: int = 100,
    seed: int = 42,
) -> dict:
    """Train SmolLM2 to consume projected vision tokens as prefix.

    Freezes SigLIP + SmolLM2, trains only the VisionProjector and a small
    LM head alignment layer.
    """
    torch.manual_seed(seed)

    # Load frozen models
    text_embedder = TextEmbedder()
    tokenizer, decoder = text_embedder._ensure()
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    d_in = 768
    d_out = 960
    projector = VisionProjector(d_in=d_in, d_out=d_out, budget=budget)

    # Load the stage 1b trained projector if available
    stage1b_path = Path("phase1b-results/vision_projector.pt")
    if stage1b_path.exists():
        projector.load_state_dict(torch.load(stage1b_path, weights_only=True))
        log.info("loaded stage 1b projector weights")

    projector.train()

    # Only train the projector (decoder is frozen)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Stack all patch states
    all_patches = torch.cat([p.double() for p in patch_states_list], dim=0).float()  # (N, 768)

    # Tokenize all captions
    encodings = tokenizer(
        [normalize_caption(c) for c in captions],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = encodings["input_ids"]  # (N, L)
    attention_mask = encodings["attention_mask"]  # (N, L)
    L = input_ids.shape[1]

    # Compute baseline: cosine between projected vision (ridge) and caption embeddings
    log.info("training decoder alignment: %d pairs, budget=%d, L=%d", len(captions), budget, L)

    log_entries: list[dict] = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Project vision tokens: (N, 768) → (N, budget, 960)
        # Reshape to (N, 1, 768) so VisionProjector treats each as a single-token sequence
        vision_prefix = projector(all_patches.unsqueeze(1))  # (N, budget, 960)

        # Get text embeddings: (N, L, 960)
        with torch.no_grad():
            text_embeds = decoder.get_input_embeddings()(input_ids)  # (N, L, 960)

        # Concatenate: vision prefix + text tokens
        combined = torch.cat([vision_prefix, text_embeds], dim=1)  # (N, budget+L, 960)

        # Create attention mask: all ones for both vision and text
        vision_mask = torch.ones(vision_prefix.shape[:2], dtype=attention_mask.dtype)
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)  # (N, budget+L)

        # Forward pass through frozen decoder
        position_ids = torch.arange(combined.shape[1]).unsqueeze(0).expand(combined.shape[0], -1)
        out = decoder(inputs_embeds=combined, attention_mask=combined_mask, position_ids=position_ids)
        logits = out.last_hidden_state  # (N, budget+L, 960)

        # LM head: project back to vocab size
        lm_head = decoder.lm_head  # (960, vocab_size)
        token_logits = logits[:, budget:, :] @ lm_head.weight.T  # (N, L, vocab_size)

        # Loss: only on caption tokens (mask out vision prefix)
        shift_logits = token_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        # Mask padding tokens
        shift_labels[shift_labels == tokenizer.pad_token_id] = -100
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                vision_pooled = vision_prefix.mean(dim=1)  # (N, 960)
                caption_pooled = text_embedder.embed_batch_cached(
                    [normalize_caption(c) for c in captions], FeaturesCache("features-cache")
                ).float()
                cos = cosine_similarity(vision_pooled, caption_pooled).item()

            log_entries.append({
                "epoch": epoch,
                "loss": loss.item(),
                "vision_caption_cosine": cos,
            })
            log.info(
                "epoch %d | loss %.4f | vision-caption cosine %.4f",
                epoch,
                loss.item(),
                cos,
            )
            best_loss = min(best_loss, loss.item())

    return {
        "n_pairs": len(captions),
        "budget": budget,
        "epochs": epochs,
        "lr": lr,
        "final_loss": loss.item(),
        "best_loss": best_loss,
        "final_vision_caption_cosine": cos,
        "training_log": log_entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2a: decoder alignment")
    parser.add_argument("--images-dir", default="bench/space")
    parser.add_argument("--cache", default="features-cache")
    parser.add_argument("--out", default="phase2a-results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    images_dir = Path(args.images_dir)
    cache = FeaturesCache(args.cache)
    patch_states_list, caption_embs_list, caption_strs = load_pairs(images_dir, cache)

    result = train_decoder(
        patch_states_list,
        caption_strs,
        budget=args.budget,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase2a.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log.info("results written to %s", out_dir)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
