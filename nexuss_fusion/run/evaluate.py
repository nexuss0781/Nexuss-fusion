"""Generation quality evaluation: decode captions + BLEU/ROUGE.

Usage:
    python -m nexuss_fusion.run.evaluate --images-dir bench/space --cache features-cache

Loads the trained VisionProjector (from phase1b-results/) + SmolLM2 decoder,
generates captions for each SPACE image via greedy decode, and scores against
reference captions with BLEU-1/4 and ROUGE-L.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import FeaturesCache
from ..extract import TextEmbedder, VisionExtractor
from ..extract.text import caption_cache_key, normalize_caption
from ..math.projector import VisionProjector

log = logging.getLogger("nexuss_fusion.evaluate")

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "main"


def load_pairs(
    images_dir: Path, cache: FeaturesCache
) -> tuple[list[torch.Tensor], list[str], list[str]]:
    captions_raw: dict[str, str] = {}
    captions_file = images_dir / "captions.json"
    if captions_file.exists():
        captions_raw = {str(k): str(v) for k, v in json.loads(captions_file.read_text()).items()}

    vision_model, vision_rev = VisionExtractor().model_id, VisionExtractor().revision
    text_model, text_rev = TextEmbedder().model_id, TextEmbedder().revision

    patches: list[torch.Tensor] = []
    ref_captions: list[str] = []
    image_names: list[str] = []

    for image_path in sorted([*images_dir.rglob("*.jpg"), *images_dir.rglob("*.png")]):
        caption = captions_raw.get(image_path.name)
        if not caption:
            continue
        img_key = cache.key(
            {"kind": "image", "model": vision_model, "revision": vision_rev, "image": str(image_path)}
        )
        txt_key = caption_cache_key(cache, text_model, text_rev, caption)
        cached_img = cache.get(img_key)
        cached_txt = cache.get(txt_key)
        if cached_img is None or cached_txt is None:
            continue
        patches.append(cached_img)
        ref_captions.append(caption)
        image_names.append(image_path.name)

    log.info("loaded %d pairs", len(patches))
    return patches, ref_captions, image_names


def simple_tokenize(text: str) -> list[str]:
    return normalize_caption(text).split()


def compute_bleu(generated: str, reference: str) -> dict[str, float]:
    gen_tokens = simple_tokenize(generated)
    ref_tokens = simple_tokenize(reference)
    if not gen_tokens or not ref_tokens:
        return {"bleu1": 0.0, "bleu4": 0.0}

    ref_set = set(ref_tokens)
    gen_set = set(gen_tokens)

    precision1 = len(gen_set & ref_set) / max(len(gen_set), 1)

    if len(gen_tokens) >= 4 and len(ref_tokens) >= 4:
        gen_grams = [tuple(gen_tokens[i : i + 4]) for i in range(len(gen_tokens) - 3)]
        ref_grams = [tuple(ref_tokens[i : i + 4]) for i in range(len(ref_tokens) - 3)]
        ref_gram_set = set(ref_grams)
        matched = sum(1 for g in gen_grams if g in ref_gram_set)
        precision4 = matched / max(len(gen_grams), 1)
    else:
        precision4 = precision1

    import math

    bp = min(1.0, math.exp(1 - max(len(ref_tokens), 1) / max(len(gen_tokens), 1)))
    bleu1 = bp * precision1
    bleu4 = bp * precision4
    return {"bleu1": bleu1, "bleu4": bleu4}


def compute_rouge_l(generated: str, reference: str) -> float:
    gen_tokens = simple_tokenize(generated)
    ref_tokens = simple_tokenize(reference)
    if not gen_tokens or not ref_tokens:
        return 0.0

    m, n = len(gen_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gen_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    precision = lcs_len / max(m, 1)
    recall = lcs_len / max(n, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def generate_caption(
    patch_states: torch.Tensor,
    projector: VisionProjector,
    tokenizer: AutoTokenizer,
    decoder: AutoModelForCausalLM,
    budget: int = 64,
    max_new_tokens: int = 64,
) -> str:
    projector.eval()
    decoder.eval()

    with torch.no_grad():
        vision_prefix = projector(patch_states.unsqueeze(0))
        bos = tokenizer.bos_token_id or tokenizer.eos_token_id
        prefix_ids = torch.tensor([[bos]], dtype=torch.long)
        prefix_embeds = decoder.get_input_embeddings()(prefix_ids)

        combined = torch.cat([vision_prefix, prefix_embeds], dim=1)
        attention_mask = torch.ones(combined.shape[:2], dtype=torch.long)
        position_ids = torch.arange(combined.shape[1]).unsqueeze(0)

        generated_ids = []
        past = None

        for _ in range(max_new_tokens):
            if past is not None:
                model_out = decoder(
                    input_ids=prefix_ids[:, -1:],
                    past_key_values=past,
                    attention_mask=attention_mask,
                )
            else:
                model_out = decoder(
                    inputs_embeds=combined,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                past = model_out.past_key_values

            next_logits = model_out.logits[:, -1, :]
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids.append(next_id.item())

            prefix_ids = next_id
            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=torch.long)], dim=1
            )

            if next_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generation quality evaluation")
    parser.add_argument("--images-dir", default="bench/space")
    parser.add_argument("--cache", default="features-cache")
    parser.add_argument("--projector", default="phase1b-results/vision_projector.pt")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="eval-results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    images_dir = Path(args.images_dir)
    cache = FeaturesCache(args.cache)
    patches_list, ref_captions, image_names = load_pairs(images_dir, cache)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    decoder = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float32)
    decoder.eval()

    projector = VisionProjector(d_in=768, d_out=960, budget=args.budget)
    projector_path = Path(args.projector)
    if projector_path.exists():
        projector.load_state_dict(torch.load(projector_path, weights_only=True))
        log.info("loaded projector from %s", projector_path)
    else:
        log.warning("no projector found at %s, using random init", projector_path)

    results: list[dict] = []
    all_bleu1, all_bleu4, all_rouge = [], [], []

    for i, (patches, ref, name) in enumerate(zip(patches_list, ref_captions, image_names, strict=True)):
        log.info("generating caption for %s (%d/%d)", name, i + 1, len(image_names))
        generated = generate_caption(
            patches.float(), projector, tokenizer, decoder,
            budget=args.budget, max_new_tokens=args.max_new_tokens,
        )
        bleu = compute_bleu(generated, ref)
        rouge = compute_rouge_l(generated, ref)

        all_bleu1.append(bleu["bleu1"])
        all_bleu4.append(bleu["bleu4"])
        all_rouge.append(rouge)

        results.append({
            "image": name,
            "reference": ref,
            "generated": generated,
            "bleu1": bleu["bleu1"],
            "bleu4": bleu["bleu4"],
            "rouge_l": rouge,
        })
        log.info("  ref:  %s", ref)
        log.info("  gen:  %s", generated)
        log.info("  bleu1=%.3f bleu4=%.3f rouge_l=%.3f", bleu["bleu1"], bleu["bleu4"], rouge)

    summary = {
        "n_pairs": len(results),
        "mean_bleu1": sum(all_bleu1) / max(len(all_bleu1), 1),
        "mean_bleu4": sum(all_bleu4) / max(len(all_bleu4), 1),
        "mean_rouge_l": sum(all_rouge) / max(len(all_rouge), 1),
        "per_image": results,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    log.info("results written to %s", out_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
