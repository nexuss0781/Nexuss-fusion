"""Phase 2 experiment: fit + evaluate the vision bridge against caption anchors.

Fully CI-runable: features come from the content-addressed cache (never
recomputed), so a clean runner can extract once, then fit/eval many times.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from ..calibration import CalibrationBridge, split_by_key
from ..data import FeaturesCache
from ..eval import alignment_report
from ..extract import TextEmbedder, VisionExtractor
from ..extract.text import caption_cache_key

log = logging.getLogger("nexuss_fusion.phase2")

MIN_COS_ABOVE_ZERO = 0.15
MIN_COS_ABOVE_RANDOM = 0.10
REL_FRO_MAX = 0.60


def load_pairs(images_dir: Path, cache: FeaturesCache) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    captions: dict[str, str] = {}
    captions_file = images_dir / "captions.json"
    if captions_file.exists():
        captions = {p.name: str(v) for p, v in json.loads(captions_file.read_text()).items()}
    source_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    keys: list[str] = []
    vision_model, vision_rev = VisionExtractor().model_id, VisionExtractor().revision
    text_model, text_rev = TextEmbedder().model_id, TextEmbedder().revision
    for image_path in sorted(images_dir.rglob("*.jpg")):
        caption = captions.get(image_path.name)
        if not caption:
            log.warning("skipping %s: no caption", image_path)
            continue
        img_key = cache.key({"kind": "image", "model": vision_model, "revision": vision_rev, "image": str(image_path)})
        txt_key = caption_cache_key(cache, text_model, text_rev, caption)
        cached_img = cache.get(img_key)
        cached_txt = cache.get(txt_key)
        if cached_img is None or cached_txt is None:
            log.warning("skipping %s: missing cache entries", image_path)
            continue
        source_rows.append(cached_img)
        target_rows.append(cached_txt)
        keys.append(str(image_path))
    if not source_rows:
        raise SystemExit("no paired cached features found; run with --extract first")
    return CalibrationBridge.assemble(source_rows, target_rows) + (keys,)


def extract(images_dir: Path, cache: FeaturesCache) -> None:
    captions: dict[str, str] = {}
    captions_file = images_dir / "captions.json"
    if captions_file.exists():
        captions = json.loads(captions_file.read_text())
    vision = VisionExtractor()
    text = TextEmbedder()
    for image_path in sorted(images_dir.rglob("*.jpg")):
        img_key = cache.key(
            {"kind": "image", "model": vision.model_id, "revision": vision.revision, "image": str(image_path)}
        )
        if cache.has(img_key):
            continue
        vision.encode_image_batch_cached([str(image_path)], cache)
        caption = captions.get(image_path.name)
        if caption:
            text.embed_batch_cached([caption], cache)


def run_phase2(
    images_dir: Path,
    cache_path: Path,
    out_dir: Path,
    seed: int = 42,
    te: float = 0.2,
    lam: float = 1e-3,
) -> dict:
    log.info("loading pairs from cache %s", cache_path)
    cache = FeaturesCache(cache_path)
    src, tgt, keys = load_pairs(images_dir, cache)

    train_keys, test_keys = split_by_key(keys, te=te, seed=seed)
    train_idx = [keys.index(k) for k in train_keys]
    test_idx = [keys.index(k) for k in test_keys]
    log.info("split: %d train / %d test", len(train_idx), len(test_idx))

    bridge = CalibrationBridge.fit(src[train_idx], tgt[train_idx], lam=lam)
    report = alignment_report(src[test_idx], tgt[test_idx], bridge, seed=seed)

    gates = {
        "cosine_ridge_above_zero": report["cosine_ridge"] - report["cosine_zero"] >= MIN_COS_ABOVE_ZERO,
        "cosine_ridge_above_random": report["cosine_ridge"] - report["cosine_random"] >= MIN_COS_ABOVE_RANDOM,
        "rel_fro_below_max": report["rel_fro_ridge"] <= REL_FRO_MAX,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    CalibrationBridge.save(
        bridge,
        out_dir,
        meta={"seed": seed, "te": te, "lam": lam, "n_train": len(train_idx), "n_test": len(test_idx)},
    )
    result = {
        "status": "ok",
        "n_pairs": len(keys),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "metrics": {k: round(float(v), 5) for k, v in report.items()},
        "gates": gates,
    }
    (out_dir / "phase2.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log.info("phase2 result written to %s", out_dir / "phase2.json")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 vision-bridge experiment")
    parser.add_argument("--images-dir", default="benchmarks/vision/images")
    parser.add_argument("--cache", default="features-cache")
    parser.add_argument("--out", default="phase2-results")
    parser.add_argument("--extract", action="store_true", help="populate the feature cache from HF extractors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--te", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    images_dir = Path(args.images_dir)
    cache = FeaturesCache(args.cache)
    if args.extract:
        extract(images_dir, cache)

    result = run_phase2(images_dir, Path(args.cache), Path(args.out), seed=args.seed, te=args.te, lam=args.lam)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())