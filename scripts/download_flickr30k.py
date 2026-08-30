"""Download Flickr30k dataset from HuggingFace.

Usage:
    python scripts/download_flickr30k.py --output bench/flickr30k --max-images 10

Downloads images + captions from HuggingFace `nlphuji/flickr30k`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Flickr30k")
    parser.add_argument("--output", default="bench/flickr30k")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading flickr30k from huggingface (max {args.max_images} images)...")
    ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True)

    import random

    random.seed(args.seed)

    captions: dict[str, list[str]] = {}
    count = 0

    for example in ds:
        if count >= args.max_images:
            break

        image = example["image"]
        filename = example["filename"]
        caption_list = example["caption"]

        image_path = images_dir / filename
        image.save(image_path)
        captions[filename] = caption_list

        count += 1
        if count % 5 == 0:
            print(f"  downloaded {count}/{args.max_images} images")

    # Write captions
    captions_file = output_dir / "captions.json"
    captions_file.write_text(json.dumps(captions, indent=2))

    # Also write a simple captions.txt for compatibility
    simple_captions: dict[str, str] = {}
    for fname, caps in captions.items():
        simple_captions[fname] = caps[0] if caps else ""
    (output_dir / "captions_simple.json").write_text(json.dumps(simple_captions, indent=2))

    print(f"done: {count} images saved to {output_dir}")
    print(f"  images: {images_dir}")
    print(f"  captions: {captions_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
