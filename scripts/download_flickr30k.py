"""Download Flickr30k dataset from GitHub releases (no HuggingFace dependency).

Usage:
    python scripts/download_flickr30k.py --output bench/flickr30k --max-images 10

Downloads from awsaf49/flickr-dataset GitHub releases.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

FLICKR30K_URLS = [
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Flickr30k")
    parser.add_argument("--output", default="bench/flickr30k")
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "flickr30k-images"

    if images_dir.exists() and any(images_dir.glob("*.jpg")):
        print(f"images already exist at {images_dir}, skipping download")
    else:
        zip_path = output_dir / "flickr30k.zip"
        if not zip_path.exists():
            print("downloading flickr30k (3 parts)...")
            for i, url in enumerate(FLICKR30K_URLS):
                part_path = output_dir / f"flickr30k_part{i:02d}"
                print(f"  part {i}: {url}")
                subprocess.run(["wget", "-q", "-O", str(part_path), url], check=True)

            print("combining parts...")
            with open(zip_path, "wb") as out:
                for i in range(3):
                    part = output_dir / f"flickr30k_part{i:02d}"
                    with open(part, "rb") as inp:
                        out.write(inp.read())
                    part.unlink()

        print("extracting images...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".jpg")]
            print(f"  {len(names)} images in archive")
            for name in names[: args.max_images + 100]:
                zf.extract(name, str(output_dir))

        zip_path.unlink()

    # Find all jpg images
    all_images = sorted(images_dir.glob("*.jpg")) if images_dir.exists() else []
    selected = all_images[: args.max_images]
    print(f"selected {len(selected)} images")

    # Create dummy captions (we'll use the same caption for all during test)
    captions = {}
    for img in selected:
        captions[img.name] = [f"a photograph of {img.stem.replace('_', ' ')}"]

    captions_file = output_dir / "captions.json"
    captions_file.write_text(json.dumps(captions, indent=2))
    print(f"captions written to {captions_file}")

    # Also copy selected images to a simpler structure
    simple_images = output_dir / "images"
    simple_images.mkdir(exist_ok=True)
    import shutil

    for img in selected:
        shutil.copy2(img, simple_images / img.name)

    print(f"done: {len(selected)} images at {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
