#!/usr/bin/env python3
"""Generate a deterministic synthetic calibration set (images + captions.json).

Usage: python scripts/make_synthetic_calib.py [out_dir] [n_pairs] [seed]
Output: <out_dir>/*.jpg + out_dir/captions.json (filename -> caption).

Intended for CI smoke experiments when no offline image set is available:
the captions describe the same visual statistics that a vision encoder can
pick up (dominant color, geometry, brightness), so Procrustes/Ridge bridges
have learnable signal even on synthetic inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (40, 80, 220),
    "yellow": (230, 200, 40),
    "purple": (150, 60, 200),
    "teal": (40, 170, 170),
    "orange": (225, 130, 40),
    "pink": (240, 120, 160),
}

SHAPES = ["circle", "square", "triangle", "stripes"]


def _color_name(rgb: tuple[int, int, int]) -> str:
    for name, ref in COLORS.items():
        if all(abs(a - b) < 30 for a, b in zip(rgb, ref)):
            return name
    return "dark"


def render(rng, out_path: Path, size: int = 256) -> dict:
    color = COLORS[list(COLORS.keys())[rng.integers(0, len(COLORS))]]
    bg = tuple(min(255, c + rng.integers(-15, 15)) for c in (245, 245, 245))
    shape = SHAPES[rng.integers(0, len(SHAPES))]
    n_shapes = int(rng.integers(1, 4))
    brightness = rng.integers(1, 4)

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    for _ in range(n_shapes):
        cx = rng.integers(30, size - 30)
        cy = rng.integers(30, size - 30)
        r = rng.integers(20, 60)
        fill = tuple(min(255, int(c * (0.6 + 0.2 * brightness))) for c in color)
        if shape == "circle":
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
        elif shape == "square":
            draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill)
        elif shape == "triangle":
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
        else:  # stripes
            stripe = max(4, r // 6)
            for i in range(-size, size, stripe):
                draw.rectangle((i, 0, i + stripe, size), fill=fill)
    img.convert("RGB").save(out_path, "JPEG")
    return {
        "background": _color_name(bg),
        "shape": shape,
        "count": int(n_shapes),
        "brightness": int(brightness),
    }


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/calib")
    n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    rng = np.random.default_rng(seed)
    captions: dict[str, str] = {}
    for i in range(n_pairs):
        name = f"calib_{i:02d}.jpg"
        params = render(rng, out_dir / name)
        captions[name] = (
            f"a {params['shape']} ({_color_name(COLORS[list(COLORS)[i % len(COLORS)]]).lower()} object) "
            f"on a {params['background']} background, {params['count']} instance, "
            f"{'bright' if params['brightness'] > 2 else 'soft'} lighting"
        )
    (out_dir / "captions.json").write_text(json.dumps(captions, indent=2, sort_keys=True))
    print(f"wrote {n_pairs} synthetic pairs to {out_dir}")


if __name__ == "__main__":
    main()