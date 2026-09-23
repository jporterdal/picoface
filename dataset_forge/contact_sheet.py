"""A contact sheet for checking by eye that every class reads as itself.

    python -m dataset_forge.contact_sheet [--config PATH] [--seed N] [--out FILE]

One row per class, in config order, of freshly rendered images, each scaled
up with nearest-neighbour so individual pixels stay visible. Worth looking
at after adding a class or changing resolution, size, or stroke settings.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dataset_forge.config import ForgeConfig
from dataset_forge.export import DEFAULT_CONFIG, OUTPUT_DIR
from dataset_forge.render import render_image
from dataset_forge.shapes import check_class_names

_SCALE = 8
_GAP = 4
_LABEL_WIDTH = 120


def contact_sheet(config: ForgeConfig, seed: int = 0, per_class: int = 12) -> Image.Image:
    check_class_names(config.class_names)
    rng = np.random.default_rng(seed)
    cell_w, cell_h = config.width * _SCALE, config.height * _SCALE
    sheet = Image.new(
        "L",
        (_LABEL_WIDTH + per_class * (cell_w + _GAP), len(config.class_names) * (cell_h + _GAP)),
        255,
    )
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(config.class_names):
        top = row * (cell_h + _GAP)
        draw.text((4, top + cell_h // 2), name, fill=0)
        for col in range(per_class):
            image = Image.fromarray(render_image(rng, config, name)[..., 0], mode="L")
            left = _LABEL_WIDTH + col * (cell_w + _GAP)
            sheet.paste(image.resize((cell_w, cell_h), Image.Resampling.NEAREST), (left, top))
    return sheet


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "contact_sheet.png")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(ForgeConfig.load(args.config), seed=args.seed).save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
