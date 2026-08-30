"""Compose existing qualitative-result images into a reproducible panel figure."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Mapping

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Composition config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Composition config must contain a YAML mapping")
    required = ("output_dir", "panels")
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    return config


def refuse_nonempty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_panel(index: int, panel: object) -> tuple[str, Path]:
    if not isinstance(panel, Mapping):
        raise ValueError(f"Panel {index} must be a mapping")
    if "title" not in panel or "path" not in panel:
        raise ValueError(f"Panel {index} requires title and path")
    title = str(panel["title"]).strip()
    path = Path(str(panel["path"])).expanduser().resolve()
    if not title:
        raise ValueError(f"Panel {index} has an empty title")
    if not path.is_file():
        raise FileNotFoundError(f"Panel {index} image not found: {path}")
    return title, path


def load_font(size: int) -> ImageFont.ImageFont:
    if size <= 0:
        raise ValueError("font_size must be positive")
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_panel(
    image_path: Path,
    title: str,
    size: int,
    header: int,
    font_size: int,
) -> Image.Image:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + header), "white")
    x = (size - image.width) // 2
    y = header + (size - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size)
    box = draw.textbbox((0, 0), title, font=font)
    draw.text(((size - (box[2] - box[0])) // 2, 14), title, fill="black", font=font)
    return canvas


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    refuse_nonempty_output(output_dir)
    shutil.copy2(config_path, output_dir / "compose.yaml")

    panels_raw = config["panels"]
    if not isinstance(panels_raw, list) or not panels_raw:
        raise ValueError("panels must be a non-empty list")
    panels = [validate_panel(index, item) for index, item in enumerate(panels_raw)]
    columns = int(config.get("columns", len(panels)))
    panel_size = int(config.get("panel_size", 520))
    header = int(config.get("header_height", 48))
    font_size = int(config.get("font_size", 10))
    if columns <= 0 or panel_size <= 0 or header < 0:
        raise ValueError("columns and panel_size must be positive; header_height >= 0")
    output_filename = str(config.get("output_filename", "qualitative_comparison.png"))
    if Path(output_filename).name != output_filename:
        raise ValueError("output_filename must not contain a directory")

    rows = math.ceil(len(panels) / columns)
    sheet = Image.new(
        "RGB",
        (columns * panel_size, rows * (panel_size + header)),
        "white",
    )
    for index, (title, image_path) in enumerate(
        tqdm(panels, desc="Compose qualitative figure", unit="panel")
    ):
        tile = make_panel(image_path, title, panel_size, header, font_size)
        sheet.paste(
            tile,
            ((index % columns) * panel_size, (index // columns) * (panel_size + header)),
        )
    output_path = output_dir / output_filename
    sheet.save(output_path)
    print(f"Wrote qualitative comparison: {output_path}")
    return output_path


def self_test() -> None:
    try:
        load_config(Path("missing.yaml"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing composition config was not rejected")
    try:
        validate_panel(0, {"title": "x", "path": "missing.png"})
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing panel image was not rejected")
    print("Existing qualitative compositor self-test passed")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.config is not None:
        run(arguments.config)
    else:
        raise SystemExit("Provide --config or --self-test")
