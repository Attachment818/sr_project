"""Render an auditable SalC parameter sweep for existing registered images."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_salc import (
    load_config,
    load_font,
    load_image,
    reference_fov_crop_box,
    refuse_nonempty_output,
    source_path,
    validate_source,
)
from tools.salc_reference import (
    as_unit_gray,
    extract_salient_intensity,
    salience_overlay,
    score_salience,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def make_panel(image: np.ndarray, title: str, size: int, font_size: int) -> Image.Image:
    header = 66
    content = Image.fromarray(image).convert("RGB")
    content = ImageOps.contain(content, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + header), "white")
    canvas.paste(content, ((size - content.width) // 2, header + (size - content.height) // 2))
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size)
    box = draw.multiline_textbbox((0, 0), title, font=font, align="center", spacing=2)
    draw.multiline_text(
        ((size - (box[2] - box[0])) / 2.0, 7),
        title,
        fill="black",
        font=font,
        align="center",
        spacing=2,
    )
    return canvas


def validate_variant(index: int, raw: object) -> dict:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Variant {index} must be a mapping")
    required = ("name", "smoothing_radius", "salient_percent", "gamma", "opacity")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"Variant {index} is missing: {missing}")
    variant = {
        "name": str(raw["name"]),
        "smoothing_radius": float(raw["smoothing_radius"]),
        "salient_percent": float(raw["salient_percent"]),
        "gamma": float(raw["gamma"]),
        "opacity": float(raw["opacity"]),
    }
    if variant["smoothing_radius"] < 0:
        raise ValueError("smoothing_radius must be non-negative")
    if not 0 < variant["salient_percent"] <= 100:
        raise ValueError("salient_percent must be in (0, 100]")
    if variant["gamma"] <= 0 or not 0 < variant["opacity"] <= 1:
        raise ValueError("gamma must be positive and opacity must be in (0, 1]")
    return variant


def run(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("Sweep config must contain a YAML mapping")
    for field in ("audit_config", "output_dir", "pair_ids", "methods", "variants"):
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    audit_path = Path(str(config["audit_config"])).expanduser().resolve()
    audit = load_config(audit_path)
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    refuse_nonempty_output(output_dir)
    shutil.copy2(config_path, output_dir / "sweep.yaml")

    reference_source = validate_source("reference", audit["reference"])
    method_sources = {
        str(name): validate_source(str(name), source)
        for name, source in audit["methods"].items()
    }
    pair_ids = [str(value) for value in config["pair_ids"]]
    methods = [str(value) for value in config["methods"]]
    unknown = [method for method in methods if method not in method_sources]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    variants = [validate_variant(i, value) for i, value in enumerate(config["variants"])]
    if not pair_ids or not methods or not variants:
        raise ValueError("pair_ids, methods and variants must be non-empty")

    channel = str(audit.get("channel", "green"))
    legacy = bool(audit.get("legacy_zero_interpolation", True))
    panel_size = int(config.get("panel_size", 480))
    columns = int(config.get("columns", 4))
    font_size = int(config.get("font_size", 18))
    crop = bool(config.get("crop_to_reference_fov", True))
    margin_fraction = float(config.get("crop_margin_fraction", 0.02))
    rows_out: list[dict] = []

    progress = tqdm(
        total=len(pair_ids) * len(variants) * (len(methods) + 1),
        desc="Render HRF parameter sweep",
        unit="salience",
    )
    try:
        for pair_id in pair_ids:
            reference_raw = load_image(source_path(reference_source, pair_id))
            reference_gray = as_unit_gray(reference_raw, channel)
            box = reference_fov_crop_box(reference_gray, margin_fraction) if crop else None
            tiles: list[Image.Image] = []
            for variant in variants:
                radius = variant["smoothing_radius"]
                percent = variant["salient_percent"]
                reference_salience = extract_salient_intensity(
                    reference_gray,
                    smoothing_radius=radius,
                    salient_percent=percent,
                    legacy_zero_interpolation=False,
                )
                progress.update(1)
                for method in methods:
                    registered_raw = load_image(source_path(method_sources[method], pair_id))
                    registered_gray = as_unit_gray(registered_raw, channel)
                    if registered_gray.shape != reference_gray.shape:
                        raise ValueError(
                            f"{pair_id} {method} shape {registered_gray.shape} differs "
                            f"from reference {reference_gray.shape}"
                        )
                    registered_salience = extract_salient_intensity(
                        registered_gray,
                        smoothing_radius=radius,
                        salient_percent=percent,
                        legacy_zero_interpolation=legacy,
                    )
                    score = score_salience(reference_salience, registered_salience)
                    overlay = salience_overlay(
                        reference_salience,
                        registered_salience,
                        gamma=variant["gamma"],
                        opacity=variant["opacity"],
                    )
                    if box is not None:
                        left, top, right, bottom = box
                        overlay = overlay[top:bottom, left:right]
                    title = (
                        f"{method} | {variant['name']}\n"
                        f"r={radius:g}, p={percent:g}, SalC={score:.3f}"
                    )
                    tiles.append(make_panel(overlay, title, panel_size, font_size))
                    rows_out.append(
                        {
                            "pair_id": pair_id,
                            "method": method,
                            "variant": variant["name"],
                            "smoothing_radius": radius,
                            "salient_percent": percent,
                            "gamma": variant["gamma"],
                            "opacity": variant["opacity"],
                            "salc": score,
                        }
                    )
                    progress.update(1)

            row_count = math.ceil(len(tiles) / columns)
            sheet = Image.new(
                "RGB",
                (columns * panel_size, row_count * (panel_size + 66)),
                "white",
            )
            for index, tile in enumerate(tiles):
                sheet.paste(
                    tile,
                    ((index % columns) * panel_size, (index // columns) * (panel_size + 66)),
                )
            sheet.save(output_dir / f"pair_{pair_id}_parameter_sweep.png")
    finally:
        progress.close()

    with (output_dir / "parameter_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote SalC parameter sweep: {output_dir}")
    return output_dir


def self_test() -> None:
    image = np.zeros((20, 30), dtype=np.float64)
    image[4:16, 6:25] = 1.0
    assert reference_fov_crop_box(image, 0.0) == (6, 4, 25, 16)
    validate_variant(
        0,
        {
            "name": "protocol",
            "smoothing_radius": 25,
            "salient_percent": 10,
            "gamma": 0.65,
            "opacity": 0.72,
        },
    )
    print("SalC parameter-sweep self-test passed")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.config is not None:
        run(arguments.config)
    else:
        raise SystemExit("Provide --config or --self-test")
