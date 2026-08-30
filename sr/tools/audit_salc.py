"""Audit pre-registered HRF/OCTA results with the supplied SalC definition.

The tool is read-only with respect to datasets and registration results.  Every
run writes to a new, empty output directory and records the effective config,
per-pair scores, aggregate means, candidate ranking, and optional figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"SalC audit config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("SalC audit config must contain a YAML mapping")
    required = ("dataset_name", "output_dir", "pair_ids", "reference", "methods")
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


def validate_source(name: str, source: Mapping[str, object]) -> Tuple[Path, str]:
    if not isinstance(source, Mapping):
        raise ValueError(f"Source {name} must be a mapping")
    if "directory" not in source or "pattern" not in source:
        raise ValueError(f"Source {name} requires directory and pattern")
    directory = Path(str(source["directory"])).expanduser().resolve()
    pattern = str(source["pattern"])
    if not directory.is_dir():
        raise FileNotFoundError(f"Source directory not found for {name}: {directory}")
    if "{pair}" not in pattern:
        raise ValueError(f"Source pattern for {name} must contain {{pair}}")
    return directory, pattern


def source_path(source: Tuple[Path, str], pair_id: str) -> Path:
    directory, pattern = source
    return directory / pattern.format(pair=pair_id)


def load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    # OpenCV 4.6 on Windows cannot reliably open non-ASCII paths via imread.
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return image


def rgb_preview(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported preview shape: {image.shape}")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    if size <= 0:
        raise ValueError("font_size must be positive")
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(image: np.ndarray, title: str, size: int, font_size: int) -> Image.Image:
    header = 38
    canvas = Image.new("RGB", (size, size + header), "white")
    content = Image.fromarray(image).convert("RGB")
    content = ImageOps.contain(content, (size, size), Image.Resampling.LANCZOS)
    x = (size - content.width) // 2
    y = header + (size - content.height) // 2
    canvas.paste(content, (x, y))
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size)
    box = draw.textbbox((0, 0), title, font=font)
    draw.text(((size - (box[2] - box[0])) // 2, 12), title, fill="black", font=font)
    return canvas


def render_pair_figure(
    pair_id: str,
    reference_source: Tuple[Path, str],
    method_sources: Mapping[str, Tuple[Path, str]],
    figure_methods: Sequence[str],
    output_path: Path,
    *,
    channel: str,
    smoothing_radius: float,
    salient_percent: float,
    legacy_zero_interpolation: bool,
    panel_size: int,
    columns: int,
    overlay_opacity: float,
    font_size: int,
) -> None:
    reference_raw = load_image(source_path(reference_source, pair_id))
    reference_gray = as_unit_gray(reference_raw, channel)
    reference_salience = extract_salient_intensity(
        reference_gray,
        smoothing_radius=smoothing_radius,
        salient_percent=salient_percent,
        legacy_zero_interpolation=False,
    )
    source_raw = load_image(source_path(method_sources["Source"], pair_id))
    source_gray = as_unit_gray(source_raw, channel)
    source_salience = extract_salient_intensity(
        source_gray,
        smoothing_radius=smoothing_radius,
        salient_percent=salient_percent,
        legacy_zero_interpolation=legacy_zero_interpolation,
    )

    images: List[Tuple[str, np.ndarray]] = [
        ("Target image", rgb_preview(reference_raw)),
        ("Source image", rgb_preview(source_raw)),
        (
            "Target and source",
            salience_overlay(
                reference_salience,
                source_salience,
                opacity=overlay_opacity,
            ),
        ),
    ]
    for method in figure_methods:
        if method == "Source":
            continue
        raw = load_image(source_path(method_sources[method], pair_id))
        gray = as_unit_gray(raw, channel)
        if gray.shape != reference_gray.shape:
            raise ValueError(
                f"{pair_id} {method} shape {gray.shape} differs from "
                f"reference {reference_gray.shape}"
            )
        method_salience = extract_salient_intensity(
            gray,
            smoothing_radius=smoothing_radius,
            salient_percent=salient_percent,
            legacy_zero_interpolation=legacy_zero_interpolation,
        )
        images.append(
            (
                method,
                salience_overlay(
                    reference_salience,
                    method_salience,
                    opacity=overlay_opacity,
                ),
            )
        )

    rows = math.ceil(len(images) / columns)
    sheet = Image.new(
        "RGB",
        (columns * panel_size, rows * (panel_size + 38)),
        "white",
    )
    for index, (title, image) in enumerate(images):
        tile = panel(image, title, panel_size, font_size)
        sheet.paste(
            tile,
            ((index % columns) * panel_size, (index // columns) * (panel_size + 38)),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    refuse_nonempty_output(output_dir)
    shutil.copy2(config_path, output_dir / "audit.yaml")

    pair_ids = [str(value) for value in config["pair_ids"]]
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_ids must be non-empty and unique")
    reference_source = validate_source("reference", config["reference"])
    methods_raw = config["methods"]
    if not isinstance(methods_raw, Mapping) or not methods_raw:
        raise ValueError("methods must be a non-empty mapping")
    method_sources = {
        str(name): validate_source(str(name), source)
        for name, source in methods_raw.items()
    }
    if "Source" not in method_sources:
        raise ValueError("methods must include Source")

    channel = str(config.get("channel", "green"))
    smoothing_radius = float(config.get("smoothing_radius", 25.0))
    salient_percent = float(config.get("salient_percent", 10.0))
    legacy_zero_interpolation = bool(config.get("legacy_zero_interpolation", True))

    rows: List[dict] = []
    progress = tqdm(
        total=len(pair_ids) * (len(method_sources) + 1),
        desc=f"SalC {config['dataset_name']}",
        unit="image",
    )
    try:
        for pair_id in pair_ids:
            reference_path = source_path(reference_source, pair_id)
            reference_raw = load_image(reference_path)
            reference_gray = as_unit_gray(reference_raw, channel)
            reference_salience = extract_salient_intensity(
                reference_gray,
                smoothing_radius=smoothing_radius,
                salient_percent=salient_percent,
                legacy_zero_interpolation=False,
            )
            progress.update(1)

            for method, method_source in method_sources.items():
                registered_path = source_path(method_source, pair_id)
                registered_raw = load_image(registered_path)
                registered_gray = as_unit_gray(registered_raw, channel)
                if registered_gray.shape != reference_gray.shape:
                    raise ValueError(
                        f"{pair_id} {method} shape {registered_gray.shape} differs "
                        f"from reference {reference_gray.shape}"
                    )
                registered_salience = extract_salient_intensity(
                    registered_gray,
                    smoothing_radius=smoothing_radius,
                    salient_percent=salient_percent,
                    legacy_zero_interpolation=legacy_zero_interpolation,
                )
                score = score_salience(reference_salience, registered_salience)
                rows.append(
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "salc": score,
                        "reference_path": str(reference_path),
                        "registered_path": str(registered_path),
                    }
                )
                progress.update(1)
    finally:
        progress.close()

    write_csv(
        output_dir / "salc_scores.csv",
        rows,
        ("pair_id", "method", "salc", "reference_path", "registered_path"),
    )

    means: Dict[str, float] = {}
    method_values: Dict[str, List[float]] = {name: [] for name in method_sources}
    pair_scores: Dict[str, Dict[str, float]] = {pair_id: {} for pair_id in pair_ids}
    for row in rows:
        score = float(row["salc"])
        method_values[str(row["method"])].append(score)
        pair_scores[str(row["pair_id"])][str(row["method"])] = score
    for method, values in method_values.items():
        means[method] = float(np.nanmean(np.asarray(values, dtype=np.float64)))

    expected_raw = config.get("expected_means", {})
    expected = {str(name): float(value) for name, value in expected_raw.items()}
    tolerance = float(config.get("expected_mean_tolerance", 0.01))
    validation = {
        method: {
            "expected": expected[method],
            "actual": means.get(method),
            "absolute_difference": (
                None if method not in means else abs(means[method] - expected[method])
            ),
            "within_tolerance": (
                method in means and abs(means[method] - expected[method]) <= tolerance
            ),
        }
        for method in expected
    }

    primary_method = str(config.get("primary_method", "Ours"))
    if primary_method not in method_sources:
        raise ValueError(f"primary_method is not configured: {primary_method}")
    comparator_methods = [
        name for name in method_sources if name not in ("Source", primary_method)
    ]
    candidate_rows: List[dict] = []
    for pair_id in pair_ids:
        primary = pair_scores[pair_id][primary_method]
        best_method = ""
        best_score = float("nan")
        if comparator_methods:
            best_method = max(
                comparator_methods,
                key=lambda name: pair_scores[pair_id][name],
            )
            best_score = pair_scores[pair_id][best_method]
        else:
            best_method = "Source"
            best_score = pair_scores[pair_id]["Source"]
        candidate_rows.append(
            {
                "pair_id": pair_id,
                "primary_method": primary_method,
                "primary_salc": primary,
                "best_comparator": best_method,
                "best_comparator_salc": best_score,
                "primary_gap": primary - best_score,
                "source_salc": pair_scores[pair_id]["Source"],
            }
        )
    candidate_rows.sort(
        key=lambda row: (float(row["primary_gap"]), float(row["primary_salc"])),
        reverse=True,
    )
    for rank, row in enumerate(candidate_rows, start=1):
        row["rank"] = rank
    write_csv(
        output_dir / "candidate_ranking.csv",
        candidate_rows,
        (
            "rank",
            "pair_id",
            "primary_method",
            "primary_salc",
            "best_comparator",
            "best_comparator_salc",
            "primary_gap",
            "source_salc",
        ),
    )

    summary = {
        "dataset_name": str(config["dataset_name"]),
        "pair_count": len(pair_ids),
        "methods": list(method_sources),
        "parameters": {
            "channel": channel,
            "smoothing_radius": smoothing_radius,
            "salient_percent": salient_percent,
            "legacy_zero_interpolation": legacy_zero_interpolation,
        },
        "mean_salc": means,
        "expected_mean_tolerance": tolerance,
        "expected_mean_validation": validation,
    }
    with (output_dir / "salc_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    visualization = config.get("visualization", {})
    top_k = int(visualization.get("top_k", 0)) if visualization else 0
    if top_k > 0:
        figure_methods = [
            str(name)
            for name in visualization.get("methods", list(method_sources))
        ]
        unknown = [name for name in figure_methods if name not in method_sources]
        if unknown:
            raise ValueError(f"Unknown visualization methods: {unknown}")
        panel_size = int(visualization.get("panel_size", 420))
        columns = int(visualization.get("columns", 5))
        overlay_opacity = float(visualization.get("overlay_opacity", 0.55))
        font_size = int(visualization.get("font_size", 10))
        for row in tqdm(
            candidate_rows[:top_k],
            desc=f"Render {config['dataset_name']}",
            unit="pair",
        ):
            pair_id = str(row["pair_id"])
            render_pair_figure(
                pair_id,
                reference_source,
                method_sources,
                figure_methods,
                output_dir / "figures" / f"{pair_id}_salc_comparison.png",
                channel=channel,
                smoothing_radius=smoothing_radius,
                salient_percent=salient_percent,
                legacy_zero_interpolation=legacy_zero_interpolation,
                panel_size=panel_size,
                columns=columns,
                overlay_opacity=overlay_opacity,
                font_size=font_size,
            )

    print(f"Wrote SalC audit: {output_dir}")
    for method, mean_value in means.items():
        print(f"  {method}: {mean_value:.6f}")
    return output_dir


def self_test() -> None:
    assert source_path((Path("images"), "{pair}_good.png"), "01") == Path(
        "images/01_good.png"
    )
    try:
        validate_source("bad", {"directory": ".", "pattern": "fixed.png"})
    except ValueError as error:
        assert "{pair}" in str(error)
    else:
        raise AssertionError("Missing pair placeholder was not rejected")
    print("SalC audit self-test passed")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.config is not None:
        run(arguments.config)
    else:
        raise SystemExit("Provide --config or --self-test")
