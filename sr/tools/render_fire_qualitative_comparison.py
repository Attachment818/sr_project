"""Render uniform FIRE qualitative comparison figures from copied results.

The renderer never modifies the source experiment folders.  It uses the existing
aligned images, recomputes transformed FIRE control points from each method's
saved transformation, and produces identical blend/marker styling for every
panel.  RetinaRegNet's historical MLE text file is not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_fire_qualitative_candidates import (
    METHOD_ORDER,
    apply_homography,
    apply_quadratic,
    invert_third_order,
    load_rgb,
    read_numeric,
    result_image,
    validate_pair_ids,
)


PANEL_ORDER = ("Target and Source", *METHOD_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Render config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Render config must contain a YAML mapping")
    for field in ("fire_root", "output_dir", "pairs"):
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    return config


def refuse_nonempty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def transform_points(fire_root: Path, pair_id: str, method: str) -> np.ndarray:
    gt = read_numeric(
        fire_root / "Ground Truth" / f"control_points_{pair_id}_1_2.txt"
    ).reshape(-1, 4)
    query = gt[:, 2:]
    if method == "Target and Source":
        return query.copy()
    if method in {"SIFT", "NCNet", "SuperPoint"}:
        folder = {"SIFT": "SIFT", "NCNet": "ncnet", "SuperPoint": "SuperPoint"}[
            method
        ]
        matrix = read_numeric(
            fire_root / folder / "Homography" / f"{pair_id}_Homography.txt"
        ).reshape(3, 3)
        return apply_homography(query, matrix)
    if method == "GeoFormer":
        matrix = read_numeric(
            fire_root
            / "GeoFormer"
            / "H_matrices"
            / f"control_points_{pair_id}_1_2.txt"
        ).reshape(3, 3)
        return apply_homography(query, matrix)
    if method == "SuperRetina":
        first = read_numeric(
            fire_root
            / "SuperRetina"
            / "Homography1"
            / f"{pair_id}_Homography.txt"
        ).reshape(3, 3)
        second = read_numeric(
            fire_root
            / "SuperRetina"
            / "Homography2"
            / f"{pair_id}_Homography.txt"
        ).reshape(3, 3)
        return apply_homography(apply_homography(query, first), second)
    if method == "RetinaRegNet":
        number = int(pair_id[1:])
        homography = read_numeric(
            fire_root
            / "RetinaRegNet"
            / "FIRE_Deformation"
            / "Homography"
            / f"{pair_id}_Homography.txt"
        ).reshape(3, 3)
        polynomial = read_numeric(
            fire_root
            / "RetinaRegNet"
            / "FIRE_Deformation"
            / "Polynomial"
            / f"{pair_id[0]}{number}_Polynomial.txt"
        )
        stage1 = apply_homography(query, homography)
        return invert_third_order(stage1, polynomial)
    if method == "Ours":
        homography = read_numeric(
            fire_root / "ours" / "Homography1" / f"{pair_id}.txt"
        ).reshape(3, 3)
        polynomial = read_numeric(fire_root / "ours" / "D2" / f"{pair_id}.txt")
        return apply_quadratic(apply_homography(query, homography), polynomial)
    raise KeyError(f"Unsupported method: {method}")


def aligned_image(fire_root: Path, pair_id: str, method: str) -> np.ndarray:
    if method == "Target and Source":
        return load_rgb(fire_root / "Images" / f"{pair_id}_2.jpg")
    image = result_image(fire_root, method, pair_id)
    if image is None:
        raise FileNotFoundError(f"Aligned image missing for {pair_id}/{method}")
    return image


def blend_valid_regions(
    reference: np.ndarray,
    aligned: np.ndarray,
    alpha: float,
    moving_brightness_scale: float = 1.0,
) -> np.ndarray:
    if aligned.shape[:2] != reference.shape[:2]:
        aligned = cv2.resize(
            aligned,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    reference_f = reference.astype(np.float32)
    aligned_f = aligned.astype(np.float32) * moving_brightness_scale
    ref_valid = np.max(reference, axis=2) >= 18
    aligned_valid = np.max(aligned, axis=2) >= 18
    both = ref_valid & aligned_valid
    output = np.zeros_like(reference_f)
    output[ref_valid] = reference_f[ref_valid]
    output[aligned_valid & ~ref_valid] = aligned_f[aligned_valid & ~ref_valid]
    output[both] = (1.0 - alpha) * reference_f[both] + alpha * aligned_f[both]
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def draw_control_points(
    image: np.ndarray,
    reference_points: np.ndarray,
    transformed_points: np.ndarray,
    radius: int,
    line_width: int,
) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    height, width = image.shape[:2]
    for points, color in ((reference_points, (0, 255, 0)), (transformed_points, (255, 0, 0))):
        for x, y in points:
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if x < -radius or y < -radius or x >= width + radius or y >= height + radius:
                continue
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color,
                width=line_width,
            )
    return np.asarray(canvas)


def labelled_panel(image: np.ndarray, title: str, panel_size: int) -> Image.Image:
    header = 54
    panel = Image.new("RGB", (panel_size, panel_size + header), "white")
    rendered = Image.fromarray(image).resize(
        (panel_size, panel_size), Image.Resampling.LANCZOS
    )
    panel.paste(rendered, (0, header))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    text_box = draw.textbbox((0, 0), title, font=font)
    text_width = text_box[2] - text_box[0]
    draw.text(((panel_size - text_width) / 2, 14), title, fill="black", font=font)
    return panel


def render_pair(
    fire_root: Path,
    pair_id: str,
    pair_dir: Path,
    alpha: float,
    panel_size: int,
    radius: int,
    line_width: int,
    moving_brightness_scale: float,
) -> List[dict]:
    pair_dir.mkdir(parents=True)
    individual_dir = pair_dir / "individual_panels"
    individual_dir.mkdir()
    reference = load_rgb(fire_root / "Images" / f"{pair_id}_1.jpg")
    gt = read_numeric(
        fire_root / "Ground Truth" / f"control_points_{pair_id}_1_2.txt"
    ).reshape(-1, 4)
    reference_points = gt[:, :2]
    panels: List[Image.Image] = []
    metrics: List[dict] = []
    for index, method in enumerate(PANEL_ORDER):
        warped = aligned_image(fire_root, pair_id, method)
        predicted = transform_points(fire_root, pair_id, method)
        blended = blend_valid_regions(
            reference, warped, alpha, moving_brightness_scale
        )
        marked = draw_control_points(
            blended, reference_points, predicted, radius, line_width
        )
        title = f"({chr(ord('a') + index)}) {method}"
        panel = labelled_panel(marked, title, panel_size)
        panels.append(panel)
        safe_name = method.lower().replace(" ", "_")
        panel.save(individual_dir / f"{index + 1:02d}_{safe_name}.png")
        errors = np.linalg.norm(predicted - reference_points, axis=1)
        metrics.append(
            {
                "pair_id": pair_id,
                "method": method,
                "mle": float(errors.mean()),
                "max_error": float(errors.max()),
                "points_under_25": int((errors < 25).sum()),
                "points_in_canvas": int(
                    (
                        (predicted[:, 0] >= 0)
                        & (predicted[:, 0] < reference.shape[1])
                        & (predicted[:, 1] >= 0)
                        & (predicted[:, 1] < reference.shape[0])
                    ).sum()
                ),
            }
        )

    columns = 4
    rows = 2
    canvas = Image.new(
        "RGB", (columns * panel_size, rows * (panel_size + 54)), "white"
    )
    for index, panel in enumerate(panels):
        canvas.paste(
            panel,
            ((index % columns) * panel_size, (index // columns) * (panel_size + 54)),
        )
    canvas.save(pair_dir / f"{pair_id}_comparison_2x4.png")
    return metrics


def run(config: dict) -> Path:
    fire_root = Path(config["fire_root"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    pairs = validate_pair_ids(config["pairs"])
    alpha = float(config.get("blend_alpha", 0.5))
    panel_size = int(config.get("panel_size", 700))
    radius = int(config.get("control_point_radius", 34))
    line_width = int(config.get("control_point_line_width", 8))
    moving_brightness_scale = float(config.get("moving_brightness_scale", 1.0))
    if not 0 <= alpha <= 1:
        raise ValueError("blend_alpha must be in [0, 1]")
    if panel_size < 256 or panel_size > 1600:
        raise ValueError("panel_size must be in [256, 1600]")
    if not 0 < moving_brightness_scale <= 1:
        raise ValueError("moving_brightness_scale must be in (0, 1]")
    refuse_nonempty_output(output_dir)

    all_metrics: List[dict] = []
    for pair_id in tqdm(pairs, desc="Render FIRE comparisons", unit="pair"):
        all_metrics.extend(
            render_pair(
                fire_root,
                pair_id,
                output_dir / pair_id,
                alpha,
                panel_size,
                radius,
                line_width,
                moving_brightness_scale,
            )
        )
    fields = ("pair_id", "method", "mle", "max_error", "points_under_25", "points_in_canvas")
    with (output_dir / "candidate_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_metrics)
    metadata = {
        "fire_root": str(fire_root),
        "pairs": pairs,
        "panel_order": PANEL_ORDER,
        "blend_alpha": alpha,
        "moving_brightness_scale": moving_brightness_scale,
        "control_point_colors": {
            "reference": "green",
            "transformed_query": "red",
        },
        "retinaregnet_historical_mle_used": False,
        "retinaregnet_polynomial_direction": "saved output-to-input warp inverted for landmark propagation",
    }
    with (output_dir / "render_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    shutil.copy2(Path(config["_config_path"]), output_dir / "render.yaml")
    print(f"Wrote FIRE qualitative comparisons: {output_dir}")
    return output_dir


def self_test() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    reference = image.copy()
    aligned = image.copy()
    reference[8:40, 8:40] = (100, 20, 10)
    aligned[24:56, 24:56] = (20, 100, 10)
    blended = blend_valid_regions(reference, aligned, 0.5)
    assert tuple(blended[12, 12]) == (100, 20, 10)
    assert tuple(blended[48, 48]) == (20, 100, 10)
    assert tuple(blended[28, 28]) == (60, 60, 10)
    darkened = blend_valid_regions(reference, aligned, 0.5, 0.5)
    assert tuple(darkened[12, 12]) == (100, 20, 10)
    assert tuple(darkened[48, 48]) == (10, 50, 5)
    assert tuple(darkened[28, 28]) == (55, 35, 8)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "new"
        refuse_nonempty_output(output)
        (output / "occupied.txt").write_text("x", encoding="utf-8")
        try:
            refuse_nonempty_output(output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Non-empty output directory was not rejected")
    print("FIRE qualitative renderer self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.render_config is None:
        raise ValueError("--render-config is required unless --self-test is used")
    config = load_config(args.render_config)
    config["_config_path"] = str(args.render_config.resolve())
    run(config)


if __name__ == "__main__":
    main()
