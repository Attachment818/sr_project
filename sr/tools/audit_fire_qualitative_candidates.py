"""Audit FIRE pairs for a fair qualitative registration comparison.

This tool is deliberately read-only with respect to the copied experiment data.
It inventories pre-rendered registration results, measures conservative image-shape
warning signals, creates contact sheets, and ranks pairs using the available MLE
files plus an independently recomputed MLE for the two-stage ``ours`` result.

RetinaRegNet's historical FIRE_RRN_MLE.txt is intentionally not read because its
late-P labels are inconsistent.  RetinaRegNet is included visually through its
category-local ``case`` images (case index = FIRE number - 1).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


METHOD_ORDER = (
    "SIFT",
    "NCNet",
    "SuperPoint",
    "GeoFormer",
    "SuperRetina",
    "RetinaRegNet",
    "Ours",
)
SUPPORTED_IMAGE_METHODS = (*METHOD_ORDER, "LoFTR")

MLE_FILES = {
    "SIFT": ("SIFT/SIFT_MLE.txt",),
    "NCNet": ("ncnet/NCnet_MLE.txt",),
    "SuperPoint": ("SuperPoint/SuperPoint_MLE.txt",),
    "GeoFormer": ("GeoFormer/GeoFormer_geoformer_noms_errors.txt",),
    "SuperRetina": ("SuperRetina/SuperRetina_MLE.txt",),
    "LoFTR": ("LoFTR/LoFTR_MLE.txt",),
    "REMPE": ("REMPE-1.1.0/REMPE_MLE.txt",),
}

CSV_NAMES = (
    "pair_summary.csv",
    "image_shape_audit.csv",
    "candidate_ranking.csv",
)


@dataclass
class ShapeMetrics:
    valid_fraction: float
    bbox_aspect: float
    bbox_fill: float
    largest_component_ratio: float
    internal_hole_fraction: float
    centroid_offset: float
    warning: bool
    severe: bool
    reasons: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Audit config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Audit config must contain a YAML mapping")
    required = ("fire_root", "output_dir", "pairs")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    return config


def validate_pair_ids(values: Sequence[str]) -> List[str]:
    pairs: List[str] = []
    for value in values:
        pair_id = str(value).upper()
        if len(pair_id) != 3 or pair_id[0] not in "APS" or not pair_id[1:].isdigit():
            raise ValueError(f"Invalid FIRE pair id: {value}")
        if pair_id in {"P01", "P37"}:
            raise ValueError(f"Pair {pair_id} is explicitly excluded from this audit")
        pairs.append(pair_id)
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate FIRE pair ids in audit config")
    return pairs


def validate_methods(values: Sequence[str]) -> Tuple[str, ...]:
    methods = tuple(str(value) for value in values)
    if not methods:
        raise ValueError("At least one image method is required")
    unknown = [method for method in methods if method not in SUPPORTED_IMAGE_METHODS]
    if unknown:
        raise ValueError(f"Unsupported audit image methods: {unknown}")
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate image methods in audit config")
    return methods


def refuse_nonempty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_mle_file(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    if not path.is_file():
        return values
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            pair_id = fields[0].upper()
            try:
                value = float(fields[1])
            except ValueError:
                continue
            values[pair_id] = value
    return values


def displacement_metrics(fire_root: Path, pair_id: str) -> dict:
    """Summarize the native query-to-reference displacement of a FIRE pair."""
    ground_truth = read_numeric(
        fire_root / "Ground Truth" / f"control_points_{pair_id}_1_2.txt"
    ).reshape(-1, 4)
    displacement = ground_truth[:, :2] - ground_truth[:, 2:]
    mean_dx, mean_dy = displacement.mean(axis=0)
    return {
        "mean_displacement_x": float(mean_dx),
        "mean_displacement_y": float(mean_dy),
        "mean_displacement_magnitude": float(np.hypot(mean_dx, mean_dy)),
        "displacement_angle_degrees": float(
            np.degrees(np.arctan2(mean_dy, mean_dx))
        ),
        "vertical_to_horizontal_ratio": float(
            abs(mean_dy) / max(abs(mean_dx), 1e-6)
        ),
    }


def method_image_path(fire_root: Path, method: str, pair_id: str) -> Path:
    category = pair_id[0]
    number = int(pair_id[1:])
    if method == "SIFT":
        return fire_root / "SIFT" / "align_image" / f"{pair_id}_2_aligned.jpg"
    if method == "NCNet":
        return fire_root / "ncnet" / "align_image" / f"{pair_id}_2_aligned.jpg"
    if method == "SuperPoint":
        return fire_root / "SuperPoint" / "align_image" / f"{pair_id}_2_aligned.jpg"
    if method == "LoFTR":
        return fire_root / "LoFTR" / "align_image" / f"{pair_id}_2_aligned.jpg"
    if method == "GeoFormer":
        return (
            fire_root
            / "GeoFormer"
            / "warped_images"
            / f"control_points_{pair_id}_1_2.png"
        )
    if method == "SuperRetina":
        return (
            fire_root
            / "SuperRetina"
            / "align_image"
            / f"{pair_id}_2_aligned.jpg"
        )
    if method == "RetinaRegNet":
        case_index = number - 1
        return (
            fire_root
            / "RetinaRegNet"
            / "FIRE_Image_Registration_Results"
            / "Final_Registration_Results"
            / category
            / f"Final_Registration_Results_for_case{case_index}_show_result_.png"
        )
    if method == "Ours":
        return fire_root / "ours" / "job2" / "align_image" / f"{pair_id}_D2.jpg"
    raise KeyError(f"Unsupported method: {method}")


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def extract_retinaregnet_deformed(image: np.ndarray) -> np.ndarray:
    """Extract the right-hand deformed panel from a 3-column RRN figure."""
    height, width = image.shape[:2]
    # Exported figures are 1800x600.  Ratios keep the crop stable if rescaled.
    x0 = int(round(width * 1209 / 1800))
    x1 = int(round(width * 1621 / 1800))
    y0 = int(round(height * 98 / 600))
    y1 = int(round(height * 508 / 600))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid RetinaRegNet panel dimensions: {width}x{height}")
    return image[y0:y1, x0:x1]


def result_image(fire_root: Path, method: str, pair_id: str) -> Optional[np.ndarray]:
    path = method_image_path(fire_root, method, pair_id)
    if not path.is_file():
        return None
    image = load_rgb(path)
    if method == "RetinaRegNet":
        image = extract_retinaregnet_deformed(image)
    return image


def valid_mask(image: np.ndarray, size: int = 192) -> np.ndarray:
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    # FIRE images have a black background; tolerate JPEG ringing near the border.
    return np.max(resized, axis=2) >= 18


def shape_metrics(image: np.ndarray) -> ShapeMetrics:
    mask = valid_mask(image)
    height, width = mask.shape
    area = int(mask.sum())
    total = height * width
    if area == 0:
        return ShapeMetrics(0, 0, 0, 0, 1, 1, True, True, "empty_valid_mask")

    ys, xs = np.nonzero(mask)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_area = bbox_w * bbox_h
    aspect = bbox_w / max(bbox_h, 1)
    fill = area / max(bbox_area, 1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([0])
    largest_ratio = float(component_areas.max() / area)

    inverse = (~mask).astype(np.uint8)
    inv_count, inv_labels, inv_stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=4
    )
    hole_area = 0
    for index in range(1, inv_count):
        left, top, comp_w, comp_h, comp_area = inv_stats[index]
        touches_border = (
            left == 0
            or top == 0
            or left + comp_w == width
            or top + comp_h == height
        )
        if not touches_border:
            hole_area += int(comp_area)
    hole_fraction = hole_area / total

    cx = float(xs.mean()) / max(width - 1, 1)
    cy = float(ys.mean()) / max(height - 1, 1)
    centroid_offset = math.hypot(cx - 0.5, cy - 0.5) / math.sqrt(0.5)
    valid_fraction = area / total

    severe_reasons: List[str] = []
    warning_reasons: List[str] = []
    if valid_fraction < 0.06:
        severe_reasons.append("tiny_valid_area")
    elif valid_fraction < 0.14:
        warning_reasons.append("small_valid_area")
    if valid_fraction > 0.88:
        severe_reasons.append("near_full_canvas")
    elif valid_fraction > 0.72:
        warning_reasons.append("large_valid_area")
    if aspect < 0.32 or aspect > 3.1:
        severe_reasons.append("extreme_aspect")
    elif aspect < 0.62 or aspect > 1.62:
        warning_reasons.append("unusual_aspect")
    if largest_ratio < 0.72:
        severe_reasons.append("fragmented_mask")
    elif largest_ratio < 0.94:
        warning_reasons.append("minor_fragments")
    if hole_fraction > 0.20:
        severe_reasons.append("large_internal_holes")
    elif hole_fraction > 0.035:
        warning_reasons.append("internal_holes")
    if centroid_offset > 0.72:
        severe_reasons.append("far_off_center")
    elif centroid_offset > 0.43:
        warning_reasons.append("off_center")

    reasons = severe_reasons + warning_reasons
    return ShapeMetrics(
        valid_fraction=float(valid_fraction),
        bbox_aspect=float(aspect),
        bbox_fill=float(fill),
        largest_component_ratio=largest_ratio,
        internal_hole_fraction=float(hole_fraction),
        centroid_offset=float(centroid_offset),
        warning=bool(reasons),
        severe=bool(severe_reasons),
        reasons=";".join(reasons),
    )


def read_numeric(path: Path) -> np.ndarray:
    values = np.loadtxt(str(path), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    transformed = homogeneous @ matrix.T
    denominator = transformed[:, 2:3]
    if np.any(np.abs(denominator) < 1e-10):
        raise ValueError("Homography maps a point to infinity")
    return transformed[:, :2] / denominator


def apply_quadratic(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coefficients.size != 12:
        raise ValueError(f"Expected 12 quadratic coefficients, got {coefficients.size}")
    x = points[:, 0]
    y = points[:, 1]
    # The copied ICME experiment exports its order-2 coefficients as
    # [x, y, xy, x^2, y^2, 1] for each output coordinate.  This differs from
    # sklearn's common [1, x, y, x^2, xy, y^2] convention.
    basis = np.column_stack((x, y, x * y, x * x, y * y, np.ones_like(x)))
    return np.column_stack(
        (basis @ coefficients[:6], basis @ coefficients[6:])
    )


def apply_third_order(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coefficients.size != 20:
        raise ValueError(f"Expected 20 cubic coefficients, got {coefficients.size}")
    x = points[:, 0]
    y = points[:, 1]
    basis = np.column_stack(
        (
            x**3,
            x * x * y,
            x * y * y,
            y**3,
            x * x,
            x * y,
            y * y,
            x,
            y,
            np.ones_like(x),
        )
    )
    return np.column_stack((basis @ coefficients[:10], basis @ coefficients[10:]))


def invert_third_order(
    mapped_points: np.ndarray,
    coefficients: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """Invert RetinaRegNet's output-to-input image-warp polynomial."""
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coefficients.size != 20:
        raise ValueError(f"Expected 20 cubic coefficients, got {coefficients.size}")
    estimate = np.asarray(mapped_points, dtype=np.float64).copy()
    for _ in range(max_iterations):
        x = estimate[:, 0]
        y = estimate[:, 1]
        residual = apply_third_order(estimate, coefficients) - mapped_points
        jacobian = np.empty((len(estimate), 2, 2), dtype=np.float64)
        for output_index, values in enumerate(
            (coefficients[:10], coefficients[10:])
        ):
            jacobian[:, output_index, 0] = (
                3 * values[0] * x * x
                + 2 * values[1] * x * y
                + values[2] * y * y
                + 2 * values[4] * x
                + values[5] * y
                + values[7]
            )
            jacobian[:, output_index, 1] = (
                values[1] * x * x
                + 2 * values[2] * x * y
                + 3 * values[3] * y * y
                + values[5] * x
                + 2 * values[6] * y
                + values[8]
            )
        try:
            update = np.linalg.solve(jacobian, residual[..., None])[..., 0]
        except np.linalg.LinAlgError as error:
            raise ValueError("Singular RetinaRegNet polynomial Jacobian") from error
        estimate -= update
        if float(np.max(np.abs(update))) < tolerance:
            reconstruction = apply_third_order(estimate, coefficients)
            if float(np.max(np.abs(reconstruction - mapped_points))) > 1e-3:
                raise ValueError("RetinaRegNet polynomial inverse did not reconstruct")
            return estimate
    raise ValueError("RetinaRegNet polynomial inverse did not converge")


def compute_ours_mle(fire_root: Path, pair_id: str) -> float:
    gt_path = fire_root / "Ground Truth" / f"control_points_{pair_id}_1_2.txt"
    homography_path = fire_root / "ours" / "Homography1" / f"{pair_id}.txt"
    polynomial_path = fire_root / "ours" / "D2" / f"{pair_id}.txt"
    if not all(path.is_file() for path in (gt_path, homography_path, polynomial_path)):
        return math.nan
    gt = read_numeric(gt_path).reshape(-1, 4)
    reference = gt[:, :2]
    query = gt[:, 2:]
    stage1 = apply_homography(query, read_numeric(homography_path).reshape(3, 3))
    predicted = apply_quadratic(stage1, read_numeric(polynomial_path))
    return float(np.linalg.norm(predicted - reference, axis=1).mean())


def compute_retinaregnet_mle(fire_root: Path, pair_id: str) -> float:
    gt_path = fire_root / "Ground Truth" / f"control_points_{pair_id}_1_2.txt"
    homography_path = (
        fire_root
        / "RetinaRegNet"
        / "FIRE_Deformation"
        / "Homography"
        / f"{pair_id}_Homography.txt"
    )
    polynomial_path = (
        fire_root
        / "RetinaRegNet"
        / "FIRE_Deformation"
        / "Polynomial"
        / f"{pair_id[0]}{int(pair_id[1:])}_Polynomial.txt"
    )
    if not all(path.is_file() for path in (gt_path, homography_path, polynomial_path)):
        return math.nan
    gt = read_numeric(gt_path).reshape(-1, 4)
    reference = gt[:, :2]
    query = gt[:, 2:]
    stage1 = apply_homography(query, read_numeric(homography_path).reshape(3, 3))
    predicted = invert_third_order(stage1, read_numeric(polynomial_path))
    return float(np.linalg.norm(predicted - reference, axis=1).mean())


def make_tile(image: Optional[np.ndarray], label: str, size: int) -> Image.Image:
    header = 34
    tile = Image.new("RGB", (size, size + header), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((8, 9), label, fill="black", font=ImageFont.load_default())
    if image is None:
        draw.rectangle((0, header, size - 1, size + header - 1), fill=(28, 28, 28))
        draw.text((size // 2 - 24, header + size // 2), "MISSING", fill="white")
        return tile
    rendered = Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS)
    tile.paste(rendered, (0, header))
    return tile


def write_contact_sheet(
    fire_root: Path,
    pair_id: str,
    method_images: Dict[str, Optional[np.ndarray]],
    output_path: Path,
    tile_size: int,
    methods: Sequence[str],
) -> None:
    panels: List[Tuple[str, Optional[np.ndarray]]] = [
        ("Reference", load_rgb(fire_root / "Images" / f"{pair_id}_1.jpg")),
        ("Query", load_rgb(fire_root / "Images" / f"{pair_id}_2.jpg")),
    ]
    panels.extend((method, method_images[method]) for method in methods)
    columns = 3
    rows = math.ceil(len(panels) / columns)
    header = 34
    canvas = Image.new(
        "RGB", (columns * tile_size, rows * (tile_size + header)), "white"
    )
    for index, (label, image) in enumerate(panels):
        tile = make_tile(image, label, tile_size)
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + header)
        canvas.paste(tile, (x, y))
    canvas.save(output_path, quality=92)


def finite_median(values: Iterable[float]) -> float:
    data = [value for value in values if math.isfinite(value)]
    return float(np.median(data)) if data else math.nan


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(config: dict) -> Path:
    fire_root = Path(config["fire_root"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    pairs = validate_pair_ids(config["pairs"])
    methods = validate_methods(config.get("methods", METHOD_ORDER))
    tile_size = int(config.get("contact_sheet_tile_size", 300))
    if tile_size < 96 or tile_size > 1024:
        raise ValueError("contact_sheet_tile_size must be in [96, 1024]")
    for required in ("Images", "Ground Truth"):
        if not (fire_root / required).is_dir():
            raise FileNotFoundError(f"FIRE subdirectory not found: {fire_root / required}")
    refuse_nonempty_output(output_dir)
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir()

    mle_data: Dict[str, Dict[str, float]] = {}
    for method, candidates in MLE_FILES.items():
        selected = next((fire_root / item for item in candidates if (fire_root / item).is_file()), None)
        mle_data[method] = parse_mle_file(selected) if selected else {}

    pair_rows: List[dict] = []
    shape_rows: List[dict] = []
    ranking_rows: List[dict] = []
    for pair_id in tqdm(pairs, desc="FIRE qualitative audit", unit="pair"):
        method_images: Dict[str, Optional[np.ndarray]] = {}
        missing: List[str] = []
        severe_methods: List[str] = []
        warning_methods: List[str] = []
        for method in methods:
            path = method_image_path(fire_root, method, pair_id)
            image = result_image(fire_root, method, pair_id)
            method_images[method] = image
            if image is None:
                missing.append(method)
                shape_rows.append(
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "path": str(path),
                        "available": False,
                        "valid_fraction": "",
                        "bbox_aspect": "",
                        "bbox_fill": "",
                        "largest_component_ratio": "",
                        "internal_hole_fraction": "",
                        "centroid_offset": "",
                        "warning": True,
                        "severe": True,
                        "reasons": "missing",
                    }
                )
                continue
            metrics = shape_metrics(image)
            if metrics.severe:
                severe_methods.append(method)
            elif metrics.warning:
                warning_methods.append(method)
            row = {
                "pair_id": pair_id,
                "method": method,
                "path": str(path),
                "available": True,
            }
            row.update(asdict(metrics))
            shape_rows.append(row)

        make_contact = bool(config.get("write_all_contact_sheets", True)) or not missing
        if make_contact:
            write_contact_sheet(
                fire_root,
                pair_id,
                method_images,
                contact_dir / f"{pair_id}_methods.jpg",
                tile_size,
                methods,
            )

        ours_mle = compute_ours_mle(fire_root, pair_id)
        retinaregnet_mle = compute_retinaregnet_mle(fire_root, pair_id)
        comparator_values = {
            method: mle_data.get(method, {}).get(pair_id, math.nan)
            for method in MLE_FILES
        }
        comparator_median = finite_median(comparator_values.values())
        advantage_count = sum(
            math.isfinite(value) and math.isfinite(ours_mle) and ours_mle < value
            for value in comparator_values.values()
        )
        eligible = not missing and not severe_methods and math.isfinite(ours_mle)
        score = (
            min(comparator_median, 25.0) - min(ours_mle, 25.0)
            if eligible and math.isfinite(comparator_median)
            else -math.inf
        )
        pair_row = {
            "pair_id": pair_id,
            "complete": not missing,
            "missing_methods": ";".join(missing),
            "severe_shape_methods": ";".join(severe_methods),
            "warning_shape_methods": ";".join(warning_methods),
            "requires_manual_shape_review": bool(warning_methods),
            "eligible_before_manual_review": eligible,
            "ours_recomputed_mle": ours_mle,
            "RetinaRegNet_recomputed_mle": retinaregnet_mle,
            "comparator_median_mle": comparator_median,
            "ours_better_method_count": advantage_count,
            "ranking_score": score if math.isfinite(score) else "",
            "contact_sheet": str(contact_dir / f"{pair_id}_methods.jpg"),
            **displacement_metrics(fire_root, pair_id),
        }
        for method, value in comparator_values.items():
            pair_row[f"{method}_reported_mle"] = value
        pair_rows.append(pair_row)
        if eligible:
            ranking_rows.append(dict(pair_row))

    ranking_rows.sort(
        key=lambda row: (
            int(row["requires_manual_shape_review"]),
            -int(row["ours_better_method_count"]),
            -float(row["ranking_score"]),
            float(row["ours_recomputed_mle"]),
        )
    )
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = rank

    pair_fields = [
        "pair_id",
        "complete",
        "missing_methods",
        "severe_shape_methods",
        "warning_shape_methods",
        "requires_manual_shape_review",
        "eligible_before_manual_review",
        "ours_recomputed_mle",
        "RetinaRegNet_recomputed_mle",
        "comparator_median_mle",
        "ours_better_method_count",
        "ranking_score",
        "mean_displacement_x",
        "mean_displacement_y",
        "mean_displacement_magnitude",
        "displacement_angle_degrees",
        "vertical_to_horizontal_ratio",
        *[f"{method}_reported_mle" for method in MLE_FILES],
        "contact_sheet",
    ]
    shape_fields = [
        "pair_id",
        "method",
        "path",
        "available",
        "valid_fraction",
        "bbox_aspect",
        "bbox_fill",
        "largest_component_ratio",
        "internal_hole_fraction",
        "centroid_offset",
        "warning",
        "severe",
        "reasons",
    ]
    ranking_fields = ["rank", *pair_fields]
    write_csv(output_dir / CSV_NAMES[0], pair_rows, pair_fields)
    write_csv(output_dir / CSV_NAMES[1], shape_rows, shape_fields)
    write_csv(output_dir / CSV_NAMES[2], ranking_rows, ranking_fields)

    summary = {
        "fire_root": str(fire_root),
        "output_dir": str(output_dir),
        "pairs_requested": pairs,
        "retinaregnet_mle_file_used": False,
        "retinaregnet_case_rule": "case_index = numeric FIRE pair id - 1",
        "pair_count": len(pair_rows),
        "complete_pair_count": sum(bool(row["complete"]) for row in pair_rows),
        "auto_eligible_count": len(ranking_rows),
        "manual_review_required": True,
        "top_candidates_before_manual_review": [
            row["pair_id"] for row in ranking_rows[:10]
        ],
    }
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    shutil.copy2(Path(config["_config_path"]), output_dir / "audit.yaml")
    print(f"Wrote FIRE qualitative audit: {output_dir}")
    print(
        "RetinaRegNet FIRE_RRN_MLE.txt was intentionally ignored; "
        "manual contact-sheet review is still required."
    )
    return output_dir


def self_test() -> None:
    points = np.array([[1.0, 2.0], [4.0, 8.0]])
    assert np.allclose(apply_homography(points, np.eye(3)), points)
    identity_poly = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    assert np.allclose(apply_quadratic(points, identity_poly), points)
    xy_poly = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    assert np.allclose(apply_quadratic(points, xy_poly)[:, 0], points.prod(axis=1))
    identity_cubic = np.zeros(20, dtype=np.float64)
    identity_cubic[7] = 1
    identity_cubic[18] = 1
    assert np.allclose(apply_third_order(points, identity_cubic), points)
    assert np.allclose(invert_third_order(points, identity_cubic), points)

    assert method_image_path(Path("FIRE"), "RetinaRegNet", "P38").name == (
        "Final_Registration_Results_for_case37_show_result_.png"
    )
    assert method_image_path(Path("FIRE"), "RetinaRegNet", "A01").name == (
        "Final_Registration_Results_for_case0_show_result_.png"
    )
    assert validate_methods(["SIFT", "LoFTR"]) == ("SIFT", "LoFTR")

    normal = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(normal, (128, 128), 92, (180, 70, 40), thickness=-1)
    assert not shape_metrics(normal).severe
    tiny = np.zeros_like(normal)
    cv2.circle(tiny, (128, 128), 18, (180, 70, 40), thickness=-1)
    assert shape_metrics(tiny).severe

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
    print("FIRE qualitative audit self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_config is None:
        raise ValueError("--audit-config is required unless --self-test is used")
    config = load_config(args.audit_config)
    config["_config_path"] = str(args.audit_config.resolve())
    run_audit(config)


if __name__ == "__main__":
    main()
