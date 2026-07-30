"""Read-only detector/descriptor stress audit under acquisition-like changes.

Each source image is matched to an appearance-transformed copy of itself.  No
geometry is changed, so spatially coincident detections are known positives and
the identity mapping is the expected registration.  The transformations model
brightness, response-curve, blur, local-contrast, and low-frequency illumination
changes; they deliberately do not simulate retinal lesions.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.common_util import pre_processing
from common.eval_util import list_fimd_pairs
from predictor import Predictor

OUTPUT_NAMES = ("appearance_stability.csv", "appearance_stability.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def apply_appearance(image, spec):
    """Apply a deterministic, geometry-preserving transform to uint8 grayscale."""
    kind = str(spec["type"]).lower()
    image_f = image.astype(np.float32) / 255.0
    if kind == "identity":
        result = image_f
    elif kind == "gamma":
        gamma = float(spec["gamma"])
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        result = np.power(image_f, gamma)
    elif kind == "brightness_contrast":
        result = image_f * float(spec.get("contrast", 1.0))
        result += float(spec.get("brightness", 0.0))
    elif kind == "gaussian_blur":
        sigma = float(spec["sigma"])
        if sigma <= 0:
            raise ValueError("gaussian blur sigma must be positive")
        result = cv2.GaussianBlur(image_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif kind == "clahe":
        clip = float(spec.get("clip_limit", 2.0))
        tiles = int(spec.get("tile_grid_size", 8))
        if clip <= 0 or tiles < 2:
            raise ValueError("CLAHE parameters must be positive and tiles >= 2")
        result = cv2.createCLAHE(
            clipLimit=clip, tileGridSize=(tiles, tiles)
        ).apply(image).astype(np.float32) / 255.0
    elif kind == "low_frequency_shading":
        amplitude = float(spec["amplitude"])
        if not 0 <= amplitude < 1:
            raise ValueError("shading amplitude must be in [0, 1)")
        h, w = image.shape
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        angle = np.deg2rad(float(spec.get("angle_degrees", 30.0)))
        field = np.cos(angle) * xx + np.sin(angle) * yy
        result = image_f * (1.0 + amplitude * field)
    else:
        raise ValueError(f"Unsupported appearance transform: {kind}")
    return np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8)


def cv_keypoints(points, predictor, shape):
    h, w = shape
    return [
        cv2.KeyPoint(
            int(point[0] / predictor.model_image_width * w),
            int(point[1] / predictor.model_image_height * h),
            30,
        )
        for point in points
    ]


def run_pair(predictor, original, transformed):
    """Return keypoints/descriptors for one identity-geometry image pair."""
    original_input = (pre_processing(original) * 255).astype(np.uint8)
    transformed_input = (pre_processing(transformed) * 255).astype(np.uint8)
    tensors = [
        predictor.trasformer(Image.fromarray(value))
        for value in (original_input, transformed_input)
    ]
    keypoints, descriptors = predictor.model_run_pair(*tensors)
    keypoints = [
        cv_keypoints(points, predictor, original.shape)
        for points in keypoints
    ]
    descriptors = [
        value.permute(1, 0).numpy().astype(np.float32, copy=False)
        for value in descriptors
    ]
    return keypoints, descriptors


def ratio_matches(matcher, first, second, threshold):
    if len(first) == 0 or len(second) < 2:
        return []
    try:
        pairs = matcher.knnMatch(first, second, k=2)
    except cv2.error:
        return []
    return [m for m, n in pairs if m.distance < threshold * n.distance]


def spatial_positive_pairs(first, second, tolerance):
    """Greedily form unique spatial positives under identity geometry."""
    if not first or not second:
        return []
    a = np.asarray([point.pt for point in first], dtype=np.float32)
    b = np.asarray([point.pt for point in second], dtype=np.float32)
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    candidates = np.argwhere(distances <= tolerance)
    ranked = sorted(
        ((float(distances[i, j]), int(i), int(j)) for i, j in candidates),
        key=lambda item: item[0],
    )
    used_a, used_b, selected = set(), set(), []
    for distance, index_a, index_b in ranked:
        if index_a not in used_a and index_b not in used_b:
            selected.append((index_a, index_b, distance))
            used_a.add(index_a)
            used_b.add(index_b)
    return selected


def grid_metrics(points, shape, grid_size):
    if not points:
        return 0, 0.0, 0.0
    h, w = shape
    cells = {
        (
            min(grid_size - 1, max(0, int(point[1] * grid_size / h))),
            min(grid_size - 1, max(0, int(point[0] * grid_size / w))),
        )
        for point in points
    }
    hull = 0.0
    if len(points) >= 3:
        contour = cv2.convexHull(
            np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        )
        hull = float(cv2.contourArea(contour) / max(1.0, float(h * w)))
    return len(cells), len(cells) / float(grid_size * grid_size), hull


def descriptor_separation(desc_a, desc_b, keypoints_b, positives, min_negative):
    positive_distances, negative_distances = [], []
    if len(desc_b) == 0:
        return None, None
    coords_b = np.asarray([point.pt for point in keypoints_b], dtype=np.float32)
    for index_a, index_b, _ in positives:
        positive_distances.append(
            float(np.linalg.norm(desc_a[index_a] - desc_b[index_b]))
        )
        spatial = np.linalg.norm(coords_b - coords_b[index_b], axis=1)
        eligible = np.flatnonzero(spatial >= min_negative)
        if eligible.size:
            distances = np.linalg.norm(
                desc_b[eligible] - desc_a[index_a][None, :], axis=1
            )
            negative_distances.append(float(distances.min()))
    return (
        float(np.mean(positive_distances)) if positive_distances else None,
        float(np.mean(negative_distances)) if negative_distances else None,
    )


def ransac_inliers(matches, keypoints_a, keypoints_b, threshold):
    if len(matches) < 4:
        return np.zeros(len(matches), dtype=bool)
    source = np.float32(
        [keypoints_a[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    target = np.float32(
        [keypoints_b[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, threshold)
    return (
        np.zeros(len(matches), dtype=bool)
        if mask is None else mask.ravel().astype(bool)
    )


def self_test():
    image = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    assert np.array_equal(image, apply_appearance(image, {"type": "identity"}))
    for spec in (
        {"type": "gamma", "gamma": 0.8},
        {"type": "brightness_contrast", "brightness": 0.1, "contrast": 0.9},
        {"type": "gaussian_blur", "sigma": 1.0},
        {"type": "clahe", "clip_limit": 2.0, "tile_grid_size": 2},
        {"type": "low_frequency_shading", "amplitude": 0.2},
    ):
        value = apply_appearance(image, spec)
        assert value.shape == image.shape and value.dtype == np.uint8
    first = [cv2.KeyPoint(2, 2, 1), cv2.KeyPoint(20, 20, 1)]
    second = [cv2.KeyPoint(2.5, 2, 1), cv2.KeyPoint(25, 25, 1)]
    pairs = spatial_positive_pairs(first, second, 1.0)
    assert len(pairs) == 1 and pairs[0][:2] == (0, 0)
    cells, coverage, hull = grid_metrics([(1, 1), (7, 7)], (8, 8), 2)
    assert cells == 2 and coverage == 0.5 and hull == 0.0
    print("appearance-stability audit self-test passed")


def validate_and_load(audit):
    dataset_root = Path(audit["dataset_root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"FIMD root not found: {dataset_root}")
    output_dir = Path(audit["output_dir"])
    occupied = [
        output_dir / name for name in OUTPUT_NAMES
        if (output_dir / name).exists()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite D26 result(s): "
            + ", ".join(map(str, occupied))
        )
    sources = list(audit.get("sources", []))
    if len(sources) < 2:
        raise ValueError("At least two model sources are required")
    for source in sources:
        for field in ("label", "test_config_path", "checkpoint_path"):
            if field not in source:
                raise KeyError(f"Source is missing {field}: {source}")
            if not Path(source[field]).is_file():
                raise FileNotFoundError(f"Source {field} not found: {source[field]}")
    transforms = list(audit.get("transforms", []))
    if not transforms or transforms[0].get("type") != "identity":
        raise ValueError("The first transform must be an identity baseline")
    names = [str(item["name"]) for item in transforms]
    if len(names) != len(set(names)):
        raise ValueError("Transform names must be unique")
    return dataset_root, output_dir, sources, transforms


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_config is None:
        raise ValueError("--audit-config is required unless --self-test is used")
    audit = yaml.safe_load(
        args.audit_config.read_text(encoding="utf-8")
    )["AUDIT"]
    dataset_root, output_dir, sources, transforms = validate_and_load(audit)
    available = {
        item["pair_name"]: item for item in list_fimd_pairs(str(dataset_root))
    }
    requested = list(audit.get("pairs", []))
    missing = [name for name in requested if name not in available]
    if missing:
        raise KeyError(f"Unknown FIMD pairs: {missing}")
    items = [available[name] for name in requested]
    if not items:
        raise ValueError("At least one explicit FIMD pair is required")

    device = str(audit["device"])
    grid_size = int(audit.get("grid_size", 8))
    tolerance = float(audit.get("repeatability_tolerance", 4.0))
    min_negative = float(audit.get("min_negative_distance", 16.0))
    ransac_threshold = float(audit.get("ransac_threshold", 3.0))
    border_margin = int(audit.get("outer_border_margin", 48))
    if grid_size < 2 or tolerance <= 0 or min_negative <= tolerance:
        raise ValueError("Invalid grid/repeatability/negative-distance settings")

    rows = []
    total = len(sources) * len(items) * len(transforms)
    progress = tqdm(total=total, desc="D26 appearance stability", unit="case")
    try:
        for source in sources:
            config = yaml.safe_load(
                Path(source["test_config_path"]).read_text(encoding="utf-8")
            )
            config["PREDICT"]["model_save_path"] = source["checkpoint_path"]
            config["PREDICT"]["device"] = device
            predictor = Predictor(config)
            predictor.set_eye_mask(None)
            for item in items:
                raw = cv2.imread(item["query_im_path"], cv2.IMREAD_COLOR)
                if raw is None:
                    raise FileNotFoundError(item["query_im_path"])
                original = raw[:, :, 1]
                for transform in transforms:
                    transformed = apply_appearance(original, transform)
                    keypoints, descriptors = run_pair(
                        predictor, original, transformed
                    )
                    first_kp, second_kp = keypoints
                    first_desc, second_desc = descriptors
                    positives = spatial_positive_pairs(
                        first_kp, second_kp, tolerance
                    )
                    positive_distance, negative_distance = descriptor_separation(
                        first_desc, second_desc, second_kp, positives, min_negative
                    )
                    ratio = ratio_matches(
                        predictor.knn_matcher, first_desc, second_desc,
                        predictor.knn_thresh,
                    )
                    mutual, _, _ = predictor.check_inverse_consistency(
                        first_kp, second_kp, first_desc, second_desc,
                        iccl=float(config["PREDICT"].get("iccl", 3.0)),
                    )
                    keep = ransac_inliers(
                        mutual, first_kp, second_kp, ransac_threshold
                    )
                    inlier_points = [
                        first_kp[match.queryIdx].pt
                        for match, retained in zip(mutual, keep) if retained
                    ]
                    cells, coverage, hull = grid_metrics(
                        inlier_points, original.shape, grid_size
                    )
                    original_border = sum(
                        point.pt[0] < border_margin
                        or point.pt[1] < border_margin
                        or point.pt[0] >= original.shape[1] - border_margin
                        or point.pt[1] >= original.shape[0] - border_margin
                        for point in first_kp
                    )
                    repeated_border = sum(
                        first_kp[index_a].pt[0] < border_margin
                        or first_kp[index_a].pt[1] < border_margin
                        or first_kp[index_a].pt[0]
                        >= original.shape[1] - border_margin
                        or first_kp[index_a].pt[1]
                        >= original.shape[0] - border_margin
                        for index_a, _, _ in positives
                    )
                    rows.append({
                        "method": source["label"],
                        "pair_id": item["pair_name"],
                        "transform": transform["name"],
                        "transform_type": transform["type"],
                        "detected_original": len(first_kp),
                        "detected_transformed": len(second_kp),
                        "spatial_positive_pairs": len(positives),
                        "detector_repeatability_original": (
                            len(positives) / len(first_kp) if first_kp else 0.0
                        ),
                        "detector_repeatability_symmetric": (
                            2 * len(positives) / (len(first_kp) + len(second_kp))
                            if first_kp or second_kp else 0.0
                        ),
                        "positive_descriptor_distance": positive_distance,
                        "nearest_spatial_negative_distance": negative_distance,
                        "descriptor_margin": (
                            negative_distance - positive_distance
                            if positive_distance is not None
                            and negative_distance is not None else None
                        ),
                        "ratio_matches": len(ratio),
                        "ratio_pass_rate": (
                            len(ratio) / len(first_kp) if first_kp else 0.0
                        ),
                        "bidirectional_matches": len(mutual),
                        "bidirectional_pass_rate": (
                            len(mutual) / len(ratio) if ratio else 0.0
                        ),
                        "ransac_inliers": int(keep.sum()),
                        "ransac_inlier_rate": (
                            float(keep.mean()) if keep.size else 0.0
                        ),
                        "inlier_grid_cells": cells,
                        "inlier_grid_coverage": coverage,
                        "inlier_hull_fraction": hull,
                        "outer_border_original": original_border,
                        "outer_border_repeated": repeated_border,
                        "outer_border_repeatability": (
                            repeated_border / original_border
                            if original_border else 0.0
                        ),
                    })
                    progress.update(1)
                    progress.set_postfix(
                        method=source["label"], pair=item["pair_name"],
                        transform=transform["name"],
                    )
    finally:
        progress.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = [output_dir / name for name in OUTPUT_NAMES]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "audit_type": "read_only_acquisition_appearance_stability",
        "audit_config": str(args.audit_config),
        "dataset_root": str(dataset_root),
        "device": device,
        "pairs": requested,
        "sources": sources,
        "transforms": transforms,
        "settings": {
            "repeatability_tolerance": tolerance,
            "min_negative_distance": min_negative,
            "ransac_threshold": ransac_threshold,
            "grid_size": grid_size,
            "outer_border_margin": border_margin,
        },
        "rows": rows,
        "interpretation": (
            "The query image is compared with a geometry-preserving appearance "
            "variant of itself. Spatial positives therefore measure acquisition "
            "appearance stability, not disease simulation or FIMD registration."
        ),
        "safety": (
            "Only the declared CSV and JSON are written. Checkpoints, datasets, "
            "training state, and existing test results are read-only."
        ),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote D26 appearance-stability audit: {output_dir}")


if __name__ == "__main__":
    main()
