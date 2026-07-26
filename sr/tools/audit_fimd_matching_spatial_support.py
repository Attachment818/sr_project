"""Read-only FIMD matching-chain and spatial-support audit.

The script replays inference with the supplied test configuration and records
detection, ratio, inverse-consistency, returned-match, RANSAC, spatial-support,
and control-point diagnostics.  It does not alter inference code, weights, or
existing test outputs.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.eval_util import list_fimd_pairs, scale_reference_gt_to_query_space
from common.inference_diagnostics import summarize_keypoints
from common.spatial_geometry import estimate_homography_with_spatial_support
from predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-config", required=True, type=Path)
    return parser.parse_args()


def ensure_output_available(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [output_dir / name for name in ("matching_chain_summary.csv", "matching_chain_summary.json", "matching_region_contributions.csv")]
    occupied = [path for path in generated if path.exists()]
    if occupied:
        raise FileExistsError("Refusing to overwrite existing D9 result(s): " + ", ".join(map(str, occupied)))


def grid_cell(point, shape, grid_size):
    height, width = shape[:2]
    x, y = point
    return (min(grid_size - 1, max(0, int(y * grid_size / height))),
            min(grid_size - 1, max(0, int(x * grid_size / width))))


def convex_hull_fraction(points, shape):
    height, width = shape[:2]
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull) / max(1.0, float(height * width)))


def load_control_points(item, resize_refer_to_query):
    values = np.loadtxt(item["gt_file"])
    if values.ndim == 1:
        values = values.reshape(1, -1)
    query_points = np.column_stack((values[:, 2], values[:, 3])).astype(np.float32)
    refer_points = np.column_stack((values[:, 0], values[:, 1])).astype(np.float32)
    if resize_refer_to_query:
        query_points, refer_points = scale_reference_gt_to_query_space(
            query_points, refer_points, item["query_im_path"], item["refer_im_path"]
        )
    return query_points, refer_points


def homography_control_error(homography, query_points, refer_points):
    if homography is None or len(query_points) == 0:
        return None, None, None
    predicted = cv2.perspectiveTransform(query_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(predicted - refer_points, axis=1)
    return float(errors.mean()), float(np.median(errors)), int((errors < 25.0).sum())


def main():
    args = parse_args()
    audit = yaml.safe_load(args.audit_config.read_text(encoding="utf-8"))["AUDIT"]
    test_config_path = Path(audit["test_config_path"])
    checkpoint_path = Path(audit["checkpoint_path"])
    dataset_root = Path(audit["dataset_root"])
    output_dir = Path(audit["output_dir"])
    for path, label in ((test_config_path, "test config"), (checkpoint_path, "checkpoint"), (dataset_root, "FIMD dataset")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    ensure_output_available(output_dir)

    config = yaml.safe_load(test_config_path.read_text(encoding="utf-8"))
    config["PREDICT"]["model_save_path"] = str(checkpoint_path)
    config["PREDICT"]["device"] = audit.get("device", config["PREDICT"].get("device", "cuda:0"))
    config.setdefault("FIMD", {})["data_root"] = str(dataset_root)
    grid_size = int(audit.get("grid_size", 8))
    border_margin = int(audit.get("outer_border_margin", 48))
    pair_filter = set(audit.get("pairs", []))
    if grid_size < 2 or border_margin < 0:
        raise ValueError("grid_size must be >= 2 and outer_border_margin must be non-negative")

    predictor = Predictor(config)
    predictor.set_eye_mask(None)
    items = list_fimd_pairs(str(dataset_root))
    if pair_filter:
        found = {item["pair_name"] for item in items}
        missing = sorted(pair_filter - found)
        if missing:
            raise KeyError(f"Unknown FIMD pair(s): {missing}")
        items = [item for item in items if item["pair_name"] in pair_filter]

    predict = config["PREDICT"]
    rows, contribution_rows = [], []
    for item in tqdm(items, desc="D9 FIMD matching-chain audit", unit="pair"):
        match_result = predictor.match_with_consistency_check(
            item["query_im_path"], item["refer_im_path"],
            use_inverse_consistency=predict.get("use_inverse_consistency", True),
            iccl=predict.get("iccl", 3.0),
            use_outlier_filter=predict.get("use_outlier_filter", True),
            outlier_criteria=predict.get("outlier_criteria", "homography"),
            outlier_threshold=predict.get("outlier_threshold", 20.0),
            return_diagnostics=True,
        )
        matches, query_keypoints, refer_keypoints, query_image, refer_image, chain = match_result
        legacy_homography, legacy_mask, _ = estimate_homography_with_spatial_support(
            matches, query_keypoints, refer_keypoints, query_image.shape,
            enabled=False, reprojection_threshold=float(predict.get("outlier_threshold", 20.0)),
        )
        legacy_mask = np.zeros(len(matches), dtype=bool) if legacy_mask is None else np.asarray(legacy_mask, dtype=bool)
        if len(matches) >= 4:
            match_query = np.float32([query_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            match_refer = np.float32([refer_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            ransac_homography, ransac_mask_raw = cv2.findHomography(
                match_query, match_refer, cv2.RANSAC, float(predict.get("outlier_threshold", 20.0))
            )
            ransac_mask = np.zeros(len(matches), dtype=bool) if ransac_mask_raw is None else ransac_mask_raw.ravel().astype(bool)
        else:
            ransac_homography, ransac_mask = None, np.zeros(len(matches), dtype=bool)
        query_points = [query_keypoints[match.queryIdx].pt for match in matches]
        inlier_points = [point for point, keep in zip(query_points, ransac_mask) if keep]
        all_cells = {grid_cell(point, query_image.shape, grid_size) for point in query_points}
        inlier_cells = {grid_cell(point, query_image.shape, grid_size) for point in inlier_points}
        control_query, control_refer = load_control_points(item, bool(predict.get("resize_refer_to_query", False)))
        legacy_control_mean, legacy_control_median, legacy_control_under25 = homography_control_error(
            legacy_homography, control_query, control_refer
        )
        ransac_control_mean, ransac_control_median, ransac_control_under25 = homography_control_error(
            ransac_homography, control_query, control_refer
        )

        detected_regions = summarize_keypoints(
            query_keypoints, query_image, grid_size=grid_size,
            vessel_backend=predict.get("diagnostic_vessel_backend", "morph"),
            vessel_threshold=float(predict.get("diagnostic_vessel_threshold", 0.25)),
            vessel_dilate=int(predict.get("diagnostic_vessel_dilate", 3)),
        )
        final_regions = summarize_keypoints(
            [query_keypoints[match.queryIdx] for match in matches], query_image, grid_size=grid_size,
            vessel_backend=predict.get("diagnostic_vessel_backend", "morph"),
            vessel_threshold=float(predict.get("diagnostic_vessel_threshold", 0.25)),
            vessel_dilate=int(predict.get("diagnostic_vessel_dilate", 3)),
        )
        inlier_keypoints = [query_keypoints[match.queryIdx] for match, keep in zip(matches, ransac_mask) if keep]
        inlier_regions = summarize_keypoints(
            inlier_keypoints, query_image, grid_size=grid_size,
            vessel_backend=predict.get("diagnostic_vessel_backend", "morph"),
            vessel_threshold=float(predict.get("diagnostic_vessel_threshold", 0.25)),
            vessel_dilate=int(predict.get("diagnostic_vessel_dilate", 3)),
        )
        outer_final = sum(
            point[0] < border_margin or point[1] < border_margin or
            point[0] >= query_image.shape[1] - border_margin or point[1] >= query_image.shape[0] - border_margin
            for point in query_points
        )
        outer_inlier = sum(
            point[0] < border_margin or point[1] < border_margin or
            point[0] >= query_image.shape[1] - border_margin or point[1] >= query_image.shape[0] - border_margin
            for point in inlier_points
        )
        row = {
            "pair_id": item["pair_name"], "seed_label": audit.get("seed_label", ""),
            "detected_query": chain["detected_query_keypoints"], "detected_refer": chain["detected_refer_keypoints"],
            "ratio_matches": chain["ratio_matches"], "inverse_consistency_matches": chain["inverse_consistency_matches"],
            "returned_matches": chain["outlier_filter_matches"], "legacy_lmeds_inliers": int(legacy_mask.sum()),
            "ransac_inliers": int(ransac_mask.sum()),
            "ransac_inlier_rate": float(ransac_mask.mean()) if len(ransac_mask) else 0.0,
            "match_grid_cells": len(all_cells), "inlier_grid_cells": len(inlier_cells),
            "inlier_grid_coverage": len(inlier_cells) / float(grid_size * grid_size),
            "match_hull_fraction": convex_hull_fraction(query_points, query_image.shape),
            "inlier_hull_fraction": convex_hull_fraction(inlier_points, query_image.shape),
            "control_point_count": len(control_query),
            "legacy_lmeds_control_mean_error": legacy_control_mean,
            "legacy_lmeds_control_median_error": legacy_control_median,
            "legacy_lmeds_control_points_under25": legacy_control_under25,
            "ransac_control_mean_error": ransac_control_mean,
            "ransac_control_median_error": ransac_control_median,
            "ransac_control_points_under25": ransac_control_under25,
            "final_outer_border_matches": int(outer_final), "inlier_outer_border_matches": int(outer_inlier),
        }
        for prefix, summary in (("detected", detected_regions), ("final", final_regions), ("inlier", inlier_regions)):
            for key in ("vessel_core_count", "vessel_edge_count", "non_vessel_count", "grid_occupied_cells", "grid_coverage", "grid_entropy"):
                row[f"{prefix}_{key}"] = summary.get(key, 0)
        rows.append(row)
        for region in ("vessel_core", "vessel_edge", "non_vessel"):
            contribution_rows.append({
                "pair_id": item["pair_name"], "region": region,
                "final_matches": final_regions[f"{region}_count"],
                "ransac_inliers": inlier_regions[f"{region}_count"],
            })

    with (output_dir / "matching_chain_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / "matching_region_contributions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=contribution_rows[0].keys())
        writer.writeheader(); writer.writerows(contribution_rows)
    summary = {
        "audit_type": "read_only_fimd_matching_chain_spatial_support",
        "test_config_path": str(test_config_path), "checkpoint_path": str(checkpoint_path),
        "dataset_root": str(dataset_root), "device": config["PREDICT"]["device"], "grid_size": grid_size,
        "outer_border_margin": border_margin, "pair_count": len(rows), "pairs": rows,
        "interpretation": "legacy_lmeds columns reproduce the test helper's first-stage homography estimator; ransac columns are an additional explicit RANSAC replay over the exact returned matches. Neither replaces the full test protocol's optional quadratic or matching-trick result.",
        "safety": "Only new CSV/JSON files are written under output_dir; the checkpoint and existing test outputs are read-only.",
    }
    (output_dir / "matching_chain_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only FIMD matching-chain audit: {output_dir}")


if __name__ == "__main__":
    main()
