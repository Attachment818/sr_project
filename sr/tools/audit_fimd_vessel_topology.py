"""Read-only topology audit for FIMD detections, matches and RANSAC inliers.

The vessel topology is a morphology-derived proxy, not a vessel annotation.
It uses the project's current online vessel-mask implementation, removes tiny
components, thins the remaining mask, then labels degree-1 skeleton pixels as
endpoints and degree >= 3 clusters as junction/branch regions.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.eval_util import list_fimd_pairs
from common.spatial_geometry import estimate_homography_with_spatial_support
from common.vessel_mask_util import compute_vessel_mask
from predictor import Predictor


STRUCTURES = ("junction", "endpoint", "vessel_segment", "non_vessel")
ROW_FIELDS = (
    "pair_id", "seed_label", "side", "stage", "structure", "x", "y",
    "distance_to_skeleton", "distance_to_junction", "distance_to_endpoint",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--seed-label", default="")
    parser.add_argument("--mask-backend", default="morph")
    parser.add_argument("--mask-threshold", type=float, default=0.25)
    parser.add_argument("--min-component-pixels", type=int, default=20)
    parser.add_argument("--proximity-pixels", type=float, default=8.0)
    return parser.parse_args()


def clean_components(mask, minimum_pixels):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum_pixels:
            cleaned[labels == index] = 1
    return cleaned


def morphological_skeleton(mask):
    """Dependency-free binary thinning suitable for a diagnostic proxy."""
    work = (mask.astype(np.uint8) * 255).copy()
    skeleton = np.zeros_like(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work) > 0:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, kernel)
    return (skeleton > 0).astype(np.uint8)


def topology_from_image(image, backend, threshold, minimum_pixels):
    mask = compute_vessel_mask(image, backend=backend, threshold=threshold, dilate_kernel=0)
    mask = clean_components(mask > 0.5, minimum_pixels)
    skeleton = morphological_skeleton(mask)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), dtype=np.int16)) - skeleton
    junction_seed = ((skeleton > 0) & (degree >= 3)).astype(np.uint8)
    endpoint_seed = ((skeleton > 0) & (degree == 1)).astype(np.uint8)
    junction_count, _, _, _ = cv2.connectedComponentsWithStats(junction_seed, connectivity=8)
    endpoint_count, _, _, _ = cv2.connectedComponentsWithStats(endpoint_seed, connectivity=8)
    return {
        "mask": mask, "skeleton": skeleton, "junction_seed": junction_seed,
        "endpoint_seed": endpoint_seed,
        "distance_to_skeleton": cv2.distanceTransform((1 - skeleton).astype(np.uint8), cv2.DIST_L2, 3),
        "distance_to_junction": cv2.distanceTransform((1 - junction_seed).astype(np.uint8), cv2.DIST_L2, 3),
        "distance_to_endpoint": cv2.distanceTransform((1 - endpoint_seed).astype(np.uint8), cv2.DIST_L2, 3),
        "junction_components": max(0, junction_count - 1),
        "endpoint_components": max(0, endpoint_count - 1),
    }


def classify(topology, point, radius):
    height, width = topology["mask"].shape
    x = min(width - 1, max(0, int(round(point[0]))))
    y = min(height - 1, max(0, int(round(point[1]))))
    skeleton_distance = float(topology["distance_to_skeleton"][y, x])
    junction_distance = float(topology["distance_to_junction"][y, x])
    endpoint_distance = float(topology["distance_to_endpoint"][y, x])
    if junction_distance <= radius:
        structure = "junction"
    elif endpoint_distance <= radius:
        structure = "endpoint"
    elif skeleton_distance <= radius:
        structure = "vessel_segment"
    else:
        structure = "non_vessel"
    return structure, skeleton_distance, junction_distance, endpoint_distance


def add_points(rows, pair_id, seed_label, side, stage, keypoints, topology, radius):
    for keypoint in keypoints:
        x, y = keypoint.pt
        structure, ds, dj, de = classify(topology, (x, y), radius)
        rows.append({
            "pair_id": pair_id, "seed_label": seed_label, "side": side, "stage": stage,
            "structure": structure, "x": x, "y": y,
            "distance_to_skeleton": ds, "distance_to_junction": dj, "distance_to_endpoint": de,
        })


def draw_overlay(image, topology, keypoints, path):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas[topology["skeleton"] > 0] = (0, 180, 0)
    canvas[topology["junction_seed"] > 0] = (0, 0, 255)
    canvas[topology["endpoint_seed"] > 0] = (255, 0, 0)
    for keypoint in keypoints:
        cv2.circle(canvas, tuple(map(int, keypoint.pt)), 2, (0, 220, 255), -1)
    cv2.imwrite(str(path), canvas)


def summary_for_rows(rows, pair_id):
    result = {}
    for side in ("query", "refer"):
        for stage in ("detected", "ratio", "final", "first_stage_inlier"):
            counter = Counter(
                row["structure"] for row in rows
                if row["pair_id"] == pair_id and row["side"] == side and row["stage"] == stage
            )
            total = sum(counter.values())
            result[f"{side}_{stage}"] = {
                "total": total,
                "counts": {name: counter.get(name, 0) for name in STRUCTURES},
                "fractions": {name: counter.get(name, 0) / max(1, total) for name in STRUCTURES},
            }
    return result


def main():
    args = parse_args()
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Both --config and --checkpoint must be existing files.")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"FIMD dataset root not found: {args.dataset_root}")
    allowed_placeholders = {".gitkeep", "audit.yaml", "test_topology.yaml"}
    if any(path.name not in allowed_placeholders for path in args.output_dir.glob("*")):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["PREDICT"]["model_save_path"] = str(args.checkpoint)
    predictor = Predictor(config)
    items = {item["pair_name"]: item for item in list_fimd_pairs(str(args.dataset_root))}
    pair_ids = args.pair or sorted(items)
    missing = [pair for pair in pair_ids if pair not in items]
    if missing:
        raise KeyError(f"Unknown FIMD pair(s): {missing}")

    rows, summaries = [], []
    for pair_id in tqdm(pair_ids, desc="Auditing FIMD vessel topology", unit="pair"):
        item = items[pair_id]
        ratio_matches, ratio_query, ratio_refer, query, refer = predictor.match(
            item["query_im_path"], item["refer_im_path"]
        )
        query_topology = topology_from_image(query, args.mask_backend, args.mask_threshold, args.min_component_pixels)
        refer_topology = topology_from_image(refer, args.mask_backend, args.mask_threshold, args.min_component_pixels)
        add_points(rows, pair_id, args.seed_label, "query", "detected", ratio_query, query_topology, args.proximity_pixels)
        add_points(rows, pair_id, args.seed_label, "refer", "detected", ratio_refer, refer_topology, args.proximity_pixels)
        add_points(rows, pair_id, args.seed_label, "query", "ratio",
                   [ratio_query[m.queryIdx] for m in ratio_matches], query_topology, args.proximity_pixels)
        add_points(rows, pair_id, args.seed_label, "refer", "ratio",
                   [ratio_refer[m.trainIdx] for m in ratio_matches], refer_topology, args.proximity_pixels)

        p = config["PREDICT"]
        final_matches, final_query, final_refer, final_query_img, final_refer_img = predictor.match_with_consistency_check(
            item["query_im_path"], item["refer_im_path"],
            use_inverse_consistency=p.get("use_inverse_consistency", True), iccl=p.get("iccl", 3.0),
            use_outlier_filter=p.get("use_outlier_filter", True),
            outlier_criteria=p.get("outlier_criteria", "homography"),
            outlier_threshold=p.get("outlier_threshold", 20.0),
        )
        if final_query_img.shape != query.shape or final_refer_img.shape != refer.shape:
            raise RuntimeError("Inconsistent preprocessing between ratio and final matching paths.")
        add_points(rows, pair_id, args.seed_label, "query", "final",
                   [final_query[m.queryIdx] for m in final_matches], query_topology, args.proximity_pixels)
        add_points(rows, pair_id, args.seed_label, "refer", "final",
                   [final_refer[m.trainIdx] for m in final_matches], refer_topology, args.proximity_pixels)
        _, inlier_mask, _ = estimate_homography_with_spatial_support(
            final_matches, final_query, final_refer, query.shape, enabled=False,
            reprojection_threshold=p.get("outlier_threshold", 20.0),
        )
        inlier_mask = np.zeros(len(final_matches), dtype=bool) if inlier_mask is None else np.asarray(inlier_mask, dtype=bool)
        inlier_matches = [match for index, match in enumerate(final_matches) if inlier_mask[index]]
        add_points(rows, pair_id, args.seed_label, "query", "first_stage_inlier",
                   [final_query[m.queryIdx] for m in inlier_matches], query_topology, args.proximity_pixels)
        add_points(rows, pair_id, args.seed_label, "refer", "first_stage_inlier",
                   [final_refer[m.trainIdx] for m in inlier_matches], refer_topology, args.proximity_pixels)
        draw_overlay(query, query_topology, ratio_query, args.output_dir / f"{pair_id}_query_topology.jpg")
        draw_overlay(refer, refer_topology, ratio_refer, args.output_dir / f"{pair_id}_refer_topology.jpg")
        summaries.append({
            "pair_id": pair_id, "ratio_matches": len(ratio_matches),
            "final_matches": len(final_matches), "first_stage_inliers": int(inlier_mask.sum()),
            "query_topology": {key: int(query_topology[key]) for key in ("junction_components", "endpoint_components")},
            "refer_topology": {key: int(refer_topology[key]) for key in ("junction_components", "endpoint_components")},
            "stages": summary_for_rows(rows, pair_id),
        })

    with (args.output_dir / "vessel_topology_points.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "audit_type": "read_only_fimd_vessel_topology", "config": str(args.config),
        "checkpoint": str(args.checkpoint), "dataset_root": str(args.dataset_root),
        "seed_label": args.seed_label, "mask_backend": args.mask_backend,
        "mask_threshold": args.mask_threshold, "min_component_pixels": args.min_component_pixels,
        "proximity_pixels": args.proximity_pixels, "pairs": summaries,
    }
    (args.output_dir / "vessel_topology_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote read-only FIMD vessel-topology audit: {args.output_dir}")


if __name__ == "__main__":
    main()
