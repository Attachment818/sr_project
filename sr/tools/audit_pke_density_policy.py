"""Read-only counterfactual audit for core-preserving density-aware PKE feedback.

Baseline and policy updates both replay the existing content-validated PKE
points on clones of the saved G0 value maps.  The policy leaves every vessel
core point at the historical weight of one and boosts only already-validated
non-core points in low-density 8x8 cells.  No model, PNG value map, or training
configuration is written.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.common_util import nms
from common.train_util import affine_images, value_map_load
from common.vessel_mask_util import compute_vessel_mask_batch
from dataset.retina_dataset import RetinaDataset
from model.pke_module import content_filter, geometric_filter, mapping_points
from model.record_module import update_value_map
from model.super_retina import SuperRetinaWithVesselOnlyMasked
from tools.audit_pke_value_map_retention import (
    REGIONS, add_point_counts, final_points_from_value_map, grid_cells,
    point_region, point_set, region_maps,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def point_cell(point, height, width, grid_size):
    x, y = int(point[0]), int(point[1])
    return min(grid_size - 1, y * grid_size // height), min(grid_size - 1, x * grid_size // width)


def cell_counts(points, height, width, grid_size):
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    iterable = points.detach().cpu().numpy() if torch.is_tensor(points) else np.empty((0, 2), dtype=np.int64)
    for point in iterable:
        row, col = point_cell(point, height, width, grid_size)
        counts[row, col] += 1
    return counts


def density_summary(points, height, width, grid_size):
    counts = cell_counts(points, height, width, grid_size)
    total = int(counts.sum())
    flat = np.sort(counts.ravel())
    return {
        "total_peaks": total,
        "occupied_cells": int((counts > 0).sum()),
        "top1_cell_fraction": float(counts.max() / max(1, total)),
        "top4_cell_fraction": float(flat[-4:].sum() / max(1, total)),
        "cell_counts": counts.tolist(),
    }


def policy_weights(content_points, mask, before_counts, height, width, grid_size, density_max, multiplier):
    """Core stays weight=1; only validated non-core points in sparse cells are boosted."""
    core, edge = region_maps(mask)
    weights, selected = [], []
    for x_raw, y_raw in content_points.detach().cpu().numpy():
        x, y = int(x_raw), int(y_raw)
        region = point_region(core, edge, x, y)
        row, col = point_cell((x, y), height, width, grid_size)
        promote = region != "vessel_core" and before_counts[row, col] <= density_max
        weights.append(multiplier if promote else 1)
        if promote:
            selected.append((x, y))
    return torch.tensor(weights, dtype=torch.long), torch.tensor(selected, dtype=torch.long).reshape(-1, 2)


def empty_region_counts():
    return {region: 0 for region in REGIONS}


def add_policy_point_stats(stats, points, mask, height, width, grid_size):
    stats["total"] += len(points)
    add_point_counts(stats["by_region"], points, mask)
    stats["grid_cells_sum"] += len(grid_cells(points, height, width, grid_size))
    stats["samples"] += 1


def finalize_point_stats(stats):
    samples = max(1, stats.pop("samples"))
    stats["mean_grid_cells_per_image"] = stats.pop("grid_cells_sum") / samples
    stats["by_region_fraction"] = {
        region: count / max(1, stats["total"])
        for region, count in stats["by_region"].items()
    }


def main():
    args = parse_args()
    audit = yaml.safe_load(args.config.read_text(encoding="utf-8"))["AUDIT"]
    train_config_path = Path(audit["train_config_path"])
    checkpoint_path = Path(audit["checkpoint_path"])
    value_map_dir = Path(audit["value_map_dir"])
    output_path = Path(audit["output_path"])
    for path, label in ((train_config_path, "training config"), (checkpoint_path, "checkpoint"),
                        (value_map_dir, "value-map directory")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not any(value_map_dir.glob("*.png")):
        raise FileNotFoundError(f"No saved value-map PNG files found in: {value_map_dir}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output_path}")

    train_yaml = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    config = {**train_yaml["MODEL"], **train_yaml["PKE"], **train_yaml["DATASET"], **train_yaml["VALUE_MAP"]}
    device_name = audit.get("device", config.get("device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    affine_passes = int(audit.get("affine_passes", 1))
    grid_size = int(audit.get("grid_size", 8))
    density_max = int(audit.get("low_density_max_peaks", 4))
    multiplier = int(audit.get("noncore_boost_multiplier", 2))
    if density_max < 0 or multiplier < 1:
        raise ValueError("low_density_max_peaks must be >= 0 and noncore_boost_multiplier must be >= 1")
    seed = int(audit.get("seed", config.get("seed", 3407)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    threshold, nms_size = int(config["value_increase_point"]), int(config["area"]) * 2
    height, width = int(config["model_image_height"]), int(config["model_image_width"])

    dataset = RetinaDataset(config["dataset_path"], split_file=config["train_split_file"], is_train=False,
                            data_shape=(height, width), auxiliary=None)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False,
                        num_workers=int(config["num_workers"]))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(str(checkpoint_path), device=device, strict=False)
    model.eval()

    selected = {"total": 0, "by_region": empty_region_counts(), "grid_cells_sum": 0, "samples": 0}
    baseline = {"total": 0, "by_region": empty_region_counts(), "grid_cells_sum": 0, "samples": 0}
    policy = {"total": 0, "by_region": empty_region_counts(), "grid_cells_sum": 0, "samples": 0}
    changes = {"policy_new_vs_baseline": 0, "policy_removed_vs_baseline": 0,
               "core_baseline_peaks": 0, "core_baseline_peaks_retained": 0,
               "baseline_saturated_pixels": 0, "policy_saturated_pixels": 0,
               "samples": 0}
    density = {"baseline_occupied_cells_sum": 0, "policy_occupied_cells_sum": 0,
               "baseline_top1_sum": 0.0, "policy_top1_sum": 0.0,
               "baseline_top4_sum": 0.0, "policy_top4_sum": 0.0}
    eligible_samples = 0

    with torch.no_grad():
        progress = tqdm(total=affine_passes * len(loader), desc="Auditing density-aware PKE policy", unit="batch")
        for _ in range(affine_passes):
            for images, input_with_label, _, label_names in loader:
                value_maps = value_map_load(str(value_map_dir), label_names, input_with_label, images.shape[-2:])
                images = images.to(device)
                detector, descriptor = model.network(images)
                affine_images_tensor, _, grid_inverse = affine_images(images, used_for="detector")
                affine_detector, affine_descriptor = model.network(affine_images_tensor)
                mapped, affine_points = mapping_points(
                    grid_inverse, nms(detector, nms_thresh=config["nms_thresh"], nms_size=config["nms_size"]),
                    detector.shape[-2], detector.shape[-1],
                )
                geometric, affine_geometric = geometric_filter(
                    affine_detector, mapped, affine_points, geometric_thresh=float(config["geometric_thresh"])
                )
                content, _, _ = content_filter(
                    descriptor, affine_descriptor, geometric, affine_geometric,
                    content_thresh=float(config["content_thresh"]), scale=8,
                    mode=config.get("pke_content_mode", "one_way"),
                    weak_feedback=bool(config.get("pke_content_weak_feedback", False)),
                    strong_feedback_multiplier=int(config.get("pke_content_strong_feedback_multiplier", 1)),
                    weak_feedback_multiplier=int(config.get("pke_content_weak_feedback_multiplier", 1)),
                    return_feedback_weights=True,
                )
                masks = compute_vessel_mask_batch(images, backend=config["vessel_mask_backend"],
                                                  threshold=config["vessel_mask_threshold"],
                                                  dilate_kernel=config["vessel_mask_dilate"])
                for index, has_label in enumerate(input_with_label):
                    if not bool(has_label):
                        continue
                    eligible_samples += 1
                    content_points = content[index].detach().cpu().long() if torch.is_tensor(content[index]) else torch.empty((0, 2), dtype=torch.long)
                    before_map = value_maps[index].clone()
                    before_points = final_points_from_value_map(before_map, threshold, nms_size)
                    before_counts = cell_counts(before_points, height, width, grid_size)
                    weights, selected_points = policy_weights(content_points, masks[index], before_counts,
                                                               height, width, grid_size, density_max, multiplier)
                    baseline_map, policy_map = before_map.clone(), before_map.clone()
                    baseline_points = update_value_map(baseline_map, content_points, config)
                    policy_points = update_value_map(policy_map, content_points, config, point_weights=weights)
                    add_policy_point_stats(selected, selected_points, masks[index], height, width, grid_size)
                    add_policy_point_stats(baseline, baseline_points, masks[index], height, width, grid_size)
                    add_policy_point_stats(policy, policy_points, masks[index], height, width, grid_size)

                    baseline_set, policy_set = point_set(baseline_points), point_set(policy_points)
                    core, edge = region_maps(masks[index])
                    baseline_core = {point for point in baseline_set if point_region(core, edge, point[0], point[1]) == "vessel_core"}
                    policy_core = {point for point in policy_set if point_region(core, edge, point[0], point[1]) == "vessel_core"}
                    changes["policy_new_vs_baseline"] += len(policy_set - baseline_set)
                    changes["policy_removed_vs_baseline"] += len(baseline_set - policy_set)
                    changes["core_baseline_peaks"] += len(baseline_core)
                    changes["core_baseline_peaks_retained"] += len(baseline_core & policy_core)
                    changes["baseline_saturated_pixels"] += int((baseline_map == 255).sum())
                    changes["policy_saturated_pixels"] += int((policy_map == 255).sum())
                    changes["samples"] += 1
                    baseline_density = density_summary(baseline_points, height, width, grid_size)
                    policy_density = density_summary(policy_points, height, width, grid_size)
                    density["baseline_occupied_cells_sum"] += baseline_density["occupied_cells"]
                    density["policy_occupied_cells_sum"] += policy_density["occupied_cells"]
                    density["baseline_top1_sum"] += baseline_density["top1_cell_fraction"]
                    density["policy_top1_sum"] += policy_density["top1_cell_fraction"]
                    density["baseline_top4_sum"] += baseline_density["top4_cell_fraction"]
                    density["policy_top4_sum"] += policy_density["top4_cell_fraction"]
                progress.update(1)
        progress.close()

    for stats in (selected, baseline, policy):
        finalize_point_stats(stats)
    samples = max(1, changes.pop("samples"))
    density_result = {
        "baseline_occupied_cells_mean_per_image": density["baseline_occupied_cells_sum"] / samples,
        "policy_occupied_cells_mean_per_image": density["policy_occupied_cells_sum"] / samples,
        "baseline_top1_cell_fraction_mean": density["baseline_top1_sum"] / samples,
        "policy_top1_cell_fraction_mean": density["policy_top1_sum"] / samples,
        "baseline_top4_cell_fraction_mean": density["baseline_top4_sum"] / samples,
        "policy_top4_cell_fraction_mean": density["policy_top4_sum"] / samples,
    }
    changes["core_baseline_peak_retention"] = changes["core_baseline_peaks_retained"] / max(1, changes["core_baseline_peaks"])
    result = {
        "audit_type": "read_only_pke_density_policy_counterfactual",
        "train_config_path": str(train_config_path), "checkpoint_path": str(checkpoint_path),
        "value_map_dir": str(value_map_dir), "device": str(device), "affine_passes": affine_passes,
        "grid_size": grid_size, "low_density_max_peaks": density_max,
        "noncore_boost_multiplier": multiplier, "eligible_labeled_samples": eligible_samples,
        "policy": "All G0 content-pass points remain. Core weight stays one; non-core points in cells with at most low_density_max_peaks existing final peaks receive the configured integer multiplier.",
        "selected_noncore_content": selected, "baseline_final": baseline, "policy_final": policy,
        "baseline_to_policy_changes": changes, "density": density_result,
        "safety": "Baseline and policy maps are independent clones. value_map_save is never called.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only density-policy counterfactual: {output_path}")


if __name__ == "__main__":
    main()
