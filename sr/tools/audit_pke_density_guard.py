"""Read-only core-support gate for the D6 density-aware PKE policy.

For each image, the script compares the ordinary G0 value-map update with the
D6 low-density non-core boost.  The boost is accepted only when its core final
peak count retains the configured fraction of baseline and its core grid
coverage does not decrease.  Otherwise that image falls back to baseline.
"""

import argparse
import json
import math
import sys
from pathlib import Path

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
from tools.audit_pke_density_policy import (
    add_policy_point_stats, cell_counts, density_summary, empty_region_counts,
    finalize_point_stats, policy_weights,
)
from tools.audit_pke_value_map_retention import final_points_from_value_map, point_region, region_maps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def subset_region(points, mask, wanted_region):
    core, edge = region_maps(mask)
    selected = []
    for x_raw, y_raw in points.detach().cpu().numpy():
        x, y = int(x_raw), int(y_raw)
        if point_region(core, edge, x, y) == wanted_region:
            selected.append((x, y))
    return torch.tensor(selected, dtype=torch.long).reshape(-1, 2)


def empty_stats():
    return {"total": 0, "by_region": empty_region_counts(), "grid_cells_sum": 0, "samples": 0}


def add_density_totals(totals, points, height, width, grid_size):
    summary = density_summary(points, height, width, grid_size)
    totals["occupied_cells"] += summary["occupied_cells"]
    totals["top1"] += summary["top1_cell_fraction"]
    totals["top4"] += summary["top4_cell_fraction"]


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
    min_core_retention = float(audit.get("min_core_peak_retention", 0.98))
    if density_max < 0 or multiplier < 1 or not 0.0 <= min_core_retention <= 1.0:
        raise ValueError("Invalid density or core-retention gate parameters")
    seed = int(audit.get("seed", config.get("seed", 3407)))
    torch.manual_seed(seed)
    threshold, nms_size = int(config["value_increase_point"]), int(config["area"]) * 2
    height, width = int(config["model_image_height"]), int(config["model_image_width"])

    dataset = RetinaDataset(config["dataset_path"], split_file=config["train_split_file"], is_train=False,
                            data_shape=(height, width), auxiliary=None)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False,
                        num_workers=int(config["num_workers"]))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(str(checkpoint_path), device=device, strict=False)
    model.eval()

    baseline, potential, guarded = empty_stats(), empty_stats(), empty_stats()
    gate = {"accepted_images": 0, "rejected_low_core_count": 0, "rejected_core_coverage": 0,
            "baseline_core_peaks": 0, "potential_core_peaks": 0, "guarded_core_peaks": 0,
            "baseline_core_cells": 0, "potential_core_cells": 0, "guarded_core_cells": 0,
            "guarded_saturated_pixels": 0, "samples": 0}
    density = {name: {"occupied_cells": 0, "top1": 0.0, "top4": 0.0}
               for name in ("baseline", "potential", "guarded")}
    eligible_samples = 0

    with torch.no_grad():
        progress = tqdm(total=affine_passes * len(loader), desc="Auditing core-support density gate", unit="batch")
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
                    weights, _ = policy_weights(content_points, masks[index], before_counts,
                                                height, width, grid_size, density_max, multiplier)
                    baseline_map, potential_map = before_map.clone(), before_map.clone()
                    baseline_points = update_value_map(baseline_map, content_points, config)
                    potential_points = update_value_map(potential_map, content_points, config, point_weights=weights)
                    baseline_core = subset_region(baseline_points, masks[index], "vessel_core")
                    potential_core = subset_region(potential_points, masks[index], "vessel_core")
                    baseline_core_cells = len(cell_counts(baseline_core, height, width, grid_size).nonzero()[0])
                    potential_core_cells = len(cell_counts(potential_core, height, width, grid_size).nonzero()[0])
                    count_ok = len(potential_core) >= math.ceil(len(baseline_core) * min_core_retention)
                    coverage_ok = potential_core_cells >= baseline_core_cells
                    accept = count_ok and coverage_ok
                    if accept:
                        guarded_points, guarded_map = potential_points, potential_map
                        gate["accepted_images"] += 1
                    else:
                        guarded_points, guarded_map = baseline_points, baseline_map
                        if not count_ok:
                            gate["rejected_low_core_count"] += 1
                        if not coverage_ok:
                            gate["rejected_core_coverage"] += 1
                    guarded_core = subset_region(guarded_points, masks[index], "vessel_core")
                    guarded_core_cells = len(cell_counts(guarded_core, height, width, grid_size).nonzero()[0])
                    for stats, points in ((baseline, baseline_points), (potential, potential_points), (guarded, guarded_points)):
                        add_policy_point_stats(stats, points, masks[index], height, width, grid_size)
                    for name, points in (("baseline", baseline_points), ("potential", potential_points), ("guarded", guarded_points)):
                        add_density_totals(density[name], points, height, width, grid_size)
                    gate["baseline_core_peaks"] += len(baseline_core)
                    gate["potential_core_peaks"] += len(potential_core)
                    gate["guarded_core_peaks"] += len(guarded_core)
                    gate["baseline_core_cells"] += baseline_core_cells
                    gate["potential_core_cells"] += potential_core_cells
                    gate["guarded_core_cells"] += guarded_core_cells
                    gate["guarded_saturated_pixels"] += int((guarded_map == 255).sum())
                    gate["samples"] += 1
                progress.update(1)
        progress.close()

    for stats in (baseline, potential, guarded):
        finalize_point_stats(stats)
    samples = max(1, gate.pop("samples"))
    density = {
        name: {"occupied_cells_mean_per_image": values["occupied_cells"] / samples,
               "top1_cell_fraction_mean": values["top1"] / samples,
               "top4_cell_fraction_mean": values["top4"] / samples}
        for name, values in density.items()
    }
    result = {
        "audit_type": "read_only_pke_density_core_support_gate",
        "train_config_path": str(train_config_path), "checkpoint_path": str(checkpoint_path),
        "value_map_dir": str(value_map_dir), "device": str(device), "affine_passes": affine_passes,
        "grid_size": grid_size, "low_density_max_peaks": density_max,
        "noncore_boost_multiplier": multiplier, "min_core_peak_retention": min_core_retention,
        "require_core_coverage_non_decrease": True, "eligible_labeled_samples": eligible_samples,
        "gate": gate, "baseline_final": baseline, "potential_policy_final": potential,
        "guarded_policy_final": guarded, "density": density,
        "safety": "Every value map is cloned. value_map_save is never called; rejected images use the baseline clone.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only core-support density gate audit: {output_path}")


if __name__ == "__main__":
    main()
