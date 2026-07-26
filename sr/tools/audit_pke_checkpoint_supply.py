"""Read-only PKE candidate-supply audit across saved training checkpoints.

This compares detector -> geometric admission -> descriptor-content admission
with identical seeded affine views for each checkpoint.  Historic value-map
snapshots are not required and are deliberately not reconstructed from the
single final uint8 map.  The script never writes a value map or a checkpoint.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.common_util import nms
from common.train_util import affine_images
from common.vessel_mask_util import compute_vessel_mask_batch
from dataset.retina_dataset import RetinaDataset
from model.pke_module import content_filter, geometric_filter, mapping_points
from model.super_retina import SuperRetinaWithVesselOnlyMasked

REGIONS = ("vessel_core", "vessel_edge", "non_vessel")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-config", required=True, type=Path)
    return parser.parse_args()


def region_maps(mask):
    vessel = (mask.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    if vessel.ndim == 3:
        vessel = vessel[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(vessel, kernel, iterations=1).astype(bool)
    edge = vessel.astype(bool) & ~core
    return core, edge


def point_region(core, edge, x, y):
    if core[y, x]:
        return "vessel_core"
    if edge[y, x]:
        return "vessel_edge"
    return "non_vessel"


def as_points(value):
    return value.detach().cpu().long() if torch.is_tensor(value) else torch.empty((0, 2), dtype=torch.long)


def point_cells(points, height, width, grid_size):
    if len(points) == 0:
        return []
    return [
        (min(grid_size - 1, max(0, int(y) * grid_size // height)),
         min(grid_size - 1, max(0, int(x) * grid_size // width)))
        for x, y in points.tolist()
    ]


def empty_stage_stats():
    return {
        "total": 0,
        "by_region": {region: 0 for region in REGIONS},
        "outer_border": 0,
        "inner_peripheral": 0,
        "grid_occupied_sum": 0,
        "top1_fraction_sum": 0.0,
        "top4_fraction_sum": 0.0,
        "low_density_point_count": 0,
        "samples": 0,
    }


def add_stats(stats, points, mask, height, width, grid_size, border_margin, low_density_max):
    core, edge = region_maps(mask)
    cells = point_cells(points, height, width, grid_size)
    counts = Counter(cells)
    for x_raw, y_raw in points.tolist():
        x = min(width - 1, max(0, int(x_raw)))
        y = min(height - 1, max(0, int(y_raw)))
        stats["by_region"][point_region(core, edge, x, y)] += 1
        if x < border_margin or y < border_margin or x >= width - border_margin or y >= height - border_margin:
            stats["outer_border"] += 1
        cell = (min(grid_size - 1, y * grid_size // height), min(grid_size - 1, x * grid_size // width))
        if cell[0] in (0, grid_size - 1) or cell[1] in (0, grid_size - 1):
            if not (x < border_margin or y < border_margin or x >= width - border_margin or y >= height - border_margin):
                stats["inner_peripheral"] += 1
        if counts[cell] <= low_density_max:
            stats["low_density_point_count"] += 1
    total = len(points)
    stats["total"] += total
    stats["grid_occupied_sum"] += len(counts)
    if total:
        ranked = sorted(counts.values(), reverse=True)
        stats["top1_fraction_sum"] += ranked[0] / total
        stats["top4_fraction_sum"] += sum(ranked[:4]) / total
    stats["samples"] += 1


def finalize(stats):
    samples, total = max(1, stats.pop("samples")), max(1, stats["total"])
    stats["mean_grid_occupied_cells"] = stats.pop("grid_occupied_sum") / samples
    stats["mean_top1_cell_fraction"] = stats.pop("top1_fraction_sum") / samples
    stats["mean_top4_cell_fraction"] = stats.pop("top4_fraction_sum") / samples
    stats["low_density_point_fraction"] = stats.pop("low_density_point_count") / total
    stats["by_region_fraction"] = {key: value / total for key, value in stats["by_region"].items()}
    stats["outer_border_fraction"] = stats["outer_border"] / total
    stats["inner_peripheral_fraction"] = stats["inner_peripheral"] / total


def ensure_output_available(output_path):
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    audit = yaml.safe_load(args.audit_config.read_text(encoding="utf-8"))["AUDIT"]
    train_config_path = Path(audit["train_config_path"])
    output_path = Path(audit["output_path"])
    if not train_config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")
    ensure_output_available(output_path)
    sources = audit["checkpoints"]
    if len(sources) < 2:
        raise ValueError("At least two checkpoints are required for a temporal supply audit")
    for source in sources:
        if not Path(source["path"]).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source['path']}")

    train_yaml = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    config = {**train_yaml["MODEL"], **train_yaml["PKE"], **train_yaml["DATASET"], **train_yaml["VALUE_MAP"]}
    device_name = audit.get("device", config.get("device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    seed = int(audit.get("seed", config.get("seed", 3407)))
    grid_size = int(audit.get("grid_size", 8))
    border_margin = int(audit.get("outer_border_margin", 48))
    low_density_max = int(audit.get("low_density_max_points_per_cell", 4))
    affine_passes = int(audit.get("affine_passes", 1))
    if grid_size < 2 or border_margin < 0 or low_density_max < 0 or affine_passes < 1:
        raise ValueError("Invalid grid, border, density, or affine-pass setting")
    height, width = int(config["model_image_height"]), int(config["model_image_width"])
    dataset = RetinaDataset(config["dataset_path"], split_file=config["train_split_file"], is_train=False,
                            data_shape=(height, width), auxiliary=None)

    result_sources = []
    for source in sources:
        torch.manual_seed(seed)
        np.random.seed(seed)
        loader = DataLoader(dataset, batch_size=int(audit.get("batch_size", config["batch_size"])),
                            shuffle=False, num_workers=int(audit.get("num_workers", 0)))
        model = SuperRetinaWithVesselOnlyMasked(config, device=device)
        model.load_pretrained_weights(str(source["path"]), device=device, strict=False)
        model.eval()
        stages = {name: empty_stage_stats() for name in ("nms_candidates", "geometric_pass", "content_pass")}
        eligible_samples = 0
        with torch.no_grad():
            progress = tqdm(total=affine_passes * len(loader), desc=f"D8 {source['label']}", unit="batch")
            for _ in range(affine_passes):
                for images, input_with_label, _, _ in loader:
                    images = images.to(device)
                    detector, descriptor = model.network(images)
                    affine_images_tensor, _, grid_inverse = affine_images(images, used_for="detector")
                    affine_detector, affine_descriptor = model.network(affine_images_tensor)
                    candidates = nms(detector, nms_thresh=config["nms_thresh"], nms_size=config["nms_size"])
                    mapped, affine_points = mapping_points(grid_inverse, candidates, detector.shape[-2], detector.shape[-1])
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
                        for name, points in (("nms_candidates", as_points(mapped[index])),
                                             ("geometric_pass", as_points(geometric[index])),
                                             ("content_pass", as_points(content[index]))):
                            add_stats(stages[name], points, masks[index], height, width, grid_size,
                                      border_margin, low_density_max)
                    progress.update(1)
            progress.close()
        for stats in stages.values():
            finalize(stats)
        result_sources.append({"label": source["label"], "checkpoint_path": source["path"],
                               "eligible_labeled_samples": eligible_samples, "stages": stages})

    result = {
        "audit_type": "read_only_pke_checkpoint_candidate_supply",
        "train_config_path": str(train_config_path), "device": str(device), "seed": seed,
        "affine_passes": affine_passes, "grid_size": grid_size, "outer_border_margin": border_margin,
        "low_density_max_points_per_cell": low_density_max, "sources": result_sources,
        "limitation": "Historic value-map snapshots were not saved. This measures candidate supply at each saved checkpoint, not historic value-map retention.",
        "safety": "The audit uses eval mode and never calls update_value_map or value_map_save.",
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only checkpoint candidate-supply audit: {output_path}")


if __name__ == "__main__":
    main()
