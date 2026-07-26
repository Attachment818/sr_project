"""Read-only audit of PKE content-pass points retained by the current value map.

The audit loads the saved G0 value maps and replays one in-memory PKE update on
clones only.  It never calls ``value_map_save`` and never changes the source
PNG maps, checkpoint, model, or training configuration.
"""

import argparse
import json
import sys
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

from common.common_util import nms, simple_nms
from common.train_util import affine_images, value_map_load
from common.vessel_mask_util import compute_vessel_mask_batch
from dataset.retina_dataset import RetinaDataset
from model.pke_module import content_filter, geometric_filter, mapping_points
from model.record_module import update_value_map
from model.super_retina import SuperRetinaWithVesselOnlyMasked


REGIONS = ("vessel_core", "vessel_edge", "non_vessel")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def region_maps(mask):
    vessel = (mask.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    if vessel.ndim == 3:
        vessel = vessel[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(vessel, kernel, iterations=1).astype(bool)
    return core, vessel.astype(bool) & ~core


def point_region(core, edge, x, y):
    if core[y, x]:
        return "vessel_core"
    if edge[y, x]:
        return "vessel_edge"
    return "non_vessel"


def final_points_from_value_map(value_map, threshold, nms_size):
    """Mirror update_value_map's final NMS without changing the input map."""
    score = simple_nms(value_map.clone().unsqueeze(0).float(), nms_size).squeeze()
    points = torch.nonzero(score >= threshold)
    return torch.flip(points, [1]).long()


def empty_counts():
    return {region: 0 for region in REGIONS}


def add_point_counts(counts, points, mask):
    core, edge = region_maps(mask)
    height, width = core.shape
    iterable = points.detach().cpu().numpy() if torch.is_tensor(points) else np.empty((0, 2), dtype=np.int64)
    for x_raw, y_raw in iterable:
        x = min(width - 1, max(0, int(x_raw)))
        y = min(height - 1, max(0, int(y_raw)))
        counts[point_region(core, edge, x, y)] += 1


def grid_cells(points, height, width, grid_size):
    cells = set()
    iterable = points.detach().cpu().numpy() if torch.is_tensor(points) else np.empty((0, 2), dtype=np.int64)
    for x_raw, y_raw in iterable:
        x = min(width - 1, max(0, int(x_raw)))
        y = min(height - 1, max(0, int(y_raw)))
        cells.add((min(grid_size - 1, y * grid_size // height),
                   min(grid_size - 1, x * grid_size // width)))
    return cells


def point_set(points):
    return {(int(x), int(y)) for x, y in points.detach().cpu().tolist()}


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
    seed = int(audit.get("seed", config.get("seed", 3407)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    threshold = int(config["value_increase_point"])
    nms_size = int(config["area"]) * 2

    dataset = RetinaDataset(
        config["dataset_path"], split_file=config["train_split_file"], is_train=False,
        data_shape=(config["model_image_height"], config["model_image_width"]), auxiliary=None,
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False,
                        num_workers=int(config["num_workers"]))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(str(checkpoint_path), device=device, strict=False)
    model.eval()

    metrics = {
        name: {"total": 0, "by_region": empty_counts(), "grid_cells_sum": 0, "samples": 0}
        for name in (
            "content_pass", "content_in_final_empty_grid_cell", "directly_retained_content",
            "final_before", "final_after", "new_final",
        )
    }
    grid_transitions = {
        "content_cells_sum": 0,
        "before_cells_sum": 0,
        "after_cells_sum": 0,
        "newly_occupied_final_cells_sum": 0,
        "samples": 0,
    }
    eligible_samples = 0
    height, width = int(config["model_image_height"]), int(config["model_image_width"])
    with torch.no_grad():
        progress = tqdm(total=affine_passes * len(loader), desc="Auditing PKE value-map retention", unit="batch")
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
                    affine_detector, mapped, affine_points, geometric_thresh=float(config["geometric_thresh"]),
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
                masks = compute_vessel_mask_batch(
                    images, backend=config["vessel_mask_backend"], threshold=config["vessel_mask_threshold"],
                    dilate_kernel=config["vessel_mask_dilate"],
                )
                for index, has_label in enumerate(input_with_label):
                    if not bool(has_label):
                        continue
                    eligible_samples += 1
                    content_points = content[index].detach().cpu().long() if torch.is_tensor(content[index]) else torch.empty((0, 2), dtype=torch.long)
                    before_map = value_maps[index].clone()
                    before_points = final_points_from_value_map(before_map, threshold, nms_size)
                    after_map = before_map.clone()
                    after_points = update_value_map(after_map, content_points, config)
                    before_set, after_set = point_set(before_points), point_set(after_points)
                    before_cells = grid_cells(before_points, height, width, grid_size)
                    after_cells = grid_cells(after_points, height, width, grid_size)
                    content_in_final_empty_grid_cell = torch.tensor(
                        [point for point in content_points.tolist()
                         if (min(grid_size - 1, int(point[1]) * grid_size // height),
                             min(grid_size - 1, int(point[0]) * grid_size // width)) not in before_cells],
                        dtype=torch.long,
                    ).reshape(-1, 2)
                    direct_points = torch.tensor(
                        [point for point in content_points.tolist() if tuple(point) in after_set], dtype=torch.long
                    ).reshape(-1, 2)
                    new_points = torch.tensor(sorted(after_set - before_set), dtype=torch.long).reshape(-1, 2)
                    for name, points in (("content_pass", content_points),
                                         ("content_in_final_empty_grid_cell", content_in_final_empty_grid_cell),
                                         ("directly_retained_content", direct_points),
                                         ("final_before", before_points), ("final_after", after_points),
                                         ("new_final", new_points)):
                        metrics[name]["total"] += len(points)
                        add_point_counts(metrics[name]["by_region"], points, masks[index])
                        metrics[name]["grid_cells_sum"] += len(grid_cells(points, height, width, grid_size))
                        metrics[name]["samples"] += 1
                    grid_transitions["content_cells_sum"] += len(grid_cells(content_points, height, width, grid_size))
                    grid_transitions["before_cells_sum"] += len(before_cells)
                    grid_transitions["after_cells_sum"] += len(after_cells)
                    grid_transitions["newly_occupied_final_cells_sum"] += len(after_cells - before_cells)
                    grid_transitions["samples"] += 1
                progress.update(1)
        progress.close()

    for value in metrics.values():
        value["mean_grid_cells_per_image"] = value.pop("grid_cells_sum") / max(1, value.pop("samples"))
        value["by_region_fraction"] = {
            region: count / max(1, value["total"]) for region, count in value["by_region"].items()
        }
    grid_sample_count = max(1, grid_transitions.pop("samples"))
    grid_transitions = {
        name.replace("_sum", "_mean_per_image"): value / grid_sample_count
        for name, value in grid_transitions.items()
    }
    result = {
        "audit_type": "read_only_pke_value_map_retention",
        "train_config_path": str(train_config_path), "checkpoint_path": str(checkpoint_path),
        "value_map_dir": str(value_map_dir), "device": str(device), "affine_passes": affine_passes,
        "grid_size": grid_size, "eligible_labeled_samples": eligible_samples,
        "metrics": metrics,
        "rates": {
            "content_directly_retained": metrics["directly_retained_content"]["total"] / max(1, metrics["content_pass"]["total"]),
            "new_final_per_content": metrics["new_final"]["total"] / max(1, metrics["content_pass"]["total"]),
            "content_in_final_empty_grid_cell": metrics["content_in_final_empty_grid_cell"]["total"] / max(1, metrics["content_pass"]["total"]),
        },
        "grid_transitions": grid_transitions,
        "safety": "Saved value maps are loaded then cloned; value_map_save is never called.",
        "interpretation": (
            "This is a terminal-state counterfactual: it measures one new PKE update against "
            "the saved epoch-149 maps. Historic per-epoch point origins cannot be reconstructed "
            "from uint8 accumulator PNGs."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only PKE value-map retention audit: {output_path}")


if __name__ == "__main__":
    main()
