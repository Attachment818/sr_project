"""Read-only PKE region and spatial-feedback audit.

This audit reproduces the candidate -> geometric -> descriptor-content gates of
PKE, but deliberately never calls ``pke_learn`` or updates a value map.  It is
therefore safe to execute against an existing G0 checkpoint.  Anatomical region
and image-border labels are diagnostic metadata only; they do not alter a
candidate's eligibility.
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

from common.common_util import nms
from common.train_util import affine_images
from common.vessel_mask_util import compute_vessel_mask_batch
from dataset.retina_dataset import RetinaDataset
from model.pke_module import content_filter, geometric_filter, mapping_points
from model.super_retina import SuperRetinaWithVesselOnlyMasked


REGIONS = ("vessel_core", "vessel_edge", "non_vessel")
BORDER_GROUPS = ("image_border", "image_interior")
STAGES = ("mapped_candidates", "geometric_pass", "content_pass")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="Audit YAML containing an AUDIT section.")
    return parser.parse_args()


def point_region_maps(mask):
    """Return mutually exclusive core/edge/non-vessel maps for one mask."""
    vessel = (mask.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    if vessel.ndim == 3:
        vessel = vessel[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(vessel, kernel, iterations=1).astype(bool)
    return core, vessel.astype(bool) & ~core


def classify_point(core, edge, x, y, height, width, border_margin):
    if core[y, x]:
        region = "vessel_core"
    elif edge[y, x]:
        region = "vessel_edge"
    else:
        region = "non_vessel"
    is_border = (
        x < border_margin or y < border_margin
        or x >= width - border_margin or y >= height - border_margin
    )
    return region, "image_border" if is_border else "image_interior"


def empty_stage_stats():
    return {
        "total_points": 0,
        "points_by_region": {key: 0 for key in REGIONS},
        "points_by_border": {key: 0 for key in BORDER_GROUPS},
        "points_by_region_and_border": {
            region: {border: 0 for border in BORDER_GROUPS}
            for region in REGIONS
        },
        "grid_cells_sum": 0,
        "grid_cells_samples": 0,
    }


def record_stage(stats, points_per_image, masks, height, width, grid_size, border_margin):
    for points, mask in zip(points_per_image, masks):
        core, edge = point_region_maps(mask)
        cells = set()
        iterable = points.detach().cpu().numpy() if torch.is_tensor(points) else np.empty((0, 2), dtype=np.int64)
        for x_raw, y_raw in iterable:
            x = min(width - 1, max(0, int(x_raw)))
            y = min(height - 1, max(0, int(y_raw)))
            region, border = classify_point(core, edge, x, y, height, width, border_margin)
            stats["total_points"] += 1
            stats["points_by_region"][region] += 1
            stats["points_by_border"][border] += 1
            stats["points_by_region_and_border"][region][border] += 1
            cells.add((min(grid_size - 1, y * grid_size // height),
                       min(grid_size - 1, x * grid_size // width)))
        stats["grid_cells_sum"] += len(cells)
        stats["grid_cells_samples"] += 1


def finalise_stage_stats(stats):
    sample_count = max(1, stats["grid_cells_samples"])
    stats["mean_occupied_grid_cells_per_image"] = stats["grid_cells_sum"] / sample_count
    for key in ("points_by_region", "points_by_border"):
        stats[f"{key}_fraction"] = {
            name: count / max(1, stats["total_points"])
            for name, count in stats[key].items()
        }
    del stats["grid_cells_sum"]
    del stats["grid_cells_samples"]


def main():
    args = parse_args()
    audit = yaml.safe_load(args.config.read_text(encoding="utf-8"))["AUDIT"]
    train_config_path = Path(audit["train_config_path"])
    checkpoint_path = Path(audit["checkpoint_path"])
    output_path = Path(audit["output_path"])
    for path, label in ((train_config_path, "training config"), (checkpoint_path, "checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output_path}")

    train_yaml = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    config = {
        **train_yaml["MODEL"], **train_yaml["PKE"],
        **train_yaml["DATASET"], **train_yaml["VALUE_MAP"],
    }
    device_name = audit.get("device", config.get("device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    affine_passes = int(audit.get("affine_passes", 5))
    grid_size = int(audit.get("grid_size", 8))
    border_margin = int(audit.get("image_border_margin", 48))
    seed = int(audit.get("seed", config.get("seed", 3407)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = RetinaDataset(
        config["dataset_path"], split_file=config["train_split_file"], is_train=False,
        data_shape=(config["model_image_height"], config["model_image_width"]), auxiliary=None,
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False,
                        num_workers=int(config["num_workers"]))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(str(checkpoint_path), device=device, strict=False)
    model.eval()

    stage_stats = {stage: empty_stage_stats() for stage in STAGES}
    score_histogram = [0] * 10
    height, width = int(config["model_image_height"]), int(config["model_image_width"])
    with torch.no_grad():
        progress = tqdm(total=affine_passes * len(loader), desc="Auditing PKE region feedback", unit="batch")
        for _ in range(affine_passes):
            for images, _, _, _ in loader:
                images = images.to(device)
                detector, descriptor = model.network(images)
                affine_images_tensor, _, grid_inverse = affine_images(images, used_for="detector")
                affine_detector, affine_descriptor = model.network(affine_images_tensor)
                mapped, affine_points = mapping_points(
                    grid_inverse, nms(detector, nms_thresh=config["nms_thresh"], nms_size=config["nms_size"]),
                    detector.shape[-2], detector.shape[-1],
                )
                masks = compute_vessel_mask_batch(
                    images, backend=config["vessel_mask_backend"],
                    threshold=config["vessel_mask_threshold"], dilate_kernel=config["vessel_mask_dilate"],
                )
                record_stage(stage_stats["mapped_candidates"], mapped, masks, height, width, grid_size, border_margin)
                for affine_for_points, detector_one in zip(affine_points, affine_detector):
                    for ax_raw, ay_raw in affine_for_points.detach().cpu().numpy():
                        ax = min(width - 1, max(0, int(ax_raw)))
                        ay = min(height - 1, max(0, int(ay_raw)))
                        score_histogram[min(9, max(0, int(float(detector_one[0, ay, ax]) * 10)))] += 1
                geometric, affine_geometric = geometric_filter(
                    affine_detector, mapped, affine_points, geometric_thresh=float(config["geometric_thresh"]),
                )
                record_stage(stage_stats["geometric_pass"], geometric, masks, height, width, grid_size, border_margin)
                content, _, _ = content_filter(
                    descriptor, affine_descriptor, geometric, affine_geometric,
                    content_thresh=float(config["content_thresh"]), scale=8,
                    mode=config.get("pke_content_mode", "one_way"),
                    weak_feedback=bool(config.get("pke_content_weak_feedback", False)),
                    strong_feedback_multiplier=int(config.get("pke_content_strong_feedback_multiplier", 1)),
                    weak_feedback_multiplier=int(config.get("pke_content_weak_feedback_multiplier", 1)),
                    return_feedback_weights=True,
                )
                record_stage(stage_stats["content_pass"], content, masks, height, width, grid_size, border_margin)
                progress.update(1)
        progress.close()

    for stats in stage_stats.values():
        finalise_stage_stats(stats)
    mapped_total = max(1, stage_stats["mapped_candidates"]["total_points"])
    geo_total = max(1, stage_stats["geometric_pass"]["total_points"])
    result = {
        "audit_type": "read_only_pke_region_feedback",
        "train_config_path": str(train_config_path), "checkpoint_path": str(checkpoint_path),
        "device": str(device), "affine_passes": affine_passes, "grid_size": grid_size,
        "image_border_margin": border_margin, "geometric_threshold": float(config["geometric_thresh"]),
        "content_threshold": float(config["content_thresh"]),
        "affine_detector_score_histogram": score_histogram, "stages": stage_stats,
        "transition_rates": {
            "mapped_to_geometric": stage_stats["geometric_pass"]["total_points"] / mapped_total,
            "geometric_to_content": stage_stats["content_pass"]["total_points"] / geo_total,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only PKE region-feedback audit: {output_path}")


if __name__ == "__main__":
    main()
