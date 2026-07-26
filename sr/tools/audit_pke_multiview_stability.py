"""Read-only two-view PKE stability-ranking counterfactual for G0.

For each original detector candidate, the audit evaluates the normal G0
geometric+content admission against two independently sampled affine views.
It reports candidates passing one view only and candidates passing both.  It
does not filter training data, update a value map, or change any model weight.
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
from common.train_util import affine_images
from common.vessel_mask_util import compute_vessel_mask_batch
from dataset.retina_dataset import RetinaDataset
from model.pke_module import content_filter, geometric_filter, mapping_points
from model.super_retina import SuperRetinaWithVesselOnlyMasked
from tools.audit_pke_checkpoint_supply import add_stats, as_points, empty_stage_stats, finalize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-config", required=True, type=Path)
    return parser.parse_args()


def content_pass_for_view(images, detector, descriptor, model, config):
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
    return content


def select_by_coordinates(points, coordinates):
    selected = [point for point in points.tolist() if tuple(point) in coordinates]
    if not selected:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(selected, dtype=torch.long).reshape(-1, 2)


def main():
    args = parse_args()
    audit = yaml.safe_load(args.audit_config.read_text(encoding="utf-8"))["AUDIT"]
    train_config_path = Path(audit["train_config_path"])
    checkpoint_path = Path(audit["checkpoint_path"])
    output_path = Path(audit["output_path"])
    for path, label in ((train_config_path, "training config"), (checkpoint_path, "checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_yaml = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    config = {**train_yaml["MODEL"], **train_yaml["PKE"], **train_yaml["DATASET"], **train_yaml["VALUE_MAP"]}
    device_name = audit.get("device", config.get("device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    seed = int(audit.get("seed", config.get("seed", 3407)))
    grid_size = int(audit.get("grid_size", 8))
    border_margin = int(audit.get("outer_border_margin", 48))
    low_density_max = int(audit.get("low_density_max_points_per_cell", 4))
    if grid_size < 2 or border_margin < 0 or low_density_max < 0:
        raise ValueError("Invalid grid, border, or density setting")
    torch.manual_seed(seed)
    np.random.seed(seed)
    height, width = int(config["model_image_height"]), int(config["model_image_width"])
    dataset = RetinaDataset(config["dataset_path"], split_file=config["train_split_file"], is_train=False,
                            data_shape=(height, width), auxiliary=None)
    loader = DataLoader(dataset, batch_size=int(audit.get("batch_size", config["batch_size"])),
                        shuffle=False, num_workers=int(audit.get("num_workers", 0)))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(str(checkpoint_path), device=device, strict=False)
    model.eval()

    buckets = {name: empty_stage_stats() for name in (
        "view1_content_pass", "view2_content_pass", "stable_both_views", "view1_only",
    )}
    eligible_samples = 0
    with torch.no_grad():
        progress = tqdm(total=len(loader), desc="D10 multiview candidate stability", unit="batch")
        for images, input_with_label, _, _ in loader:
            images = images.to(device)
            detector, descriptor = model.network(images)
            content1 = content_pass_for_view(images, detector, descriptor, model, config)
            content2 = content_pass_for_view(images, detector, descriptor, model, config)
            masks = compute_vessel_mask_batch(images, backend=config["vessel_mask_backend"],
                                              threshold=config["vessel_mask_threshold"],
                                              dilate_kernel=config["vessel_mask_dilate"])
            for index, has_label in enumerate(input_with_label):
                if not bool(has_label):
                    continue
                eligible_samples += 1
                first, second = as_points(content1[index]), as_points(content2[index])
                first_set = {tuple(point) for point in first.tolist()}
                second_set = {tuple(point) for point in second.tolist()}
                stable = select_by_coordinates(first, first_set & second_set)
                first_only = select_by_coordinates(first, first_set - second_set)
                for name, points in (("view1_content_pass", first), ("view2_content_pass", second),
                                     ("stable_both_views", stable), ("view1_only", first_only)):
                    add_stats(buckets[name], points, masks[index], height, width, grid_size,
                              border_margin, low_density_max)
            progress.update(1)
        progress.close()

    for stats in buckets.values():
        finalize(stats)
    stable = buckets["stable_both_views"]
    first = buckets["view1_content_pass"]
    result = {
        "audit_type": "read_only_pke_multiview_stability_ranking",
        "train_config_path": str(train_config_path), "checkpoint_path": str(checkpoint_path),
        "device": str(device), "seed": seed, "eligible_labeled_samples": eligible_samples,
        "grid_size": grid_size, "outer_border_margin": border_margin,
        "low_density_max_points_per_cell": low_density_max, "buckets": buckets,
        "rates": {
            "stable_among_view1_content": stable["total"] / max(1, first["total"]),
            "stable_noncore_among_view1_noncore": (
                (stable["by_region"]["vessel_edge"] + stable["by_region"]["non_vessel"]) /
                max(1, first["by_region"]["vessel_edge"] + first["by_region"]["non_vessel"])
            ),
        },
        "interpretation": "Stable candidates are a ranking bucket only. View1-only candidates remain counted; no hard filtering or value-map update is performed.",
        "safety": "The audit uses eval mode and never calls update_value_map or value_map_save.",
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only multiview stability audit: {output_path}")


if __name__ == "__main__":
    main()
