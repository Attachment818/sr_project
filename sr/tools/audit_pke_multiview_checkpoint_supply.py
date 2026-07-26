"""Read-only D11 audit of two-view PKE stability across saved checkpoints.

This is a temporal extension of D10.  It uses identical seeded affine-view
sampling for every checkpoint and reports when stable non-core candidates are
available in sufficient quantity.  It never updates value maps or weights.
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

from common.vessel_mask_util import compute_vessel_mask_batch
from dataset.retina_dataset import RetinaDataset
from model.super_retina import SuperRetinaWithVesselOnlyMasked
from tools.audit_pke_checkpoint_supply import add_stats, as_points, empty_stage_stats, finalize
from tools.audit_pke_multiview_stability import content_pass_for_view, select_by_coordinates


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-config", required=True, type=Path)
    return parser.parse_args()


def audit_checkpoint(source, dataset, config, audit, device, seed, height, width, grid_size, border_margin, low_density_max):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(dataset, batch_size=int(audit.get("batch_size", config["batch_size"])),
                        shuffle=False, num_workers=int(audit.get("num_workers", 0)))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(str(source["path"]), device=device, strict=False)
    model.eval()
    buckets = {name: empty_stage_stats() for name in (
        "view1_content_pass", "view2_content_pass", "stable_both_views", "view1_only",
    )}
    eligible_samples = 0
    with torch.no_grad():
        progress = tqdm(total=len(loader), desc=f"D11 {source['label']}", unit="batch")
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
                stable = select_by_coordinates(first, first_set & {tuple(point) for point in second.tolist()})
                first_only = select_by_coordinates(first, first_set - {tuple(point) for point in second.tolist()})
                for name, points in (("view1_content_pass", first), ("view2_content_pass", second),
                                     ("stable_both_views", stable), ("view1_only", first_only)):
                    add_stats(buckets[name], points, masks[index], height, width, grid_size,
                              border_margin, low_density_max)
            progress.update(1)
        progress.close()
    for stats in buckets.values():
        finalize(stats)
    stable, first = buckets["stable_both_views"], buckets["view1_content_pass"]
    return {
        "label": source["label"], "checkpoint_path": source["path"],
        "eligible_labeled_samples": eligible_samples, "buckets": buckets,
        "rates": {
            "stable_among_view1_content": stable["total"] / max(1, first["total"]),
            "stable_noncore_among_view1_noncore": (
                (stable["by_region"]["vessel_edge"] + stable["by_region"]["non_vessel"]) /
                max(1, first["by_region"]["vessel_edge"] + first["by_region"]["non_vessel"])
            ),
        },
    }


def main():
    args = parse_args()
    audit = yaml.safe_load(args.audit_config.read_text(encoding="utf-8"))["AUDIT"]
    train_config_path, output_path = Path(audit["train_config_path"]), Path(audit["output_path"])
    if not train_config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit result: {output_path}")
    sources = audit.get("checkpoints", [])
    if len(sources) < 2:
        raise ValueError("D11 requires at least two checkpoint sources")
    for source in sources:
        if not Path(source["path"]).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source['path']}")
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
    height, width = int(config["model_image_height"]), int(config["model_image_width"])
    dataset = RetinaDataset(config["dataset_path"], split_file=config["train_split_file"], is_train=False,
                            data_shape=(height, width), auxiliary=None)
    sources_result = [
        audit_checkpoint(source, dataset, config, audit, device, seed, height, width, grid_size,
                         border_margin, low_density_max)
        for source in sources
    ]
    result = {
        "audit_type": "read_only_pke_multiview_checkpoint_supply",
        "train_config_path": str(train_config_path), "device": str(device), "seed": seed,
        "grid_size": grid_size, "outer_border_margin": border_margin,
        "low_density_max_points_per_cell": low_density_max, "sources": sources_result,
        "interpretation": "This determines the available supply for a possible ranking-only feedback mechanism. No candidate is filtered or written to a value map.",
        "safety": "The audit uses eval mode and never calls update_value_map or value_map_save.",
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote read-only multiview checkpoint-supply audit: {output_path}")


if __name__ == "__main__":
    main()
