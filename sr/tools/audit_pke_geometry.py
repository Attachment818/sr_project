"""Read-only audit of why PKE candidates pass/fail geometric filtering.

This script intentionally reuses the current PKE candidate and affine mapping
functions, but never calls loss.backward(), pke_learn(), or value-map updates.
It is therefore safe to run against an existing checkpoint.
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

# Scripts launched as ``python tools/<script>.py`` need the sr/ directory on
# sys.path; normal train/test entrypoints already start from that directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common_util import nms
from common.train_util import affine_images
from common.vessel_mask_util import compute_vessel_mask_batch
from dataset.retina_dataset import RetinaDataset
from model.pke_module import mapping_points
from model.super_retina import SuperRetinaWithVesselOnlyMasked


REGIONS = ('vessel_core', 'vessel_edge', 'non_vessel')


def empty_region_stats():
    return {
        'count': 0,
        'geometric_pass_count': 0,
        'local_rescue_count': 0,
        'affine_score_sum': 0.0,
        'local_max_score_sum': 0.0,
        'affine_score_histogram': [0] * 10,
    }


def classify_point(vessel_mask, x, y):
    mask = (vessel_mask.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(mask, kernel, iterations=1)
    edge = (cv2.dilate(mask, kernel, iterations=1) > 0) & (core == 0)
    if core[y, x]:
        return 'vessel_core'
    if edge[y, x]:
        return 'vessel_edge'
    return 'non_vessel'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Audit YAML path')
    args = parser.parse_args()
    audit = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train_config_path = audit['AUDIT']['train_config_path']
    train_yaml = yaml.safe_load(Path(train_config_path).read_text(encoding='utf-8'))
    config = {**train_yaml['MODEL'], **train_yaml['PKE'], **train_yaml['DATASET'], **train_yaml['VALUE_MAP']}
    device = torch.device(audit['AUDIT'].get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    checkpoint_path = audit['AUDIT']['checkpoint_path']
    passes = int(audit['AUDIT'].get('affine_passes', 5))
    local_radius = int(audit['AUDIT'].get('local_radius', 8))
    threshold = float(audit['AUDIT'].get('geometric_threshold', config['geometric_thresh']))
    output_path = Path(audit['AUDIT']['output_path'])

    dataset = RetinaDataset(
        config['dataset_path'], split_file=config['train_split_file'],
        is_train=False, data_shape=(config['model_image_height'], config['model_image_width']),
        auxiliary=None,
    )
    loader = DataLoader(dataset, batch_size=int(config['batch_size']), shuffle=False, num_workers=int(config['num_workers']))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(checkpoint_path, device=device, strict=False)
    model.eval()

    aggregate = {
        'raw_nms_candidates': 0,
        'in_bounds_candidates': 0,
        'out_of_bounds_candidates': 0,
        'regions': {region: empty_region_stats() for region in REGIONS},
    }
    with torch.no_grad():
        for _ in range(passes):
            for images, input_with_label, _, _ in loader:
                images = images.to(device)
                detector, _ = model.network(images)
                affine_images_tensor, _, grid_inverse = affine_images(images, used_for='detector')
                affine_detector, _ = model.network(affine_images_tensor)
                raw_points = nms(detector, nms_thresh=config['nms_thresh'], nms_size=config['nms_size'])
                mapped_points, affine_points = mapping_points(grid_inverse, raw_points, detector.shape[-2], detector.shape[-1])
                vessel_masks = compute_vessel_mask_batch(
                    images, backend=config['vessel_mask_backend'],
                    threshold=config['vessel_mask_threshold'], dilate_kernel=config['vessel_mask_dilate'],
                )
                for batch_index, (raw, points, affine_points_one) in enumerate(zip(raw_points, mapped_points, affine_points)):
                    aggregate['raw_nms_candidates'] += len(raw)
                    aggregate['in_bounds_candidates'] += len(points)
                    aggregate['out_of_bounds_candidates'] += len(raw) - len(points)
                    height, width = detector.shape[-2:]
                    for point, affine_point in zip(points, affine_points_one):
                        x, y = int(point[0]), int(point[1])
                        ax = min(width - 1, max(0, int(affine_point[0])))
                        ay = min(height - 1, max(0, int(affine_point[1])))
                        score = float(affine_detector[batch_index, 0, ay, ax])
                        y0, y1 = max(0, ay - local_radius), min(height, ay + local_radius + 1)
                        x0, x1 = max(0, ax - local_radius), min(width, ax + local_radius + 1)
                        local_max = float(affine_detector[batch_index, 0, y0:y1, x0:x1].max())
                        region = classify_point(vessel_masks[batch_index], x, y)
                        stats = aggregate['regions'][region]
                        stats['count'] += 1
                        stats['geometric_pass_count'] += int(score >= threshold)
                        stats['local_rescue_count'] += int(score < threshold <= local_max)
                        stats['affine_score_sum'] += score
                        stats['local_max_score_sum'] += local_max
                        stats['affine_score_histogram'][min(9, int(score * 10))] += 1

    for stats in aggregate['regions'].values():
        count = max(1, stats['count'])
        stats['geometric_pass_rate'] = stats['geometric_pass_count'] / count
        stats['local_rescue_rate'] = stats['local_rescue_count'] / count
        stats['mean_affine_score'] = stats['affine_score_sum'] / count
        stats['mean_local_max_score'] = stats['local_max_score_sum'] / count
        del stats['affine_score_sum']
        del stats['local_max_score_sum']
    aggregate['config'] = {'affine_passes': passes, 'local_radius': local_radius, 'geometric_threshold': threshold}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote read-only geometric audit: {output_path}')


if __name__ == '__main__':
    main()
