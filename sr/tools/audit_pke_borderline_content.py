"""Read-only descriptor audit for PKE geometric-score bands.

It evaluates the same mutual-nearest-neighbour plus ratio criterion used by
PKE content_filter, including candidates that normally fail geometric filtering.
No model, loss, or value-map state is modified.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common_util import nms, sample_keypoint_desc
from common.train_util import affine_images
from common.vessel_mask_util import compute_vessel_mask_batch
from dataset.retina_dataset import RetinaDataset
from model.pke_module import mapping_points
from model.super_retina import SuperRetinaWithVesselOnlyMasked


REGIONS = ('vessel_core', 'vessel_edge', 'non_vessel')
BANDS = ('low_below_04', 'borderline_04_to_05', 'standard_at_least_05')


def classify_point(mask, x, y):
    mask = (mask.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
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


def score_band(score):
    if score < 0.4:
        return 'low_below_04'
    if score < 0.5:
        return 'borderline_04_to_05'
    return 'standard_at_least_05'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Audit YAML path')
    args = parser.parse_args()
    audit = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['AUDIT']
    train_yaml = yaml.safe_load(Path(audit['train_config_path']).read_text(encoding='utf-8'))
    config = {**train_yaml['MODEL'], **train_yaml['PKE'], **train_yaml['DATASET'], **train_yaml['VALUE_MAP']}
    device = torch.device(audit.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    passes = int(audit.get('affine_passes', 5))
    output_path = Path(audit['output_path'])

    dataset = RetinaDataset(config['dataset_path'], split_file=config['train_split_file'], is_train=False,
                            data_shape=(config['model_image_height'], config['model_image_width']), auxiliary=None)
    loader = DataLoader(dataset, batch_size=int(config['batch_size']), shuffle=False, num_workers=int(config['num_workers']))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(audit['checkpoint_path'], device=device, strict=False)
    model.eval()
    results = {region: {band: {'count': 0, 'content_pass_count': 0} for band in BANDS} for region in REGIONS}

    with torch.no_grad():
        for _ in range(passes):
            for images, _, _, _ in loader:
                images = images.to(device)
                detector, descriptor = model.network(images)
                affine_tensor, _, grid_inverse = affine_images(images, used_for='detector')
                affine_detector, affine_descriptor = model.network(affine_tensor)
                raw_points = nms(detector, nms_thresh=config['nms_thresh'], nms_size=config['nms_size'])
                points, affine_points = mapping_points(grid_inverse, raw_points, detector.shape[-2], detector.shape[-1])
                vessel_masks = compute_vessel_mask_batch(images, backend=config['vessel_mask_backend'],
                                                         threshold=config['vessel_mask_threshold'], dilate_kernel=config['vessel_mask_dilate'])
                height, width = detector.shape[-2:]
                for batch_index, (points_one, affine_one) in enumerate(zip(points, affine_points)):
                    if len(points_one) < 2:
                        continue
                    desc = sample_keypoint_desc(points_one[None], descriptor[batch_index:batch_index + 1], s=8)[0].permute(1, 0)
                    affine_desc = sample_keypoint_desc(affine_one[None], affine_descriptor[batch_index:batch_index + 1], s=8)[0].permute(1, 0)
                    distances = torch.cdist(desc, affine_desc, p=2)
                    values, indices = torch.topk(distances, 2, dim=1, largest=False)
                    order = torch.arange(len(points_one), device=device)
                    content_pass = (indices[:, 0] == order) & (values[:, 0] < values[:, 1] * config['content_thresh'])
                    for index, (point, affine_point) in enumerate(zip(points_one, affine_one)):
                        x, y = int(point[0]), int(point[1])
                        ax = min(width - 1, max(0, int(affine_point[0])))
                        ay = min(height - 1, max(0, int(affine_point[1])))
                        band = score_band(float(affine_detector[batch_index, 0, ay, ax]))
                        region = classify_point(vessel_masks[batch_index], x, y)
                        results[region][band]['count'] += 1
                        results[region][band]['content_pass_count'] += int(content_pass[index])

    for region in REGIONS:
        for band in BANDS:
            item = results[region][band]
            item['content_pass_rate'] = item['content_pass_count'] / item['count'] if item['count'] else 0.0
    output = {'config': {'affine_passes': passes, 'content_threshold': config['content_thresh']}, 'regions': results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote read-only borderline descriptor audit: {output_path}')


if __name__ == '__main__':
    main()
