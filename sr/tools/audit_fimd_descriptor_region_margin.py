"""Read-only regional descriptor-margin audit for selected FIMD pairs."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.eval_util import list_fimd_pairs
from common.vessel_mask_util import compute_vessel_mask
from predictor import Predictor


REGIONS = ('vessel_core', 'vessel_edge', 'non_vessel')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', required=True, type=Path)
    return parser.parse_args()


def region_maps(image, backend, threshold, dilate):
    vessel = compute_vessel_mask(image, backend=backend, threshold=threshold,
                                 dilate_kernel=dilate).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(vessel, kernel, iterations=1) > 0
    expanded = cv2.dilate(vessel, kernel, iterations=1) > 0
    edge = expanded & ~core
    return core, edge


def query_regions(points, image_shape, model_width, model_height, core, edge):
    height, width = image_shape[:2]
    labels = []
    for x_model, y_model in points.tolist():
        x = min(width - 1, max(0, int(x_model / model_width * width)))
        y = min(height - 1, max(0, int(y_model / model_height * height)))
        labels.append('vessel_core' if core[y, x] else ('vessel_edge' if edge[y, x] else 'non_vessel'))
    return np.asarray(labels, dtype=object)


def safe_stats(values):
    if len(values) == 0:
        return {'mean': None, 'median': None, 'p90': None}
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p90': float(np.quantile(values, 0.9)),
    }


def main():
    args = parse_args()
    audit = yaml.safe_load(args.audit_config.read_text(encoding='utf-8'))['AUDIT']
    output_dir = Path(audit['output_dir'])
    generated = [output_dir / name for name in (
        'descriptor_region_margin.csv', 'descriptor_region_margin.json',
    )]
    if any(path.exists() for path in generated):
        raise FileExistsError('Refusing to overwrite existing D13 output')
    dataset_root = Path(audit['dataset_root'])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'FIMD dataset root not found: {dataset_root}')
    sources = audit.get('sources', [])
    if len(sources) < 2:
        raise ValueError('D13 requires at least two source checkpoints')
    pair_ids = list(audit.get('pairs', []))
    if not pair_ids:
        raise ValueError('D13 requires an explicit non-empty pair list')
    for source in sources:
        for field in ('label', 'test_config_path', 'checkpoint_path'):
            if field not in source:
                raise KeyError(f'D13 source is missing {field}')
        if not Path(source['test_config_path']).is_file():
            raise FileNotFoundError(f"Test config not found: {source['test_config_path']}")
        if not Path(source['checkpoint_path']).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source['checkpoint_path']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = audit.get('vessel_mask_backend', 'morph')
    threshold = float(audit.get('vessel_mask_threshold', 0.25))
    dilate = int(audit.get('vessel_mask_dilate', 3))
    device = audit.get('device', 'cuda:0')
    rows = []
    for source in tqdm(sources, desc='D13 loading checkpoints', unit='model'):
        config = yaml.safe_load(Path(source['test_config_path']).read_text(encoding='utf-8'))
        config['PREDICT']['model_save_path'] = source['checkpoint_path']
        config['PREDICT']['device'] = device
        config.setdefault('FIMD', {})['data_root'] = str(dataset_root)
        predictor = Predictor(config)
        predictor.set_eye_mask(None)
        pairs = {item['pair_name']: item for item in list_fimd_pairs(str(dataset_root))}
        missing = [pair_id for pair_id in pair_ids if pair_id not in pairs]
        if missing:
            raise KeyError(f"Unknown FIMD pair(s): {missing}")
        for pair_id in tqdm(pair_ids, desc=f"D13 {source['label']}", leave=False, unit='pair'):
            item = pairs[pair_id]
            query_image, refer_image = predictor.image_read(item['query_im_path'], item['refer_im_path'])
            query_tensor = predictor.trasformer(Image.fromarray(query_image))
            refer_tensor = predictor.trasformer(Image.fromarray(refer_image))
            keypoints, descriptors = predictor.model_run_pair(query_tensor, refer_tensor)
            query_desc = descriptors[0].permute(1, 0).numpy()
            refer_desc = descriptors[1].permute(1, 0).numpy()
            core, edge = region_maps(query_image, backend, threshold, dilate)
            labels = query_regions(keypoints[0], query_image.shape,
                                   predictor.model_image_width, predictor.model_image_height,
                                   core, edge)
            if len(query_desc) and len(refer_desc) >= 2:
                distances = np.linalg.norm(query_desc[:, None] - refer_desc[None], axis=2)
                order = np.argsort(distances, axis=1)
                best = distances[np.arange(len(query_desc)), order[:, 0]]
                second = distances[np.arange(len(query_desc)), order[:, 1]]
                ratio = best / (second + 1e-12)
                reverse_nearest = np.argmin(distances, axis=0)
                mutual = reverse_nearest[order[:, 0]] == np.arange(len(query_desc))
            else:
                best = np.empty(0, dtype=np.float32)
                second = np.empty(0, dtype=np.float32)
                ratio = np.empty(0, dtype=np.float32)
                mutual = np.empty(0, dtype=bool)
                labels = np.empty(0, dtype=object)
            ratio_pass = ratio < float(config['PREDICT']['knn_thresh'])
            for region in REGIONS:
                mask = labels == region
                selected_ratio = ratio[mask]
                selected_best = best[mask]
                selected_margin = (second - best)[mask]
                selected_ratio_pass = ratio_pass[mask]
                selected_mutual = mutual[mask] & selected_ratio_pass
                ratio_stats = safe_stats(selected_ratio)
                best_stats = safe_stats(selected_best)
                margin_stats = safe_stats(selected_margin)
                count = int(mask.sum())
                rows.append({
                    'method': source['label'], 'pair_id': pair_id, 'region': region,
                    'detected_query_count': count,
                    'ratio_pass_count': int(selected_ratio_pass.sum()),
                    'mutual_ratio_count': int(selected_mutual.sum()),
                    'ratio_pass_rate': float(selected_ratio_pass.mean()) if count else None,
                    'mutual_ratio_rate': float(selected_mutual.mean()) if count else None,
                    'ratio_mean': ratio_stats['mean'], 'ratio_median': ratio_stats['median'],
                    'ratio_p90': ratio_stats['p90'], 'best_distance_mean': best_stats['mean'],
                    'distance_margin_mean': margin_stats['mean'],
                })
    with generated[0].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    payload = {
        'audit_type': 'read_only_fimd_descriptor_region_margin',
        'dataset_root': str(dataset_root), 'device': device, 'pairs': pair_ids,
        'vessel_mask': {'backend': backend, 'threshold': threshold, 'dilate': dilate},
        'rows': rows,
        'interpretation': 'ratio_pass_rate and mutual_ratio_rate are query-descriptor observables before geometric filtering. Lower ratio or higher distance margin means stronger nearest-neighbour separation.',
        'safety': 'The audit only runs inference and writes new CSV/JSON files in output_dir.',
    }
    generated[1].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote D13 descriptor-region audit: {output_dir}')


if __name__ == '__main__':
    main()
