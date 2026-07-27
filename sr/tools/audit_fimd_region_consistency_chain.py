"""Read-only regional replay of the exact FIMD ratio/IC/RANSAC matching chain."""

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
OUTPUT_NAMES = ('region_consistency_chain.csv', 'region_consistency_chain.json')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', required=True, type=Path)
    return parser.parse_args()


def region_maps(image, backend, threshold, dilate):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vessel = compute_vessel_mask(gray, backend=backend, threshold=threshold,
                                 dilate_kernel=dilate).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(vessel, kernel, iterations=1) > 0
    edge = (cv2.dilate(vessel, kernel, iterations=1) > 0) & ~core
    return core, edge


def labels_for_keypoints(keypoints, image, backend, threshold, dilate):
    core, edge = region_maps(image, backend, threshold, dilate)
    height, width = image.shape[:2]
    labels = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        col = min(width - 1, max(0, int(round(x))))
        row = min(height - 1, max(0, int(round(y))))
        labels.append('vessel_core' if core[row, col]
                      else ('vessel_edge' if edge[row, col] else 'non_vessel'))
    return labels


def raw_pair(predictor, item):
    query_image, refer_image = predictor.image_read(item['query_im_path'], item['refer_im_path'])
    query_tensor = predictor.trasformer(Image.fromarray(query_image))
    refer_tensor = predictor.trasformer(Image.fromarray(refer_image))
    keypoints, descriptors = predictor.model_run_pair(query_tensor, refer_tensor)
    query_keypoints = [
        cv2.KeyPoint(int(point[0] / predictor.model_image_width * predictor.image_width),
                     int(point[1] / predictor.model_image_height * predictor.image_height), 30)
        for point in keypoints[0]
    ]
    refer_keypoints = [
        cv2.KeyPoint(int(point[0] / predictor.model_image_width * predictor.image_width),
                     int(point[1] / predictor.model_image_height * predictor.image_height), 30)
        for point in keypoints[1]
    ]
    return query_image, refer_image, query_keypoints, refer_keypoints, descriptors


def forward_ratio_matches(predictor, query_desc, refer_desc):
    matches = predictor.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
    return [match for match, next_match in matches
            if match.distance < predictor.knn_thresh * next_match.distance]


def ransac_mask(matches, query_keypoints, refer_keypoints, threshold):
    if len(matches) < 4:
        return np.zeros(len(matches), dtype=bool)
    src = np.float32([query_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst = np.float32([refer_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    return np.zeros(len(matches), dtype=bool) if mask is None else mask.ravel().astype(bool)


def count_matches_by_region(matches, labels):
    return {region: sum(labels[match.queryIdx] == region for match in matches) for region in REGIONS}


def main():
    args = parse_args()
    audit = yaml.safe_load(args.audit_config.read_text(encoding='utf-8'))['AUDIT']
    output_dir = Path(audit['output_dir'])
    generated = [output_dir / name for name in OUTPUT_NAMES]
    occupied = [path for path in generated if path.exists()]
    if occupied:
        raise FileExistsError('Refusing to overwrite D13_fix result(s): ' + ', '.join(map(str, occupied)))
    dataset_root = Path(audit['dataset_root'])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'FIMD dataset root not found: {dataset_root}')
    sources = audit.get('sources', [])
    pair_ids = list(audit.get('pairs', []))
    if len(sources) < 2 or not pair_ids:
        raise ValueError('D13_fix requires at least two sources and explicit pairs')
    for source in sources:
        for field in ('label', 'test_config_path', 'checkpoint_path'):
            if field not in source:
                raise KeyError(f'D13_fix source is missing {field}')
            if field.endswith('_path') and not Path(source[field]).is_file():
                raise FileNotFoundError(f'D13_fix missing {field}: {source[field]}')
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = audit.get('vessel_mask_backend', 'morph')
    threshold = float(audit.get('vessel_mask_threshold', 0.25))
    dilate = int(audit.get('vessel_mask_dilate', 3))
    device = audit.get('device', 'cuda:0')
    rows = []
    for source in tqdm(sources, desc='D13_fix loading checkpoints', unit='model'):
        config = yaml.safe_load(Path(source['test_config_path']).read_text(encoding='utf-8'))
        config['PREDICT']['model_save_path'] = source['checkpoint_path']
        config['PREDICT']['device'] = device
        config.setdefault('FIMD', {})['data_root'] = str(dataset_root)
        predict = config['PREDICT']
        predictor = Predictor(config)
        predictor.set_eye_mask(None)
        items = {item['pair_name']: item for item in list_fimd_pairs(str(dataset_root))}
        missing = [pair_id for pair_id in pair_ids if pair_id not in items]
        if missing:
            raise KeyError(f'Unknown FIMD pair(s): {missing}')
        for pair_id in tqdm(pair_ids, desc=f"D13_fix {source['label']}", leave=False, unit='pair'):
            item = items[pair_id]
            query_image, _, raw_query, raw_refer, descriptors = raw_pair(predictor, item)
            query_desc = descriptors[0].permute(1, 0).numpy()
            refer_desc = descriptors[1].permute(1, 0).numpy()
            forward = forward_ratio_matches(predictor, query_desc, refer_desc)
            bidirectional, _, _ = predictor.check_inverse_consistency(
                raw_query, raw_refer, query_desc, refer_desc, iccl=float(predict.get('iccl', 3.0))
            )
            raw_labels = labels_for_keypoints(raw_query, query_image, backend, threshold, dilate)

            exact = predictor.match_with_consistency_check(
                item['query_im_path'], item['refer_im_path'],
                use_inverse_consistency=predict.get('use_inverse_consistency', True),
                iccl=float(predict.get('iccl', 3.0)),
                use_outlier_filter=predict.get('use_outlier_filter', True),
                outlier_criteria=predict.get('outlier_criteria', 'homography'),
                outlier_threshold=float(predict.get('outlier_threshold', 20.0)),
                return_diagnostics=True,
            )
            returned, returned_query, returned_refer, exact_query_image, _, chain = exact
            if chain['ratio_matches'] != len(forward):
                raise RuntimeError(f'{pair_id}: forward ratio replay mismatch')
            if chain['inverse_consistency_matches'] != len(bidirectional):
                raise RuntimeError(f'{pair_id}: bidirectional replay mismatch')
            returned_labels = labels_for_keypoints(returned_query, exact_query_image, backend, threshold, dilate)
            keep = ransac_mask(returned, returned_query, returned_refer,
                               float(predict.get('outlier_threshold', 20.0)))
            inliers = [match for match, retained in zip(returned, keep) if retained]

            counts = {
                'detected': {region: raw_labels.count(region) for region in REGIONS},
                'forward_ratio': count_matches_by_region(forward, raw_labels),
                'bidirectional_ratio_iccl': count_matches_by_region(bidirectional, raw_labels),
                'returned_after_outlier': count_matches_by_region(returned, returned_labels),
                'ransac_inlier': count_matches_by_region(inliers, returned_labels),
            }
            for region in REGIONS:
                rows.append({
                    'method': source['label'], 'pair_id': pair_id, 'region': region,
                    'detected_query_count': counts['detected'][region],
                    'forward_ratio_count': counts['forward_ratio'][region],
                    'bidirectional_ratio_iccl_count': counts['bidirectional_ratio_iccl'][region],
                    'returned_after_outlier_count': counts['returned_after_outlier'][region],
                    'ransac_inlier_count': counts['ransac_inlier'][region],
                    'forward_ratio_rate': (counts['forward_ratio'][region] / counts['detected'][region]
                                           if counts['detected'][region] else None),
                    'bidirectional_rate': (counts['bidirectional_ratio_iccl'][region] / counts['detected'][region]
                                           if counts['detected'][region] else None),
                })
    with generated[0].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'audit_type': 'read_only_fimd_region_consistency_chain',
        'device': device, 'pairs': pair_ids, 'rows': rows,
        'stage_definition': {
            'forward_ratio_count': 'Exact forward BFMatcher ratio test from Predictor.',
            'bidirectional_ratio_iccl_count': 'Exact Predictor.check_inverse_consistency: both directions pass ratio and return index is identical or within ICCL.',
            'returned_after_outlier_count': 'Exact Predictor.match_with_consistency_check output after configured outlier filter.',
            'ransac_inlier_count': 'Explicit cv2.RANSAC replay over exact returned matches; this is a diagnostic stage, not a replacement for the full FIMD protocol.',
        },
        'safety': 'Only CSV/JSON in this new output directory are written. Checkpoints and existing test outputs are read-only.',
    }
    generated[1].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote D13_fix regional consistency-chain audit: {output_dir}')


if __name__ == '__main__':
    main()
