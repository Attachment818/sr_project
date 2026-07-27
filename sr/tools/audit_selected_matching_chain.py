"""Read-only matching-chain audit for explicit FIRE or FIMD pairs and models."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.eval_util import list_fimd_pairs
from predictor import Predictor

OUTPUT_NAMES = ('selected_matching_chain.csv', 'selected_matching_chain.json')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    return parser.parse_args()


def fire_pair_item(dataset_root, pair_name):
    prefix = 'control_points_'
    stem = pair_name[:-4] if pair_name.endswith('.txt') else pair_name
    if not stem.startswith(prefix):
        raise ValueError(f'Invalid FIRE pair name: {pair_name}')
    fields = stem[len(prefix):].split('_')
    if len(fields) != 3:
        raise ValueError(f'Invalid FIRE pair name: {pair_name}')
    image_id, refer_index, query_index = fields
    return {
        'pair_name': stem,
        'gt_file': str(dataset_root / 'Ground Truth' / f'{stem}.txt'),
        'query_im_path': str(dataset_root / 'Images' / f'{image_id}_{query_index}.jpg'),
        'refer_im_path': str(dataset_root / 'Images' / f'{image_id}_{refer_index}.jpg'),
    }


def grid_metrics(points, image_shape, grid_size):
    height, width = image_shape[:2]
    if not points:
        return 0, 0.0, 0.0
    cells = {
        (
            min(grid_size - 1, max(0, int(y * grid_size / height))),
            min(grid_size - 1, max(0, int(x * grid_size / width))),
        )
        for x, y in points
    }
    hull_fraction = 0.0
    if len(points) >= 3:
        hull = cv2.convexHull(
            np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        )
        hull_fraction = float(
            cv2.contourArea(hull) / max(1.0, float(height * width))
        )
    return len(cells), len(cells) / float(grid_size * grid_size), hull_fraction


def ransac_inliers(matches, query_keypoints, refer_keypoints, threshold):
    if len(matches) < 4:
        return np.zeros(len(matches), dtype=bool)
    source = np.float32([
        query_keypoints[match.queryIdx].pt for match in matches
    ]).reshape(-1, 1, 2)
    target = np.float32([
        refer_keypoints[match.trainIdx].pt for match in matches
    ]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, threshold)
    return (
        np.zeros(len(matches), dtype=bool)
        if mask is None else mask.ravel().astype(bool)
    )


def self_test():
    root = Path('/dataset/FIRE')
    item = fire_pair_item(root, 'control_points_P18_1_2')
    assert Path(item['refer_im_path']).name == 'P18_1.jpg'
    assert Path(item['query_im_path']).name == 'P18_2.jpg'
    assert Path(item['gt_file']).name == 'control_points_P18_1_2.txt'
    cells, coverage, hull = grid_metrics(
        [(0, 0), (63, 63), (32, 32)], (64, 64), 4
    )
    assert cells == 3 and coverage == 3 / 16 and hull >= 0
    assert ransac_inliers([], [], [], 20.0).size == 0
    print('selected matching-chain audit self-test passed')


def load_items(dataset, dataset_root, pair_ids):
    if dataset == 'FIMD':
        available = {
            item['pair_name']: item for item in list_fimd_pairs(str(dataset_root))
        }
        missing = [pair_id for pair_id in pair_ids if pair_id not in available]
        if missing:
            raise KeyError(f'Unknown FIMD pair(s): {missing}')
        return [available[pair_id] for pair_id in pair_ids]
    items = [fire_pair_item(dataset_root, pair_id) for pair_id in pair_ids]
    for item in items:
        for field in ('gt_file', 'query_im_path', 'refer_im_path'):
            if not Path(item[field]).is_file():
                raise FileNotFoundError(f"Missing FIRE input: {item[field]}")
    return items


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_config is None:
        raise ValueError('--audit-config is required unless --self-test is used')
    audit = yaml.safe_load(
        args.audit_config.read_text(encoding='utf-8')
    )['AUDIT']
    dataset = str(audit['dataset']).upper()
    if dataset not in {'FIRE', 'FIMD'}:
        raise ValueError(f'Unsupported dataset: {dataset}')
    dataset_root = Path(audit['dataset_root'])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root not found: {dataset_root}')
    pair_ids = list(audit.get('pairs', []))
    sources = list(audit.get('sources', []))
    if not pair_ids or len(sources) < 2:
        raise ValueError('Explicit pairs and at least two model sources are required')

    output_dir = Path(audit['output_dir'])
    generated = [output_dir / name for name in OUTPUT_NAMES]
    occupied = [path for path in generated if path.exists()]
    if occupied:
        raise FileExistsError(
            'Refusing to overwrite selected-chain result(s): '
            + ', '.join(map(str, occupied))
        )
    for source in sources:
        for field in ('label', 'test_config_path', 'checkpoint_path'):
            if field not in source:
                raise KeyError(f'Source is missing {field}: {source}')
        if not Path(source['test_config_path']).is_file():
            raise FileNotFoundError(
                f"Test config not found: {source['test_config_path']}"
            )
        if not Path(source['checkpoint_path']).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {source['checkpoint_path']}"
            )
    items = load_items(dataset, dataset_root, pair_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = audit['device']
    grid_size = int(audit.get('grid_size', 8))
    if grid_size < 2:
        raise ValueError('grid_size must be at least 2')
    rows = []
    progress = tqdm(
        total=len(sources) * len(items),
        desc=f'{dataset} selected matching-chain audit',
        unit='pair',
    )
    try:
        for source in sources:
            config = yaml.safe_load(
                Path(source['test_config_path']).read_text(encoding='utf-8')
            )
            config['PREDICT']['model_save_path'] = source['checkpoint_path']
            config['PREDICT']['device'] = device
            if dataset == 'FIMD':
                config.setdefault('FIMD', {})['data_root'] = str(dataset_root)
            predictor = Predictor(config)
            if dataset == 'FIMD':
                predictor.set_eye_mask(None)
            predict = config['PREDICT']
            for item in items:
                result = predictor.match_with_consistency_check(
                    item['query_im_path'],
                    item['refer_im_path'],
                    use_inverse_consistency=predict.get(
                        'use_inverse_consistency', True
                    ),
                    iccl=float(predict.get('iccl', 3.0)),
                    use_outlier_filter=predict.get('use_outlier_filter', True),
                    outlier_criteria=predict.get(
                        'outlier_criteria', 'homography'
                    ),
                    outlier_threshold=float(
                        predict.get('outlier_threshold', 20.0)
                    ),
                    return_diagnostics=True,
                )
                matches, query_kp, refer_kp, query_image, _, chain = result
                keep = ransac_inliers(
                    matches, query_kp, refer_kp,
                    float(predict.get('outlier_threshold', 20.0)),
                )
                match_points = [
                    query_kp[match.queryIdx].pt for match in matches
                ]
                inlier_points = [
                    point for point, retained in zip(match_points, keep)
                    if retained
                ]
                match_cells, match_coverage, match_hull = grid_metrics(
                    match_points, query_image.shape, grid_size
                )
                inlier_cells, inlier_coverage, inlier_hull = grid_metrics(
                    inlier_points, query_image.shape, grid_size
                )
                returned = len(matches)
                inliers = int(keep.sum())
                rows.append({
                    'dataset': dataset,
                    'method': source['label'],
                    'pair_id': item['pair_name'],
                    'detected_query': chain['detected_query_keypoints'],
                    'detected_refer': chain['detected_refer_keypoints'],
                    'ratio_matches': chain['ratio_matches'],
                    'bidirectional_matches': chain[
                        'inverse_consistency_matches'
                    ],
                    'returned_after_outlier': chain['outlier_filter_matches'],
                    'ransac_inliers': inliers,
                    'ransac_inlier_rate': inliers / returned if returned else 0.0,
                    'match_grid_cells': match_cells,
                    'match_grid_coverage': match_coverage,
                    'match_hull_fraction': match_hull,
                    'inlier_grid_cells': inlier_cells,
                    'inlier_grid_coverage': inlier_coverage,
                    'inlier_hull_fraction': inlier_hull,
                    'mean_returned_match_distance': (
                        float(np.mean([match.distance for match in matches]))
                        if matches else None
                    ),
                })
                progress.update(1)
                progress.set_postfix(
                    method=source['label'], pair=item['pair_name'],
                    returned=returned, inliers=inliers,
                )
    finally:
        progress.close()

    with generated[0].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'audit_type': 'selected_exact_matching_chain',
        'dataset': dataset,
        'dataset_root': str(dataset_root),
        'device': device,
        'grid_size': grid_size,
        'pairs': pair_ids,
        'sources': sources,
        'rows': rows,
        'stage_definition': {
            'ratio_matches': 'Forward BFMatcher ratio test from Predictor.',
            'bidirectional_matches': (
                'Predictor inverse consistency: both directions satisfy ratio '
                'and the return index is identical or within ICCL.'
            ),
            'returned_after_outlier': (
                'Exact matches returned by Predictor after configured outlier filtering.'
            ),
            'ransac_inliers': (
                'Explicit cv2.RANSAC replay over the exact returned matches.'
            ),
        },
        'safety': (
            'Only the two declared files in the new output directory are written; '
            'checkpoints and existing test results are read-only.'
        ),
    }
    generated[1].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'Wrote selected matching-chain audit: {output_dir}')


if __name__ == '__main__':
    main()
