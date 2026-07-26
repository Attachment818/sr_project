"""Read-only coordinate audit for final FIMD matches and first-stage inliers."""

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
from common.spatial_geometry import estimate_homography_with_spatial_support
from predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--pair', action='append', default=[])
    parser.add_argument('--seed-label', default=None)
    return parser.parse_args()


def grid_cell(point, shape, grid_size=4):
    h, w = shape[:2]
    return min(grid_size - 1, int(point[1] * grid_size / h)), min(grid_size - 1, int(point[0] * grid_size / w))


def draw_matches(query, refer, query_kpts, refer_kpts, matches, mask, path):
    h, w = query.shape[:2]
    canvas = np.zeros((max(h, refer.shape[0]), w + refer.shape[1], 3), dtype=np.uint8)
    canvas[:h, :w] = cv2.cvtColor(query, cv2.COLOR_GRAY2BGR)
    canvas[:refer.shape[0], w:] = cv2.cvtColor(refer, cv2.COLOR_GRAY2BGR)
    for match, is_inlier in zip(matches, mask):
        q = tuple(map(int, query_kpts[match.queryIdx].pt))
        r0 = tuple(map(int, refer_kpts[match.trainIdx].pt))
        r = (r0[0] + w, r0[1])
        color = (0, 220, 0) if is_inlier else (0, 80, 255)
        cv2.line(canvas, q, r, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, q, 2, color, -1)
        cv2.circle(canvas, r, 2, color, -1)
    for i in range(1, 4):
        cv2.line(canvas, (i * w // 4, 0), (i * w // 4, h), (255, 255, 0), 1)
        cv2.line(canvas, (w + i * refer.shape[1] // 4, 0), (w + i * refer.shape[1] // 4, refer.shape[0]), (255, 255, 0), 1)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    config['PREDICT']['model_save_path'] = str(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(config)
    pair_ids = args.pair or ['39_r_t', '40_r_t', '35_r_t']
    items = {item['pair_name']: item for item in list_fimd_pairs(str(args.dataset_root))}
    missing = [pair_id for pair_id in pair_ids if pair_id not in items]
    if missing:
        raise KeyError(f'Unknown FIMD pair(s): {missing}')

    rows, summaries = [], []
    for pair_id in tqdm(pair_ids, desc='Auditing FIMD inlier geometry', unit='pair'):
        item = items[pair_id]
        p = config['PREDICT']
        matches, query_kpts, refer_kpts, query, refer = predictor.match_with_consistency_check(
            item['query_im_path'], item['refer_im_path'],
            use_inverse_consistency=p.get('use_inverse_consistency', True),
            iccl=p.get('iccl', 3.0), use_outlier_filter=p.get('use_outlier_filter', True),
            outlier_criteria=p.get('outlier_criteria', 'homography'),
            outlier_threshold=p.get('outlier_threshold', 20.0),
        )
        _, mask, _ = estimate_homography_with_spatial_support(
            matches, query_kpts, refer_kpts, query.shape,
            enabled=False, reprojection_threshold=p.get('outlier_threshold', 20.0),
        )
        mask = np.zeros(len(matches), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        inlier_cells = set()
        for index, match in enumerate(matches):
            q = query_kpts[match.queryIdx].pt
            r = refer_kpts[match.trainIdx].pt
            cell = grid_cell(q, query.shape)
            if mask[index]:
                inlier_cells.add(cell)
            rows.append({
                'pair_id': item['file_name'], 'seed_label': args.seed_label or '', 'match_index': index,
                'is_first_stage_inlier': int(mask[index]), 'distance': float(match.distance),
                'query_x': q[0], 'query_y': q[1], 'refer_x': r[0], 'refer_y': r[1],
                'query_grid_row': cell[0], 'query_grid_col': cell[1],
            })
        summaries.append({'pair_id': item['file_name'], 'final_matches': len(matches),
                          'first_stage_inliers': int(mask.sum()), 'inlier_grid_cells': len(inlier_cells),
                          'inlier_grid_coverage': len(inlier_cells) / 16})
        draw_matches(query, refer, query_kpts, refer_kpts, matches, mask,
                     args.output_dir / f'{item["file_name"]}_inlier_geometry.jpg')
    with (args.output_dir / 'inlier_match_coordinates.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / 'inlier_geometry_summary.json').write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote coordinate audit to: {args.output_dir}')


if __name__ == '__main__':
    main()
