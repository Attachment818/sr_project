"""Read-only spatial and score-distribution audit of saved PKE value maps."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Audit YAML path')
    args = parser.parse_args()
    audit = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['AUDIT']
    value_map_dir = Path(audit['value_map_dir'])
    output_path = Path(audit['output_path'])
    grid_size = int(audit.get('grid_size', 8))
    point_threshold = int(audit.get('point_threshold', 5))
    nms_radius = int(audit.get('nms_radius', 16))
    thresholds = [5, 10, 20, 50, 100]
    records = []
    for path in sorted(value_map_dir.glob('*.png')):
        value_map = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if value_map is None:
            continue
        height, width = value_map.shape
        kernel = np.ones((2 * nms_radius + 1, 2 * nms_radius + 1), np.uint8)
        local_max = cv2.dilate(value_map, kernel)
        peaks = (value_map >= point_threshold) & (value_map == local_max)
        ys, xs = np.where(peaks)
        cell_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
        for x, y in zip(xs, ys):
            cell_counts[min(grid_size - 1, y * grid_size // height), min(grid_size - 1, x * grid_size // width)] += 1
        peak_count = int(peaks.sum())
        records.append({
            'image_name': path.name,
            'max_value': int(value_map.max()),
            'pixels_at_or_above': {str(t): int((value_map >= t).sum()) for t in thresholds},
            'approximate_peak_count': peak_count,
            'occupied_cells': int((cell_counts > 0).sum()),
            'top1_cell_fraction': float(cell_counts.max() / peak_count) if peak_count else 0.0,
            'top4_cell_fraction': float(np.sort(cell_counts.ravel())[-4:].sum() / peak_count) if peak_count else 0.0,
        })
    output = {'config': {'grid_size': grid_size, 'point_threshold': point_threshold, 'nms_radius': nms_radius}, 'images': records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote read-only value-map audit: {output_path}')


if __name__ == '__main__':
    main()
