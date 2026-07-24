"""Read-only diagnostics for the PKE candidate pipeline."""

import cv2
import numpy as np


PKE_STAGES = (
    'detector_candidates',
    'geometric_pass',
    'content_pass',
    'content_strong_pass',
    'content_weak_pass',
    'value_map_points',
)


def _summarize_points(points, vessel_mask, image_shape, grid_size):
    height, width = image_shape
    result = {
        'count': int(len(points)),
        'vessel_core_count': 0,
        'vessel_edge_count': 0,
        'non_vessel_count': 0,
        'grid_occupied_cells': 0,
        'grid_coverage': 0.0,
    }
    if len(points) == 0:
        return result

    vessel = vessel_mask.detach().float().cpu().numpy()
    if vessel.ndim == 3:
        vessel = vessel[0]
    vessel = (vessel > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(vessel, kernel, iterations=1)
    edge = (cv2.dilate(vessel, kernel, iterations=1) > 0) & (core == 0)
    cells = set()
    for point in points.detach().cpu().numpy():
        x = min(width - 1, max(0, int(round(float(point[0])))))
        y = min(height - 1, max(0, int(round(float(point[1])))))
        cells.add((min(grid_size - 1, y * grid_size // height),
                   min(grid_size - 1, x * grid_size // width)))
        if core[y, x]:
            result['vessel_core_count'] += 1
        elif edge[y, x]:
            result['vessel_edge_count'] += 1
        else:
            result['non_vessel_count'] += 1
    result['grid_occupied_cells'] = len(cells)
    result['grid_coverage'] = len(cells) / float(grid_size * grid_size)
    return result


def summarize_pke_stages(stage_points, vessel_masks, image_shape, grid_size=8):
    """Summarize PKE stages per labelled input, without changing training state."""
    records = []
    for index in range(len(vessel_masks)):
        record = {'grid_size': int(grid_size), 'stages': {}}
        for stage in PKE_STAGES:
            # Older saved diagnostic callers may not provide the optional G1
            # split; represent it as a true zero-count stage rather than
            # changing their training behaviour or failing the audit.
            points = stage_points.get(stage)
            if points is None:
                points = [vessel_masks.new_empty((0, 2))] * len(vessel_masks)
            record['stages'][stage] = _summarize_points(
                points[index], vessel_masks[index], image_shape, grid_size
            )
        records.append(record)
    return records
