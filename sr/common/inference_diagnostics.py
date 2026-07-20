"""Non-invasive diagnostics for SuperRetina inference experiments.

These helpers only observe detector/matcher outputs.  They never alter the
selected keypoints, descriptors, matches, or registration result.
"""

import json
import os

import cv2
import numpy as np

from common.vessel_mask_util import compute_vessel_mask


def summarize_keypoints(
    keypoints,
    image,
    grid_size=4,
    vessel_backend='morph',
    vessel_threshold=0.25,
    vessel_dilate=3,
):
    """Return count, spatial coverage, and pseudo-vessel region statistics."""
    h, w = image.shape[:2]
    count = len(keypoints) if keypoints is not None else 0
    result = {
        'count': int(count),
        'grid_size': int(grid_size),
        'grid_occupied_cells': 0,
        'grid_coverage': 0.0,
        'grid_entropy': 0.0,
        'vessel_core_count': 0,
        'vessel_edge_count': 0,
        'non_vessel_count': 0,
    }
    if count == 0 or h <= 0 or w <= 0:
        return result

    cell_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    for keypoint in keypoints:
        x, y = keypoint.pt
        col = min(grid_size - 1, max(0, int(x / w * grid_size)))
        row = min(grid_size - 1, max(0, int(y / h * grid_size)))
        cell_counts[row, col] += 1
    occupied = cell_counts[cell_counts > 0]
    result['grid_occupied_cells'] = int(len(occupied))
    result['grid_coverage'] = float(len(occupied) / float(grid_size * grid_size))
    if len(occupied) > 1:
        probabilities = occupied.astype(np.float64) / occupied.sum()
        result['grid_entropy'] = float(
            -(probabilities * np.log(probabilities)).sum() / np.log(grid_size * grid_size)
        )

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vessel = compute_vessel_mask(
        gray,
        backend=vessel_backend,
        threshold=vessel_threshold,
        dilate_kernel=vessel_dilate,
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(vessel, kernel, iterations=1)
    expanded = cv2.dilate(vessel, kernel, iterations=1)
    edge = (expanded > 0) & (core == 0)
    for keypoint in keypoints:
        x, y = keypoint.pt
        col = min(w - 1, max(0, int(round(x))))
        row = min(h - 1, max(0, int(round(y))))
        if core[row, col]:
            result['vessel_core_count'] += 1
        elif edge[row, col]:
            result['vessel_edge_count'] += 1
        else:
            result['non_vessel_count'] += 1
    return result


def write_jsonl(path, record):
    """Append one JSON-safe per-pair diagnostic record."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
