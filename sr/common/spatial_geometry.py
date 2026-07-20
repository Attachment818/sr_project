"""Optional spatial-support check for homography estimation."""

from collections import defaultdict

import cv2
import numpy as np


def _points(matches, query_keypoints, refer_keypoints):
    src = np.float32([query_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([refer_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src, dst


def _coverage(points, image_shape, grid_size):
    height, width = image_shape[:2]
    cells = set()
    for x, y in np.asarray(points).reshape(-1, 2):
        col = min(grid_size - 1, max(0, int(x / width * grid_size)))
        row = min(grid_size - 1, max(0, int(y / height * grid_size)))
        cells.add((row, col))
    return len(cells)


def _residual_inliers(homography, src, dst, threshold):
    if homography is None:
        return np.zeros(len(src), dtype=bool)
    projected = cv2.perspectiveTransform(src, homography)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - dst.reshape(-1, 2), axis=1)
    return residuals <= threshold


def _balanced_indices(matches, query_keypoints, image_shape, grid_size, max_per_cell):
    """Use low-distance matches from each query cell in round-robin order."""
    height, width = image_shape[:2]
    cells = defaultdict(list)
    for index, match in enumerate(matches):
        x, y = query_keypoints[match.queryIdx].pt
        cell = (
            min(grid_size - 1, max(0, int(y / height * grid_size))),
            min(grid_size - 1, max(0, int(x / width * grid_size))),
        )
        cells[cell].append(index)
    for indices in cells.values():
        indices.sort(key=lambda index: matches[index].distance)
    return [
        indices[rank]
        for rank in range(max_per_cell)
        for _, indices in sorted(cells.items())
        if rank < len(indices)
    ]


def estimate_homography_with_spatial_support(
    matches, query_keypoints, refer_keypoints, image_shape, *, enabled=False,
    grid_size=4, max_per_cell=3, reprojection_threshold=20.0,
    min_inlier_retention=0.9, min_coverage_gain=1,
):
    """Return the standard estimate unless a diverse candidate is demonstrably safer.

    No correspondence is fabricated: the candidate is fitted only from the
    existing matches.  The returned mask always indexes the original matches.
    """
    metadata = {
        'spatial_support_enabled': bool(enabled),
        'spatial_candidate_considered': False,
        'spatial_candidate_selected': False,
        'spatial_baseline_inliers': 0,
        'spatial_baseline_occupied_cells': 0,
        'spatial_candidate_inliers': 0,
        'spatial_candidate_occupied_cells': 0,
    }
    if len(matches) < 4:
        return None, None, metadata
    src, dst = _points(matches, query_keypoints, refer_keypoints)
    baseline_h, baseline_lmeds = cv2.findHomography(src, dst, cv2.LMEDS)
    if baseline_h is None:
        return None, None, metadata
    # Preserve the legacy path exactly when the experiment is disabled.
    baseline_mask = baseline_lmeds.ravel().astype(bool)
    baseline_count = int(baseline_mask.sum())
    baseline_coverage = _coverage(src[baseline_mask], image_shape, grid_size)
    metadata['spatial_baseline_inliers'] = baseline_count
    metadata['spatial_baseline_occupied_cells'] = baseline_coverage
    if not enabled:
        return baseline_h, baseline_mask, metadata

    indices = _balanced_indices(matches, query_keypoints, image_shape, grid_size, max_per_cell)
    if len(indices) < 4 or len(indices) == len(matches):
        return baseline_h, baseline_mask, metadata
    metadata['spatial_candidate_considered'] = True
    candidate_h, _ = cv2.findHomography(src[indices], dst[indices], cv2.LMEDS)
    if candidate_h is None:
        return baseline_h, baseline_mask, metadata
    candidate_mask = _residual_inliers(candidate_h, src, dst, reprojection_threshold)
    candidate_count = int(candidate_mask.sum())
    candidate_coverage = _coverage(src[candidate_mask], image_shape, grid_size)
    metadata['spatial_candidate_inliers'] = candidate_count
    metadata['spatial_candidate_occupied_cells'] = candidate_coverage
    if (candidate_count >= int(np.ceil(baseline_count * min_inlier_retention))
            and candidate_coverage >= baseline_coverage + min_coverage_gain):
        metadata['spatial_candidate_selected'] = True
        return candidate_h, candidate_mask, metadata
    return baseline_h, baseline_mask, metadata
