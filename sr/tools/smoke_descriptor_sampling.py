"""Fast CPU/GPU-independent regression check for descriptor sampling alignment.

It intentionally maps one NMS point outside the affine image.  The returned
valid keypoint list must therefore have exactly the same length as both sampled
descriptor lists.  This guards G3 spatial hard-negative masking from using the
pre-filter NMS points.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.common_util import sample_descriptors


def main():
    detector = torch.zeros((1, 1, 16, 16))
    detector[0, 0, 3, 3] = 1.0
    detector[0, 0, 12, 12] = 1.0
    descriptor = torch.randn((1, 8, 2, 2))
    affine_descriptor = torch.randn((1, 8, 2, 2))
    grid_inverse = torch.zeros((1, 16, 16, 2))
    grid_inverse[..., 0] = 0.0
    grid_inverse[..., 1] = 0.0
    grid_inverse[0, 12, 12, 0] = 1.1  # reject exactly one NMS candidate

    descriptors, affine_descriptors, valid_points = sample_descriptors(
        detector, descriptor, affine_descriptor, grid_inverse,
        nms_size=1, nms_thresh=0.1, scale=8, return_valid_keypoints=True,
    )
    count = valid_points[0].shape[0]
    assert count == descriptors[0].shape[1] == affine_descriptors[0].shape[1], (
        count, descriptors[0].shape, affine_descriptors[0].shape
    )
    assert count == 1, f'Expected one in-bounds candidate, got {count}'
    print('Descriptor sampling alignment smoke test passed.')


if __name__ == '__main__':
    main()
