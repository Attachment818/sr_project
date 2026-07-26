"""Small deterministic preflight for the optional G4 PKE feedback path.

This does not load data, weights, or write files.  It verifies that the new
centre bonus is opt-in and that the multiview selector respects its per-image
cap on a synthetic identity-affine example.
"""

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import pke_module
from model.record_module import update_value_map


def main():
    map_config = {
        'area': 8,
        'value_increase_point': 5,
        'value_increase_area': 1,
        'value_decay': 1,
    }
    points = torch.tensor([[4, 4]], dtype=torch.long)
    legacy_map = torch.zeros((1, 16, 16), dtype=torch.uint8)
    explicit_zero_map = legacy_map.clone()
    update_value_map(legacy_map, points, map_config)
    update_value_map(explicit_zero_map, points, map_config,
                     point_bonuses=torch.zeros(1, dtype=torch.long))
    assert torch.equal(legacy_map, explicit_zero_map), 'zero bonus changed legacy update'

    bonus_map = torch.zeros((1, 16, 16), dtype=torch.uint8)
    update_value_map(bonus_map, points, map_config,
                     point_bonuses=torch.ones(1, dtype=torch.long))
    assert int(bonus_map[0, 4, 4]) == int(legacy_map[0, 4, 4]) + 1

    height = width = 32
    identity = F.affine_grid(
        torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]), (1, 1, height, width),
        align_corners=True,
    )
    candidate_points = [torch.tensor([[8, 8], [20, 20]], dtype=torch.long)]
    content_points = [candidate_points[0].clone()]
    descriptor = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    detector = torch.ones((1, 1, height, width), dtype=torch.float32)
    vessel_mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
    config = {
        'geometric_thresh': 0.4,
        'content_thresh': 0.7,
        'pke_content_mode': 'one_way',
        'pke_content_weak_feedback': False,
        'pke_content_strong_feedback_multiplier': 1,
        'pke_content_weak_feedback_multiplier': 1,
        'pke_multiview_noncore_grid_size': 8,
        'pke_multiview_noncore_border_margin': 0,
        'pke_multiview_noncore_low_density_max': 4,
        'pke_multiview_noncore_max_per_image': 1,
        'pke_multiview_noncore_bonus': 1,
    }
    # The existing content filter is separately exercised by G0/D10.  This
    # preflight isolates the *new* selector and cap from floating-point
    # descriptor interpolation details by supplying a deterministic second
    # view content-pass result.
    original_content_filter = pke_module.content_filter

    def deterministic_content_filter(_descriptor, _affine_descriptor,
                                     geo_points, affine_geo_points, **_kwargs):
        weights = [torch.ones(len(point), dtype=torch.long, device=point.device)
                   for point in geo_points]
        return geo_points, affine_geo_points, weights

    pke_module.content_filter = deterministic_content_filter
    try:
        bonuses = pke_module.multiview_noncore_feedback_bonuses(
            content_points, candidate_points, descriptor, identity, detector,
            descriptor.clone(), vessel_mask, config,
        )
    finally:
        pke_module.content_filter = original_content_filter
    assert len(bonuses) == 1 and bonuses[0].shape == (2,)
    assert int(bonuses[0].sum()) == 1, 'multiview per-image cap was not respected'
    print('G4 multiview feedback smoke test: OK')


if __name__ == '__main__':
    main()
