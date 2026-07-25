"""Regression check for PKE diagnostics with a valid empty content-point list."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pke_module import select_diagnostic_feedback_points


def main():
    device = torch.device('cpu')
    empty = select_diagnostic_feedback_points([], torch.empty(0, dtype=torch.long), 1, device)
    assert empty.shape == (0, 2) and empty.device == device

    points = torch.tensor([[2, 3], [4, 5]], dtype=torch.long)
    selected = select_diagnostic_feedback_points(points, torch.tensor([1, 2]), 2, device)
    assert torch.equal(selected, torch.tensor([[4, 5]], dtype=torch.long))
    print('PKE empty-content diagnostic smoke test passed.')


if __name__ == '__main__':
    main()
