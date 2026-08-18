"""CPU smoke test for the independent anatomy-context feature prototype."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.anatomy_context.network import AnatomyContextFeatureNet
from model.anatomy_context.pretraining import (
    AnatomyContextPretrainLoss,
    apply_evidence_corruption,
    make_block_mask,
    make_ema_teacher,
    update_ema_teacher,
)


def main():
    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(23)
    student = AnatomyContextFeatureNet(
        in_channels=1, descriptor_dim=32, feature_dim=64
    )
    teacher = make_ema_teacher(student)
    images = torch.rand(2, 1, 64, 64, generator=generator)
    mask = make_block_mask(
        images, min_fraction=0.2, max_fraction=0.3,
        generator=generator,
    )
    fraction = float(mask.mean().item())
    assert 0.2 <= fraction <= 0.5, fraction
    corrupted = apply_evidence_corruption(
        images, mask, generator=generator
    )
    assert corrupted.shape == images.shape
    assert torch.isfinite(corrupted).all()
    assert not torch.allclose(corrupted, images)

    with torch.no_grad():
        teacher_output = teacher(images)
    student_output = student(corrupted)
    assert student_output["descriptor"].shape == (2, 32, 8, 8)
    assert student_output["local_gate"].shape == (2, 1, 8, 8)
    assert student_output["repeatability"].shape == (2, 1, 8, 8)
    assert student_output["reliability"].shape == (2, 1, 8, 8)
    descriptor_norm = torch.linalg.vector_norm(
        student_output["descriptor"], dim=1
    )
    assert torch.allclose(
        descriptor_norm, torch.ones_like(descriptor_norm), atol=1e-5
    )
    gate = student_output["local_gate"]
    assert bool(((gate > 0) & (gate < 1)).all())

    criterion = AnatomyContextPretrainLoss()
    losses = criterion(student_output, teacher_output, mask)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    required = {
        "local_projection.0.weight",
        "context_fusion.0.0.weight",
        "fusion_gate.0.weight",
        "descriptor_head.1.weight",
    }
    parameters = dict(student.named_parameters())
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None, f"missing gradient: {name}"
        assert torch.isfinite(gradient).all(), f"invalid gradient: {name}"
        assert float(gradient.abs().sum().item()) > 0, name
    assert all(parameter.grad is None for parameter in teacher.parameters())

    teacher_before = next(teacher.parameters()).detach().clone()
    with torch.no_grad():
        next(student.parameters()).add_(0.1)
    update_ema_teacher(student, teacher, decay=0.9)
    teacher_after = next(teacher.parameters()).detach()
    assert not torch.equal(teacher_before, teacher_after)
    assert torch.allclose(
        teacher_after, teacher_before + 0.01, atol=1e-6
    )
    print("anatomy-context feature smoke test passed")


if __name__ == "__main__":
    main()
