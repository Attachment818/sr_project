"""Masked-evidence self-distillation utilities for anatomy-context features."""

import copy

import torch
from torch import nn
from torch.nn import functional as F


def make_block_mask(
    images,
    min_fraction=0.18,
    max_fraction=0.42,
    block_fraction=(0.08, 0.28),
    generator=None,
):
    """Create irregular unions of rectangles without pretending to be lesions."""
    if images.ndim != 4:
        raise ValueError("images must have shape (B,C,H,W)")
    if not 0 < min_fraction <= max_fraction < 1:
        raise ValueError("invalid mask fraction range")
    if not 0 < block_fraction[0] <= block_fraction[1] < 1:
        raise ValueError("invalid block fraction range")

    batch, _, height, width = images.shape
    mask = torch.zeros(
        (batch, 1, height, width), device=images.device, dtype=images.dtype
    )
    for batch_index in range(batch):
        target = min_fraction + (max_fraction - min_fraction) * float(
            torch.rand((), generator=generator).item()
        )
        attempts = 0
        while float(mask[batch_index].mean().item()) < target:
            attempts += 1
            if attempts > 128:
                raise RuntimeError("could not reach requested mask coverage")
            scale_h = block_fraction[0] + (
                block_fraction[1] - block_fraction[0]
            ) * float(torch.rand((), generator=generator).item())
            scale_w = block_fraction[0] + (
                block_fraction[1] - block_fraction[0]
            ) * float(torch.rand((), generator=generator).item())
            block_h = max(1, int(round(height * scale_h)))
            block_w = max(1, int(round(width * scale_w)))
            top = int(torch.randint(
                0, max(1, height - block_h + 1), (), generator=generator
            ).item())
            left = int(torch.randint(
                0, max(1, width - block_w + 1), (), generator=generator
            ).item())
            mask[
                batch_index, :, top:top + block_h, left:left + block_w
            ] = 1
    return mask


def apply_evidence_corruption(
    images,
    mask,
    brightness=0.12,
    contrast=0.25,
    noise_std=0.025,
    generator=None,
):
    """Remove local evidence and apply geometry-preserving acquisition changes."""
    if images.ndim != 4 or mask.shape != images[:, :1].shape:
        raise ValueError("images/mask shapes are incompatible")
    batch = images.shape[0]
    random_shape = (batch, 1, 1, 1)
    random_device = images.device
    random_dtype = images.dtype

    def uniform(low, high):
        value = torch.rand(
            random_shape, generator=generator,
            device=random_device, dtype=random_dtype,
        )
        return low + (high - low) * value

    image_mean = images.mean(dim=(-2, -1), keepdim=True)
    corrupted = images * uniform(1.0 - contrast, 1.0 + contrast)
    corrupted = corrupted + uniform(-brightness, brightness)
    if noise_std > 0:
        noise = torch.randn(
            images.shape, generator=generator,
            device=random_device, dtype=random_dtype,
        )
        corrupted = corrupted + noise_std * noise
    corrupted = corrupted * (1.0 - mask) + image_mean * mask
    return corrupted.clamp(0.0, 1.0)


@torch.no_grad()
def update_ema_teacher(student, teacher, decay):
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0,1)")
    student_state = dict(student.named_parameters())
    for name, teacher_parameter in teacher.named_parameters():
        teacher_parameter.mul_(decay).add_(
            student_state[name].detach(), alpha=1.0 - decay
        )
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        teacher_buffer.copy_(student_buffers[name])


def _variance_loss(features, target_std):
    tokens = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    std = torch.sqrt(tokens.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(target_std - std).mean()


def _covariance_loss(features):
    tokens = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    covariance = tokens.T.matmul(tokens) / max(1, tokens.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag(diagonal)
    return off_diagonal.square().sum() / features.shape[1]


class AnatomyContextPretrainLoss(nn.Module):
    """Predict clean teacher features when the student's local evidence is absent."""

    def __init__(
        self,
        masked_weight=1.0,
        visible_weight=0.1,
        variance_weight=0.2,
        covariance_weight=0.02,
        target_std=0.08,
    ):
        super().__init__()
        self.masked_weight = float(masked_weight)
        self.visible_weight = float(visible_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.target_std = float(target_std)
        if self.masked_weight <= 0 or self.visible_weight < 0:
            raise ValueError("invalid feature loss weights")

    def forward(self, student_output, teacher_output, pixel_mask):
        student = student_output["descriptor"]
        teacher = teacher_output["descriptor"].detach()
        feature_mask = F.interpolate(
            pixel_mask, size=student.shape[-2:], mode="nearest"
        )
        distance = 1.0 - (student * teacher).sum(dim=1, keepdim=True)
        masked_count = feature_mask.sum().clamp_min(1.0)
        visible = 1.0 - feature_mask
        visible_count = visible.sum().clamp_min(1.0)
        masked_loss = (distance * feature_mask).sum() / masked_count
        visible_loss = (distance * visible).sum() / visible_count
        variance_loss = _variance_loss(student, self.target_std)
        covariance_loss = _covariance_loss(student)
        total = (
            self.masked_weight * masked_loss
            + self.visible_weight * visible_loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
        )
        return {
            "loss": total,
            "masked_feature_loss": masked_loss,
            "visible_feature_loss": visible_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
            "masked_feature_fraction": feature_mask.mean().detach(),
        }


def make_ema_teacher(student):
    """Return a frozen, same-state teacher for standalone pretraining."""
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher
