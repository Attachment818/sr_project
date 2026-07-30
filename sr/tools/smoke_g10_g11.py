"""Fail-fast CPU/CUDA checks for G10 dense supervision and G11 PCGrad."""

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.train_util import (
    conflict_projected_backward,
    snapshot_value_map_directory,
    training_resource_summary,
)
from model.super_retina import (
    SuperRetinaWithResidualMultiScaleDescriptor,
    SuperRetinaWithVesselOnlyMasked,
    SuperRetinaWithMultiScaleDetectorResidual,
    SuperRetinaWithZeroStartResidualMultiScaleDescriptor,
    SuperRetinaWithNormControlledZeroStartMultiScaleDescriptor,
)


def minimal_config(**updates):
    config = {
        'nms_size': 10,
        'nms_thresh': 0.1,
        'gaussian_kernel_size': 13,
        'gaussian_sigma': 2,
        'vessel_weight': 0.3,
        'vessel_schedule_mode': 'constant',
        'geometric_thresh': 0.4,
        'dense_descriptor_weight': 0.1,
        'dense_descriptor_ramp_epochs': 10,
        'dense_descriptor_grid_size': 4,
        'dense_descriptor_structure_per_cell': 1,
        'dense_descriptor_uniform_per_cell': 1,
        'dense_descriptor_border_margin': 2,
        'dense_descriptor_valid_intensity': 0.0,
        'dense_descriptor_min_negative_distance': 4.0,
        'dense_descriptor_margin': 0.2,
        'log_dense_descriptor_stats': True,
    }
    config.update(updates)
    return config


def identity_grid(batch, height, width, device):
    theta = torch.eye(2, 3, device=device)[None].repeat(batch, 1, 1)
    return torch.nn.functional.affine_grid(
        theta, (batch, 1, height, width), align_corners=True
    )


def check_dense_descriptor():
    torch.manual_seed(3407)
    model = SuperRetinaWithVesselOnlyMasked(
        minimal_config(), device='cpu'
    )
    image = torch.rand(2, 1, 64, 64)
    descriptor = torch.randn(2, 16, 8, 8, requires_grad=True)
    affine_descriptor = (
        descriptor.detach() + 0.02 * torch.randn_like(descriptor)
    ).requires_grad_(True)
    loss = model._balanced_dense_descriptor_loss(
        image, descriptor, affine_descriptor,
        identity_grid(2, 64, 64, image.device),
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert descriptor.grad is not None and torch.isfinite(descriptor.grad).all()
    assert (
        affine_descriptor.grad is not None
        and torch.isfinite(affine_descriptor.grad).all()
    )
    summary = model.dense_descriptor_epoch_summary()
    assert summary['calls'] == 1
    assert summary['sampled_points_per_call'] > 0
    assert summary['valid_pairs_per_call'] > 0
    assert summary['occupied_cells_per_call'] > 0

    g10 = SuperRetinaWithResidualMultiScaleDescriptor(
        minimal_config(descriptor_multiscale_gate_init=0.1),
        device='cpu',
    )
    g10.eval()
    probe = torch.rand(1, 1, 64, 64)
    _, full_descriptor = g10.network(probe)
    descriptor_only = g10.network(probe, descriptor_only=True)
    assert torch.equal(full_descriptor, descriptor_only), (
        'descriptor-only forward must be numerically identical'
    )


class ToyConflictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Module()
        self.network.conv1a = nn.Linear(1, 1, bias=False)
        self.detector_head = nn.Linear(1, 1, bias=False)
        self.descriptor_head = nn.Linear(1, 1, bias=False)


def check_pcgrad():
    model = ToyConflictModel()
    shared = model.network.conv1a.weight
    detector_loss = shared.sum() + model.detector_head.weight.sum()
    descriptor_loss = -2 * shared.sum() + model.descriptor_head.weight.sum()
    stats = conflict_projected_backward(
        model, detector_loss, descriptor_loss
    )
    assert stats['conflict'] == 1.0
    assert stats['cosine_before'] < 0
    assert stats['cosine_after'] >= -1e-6
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def check_g13_detector_residual():
    config = minimal_config(
        dense_descriptor_weight=0.0,
        detector_multiscale_gate_init=0.05,
        detector_multiscale_gate_max=0.15,
        detector_multiscale_hidden_channels=8,
        log_detector_residual_stats=True,
    )
    base = SuperRetinaWithVesselOnlyMasked(config, device='cpu')
    g13 = SuperRetinaWithMultiScaleDetectorResidual(
        config, device='cpu'
    )
    g13.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    g13.eval()
    probe = torch.rand(1, 1, 64, 64)
    base_detector, base_descriptor = base.network(probe)
    g13_detector, g13_descriptor = g13.network(probe)
    assert torch.equal(base_detector, g13_detector), (
        'zero-initialized G13 detector must exactly equal G0'
    )
    assert torch.equal(base_descriptor, g13_descriptor), (
        'G13 must preserve the G0 descriptor'
    )
    assert torch.count_nonzero(
        g13.detector_residual_fusion[-1].weight
    ) == 0
    g13.train()
    detector, _ = g13.network(probe)
    detector.mean().backward()
    assert g13.detector_residual_fusion[-1].weight.grad is not None
    assert torch.isfinite(
        g13.detector_residual_fusion[-1].weight.grad
    ).all()


def check_g15_zero_start_descriptor():
    config = minimal_config(
        dense_descriptor_weight=0.0,
        descriptor_multiscale_gate_init=0.1,
        log_descriptor_gate_stats=True,
    )
    base = SuperRetinaWithVesselOnlyMasked(config, device='cpu')
    g15 = SuperRetinaWithZeroStartResidualMultiScaleDescriptor(
        config, device='cpu'
    )
    g15.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    g15.eval()
    probe = torch.rand(1, 1, 64, 64)
    base_detector, base_descriptor = base.network(probe)
    g15_detector, g15_descriptor = g15.network(probe)
    assert torch.equal(base_detector, g15_detector)
    assert torch.equal(base_descriptor, g15_descriptor), (
        'zero-start G15 descriptor must exactly equal G0'
    )
    projection = g15.descriptor_residual_fusion[-1]
    assert torch.count_nonzero(projection.weight) == 0
    g15.train()
    _, descriptor = g15.network(probe)
    descriptor.square().mean().backward()
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert float(projection.weight.grad.abs().sum()) > 0


def check_g16_norm_controlled_descriptor():
    config = minimal_config(
        dense_descriptor_weight=0.0,
        descriptor_multiscale_gate_init=0.1,
        descriptor_injection_norm_control_enabled=True,
        descriptor_injection_ratio_cap=0.2,
        log_descriptor_gate_stats=True,
    )
    base = SuperRetinaWithVesselOnlyMasked(config, device='cpu')
    g16 = SuperRetinaWithNormControlledZeroStartMultiScaleDescriptor(
        config, device='cpu'
    )
    g16.load_state_dict(base.state_dict(), strict=False)
    probe = torch.rand(1, 1, 64, 64)
    base.eval()
    g16.eval()
    base_detector, base_descriptor = base.network(probe)
    g16_detector, g16_descriptor = g16.network(probe)
    assert torch.equal(base_detector, g16_detector)
    assert torch.equal(base_descriptor, g16_descriptor), (
        'zero-start G16 descriptor must exactly equal G0'
    )

    # The control must leave a small residual untouched and cap a large one.
    main = torch.ones(1, 256, 2, 2)
    small = torch.full_like(main, 0.01)
    large = torch.full_like(main, 100.0)
    g16.eval()
    small_controlled = g16._descriptor_injection(main, small)
    gate = torch.sigmoid(g16.descriptor_multiscale_gate_logit)
    assert torch.allclose(small_controlled, gate * small)
    large_controlled = g16._descriptor_injection(main, large)
    ratio = (
        torch.norm(large_controlled, p=2, dim=1)
        / torch.norm(main, p=2, dim=1)
    )
    assert float(ratio.max()) <= 0.200001

    projection = g16.descriptor_residual_fusion[-1]
    g16.train()
    _, descriptor = g16.network(probe)
    descriptor.square().mean().backward()
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert float(projection.weight.grad.abs().sum()) > 0


def check_recovery_infrastructure():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / 'value_maps'
        snapshots = root / 'snapshots'
        source.mkdir()
        (source / 'sample.png').write_bytes(b'value-map')
        observed = Path(snapshot_value_map_directory(
            str(source), str(snapshots), 29
        ))
        assert (observed / 'sample.png').read_bytes() == b'value-map'
        try:
            snapshot_value_map_directory(
                str(source), str(snapshots), 29
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError('snapshot overwrite was not refused')
    resources = training_resource_summary(torch.device('cpu'))
    assert isinstance(resources, dict)


def check_cuda(config_path):
    source = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    config = {
        **source['MODEL'], **source['PKE'],
        **source['DATASET'], **source['VALUE_MAP'],
    }
    device = torch.device(config['device'])
    if not torch.cuda.is_available() or device.type != 'cuda':
        raise RuntimeError(f'Configured CUDA device is unavailable: {device}')
    model_variant = config['model_variant']
    model_classes = {
        'vessel_masked_residual_multiscale':
            SuperRetinaWithResidualMultiScaleDescriptor,
        'vessel_masked_pcgrad':
            SuperRetinaWithVesselOnlyMasked,
        'vessel_masked_detector_residual_multiscale':
            SuperRetinaWithMultiScaleDetectorResidual,
        'vessel_masked_zero_start_residual_multiscale':
            SuperRetinaWithZeroStartResidualMultiScaleDescriptor,
        'vessel_masked_norm_controlled_zero_start_multiscale':
            SuperRetinaWithNormControlledZeroStartMultiScaleDescriptor,
    }
    if model_variant not in model_classes:
        raise ValueError(
            f'Unsupported G10/G11/G13 preflight variant: {model_variant}'
        )
    model_class = model_classes[model_variant]
    model = model_class(config, device=device)
    model.train()
    model.nms_thresh = 2.0
    model.PKE_learn = False
    model._capture_gradient_audit_losses = (
        config.get('shared_gradient_mode', 'standard') == 'pcgrad'
    )
    batch_size = int(config['batch_size'])
    height = int(config['model_image_height'])
    width = int(config['model_image_width'])
    image = torch.rand(batch_size, 1, height, width, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    labels = torch.zeros(
        batch_size, 1, height, width, device=device
    )
    for y in range(96, height, 192):
        for x in range(96, width, 192):
            labels[:, 0, y, x] = 1
    value_maps = torch.zeros_like(labels, dtype=torch.uint8)
    learn_index = (torch.arange(batch_size, device=device),)
    output = model(image, labels, value_maps, learn_index)
    loss = output[0]
    if config.get('shared_gradient_mode', 'standard') == 'pcgrad':
        stats = conflict_projected_backward(
            model, *model._gradient_audit_losses
        )
        assert all(
            torch.isfinite(torch.tensor(value))
            for value in stats.values()
        )
    else:
        loss.backward()
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    print(
        f'CUDA preflight passed: variant={model_variant}, device={device}, '
        f'batch={batch_size}, image={height}x{width}, '
        f'peak_allocated={peak_gib:.2f} GiB'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-config', type=Path)
    args = parser.parse_args()
    check_dense_descriptor()
    check_pcgrad()
    check_g13_detector_residual()
    check_g15_zero_start_descriptor()
    check_g16_norm_controlled_descriptor()
    check_recovery_infrastructure()
    print('G10/G11/G13/G15/G16 CPU smoke tests passed')
    if args.cuda_config is not None:
        check_cuda(args.cuda_config)


if __name__ == '__main__':
    main()
