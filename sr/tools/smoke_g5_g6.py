"""Smoke tests for G5 negatives and the G6/G7 descriptor architectures."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.super_retina import (
    SuperRetina,
    SuperRetinaWithDecoupledMultiScaleDescriptor,
    SuperRetinaWithResidualMultiScaleDescriptor,
    chunked_hard_negative_indices,
)
import model.super_retina as super_retina_module


def legacy_indices(descriptor, affine_descriptor, keypoints=None, min_distance=0.0):
    n = descriptor.shape[1]
    dis = torch.norm(
        descriptor[:, :, None] - affine_descriptor[:, None, :], dim=0
    )
    rows = torch.arange(n)
    dis[rows, rows] = dis.max() + 1
    if min_distance > 0:
        spatial = torch.cdist(keypoints.float(), keypoints.float())
        dis[spatial < min_distance] = dis.max() + 1
    return dis.argmin(dim=1)


def minimal_config():
    return {
        'nms_size': 10,
        'nms_thresh': 0.1,
        'gaussian_kernel_size': 13,
        'gaussian_sigma': 2,
        'vessel_weight': 0.3,
        'vessel_schedule_mode': 'constant',
        'geometric_thresh': 0.4,
        'descriptor_hard_negative_mode': 'chunked',
        'descriptor_hard_negative_chunk_size': 16,
    }


def check_chunked_equivalence():
    torch.manual_seed(3407)
    descriptor = torch.randn(16, 73)
    affine_descriptor = torch.randn(16, 73)
    keypoints = torch.rand(73, 2) * 64
    for minimum_distance in (0.0, 8.0):
        expected = legacy_indices(
            descriptor, affine_descriptor, keypoints, minimum_distance
        )
        for chunk_size in (1, 7, 32, 256):
            observed = chunked_hard_negative_indices(
                descriptor,
                affine_descriptor,
                keypoints=keypoints,
                min_negative_distance=minimum_distance,
                chunk_size=chunk_size,
            )
            assert torch.equal(expected, observed), (
                f'chunked hard negatives differ: distance={minimum_distance}, '
                f'chunk={chunk_size}'
            )
    large = chunked_hard_negative_indices(
        torch.randn(4, 1001), torch.randn(4, 1001), chunk_size=64
    )
    assert large.shape == (1001,)


def check_over_limit_descriptor_loss():
    model = SuperRetina.__new__(SuperRetina)
    nn.Module.__init__(model)
    model.PKE_learn = True
    model.nms_size = 10
    model.nms_thresh = 0.1
    model.scale = 8
    model.config = {
        'descriptor_hard_negative_mode': 'chunked',
        'descriptor_hard_negative_chunk_size': 64,
    }
    n = 1001
    original_sampler = super_retina_module.sample_descriptors

    def fake_sampler(*_args, **_kwargs):
        anchor = torch.randn(4, n)
        positive = anchor + torch.randn_like(anchor) * 0.01
        return [anchor], [positive], [torch.rand(n, 2) * 64]

    super_retina_module.sample_descriptors = fake_sampler
    try:
        detector = torch.zeros(1, 1, 2, 2)
        labels = torch.zeros_like(detector)
        descriptor = torch.zeros(1, 4, 1, 1)
        loss, trained = model.descriptor_loss(
            detector, labels, descriptor, descriptor, torch.zeros(1, 2, 2, 2)
        )
        assert trained is True
        assert loss.ndim == 0 and torch.isfinite(loss)
    finally:
        super_retina_module.sample_descriptors = original_sampler


def check_g6_structure_and_gradients():
    model = SuperRetinaWithDecoupledMultiScaleDescriptor(
        minimal_config(), device='cpu'
    )
    model.eval()
    image = torch.randn(1, 1, 64, 64)
    detector, descriptor = model.network(image)
    assert detector.shape == (1, 1, 64, 64)
    assert descriptor.shape[1] == 256
    assert torch.isfinite(detector).all() and torch.isfinite(descriptor).all()

    descriptor_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith('descriptor_')
    ]
    detector_deep_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(('conv3a.', 'conv3b.', 'conv4a.', 'conv4b.'))
    ]
    detector_to_descriptor = torch.autograd.grad(
        detector.sum(), descriptor_parameters, allow_unused=True, retain_graph=True
    )
    descriptor_to_detector = torch.autograd.grad(
        descriptor.sum(), detector_deep_parameters, allow_unused=True
    )
    assert all(gradient is None for gradient in detector_to_descriptor)
    assert all(gradient is None for gradient in descriptor_to_detector)

    state = model.state_dict()
    restored = SuperRetinaWithDecoupledMultiScaleDescriptor(
        minimal_config(), device='cpu'
    )
    restored.load_state_dict(state, strict=True)


def check_g7_residual_structure_and_gradients():
    config = minimal_config()
    config['descriptor_hard_negative_mode'] = 'legacy'
    config['descriptor_multiscale_gate_init'] = 0.1
    model = SuperRetinaWithResidualMultiScaleDescriptor(config, device='cpu')
    model.eval()
    image = torch.randn(1, 1, 64, 64)
    detector, descriptor = model.network(image)
    assert detector.shape == (1, 1, 64, 64)
    assert descriptor.shape[1] == 256
    assert torch.isfinite(detector).all() and torch.isfinite(descriptor).all()
    assert hasattr(model, 'convDa'), 'G7 must retain the G0 convDa path'
    observed_gate = torch.sigmoid(
        model.descriptor_multiscale_gate_logit
    ).item()
    assert abs(observed_gate - 0.1) < 1e-6

    residual_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith('descriptor_residual_')
    ]
    detector_to_residual = torch.autograd.grad(
        detector.sum(), residual_parameters, allow_unused=True, retain_graph=True
    )
    descriptor_to_residual = torch.autograd.grad(
        descriptor.sum(), residual_parameters, allow_unused=True
    )
    assert all(gradient is None for gradient in detector_to_residual)
    assert all(gradient is not None for gradient in descriptor_to_residual)

    state = model.state_dict()
    restored = SuperRetinaWithResidualMultiScaleDescriptor(config, device='cpu')
    restored.load_state_dict(state, strict=True)


def check_cuda_memory(config_path):
    source = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    config = {**source['MODEL'], **source['PKE'], **source['DATASET'], **source['VALUE_MAP']}
    model_classes = {
        'vessel_masked_decoupled_multiscale':
            SuperRetinaWithDecoupledMultiScaleDescriptor,
        'vessel_masked_residual_multiscale':
            SuperRetinaWithResidualMultiScaleDescriptor,
    }
    model_variant = config['model_variant']
    if model_variant not in model_classes:
        raise ValueError(
            f'CUDA preflight does not support model variant: {model_variant}'
        )
    device = torch.device(config['device'])
    if not torch.cuda.is_available() or device.type != 'cuda':
        raise RuntimeError(f'Configured CUDA device is unavailable: {device}')
    model = model_classes[model_variant](config, device=device)
    model.train()
    batch_size = int(config['batch_size'])
    height = int(config['model_image_height'])
    width = int(config['model_image_width'])
    image = torch.randn(batch_size, 1, height, width, device=device)
    torch.cuda.reset_peak_memory_stats(device)

    detector, descriptor, decoder = model.network(image, return_cPa=True)
    with torch.no_grad():
        auxiliary_detector, auxiliary_descriptor = model.network(image)
    _, affine_descriptor = model.network(image)
    loss = (
        detector.mean() + descriptor.square().mean() + decoder.square().mean()
        + affine_descriptor.square().mean()
    )
    loss.backward()
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    assert torch.isfinite(loss)
    assert auxiliary_detector.shape == detector.shape
    assert auxiliary_descriptor.shape == descriptor.shape
    print(
        f'CUDA preflight passed: variant={model_variant}, device={device}, '
        f'batch={batch_size}, '
        f'image={height}x{width}, peak_allocated={peak_gib:.2f} GiB'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-config', type=Path)
    args = parser.parse_args()
    check_chunked_equivalence()
    check_over_limit_descriptor_loss()
    check_g6_structure_and_gradients()
    check_g7_residual_structure_and_gradients()
    print('G5/G6/G7 CPU smoke tests passed')
    if args.cuda_config is not None:
        check_cuda_memory(args.cuda_config)


if __name__ == '__main__':
    main()
