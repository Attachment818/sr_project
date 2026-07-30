"""Focused invariants for the G14 EMA-teacher training mechanism."""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.super_retina import (  # noqa: E402
    SuperRetinaWithEMATeacherPKE,
    SuperRetinaWithVesselOnlyMasked,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-config')
    args = parser.parse_args()

    base_config = {
        'nms_size': 10, 'nms_thresh': 0.1,
        'geometric_thresh': 0.4, 'content_thresh': 0.7,
        'gaussian_kernel_size': 13, 'gaussian_sigma': 2,
        'vessel_weight': 0.3, 'vessel_schedule_mode': 'constant',
    }
    torch.manual_seed(7)
    legacy = SuperRetinaWithVesselOnlyMasked(base_config, device='cpu')
    torch.manual_seed(7)
    disabled = SuperRetinaWithEMATeacherPKE(
        {**base_config, 'pke_ema_teacher_enabled': False}, device='cpu'
    )
    disabled_student = {
        name: tensor for name, tensor in disabled.state_dict().items()
        if not name.startswith('ema_teacher.')
    }
    for name, tensor in legacy.state_dict().items():
        assert torch.equal(tensor, disabled_student[name]), (
            f'default-disabled G14 changed student tensor: {name}'
        )
    assert all(
        not parameter.requires_grad
        for parameter in disabled.ema_teacher.parameters()
    )
    optimizer_parameters = [
        parameter for parameter in disabled.parameters()
        if parameter.requires_grad
    ]
    teacher_ids = {id(p) for p in disabled.ema_teacher.parameters()}
    assert not teacher_ids.intersection(map(id, optimizer_parameters))

    if args.cuda_config:
        with open(args.cuda_config, encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        train_config = {
            **config['MODEL'], **config['PKE'],
            **config['DATASET'], **config['VALUE_MAP'],
        }
        device = torch.device(train_config['device'])
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA config supplied but CUDA is unavailable')
        model = SuperRetinaWithEMATeacherPKE(
            train_config, device=device
        )
        model.current_epoch = model.pke_ema_teacher_start_epoch
        model.train()
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters()
             if parameter.requires_grad),
            lr=1e-4,
        )
        batch_size = int(train_config['batch_size'])
        height = int(train_config['model_image_height'])
        width = int(train_config['model_image_width'])
        images = torch.rand(
            batch_size, 1, height, width, device=device
        )
        labels = torch.zeros_like(images)
        for y in range(96, height, 192):
            for x in range(96, width, 192):
                labels[:, 0, y, x] = 1
        value_maps = torch.zeros_like(labels, dtype=torch.uint8)
        learn_index = (torch.arange(batch_size, device=device),)
        before = next(model.ema_teacher.parameters()).detach().clone()
        torch.cuda.reset_peak_memory_stats(device)
        output = model(images, labels, value_maps, learn_index)
        loss = output[0]
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for name, parameter in model.named_parameters()
            if not name.startswith('ema_teacher.')
        )
        assert all(
            parameter.grad is None
            for parameter in model.ema_teacher.parameters()
        )
        optimizer.step()
        model.update_ema_teacher()
        after = next(model.ema_teacher.parameters()).detach()
        assert not torch.equal(before, after)
        assert all(p.grad is None for p in model.ema_teacher.parameters())
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(
            f'G14 full-batch CUDA preflight passed on {device}; '
            f'batch={batch_size}, image={height}x{width}, '
            f'teacher_delta={float((after - before).norm()):.6f}, '
            f'peak_allocated={peak_gib:.2f} GiB'
        )
    print('G14 EMA teacher invariant self-test passed')


if __name__ == '__main__':
    main()
