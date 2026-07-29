"""Read-only tensor statistics and changes across an ordered checkpoint series."""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml


def tensor_stats(tensor, transform='identity'):
    value = tensor.detach().cpu().float()
    record = {
        'shape': list(value.shape),
        'numel': value.numel(),
        'l1_mean': float(value.abs().mean()),
        'l2_norm': float(torch.linalg.vector_norm(value)),
        'max_abs': float(value.abs().max()),
        'nonzero_fraction': float(torch.count_nonzero(value) / value.numel()),
    }
    if value.numel() == 1:
        raw = float(value.item())
        record['raw_value'] = raw
        if transform == 'sigmoid':
            record['transformed_value'] = 1.0 / (1.0 + math.exp(-raw))
        elif transform != 'identity':
            raise ValueError(f'Unsupported transform: {transform}')
    elif transform != 'identity':
        raise ValueError('Transforms are supported only for scalar tensors')
    return record


def self_test():
    zero = torch.zeros(2, 2)
    one = torch.ones(2, 2)
    assert tensor_stats(zero)['nonzero_fraction'] == 0.0
    assert tensor_stats(one)['l2_norm'] == 2.0
    scalar = tensor_stats(torch.tensor(0.0), 'sigmoid')
    assert scalar['transformed_value'] == 0.5
    print('checkpoint tensor-series audit self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_config is None:
        raise ValueError('--audit-config is required unless --self-test is used')

    audit = yaml.safe_load(
        args.audit_config.read_text(encoding='utf-8')
    )['AUDIT']
    output_path = Path(audit['output_path'])
    if output_path.exists():
        raise FileExistsError(
            f'Refusing to overwrite tensor-series audit: {output_path}'
        )
    if output_path.suffix.lower() != '.json':
        raise ValueError('output_path must use the .json extension')

    checkpoint_specs = list(audit['checkpoints'])
    parameter_specs = list(audit['parameters'])
    if not checkpoint_specs:
        raise ValueError('At least one checkpoint must be configured')
    if not parameter_specs:
        raise ValueError('At least one parameter must be configured')
    checkpoint_labels = [item['label'] for item in checkpoint_specs]
    checkpoint_paths = [item['path'] for item in checkpoint_specs]
    parameter_names = [item['name'] for item in parameter_specs]
    if len(set(checkpoint_labels)) != len(checkpoint_labels):
        raise ValueError('Checkpoint labels must be unique')
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError('Checkpoint paths must be unique')
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError('Parameter names must be unique')
    previous = {}
    checkpoints = []
    for item in checkpoint_specs:
        checkpoint_path = Path(item['path'])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('net', checkpoint)
        records = []
        current = {}
        for spec in parameter_specs:
            name = spec['name']
            if name not in state:
                raise KeyError(
                    f'Checkpoint parameter not found in {checkpoint_path}: {name}'
                )
            tensor = state[name].detach().cpu().float()
            if tensor.numel() == 0:
                raise ValueError(
                    f'Checkpoint parameter is empty in {checkpoint_path}: {name}'
                )
            current[name] = tensor
            record = {
                'parameter_name': name,
                **tensor_stats(tensor, spec.get('transform', 'identity')),
            }
            if name in previous:
                delta = tensor - previous[name]
                record['delta_l2_from_previous'] = float(
                    torch.linalg.vector_norm(delta)
                )
                denominator = max(
                    float(torch.linalg.vector_norm(previous[name])), 1e-12
                )
                record['relative_delta_l2_from_previous'] = (
                    record['delta_l2_from_previous'] / denominator
                )
            records.append(record)
        checkpoints.append({
            'label': item['label'],
            'path': str(checkpoint_path),
            'saved_epoch': checkpoint.get('epoch'),
            'records': records,
        })
        previous = current

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        'audit_type': 'checkpoint_tensor_series',
        'checkpoints': checkpoints,
        'safety': 'All checkpoints were loaded read-only on CPU.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote checkpoint tensor-series audit: {output_path}')


if __name__ == '__main__':
    main()
