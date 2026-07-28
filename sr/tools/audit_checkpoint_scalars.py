"""Read scalar checkpoint parameters without modifying the checkpoint."""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    return parser.parse_args()


def scalar_record(state_dict, parameter_name, transform):
    if parameter_name not in state_dict:
        raise KeyError(f'Checkpoint parameter not found: {parameter_name}')
    tensor = state_dict[parameter_name]
    if not torch.is_tensor(tensor) or tensor.numel() != 1:
        raise ValueError(
            f'Checkpoint parameter must be a scalar tensor: {parameter_name}'
        )
    raw_value = float(tensor.detach().cpu().item())
    if transform == 'identity':
        transformed_value = raw_value
    elif transform == 'sigmoid':
        transformed_value = 1.0 / (1.0 + math.exp(-raw_value))
    else:
        raise ValueError(f'Unsupported scalar transform: {transform}')
    return {
        'parameter_name': parameter_name,
        'transform': transform,
        'raw_value': raw_value,
        'transformed_value': transformed_value,
    }


def self_test():
    state = {'gate': torch.tensor(math.log(0.1 / 0.9))}
    record = scalar_record(state, 'gate', 'sigmoid')
    assert abs(record['transformed_value'] - 0.1) < 1e-6
    try:
        scalar_record(state, 'missing', 'identity')
    except KeyError:
        pass
    else:
        raise AssertionError('Missing checkpoint parameter was not rejected')
    print('checkpoint scalar audit self-test passed')


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_config is None:
        raise ValueError('--audit-config is required unless --self-test is used')
    audit = yaml.safe_load(
        args.audit_config.read_text(encoding='utf-8')
    )['AUDIT']
    checkpoint_path = Path(audit['checkpoint_path'])
    output_path = Path(audit['output_path'])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    if output_path.exists():
        raise FileExistsError(
            f'Refusing to overwrite checkpoint scalar audit: {output_path}'
        )
    if output_path.suffix.lower() != '.json':
        raise ValueError('output_path must use the .json extension')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('net', checkpoint)
    records = [
        scalar_record(
            state_dict,
            item['parameter_name'],
            item.get('transform', 'identity'),
        )
        for item in audit['parameters']
    ]
    payload = {
        'audit_type': 'checkpoint_scalar_parameters',
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'records': records,
        'safety': 'The checkpoint is loaded read-only on CPU and is not modified.',
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'Wrote checkpoint scalar audit: {output_path}')


if __name__ == '__main__':
    main()
