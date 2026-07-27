"""Measure detector/descriptor gradient conflict in the shared G0 encoder."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.train_util import value_map_load
from dataset.retina_dataset import RetinaDataset
from model.super_retina import SuperRetinaWithVesselOnlyMasked


OUTPUT_NAMES = (
    'gradient_conflict_per_batch.csv',
    'gradient_conflict_summary.json',
    'gradient_conflict_report.md',
)
STAGES = {
    'conv1': ('conv1a.', 'conv1b.'),
    'conv2': ('conv2a.', 'conv2b.'),
    'conv3': ('conv3a.', 'conv3b.'),
    'conv4': ('conv4a.', 'conv4b.'),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gradient_metrics(detector_grads, descriptor_grads, indices):
    dot = 0.0
    detector_sq = 0.0
    descriptor_sq = 0.0
    tensor_count = 0
    for index in indices:
        detector = detector_grads[index]
        descriptor = descriptor_grads[index]
        if detector is None or descriptor is None:
            continue
        detector = detector.detach().float()
        descriptor = descriptor.detach().float()
        dot += float((detector * descriptor).sum().cpu())
        detector_sq += float((detector * detector).sum().cpu())
        descriptor_sq += float((descriptor * descriptor).sum().cpu())
        tensor_count += 1
    detector_norm = math.sqrt(detector_sq)
    descriptor_norm = math.sqrt(descriptor_sq)
    cosine = None if detector_norm == 0 or descriptor_norm == 0 else dot / (detector_norm * descriptor_norm)
    return {
        'cosine': cosine,
        'detector_grad_norm': detector_norm,
        'descriptor_grad_norm': descriptor_norm,
        'dot_product': dot,
        'tensor_count': tensor_count,
    }


def self_test():
    detector = [torch.tensor([1.0, 0.0]), torch.tensor([1.0])]
    aligned = [torch.tensor([1.0, 0.0]), torch.tensor([1.0])]
    opposed = [torch.tensor([-1.0, 0.0]), torch.tensor([-1.0])]
    assert gradient_metrics(detector, aligned, [0, 1])['cosine'] > 0.999
    assert gradient_metrics(detector, opposed, [0, 1])['cosine'] < -0.999
    print('gradient conflict self-test passed')


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def median(values):
    values = [float(value) for value in values if value is not None]
    return float(np.median(values)) if values else None


def load_configuration(audit):
    train_path = Path(audit['train_config_path'])
    checkpoint = Path(audit['checkpoint_path'])
    value_map_dir = Path(audit['value_map_dir'])
    for path, label in ((train_path, 'train config'), (checkpoint, 'checkpoint'),
                        (value_map_dir, 'value-map directory')):
        if not path.exists():
            raise FileNotFoundError(f'{label} not found: {path}')
    source = yaml.safe_load(train_path.read_text(encoding='utf-8'))
    config = {**source['MODEL'], **source['PKE'], **source['DATASET'], **source['VALUE_MAP']}
    config['device'] = audit.get('device', config.get('device', 'cuda:0'))
    return config, checkpoint, value_map_dir


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_config is None:
        raise ValueError('--audit-config is required unless --self-test is used')
    audit = yaml.safe_load(args.audit_config.read_text(encoding='utf-8'))['AUDIT']
    output_dir = Path(audit['output_dir'])
    generated = [output_dir / name for name in OUTPUT_NAMES]
    occupied = [path for path in generated if path.exists()]
    if occupied:
        raise FileExistsError('Refusing to overwrite gradient audit result(s): ' + ', '.join(map(str, occupied)))
    output_dir.mkdir(parents=True, exist_ok=True)
    config, checkpoint, value_map_dir = load_configuration(audit)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Gradient audit requires the configured CUDA device')
    seed = int(audit.get('seed', config.get('seed', 3407)))
    set_seed(seed)

    dataset = RetinaDataset(
        config['dataset_path'], split_file=config['train_split_file'], is_train=True,
        data_shape=(config['model_image_height'], config['model_image_width']),
        auxiliary=config.get('auxiliary'),
    )
    loader = DataLoader(dataset, batch_size=int(audit.get('batch_size', 2)), shuffle=False,
                        num_workers=int(audit.get('num_workers', 0)))
    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(str(checkpoint), device=device, strict=False)
    model.current_epoch = int(audit.get('epoch', 149))
    model.PKE_learn = True
    model.train()
    model._capture_gradient_audit_losses = True

    named = [(name, parameter) for name, parameter in model.named_parameters()
             if any(name.startswith(prefix) for prefixes in STAGES.values() for prefix in prefixes)]
    if not named:
        raise RuntimeError('No shared encoder parameters found for gradient audit')
    parameters = [parameter for _, parameter in named]
    stage_indices = {
        stage: [index for index, (name, _) in enumerate(named)
                if any(name.startswith(prefix) for prefix in prefixes)]
        for stage, prefixes in STAGES.items()
    }
    if any(not indices for indices in stage_indices.values()):
        raise RuntimeError(f'Incomplete shared encoder stage mapping: {stage_indices}')

    rows = []
    maximum_batches = int(audit.get('num_batches', 8))
    completed = 0
    for batch_index, batch in enumerate(tqdm(loader, desc='gradient conflict audit', unit='batch')):
        images, input_with_label, keypoint_positions, label_names = batch
        learn_index = torch.where(input_with_label)
        if len(learn_index[0]) == 0:
            continue
        images = images.to(device)
        keypoint_positions = keypoint_positions.to(device)
        value_maps = value_map_load(str(value_map_dir), label_names, input_with_label,
                                    images.shape[-2:]).to(device)
        model.zero_grad(set_to_none=True)
        model(images, keypoint_positions, value_maps, learn_index)
        captured = getattr(model, '_gradient_audit_losses', None)
        if captured is None:
            raise RuntimeError('Model did not expose gradient-audit loss components')
        detector_loss, descriptor_loss = captured
        detector_grads = torch.autograd.grad(
            detector_loss, parameters, retain_graph=True, allow_unused=True,
        )
        descriptor_grads = torch.autograd.grad(
            descriptor_loss, parameters, retain_graph=False, allow_unused=True,
        )
        batch_metrics = {'shared_all': gradient_metrics(
            detector_grads, descriptor_grads, list(range(len(parameters))))}
        batch_metrics.update({stage: gradient_metrics(detector_grads, descriptor_grads, indices)
                              for stage, indices in stage_indices.items()})
        for stage, metrics in batch_metrics.items():
            rows.append({
                'batch_index': completed, 'stage': stage,
                'detector_loss': float(detector_loss.detach().cpu()),
                'descriptor_loss': float(descriptor_loss.detach().cpu()),
                **metrics,
                'conflict': None if metrics['cosine'] is None else int(metrics['cosine'] < 0),
            })
        model._gradient_audit_losses = None
        completed += 1
        del detector_loss, descriptor_loss, detector_grads, descriptor_grads, images, value_maps
        torch.cuda.empty_cache()
        if completed >= maximum_batches:
            break
    if completed < maximum_batches:
        raise RuntimeError(f'Only {completed} labelled batches were available; expected {maximum_batches}')

    with generated[0].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {
        'audit_type': 'g0_detector_descriptor_shared_encoder_gradient_conflict',
        'checkpoint': str(checkpoint), 'device': str(device), 'batches': completed,
        'stages': {},
        'interpretation': 'Negative cosine means detector and descriptor losses request opposing updates in that shared encoder stage.',
    }
    for stage in ('shared_all', *STAGES.keys()):
        selected = [row for row in rows if row['stage'] == stage]
        cosines = [row['cosine'] for row in selected]
        summary['stages'][stage] = {
            'cosine_mean': mean(cosines), 'cosine_median': median(cosines),
            'negative_fraction': mean([row['conflict'] for row in selected]),
            'detector_grad_norm_mean': mean([row['detector_grad_norm'] for row in selected]),
            'descriptor_grad_norm_mean': mean([row['descriptor_grad_norm'] for row in selected]),
        }
    candidates = [stage for stage in STAGES
                  if summary['stages'][stage]['negative_fraction'] is not None
                  and summary['stages'][stage]['negative_fraction'] >= 0.5]
    summary['candidate_split_stage'] = candidates[0] if candidates else None
    summary['limitations'] = [
        'This is a local optimization diagnostic on G0 training batches, not proof that architectural separation improves AUC.',
        'The selected split stage must also consider memory, parameter count, and detector/descriptor ablations.',
    ]
    generated[1].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ['# G0 detector/descriptor gradient conflict', '',
              f'Checkpoint: {checkpoint}', f'Batches: {completed}', '',
              '| Stage | Mean cosine | Median cosine | Negative fraction |',
              '|---|---:|---:|---:|']
    for stage, metrics in summary['stages'].items():
        report.append('| {} | {:.4f} | {:.4f} | {:.4f} |'.format(
            stage, metrics['cosine_mean'], metrics['cosine_median'], metrics['negative_fraction']))
    report.extend(['', f"Candidate split stage: {summary['candidate_split_stage']}", ''])
    generated[2].write_text('\n'.join(report), encoding='utf-8')
    print(f'Wrote gradient conflict audit: {output_dir}')


if __name__ == '__main__':
    main()
