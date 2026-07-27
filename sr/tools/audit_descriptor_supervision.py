"""Audit when and why G0 descriptor supervision is skipped during training."""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.train_util import value_map_load
from dataset.retina_dataset import RetinaDataset
import model.super_retina as super_retina_module
from model.super_retina import SuperRetinaWithVesselOnlyMasked

OUTPUT_NAMES = (
    'descriptor_supervision_per_image.csv',
    'descriptor_supervision_per_batch.csv',
    'descriptor_supervision_summary.json',
    'descriptor_supervision_report.md',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    return parser.parse_args()


def classify_counts(sample_counts, sample_limit=1000):
    over_limit = [i for i, count in enumerate(sample_counts) if count > sample_limit]
    nonempty = [i for i, count in enumerate(sample_counts) if count > 0]
    if over_limit:
        return 'over_limit_batch_abort', over_limit, nonempty, []
    if nonempty:
        return 'trained', over_limit, nonempty, nonempty
    return 'all_images_empty', over_limit, nonempty, []


def self_test():
    assert classify_counts([500, 700])[0] == 'trained'
    assert classify_counts([0, 0])[0] == 'all_images_empty'
    reason, over, nonempty, participating = classify_counts([1001, 20])
    assert reason == 'over_limit_batch_abort'
    assert over == [0] and nonempty == [0, 1] and participating == []
    assert classify_counts([1000])[0] == 'trained'

    # Exercise the real descriptor_loss hook without constructing the network.
    model = super_retina_module.SuperRetina.__new__(super_retina_module.SuperRetina)
    nn.Module.__init__(model)
    model.PKE_learn = True
    model.nms_size = 10
    model.nms_thresh = 0.1
    model.scale = 8
    model.config = {}
    model._capture_descriptor_supervision_audit = True
    original_sampler = super_retina_module.sample_descriptors

    def fake_sampler(*_args, **_kwargs):
        descriptors = [torch.zeros(4, 1001), torch.zeros(4, 20)]
        affine_descriptors = [torch.zeros(4, 1001), torch.zeros(4, 20)]
        keypoints = [torch.zeros(1001, 2), torch.zeros(20, 2)]
        return descriptors, affine_descriptors, keypoints

    super_retina_module.sample_descriptors = fake_sampler
    try:
        detector = torch.zeros(2, 1, 2, 2)
        labels = torch.zeros_like(detector)
        descriptor = torch.zeros(2, 4, 1, 1)
        loss, trained = model.descriptor_loss(
            detector, labels, descriptor, descriptor, torch.zeros(2, 2, 2, 2)
        )
        assert float(loss) == 0.0 and trained is False
        captured = model._descriptor_supervision_audit
        assert captured['sample_counts'] == [1001, 20]
        assert captured['exit_reason'] == 'over_limit_batch_abort'
        assert captured['participating_indices'] == []
    finally:
        super_retina_module.sample_descriptors = original_sampler
    print('descriptor supervision audit self-test passed')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_configuration(audit):
    train_path = Path(audit['train_config_path'])
    checkpoint = Path(audit['checkpoint_path'])
    value_map_dir = Path(audit['value_map_dir'])
    for path, label in (
        (train_path, 'train config'),
        (checkpoint, 'checkpoint'),
        (value_map_dir, 'value-map directory'),
    ):
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
        raise FileExistsError(
            'Refusing to overwrite descriptor audit result(s): ' + ', '.join(map(str, occupied))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config, checkpoint, value_map_dir = load_configuration(audit)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Descriptor supervision audit requires the configured CUDA device')
    seed = int(audit.get('seed', config.get('seed', 3407)))
    set_seed(seed)

    dataset = RetinaDataset(
        config['dataset_path'], split_file=config['train_split_file'], is_train=True,
        data_shape=(config['model_image_height'], config['model_image_width']),
        auxiliary=config.get('auxiliary'),
    )
    loader = DataLoader(
        dataset, batch_size=int(audit.get('batch_size', 2)), shuffle=False,
        num_workers=int(audit.get('num_workers', 0)),
    )
    maximum_batches = int(audit.get('num_batches', 64))
    if maximum_batches <= 0:
        raise ValueError('num_batches must be positive')

    model = SuperRetinaWithVesselOnlyMasked(config, device=device)
    model.load_pretrained_weights(str(checkpoint), device=device, strict=False)
    model.current_epoch = int(audit.get('epoch', 149))
    model.PKE_learn = True
    model.train()
    model._capture_descriptor_supervision_audit = True

    image_rows = []
    batch_rows = []
    completed = 0
    progress = tqdm(total=maximum_batches, desc='descriptor supervision audit', unit='batch')
    try:
        for batch in loader:
            images, input_with_label, keypoint_positions, label_names = batch
            learn_index = torch.where(input_with_label)
            if len(learn_index[0]) == 0:
                continue
            images = images.to(device)
            keypoint_positions = keypoint_positions.to(device)
            value_maps = value_map_load(
                str(value_map_dir), label_names, input_with_label, images.shape[-2:]
            ).to(device)
            model._descriptor_supervision_audit = None
            with torch.no_grad():
                model(images, keypoint_positions, value_maps, learn_index)
            captured = model._descriptor_supervision_audit
            if captured is None:
                raise RuntimeError('Model did not expose descriptor-supervision audit data')

            counts = captured['sample_counts']
            expected = classify_counts(counts, captured['sample_limit'])
            observed = (
                captured['exit_reason'], captured['over_limit_indices'],
                captured['nonempty_indices'], captured['participating_indices'],
            )
            if observed != expected:
                raise RuntimeError('Descriptor audit classification disagrees with model path')
            reason = captured['exit_reason']
            participating = set(captured['participating_indices'])
            over_limit = set(captured['over_limit_indices'])
            total_samples = sum(counts)
            used_samples = sum(counts[i] for i in participating)
            batch_rows.append({
                'batch_index': completed,
                'exit_reason': reason,
                'image_count': len(counts),
                'nonempty_image_count': len(captured['nonempty_indices']),
                'over_limit_image_count': len(over_limit),
                'participating_image_count': len(participating),
                'total_correspondences': total_samples,
                'used_correspondences': used_samples,
                'discarded_correspondences': total_samples - used_samples,
                'maximum_image_correspondences': max(counts, default=0),
            })
            for image_index, count in enumerate(counts):
                image_rows.append({
                    'batch_index': completed,
                    'image_index': image_index,
                    'label_name': str(label_names[image_index]),
                    'correspondence_count': count,
                    'is_empty': int(count == 0),
                    'is_over_limit': int(image_index in over_limit),
                    'participated_in_loss': int(image_index in participating),
                    'batch_exit_reason': reason,
                })
            completed += 1
            progress.update(1)
            progress.set_postfix(
                skipped=sum(row['exit_reason'] != 'trained' for row in batch_rows),
                discarded=sum(row['discarded_correspondences'] for row in batch_rows),
            )
            del images, keypoint_positions, value_maps
            if completed >= maximum_batches:
                break
    finally:
        progress.close()
    if completed < maximum_batches:
        raise RuntimeError(
            f'Only {completed} labelled batches were available; expected {maximum_batches}'
        )

    for path, rows in ((generated[0], image_rows), (generated[1], batch_rows)):
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    reasons = Counter(row['exit_reason'] for row in batch_rows)
    counts = [row['correspondence_count'] for row in image_rows]
    total = sum(row['total_correspondences'] for row in batch_rows)
    used = sum(row['used_correspondences'] for row in batch_rows)
    discarded = total - used
    summary = {
        'audit_type': 'g0_descriptor_supervision_effective_rate',
        'checkpoint': str(checkpoint),
        'device': str(device),
        'batches': completed,
        'images': len(image_rows),
        'sample_limit': 1000,
        'batch_exit_reason_counts': dict(reasons),
        'batch_skip_fraction': (completed - reasons.get('trained', 0)) / completed,
        'over_limit_image_count': sum(row['is_over_limit'] for row in image_rows),
        'empty_image_count': sum(row['is_empty'] for row in image_rows),
        'correspondences': {
            'total': total,
            'used': used,
            'discarded': discarded,
            'effective_fraction': used / total if total else None,
            'per_image_mean': float(np.mean(counts)) if counts else None,
            'per_image_median': float(np.median(counts)) if counts else None,
            'per_image_maximum': max(counts, default=0),
            'per_image_p90': float(np.percentile(counts, 90)) if counts else None,
            'per_image_p95': float(np.percentile(counts, 95)) if counts else None,
            'per_image_p99': float(np.percentile(counts, 99)) if counts else None,
        },
        'interpretation': {
            'over_limit_batch_abort': (
                'At least one image exceeded 1000 correspondences, so the current code '
                'discarded descriptor supervision for the entire batch.'
            ),
            'all_images_empty': 'No valid descriptor correspondence was sampled in the batch.',
            'trained': 'All non-empty images contributed to the descriptor loss.',
        },
        'limitations': [
            'This replays G0 epoch-149 training batches and does not alter the checkpoint.',
            'The audit measures supervision admission, not descriptor quality or test AUC.',
        ],
    }
    generated[2].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    effective = summary['correspondences']['effective_fraction']
    report = [
        '# G0 descriptor supervision effective-rate audit', '',
        f'Checkpoint: {checkpoint}', f'Batches/images: {completed}/{len(image_rows)}', '',
        '| Exit reason | Batches | Fraction |', '|---|---:|---:|',
    ]
    for reason in ('trained', 'over_limit_batch_abort', 'all_images_empty'):
        count = reasons.get(reason, 0)
        report.append(f'| {reason} | {count} | {count / completed:.4f} |')
    report.extend([
        '', '| Correspondences | Count |', '|---|---:|',
        f'| Total | {total} |', f'| Used by descriptor loss | {used} |',
        f'| Discarded | {discarded} |',
        f"| Effective fraction | {effective:.4f} |" if effective is not None
        else '| Effective fraction | n/a |',
        '', f"Over-limit images: {summary['over_limit_image_count']}",
        f"Empty images: {summary['empty_image_count']}", '',
    ])
    generated[3].write_text('\n'.join(report), encoding='utf-8')
    print(f'Wrote descriptor supervision audit: {output_dir}')


if __name__ == '__main__':
    main()
