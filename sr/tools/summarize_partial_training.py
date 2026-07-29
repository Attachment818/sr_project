"""Summarize per-epoch training diagnostics from an interrupted train.log."""

import argparse
import csv
import json
import re
from pathlib import Path


FLOAT = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def parse_key_values(text):
    return {
        key: float(value)
        for key, value in re.findall(
            rf'([A-Za-z_][A-Za-z0-9_]*)=({FLOAT})', text
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.log.is_file():
        raise FileNotFoundError(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path for path in args.output_dir.iterdir()
        if path.name != 'audit.yaml'
    ]
    if existing:
        raise FileExistsError(
            f'Refusing to overwrite non-empty output directory: '
            f'{args.output_dir}'
        )

    text = args.log.read_text(encoding='utf-8', errors='replace')
    epoch_matches = list(re.finditer(r'^Epoch\s+(\d+)/\d+', text, re.M))
    rows = []
    for index, match in enumerate(epoch_matches):
        start = match.start()
        end = (
            epoch_matches[index + 1].start()
            if index + 1 < len(epoch_matches) else len(text)
        )
        block = text[start:end]
        row = {'epoch': int(match.group(1))}
        dense = re.search(
            r'balanced dense descriptor:\s*(.+)', block
        )
        if dense:
            row.update({
                f'dense_{key}': value
                for key, value in parse_key_values(dense.group(1)).items()
            })
        training = re.search(
            rf'train overall loss:\s*({FLOAT}).*?'
            rf'detector_loss:\s*({FLOAT}).*?'
            rf'#avg learned keypoints:({FLOAT}).*?'
            rf'descriptor_loss:\s*({FLOAT})',
            block, re.S,
        )
        if training:
            row.update({
                'overall_loss': float(training.group(1)),
                'detector_loss': float(training.group(2)),
                'avg_learned_keypoints': float(training.group(3)),
                'descriptor_loss': float(training.group(4)),
            })
        resources = re.search(r'training resources:\s*(.+)', block)
        if resources:
            row.update({
                f'resource_{key}': value
                for key, value in parse_key_values(
                    resources.group(1)
                ).items()
            })
        rows.append(row)

    if not rows:
        raise RuntimeError(f'No epoch records found in {args.log}')
    fields = ['epoch'] + sorted({
        key for row in rows for key in row if key != 'epoch'
    })
    csv_path = args.output_dir / 'partial_training_epochs.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    dense_rows = [
        row for row in rows if 'dense_loss' in row
    ]
    summary = {
        'source_log': str(args.log.resolve()),
        'epoch_records': len(rows),
        'first_epoch': rows[0]['epoch'],
        'last_epoch': rows[-1]['epoch'],
        'dense_records': len(dense_rows),
    }
    for key in (
        'dense_sampled_points_per_call',
        'dense_valid_pairs_per_call',
        'dense_occupied_cells_per_call',
        'dense_positive_distance',
        'dense_negative_distance',
        'dense_loss',
        'descriptor_loss',
        'avg_learned_keypoints',
    ):
        values = [row[key] for row in rows if key in row]
        if values:
            summary[key] = {
                'first': values[0],
                'last': values[-1],
                'minimum': min(values),
                'maximum': max(values),
                'mean': sum(values) / len(values),
            }
    json_path = args.output_dir / 'partial_training_summary.json'
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
