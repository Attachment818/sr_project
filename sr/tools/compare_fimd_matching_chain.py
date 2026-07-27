"""Compare two existing read-only FIMD matching-chain audit CSV files."""

import argparse
import csv
import json
from pathlib import Path

import yaml


METRICS = (
    'detected_query', 'ratio_matches', 'inverse_consistency_matches',
    'returned_matches', 'ransac_inliers', 'match_grid_cells',
    'inlier_grid_cells', 'match_hull_fraction', 'inlier_hull_fraction',
    'legacy_lmeds_control_mean_error',
)


def mean(values):
    return sum(values) / len(values) if values else None


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {row['pair_id']: row for row in csv.DictReader(handle)}


def as_float(row, field):
    value = row.get(field)
    return None if value in (None, '') else float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', required=True, type=Path)
    args = parser.parse_args()
    audit = yaml.safe_load(args.audit_config.read_text(encoding='utf-8'))['AUDIT']
    source_a = Path(audit['source_a_csv'])
    source_b = Path(audit['source_b_csv'])
    output_dir = Path(audit['output_dir'])
    for path, label in ((source_a, 'source_a_csv'), (source_b, 'source_b_csv')):
        if not path.is_file():
            raise FileNotFoundError(f'{label} not found: {path}')
    generated = [output_dir / name for name in (
        'g0_g4_matching_chain_comparison.csv',
        'g0_g4_matching_chain_comparison.json',
        'g0_g4_matching_chain_report.md',
    )]
    if any(path.exists() for path in generated):
        raise FileExistsError('Refusing to overwrite existing D12 comparison output')
    output_dir.mkdir(parents=True, exist_ok=True)

    first, second = read_rows(source_a), read_rows(source_b)
    common_pairs = sorted(set(first) & set(second))
    if not common_pairs:
        raise ValueError('No common FIMD pair IDs between the two audits')
    rows = []
    for pair_id in common_pairs:
        row = {'pair_id': pair_id}
        for metric in METRICS:
            a, b = as_float(first[pair_id], metric), as_float(second[pair_id], metric)
            row[f'g0_{metric}'] = a
            row[f'g4_{metric}'] = b
            row[f'delta_g4_minus_g0_{metric}'] = None if a is None or b is None else b - a
        rows.append(row)
    with generated[0].open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    aggregate = {
        metric: {
            'g0_mean': mean([row[f'g0_{metric}'] for row in rows if row[f'g0_{metric}'] is not None]),
            'g4_mean': mean([row[f'g4_{metric}'] for row in rows if row[f'g4_{metric}'] is not None]),
            'g4_minus_g0_mean': mean([row[f'delta_g4_minus_g0_{metric}'] for row in rows if row[f'delta_g4_minus_g0_{metric}'] is not None]),
        }
        for metric in METRICS
    }
    payload = {
        'audit_type': 'read_only_g0_g4_fimd_matching_chain_comparison',
        'source_a_csv': str(source_a), 'source_b_csv': str(source_b),
        'pair_count': len(rows), 'metrics': aggregate, 'pairs': rows,
        'focus_pairs': audit.get('focus_pairs', []),
    }
    generated[1].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ['# D12: G0 vs G4 FIMD matching chain', '', f'Common pairs: {len(rows)}', '', '| Metric | G0 mean | G4 mean | G4-G0 |', '|---|---:|---:|---:|']
    for metric, values in aggregate.items():
        report.append('| {} | {:.3f} | {:.3f} | {:+.3f} |'.format(metric, values['g0_mean'], values['g4_mean'], values['g4_minus_g0_mean']))
    generated[2].write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Wrote D12 comparison: {output_dir}')


if __name__ == '__main__':
    main()
