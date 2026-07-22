"""Aggregate read-only FIMD inference diagnostics across model variants.

The test runner writes one JSON record per image pair when
``save_inference_diagnostics: true``.  This tool only reads those records and
produces a per-pair comparison CSV plus a compact Markdown report.  It never
changes inference or training behaviour.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FIELDS = (
    'detected_query_keypoints', 'detected_refer_keypoints', 'ratio_matches',
    'inverse_consistency_matches', 'outlier_filter_matches',
    'matches_after_consistency', 'geometry_inliers', 'geometry_inlier_rate',
    'average_control_point_error', 'registration_success', 'failure_reason',
)


def parse_source(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Each --source must use LABEL=PATH.')
    label, file_name = value.split('=', 1)
    if not label or not file_name:
        raise argparse.ArgumentTypeError('Each --source must use LABEL=PATH.')
    return label, Path(file_name)


def read_jsonl(path):
    records = {}
    with path.open(encoding='utf-8') as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            pair_id = item.get('pair_id')
            if not pair_id:
                raise ValueError(f'{path}:{line_num} has no pair_id')
            records[pair_id] = item
    return records


def region_value(record, field):
    region = record.get(field) or {}
    return region.get('grid_coverage')


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def fmt(value, digits=3):
    if value is None:
        return '—'
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, type=parse_source,
                        help='LABEL=diagnostics.jsonl; may be repeated.')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--focus-pair', action='append', default=['control_points_39_r_t', 'control_points_40_r_t'],
                        help='Pair ID highlighted in the Markdown report; may be repeated.')
    args = parser.parse_args()

    sources = dict(args.source)
    records_by_label = {label: read_jsonl(path) for label, path in sources.items()}
    pair_ids = sorted(set().union(*(records.keys() for records in records_by_label.values())))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = ['pair_id']
    for label in sources:
        columns += [f'{label}__{field}' for field in FIELDS]
        columns += [f'{label}__query_grid_coverage', f'{label}__refer_grid_coverage']
    csv_path = args.output_dir / 'pair_diagnostics_comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for pair_id in pair_ids:
            row = {'pair_id': pair_id}
            for label, records in records_by_label.items():
                record = records.get(pair_id, {})
                for field in FIELDS:
                    row[f'{label}__{field}'] = record.get(field)
                row[f'{label}__query_grid_coverage'] = region_value(record, 'returned_query_regions')
                row[f'{label}__refer_grid_coverage'] = region_value(record, 'returned_refer_regions')
            writer.writerow(row)

    report = ['# seed3409 FIMD 逐对推理诊断', '', '## 全体 70 对的聚合统计', '',
              '| 方法 | 有记录对数 | 配准成功率 | 最终匹配数均值 | 第一阶段几何内点均值 | 最终几何内点率均值 | Query 覆盖均值 | Refer 覆盖均值 | 成功对控制点误差均值 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, records in records_by_label.items():
        rows = list(records.values())
        report.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            label, len(rows), fmt(mean([r.get('registration_success') for r in rows])),
            fmt(mean([r.get('matches_after_consistency') for r in rows])),
            fmt(mean([r.get('geometry_inliers') for r in rows])),
            fmt(mean([r.get('geometry_inlier_rate') for r in rows])),
            fmt(mean([region_value(r, 'returned_query_regions') for r in rows])),
            fmt(mean([region_value(r, 'returned_refer_regions') for r in rows])),
            fmt(mean([r.get('average_control_point_error') for r in rows if r.get('registration_success')]))))

    report += ['', '## 重点困难对', '']
    for pair_id in args.focus_pair:
        report += [f'### {pair_id}', '', '| 方法 | 最终匹配 | 第一阶段几何内点 | 最终几何内点率 | Query 覆盖 | Refer 覆盖 | 配准成功 | 控制点误差 | 失败原因 |',
                   '|---|---:|---:|---:|---:|---:|---|---:|---|']
        for label, records in records_by_label.items():
            r = records.get(pair_id, {})
            report.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                label, fmt(r.get('matches_after_consistency')), fmt(r.get('geometry_inliers')),
                fmt(r.get('geometry_inlier_rate')), fmt(region_value(r, 'returned_query_regions')),
                fmt(region_value(r, 'returned_refer_regions')), fmt(r.get('registration_success')),
                fmt(r.get('average_control_point_error')), str(r.get('failure_reason') or '—').replace('|', '/')))
        report.append('')

    report += ['## 使用说明', '',
               '- `ratio_matches → inverse_consistency_matches → outlier_filter_matches` 的下降位置，可区分 descriptor 比率筛选、逆一致性、几何异常值过滤造成的匹配损失。',
               '- `geometry_inliers` 是第一阶段几何估计的内点数；若启用 matching trick，`geometry_inlier_rate` 可能来自第二阶段，因此二者不应直接相除。',
               '- `grid_coverage` 是 4×4 网格的已占用比例，只作空间覆盖诊断；不会参与此次测试的匹配或几何估计。',
               '- 报告仅比较同一 FIMD 协议、同一 seed、同一 ep149 权重下的推理观测。']
    (args.output_dir / 'pair_diagnostics_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Wrote {csv_path} and {args.output_dir / "pair_diagnostics_report.md"}')


if __name__ == '__main__':
    main()
