"""Summarize read-only PKE diagnostic JSONL records by epoch and stage."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


STAGES = ('detector_candidates', 'geometric_pass', 'content_pass', 'value_map_points')
METRICS = (
    'count', 'vessel_core_count', 'vessel_edge_count', 'non_vessel_count',
    'grid_occupied_cells', 'grid_coverage',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to pke_diagnostics.jsonl')
    parser.add_argument('--output', required=True, help='Path to summary JSON')
    args = parser.parse_args()

    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with open(args.input, encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            epoch = str(record['epoch'])
            for stage in STAGES:
                summary = record['stages'][stage]
                bucket = buckets[epoch][stage]
                bucket['images'] += 1
                for metric in METRICS:
                    bucket[metric] += float(summary[metric])

    output = {'epochs': {}}
    for epoch, stage_buckets in sorted(buckets.items(), key=lambda item: int(item[0])):
        output['epochs'][epoch] = {}
        for stage, totals in stage_buckets.items():
            images = totals['images']
            output['epochs'][epoch][stage] = {
                'images': int(images),
                **{f'average_{metric}': totals[metric] / images for metric in METRICS},
            }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote PKE diagnostic summary: {output_path}')


if __name__ == '__main__':
    main()
