#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


SUMMARY_FIELDS = [
    'road_scenario',
    'traffic_level',
    'checkpoint_prefix',
    'eval_episodes',
    'mean_return',
    'success_rate',
    'collision_rate',
    'timeout_rate',
    'mean_length',
    'mean_speed',
    'mean_progress',
    'termination_reason_distribution',
    'collision_count',
    'collision_mean_speed',
    'collision_mean_progress',
    'startup_count',
    'car_following_count',
    'lane_change_count',
    'overtake_count',
    'interaction_count',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Aggregate chunked eval detail CSVs into final eval artifacts.')
    parser.add_argument('--summary-csvs', nargs='+', type=Path, required=True)
    parser.add_argument('--detail-csvs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifact-stem', type=str, required=True)
    return parser.parse_args()


def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as file_obj:
        return list(csv.DictReader(file_obj))


def write_rows(path: Path, fieldnames, rows) -> None:
    with path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row, key):
    return float(row.get(key, 0.0) or 0.0)


def as_int(row, key):
    return int(float(row.get(key, 0) or 0))


def phase_counts(collision_rows):
    counter = Counter(row.get('collision_phase', '') for row in collision_rows)
    return {
        'startup_count': counter.get('startup', 0),
        'car_following_count': counter.get('car_following', 0),
        'lane_change_count': counter.get('lane_change', 0),
        'overtake_count': counter.get('overtake', 0),
        'interaction_count': counter.get('interaction', 0),
    }


def aggregate_summary(summary_rows, detail_rows):
    if not summary_rows:
        raise RuntimeError('No chunk summary rows were provided.')
    if not detail_rows:
        raise RuntimeError('No chunk detail rows were provided.')

    first = summary_rows[0]
    detail_rows = sorted(detail_rows, key=lambda row: as_int(row, 'episode_index'))
    eval_episodes = len(detail_rows)
    reason_counter = Counter(row.get('termination_reason', '') for row in detail_rows)
    reason_parts = [
        f'{reason}:{count}/{eval_episodes}'
        for reason, count in sorted(reason_counter.items())
        if reason
    ]

    collision_rows = [row for row in detail_rows if as_int(row, 'collision') > 0]
    collision_speeds = [as_float(row, 'mean_speed') for row in collision_rows]
    collision_progress = [as_float(row, 'final_progress') for row in collision_rows]
    counts = phase_counts(collision_rows)

    row = {
        'road_scenario': first['road_scenario'],
        'traffic_level': first['traffic_level'],
        'checkpoint_prefix': first['checkpoint_prefix'],
        'eval_episodes': eval_episodes,
        'mean_return': np.mean([as_float(row, 'episode_return') for row in detail_rows]),
        'success_rate': np.mean([as_float(row, 'success') for row in detail_rows]),
        'collision_rate': np.mean([as_float(row, 'collision') for row in detail_rows]),
        'timeout_rate': np.mean([as_float(row, 'timeout') for row in detail_rows]),
        'mean_length': np.mean([as_float(row, 'episode_length') for row in detail_rows]),
        'mean_speed': np.mean([as_float(row, 'mean_speed') for row in detail_rows]),
        'mean_progress': np.mean([as_float(row, 'final_progress') for row in detail_rows]),
        'termination_reason_distribution': '|'.join(reason_parts),
        'collision_count': len(collision_rows),
        'collision_mean_speed': np.mean(collision_speeds) if collision_speeds else 0.0,
        'collision_mean_progress': np.mean(collision_progress) if collision_progress else 0.0,
        **counts,
    }
    return {key: f'{value:.6f}' if isinstance(value, float) else value for key, value in row.items()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for path in args.summary_csvs:
        rows = read_rows(path.resolve())
        if len(rows) != 1:
            raise RuntimeError(f'Expected one summary row in {path}')
        summary_rows.append(rows[0])

    detail_rows = []
    detail_fieldnames = None
    for path in args.detail_csvs:
        rows = read_rows(path.resolve())
        if rows and detail_fieldnames is None:
            detail_fieldnames = list(rows[0].keys())
        detail_rows.extend(rows)
    if detail_fieldnames is None:
        raise RuntimeError('No detail rows found.')
    detail_rows = sorted(detail_rows, key=lambda row: as_int(row, 'episode_index'))

    summary_path = args.output_dir / f'eval_summary_{args.artifact_stem}.csv'
    detail_path = args.output_dir / f'eval_detail_{args.artifact_stem}.csv'
    failure_path = args.output_dir / f'eval_failure_{args.artifact_stem}.csv'

    final_summary = aggregate_summary(summary_rows, detail_rows)
    write_rows(summary_path, SUMMARY_FIELDS, [final_summary])
    write_rows(detail_path, detail_fieldnames, detail_rows)
    failure_rows = [
        row for row in detail_rows
        if as_int(row, 'success') == 0
    ]
    write_rows(failure_path, detail_fieldnames, failure_rows)

    print(f'Wrote: {summary_path}')
    print(f'Wrote: {detail_path}')
    print(f'Wrote: {failure_path}')


if __name__ == '__main__':
    main()
