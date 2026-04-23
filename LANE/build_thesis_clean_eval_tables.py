#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build thesis-ready clean evaluation tables.')
    parser.add_argument('--standard-summary', type=Path, required=True)
    parser.add_argument('--dense-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def read_single_row(csv_path: Path):
    with csv_path.open('r', encoding='utf-8', newline='') as file_obj:
        rows = list(csv.DictReader(file_obj))
    if len(rows) != 1:
        raise RuntimeError(f'Expected exactly one row in {csv_path}')
    return rows[0]


def to_paper_row(summary_row):
    return {
        'scenario': f"{summary_row['road_scenario']}-{summary_row['traffic_level']}",
        'road_scenario': summary_row['road_scenario'],
        'traffic_level': summary_row['traffic_level'],
        'checkpoint_prefix': summary_row['checkpoint_prefix'],
        'eval_episodes': int(summary_row['eval_episodes']),
        'success_rate': float(summary_row['success_rate']),
        'collision_rate': float(summary_row['collision_rate']),
        'timeout_rate': float(summary_row['timeout_rate']),
        'average_reward': float(summary_row['mean_return']),
        'average_episode_length': float(summary_row['mean_length']),
        'average_speed': float(summary_row['mean_speed']),
        'average_progress': float(summary_row['mean_progress']),
    }


def write_csv(output_path: Path, rows) -> None:
    fieldnames = list(rows[0].keys())
    with output_path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output_path: Path, rows) -> None:
    lines = [
        '| Scenario | Success Rate | Collision Rate | Timeout Rate | Average Reward | Average Episode Length | Average Speed | Average Progress |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            '| {scenario} | {success_rate:.4f} | {collision_rate:.4f} | {timeout_rate:.4f} | {average_reward:.4f} | {average_episode_length:.4f} | {average_speed:.4f} | {average_progress:.4f} |'.format(
                **row
            )
        )
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        to_paper_row(read_single_row(args.standard_summary.resolve())),
        to_paper_row(read_single_row(args.dense_summary.resolve())),
    ]

    csv_path = args.output_dir / 'highway_clean_evaluation_table.csv'
    md_path = args.output_dir / 'highway_clean_evaluation_table.md'
    txt_path = args.output_dir / 'highway_clean_evaluation_table.txt'

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    txt_path.write_text(md_path.read_text(encoding='utf-8'), encoding='utf-8')

    print(f'Wrote: {csv_path}')
    print(f'Wrote: {md_path}')
    print(f'Wrote: {txt_path}')


if __name__ == '__main__':
    main()
