#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path('/root/autodl-tmp/SVPG2023/LANE')
RUNNER = REPO_ROOT / 'run_RL_ours.py'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'training_runs' / 'ours' / 'highway_eval_reports'
SUPPORTED_SCENARIOS = ['standard', 'dense']


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate one best checkpoint on highway standard and dense.')
    parser.add_argument('--checkpoint_prefix', type=str, required=True)
    parser.add_argument('--eval_episodes', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--model', type=str, default='rwtaspk')
    parser.add_argument('--road_scenario', type=str, default='highway', choices=['highway'])
    parser.add_argument('--traffic_levels', nargs='+', default=['standard', 'dense'])
    parser.add_argument('--lane_profile', type=str, default='auto', choices=['auto', 'legacy'])
    parser.add_argument('--highway_reward_stage', type=str, default='c_success')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def validate_args(args):
    requested = []
    for traffic_level in args.traffic_levels:
        if traffic_level == 'light':
            raise SystemExit('Light evaluation is intentionally disabled for this batch script.')
        if traffic_level not in SUPPORTED_SCENARIOS:
            raise SystemExit(f'Unsupported traffic level: {traffic_level}')
        if traffic_level not in requested:
            requested.append(traffic_level)
    args.traffic_levels = requested


def read_single_row_csv(csv_path):
    with csv_path.open('r', encoding='utf-8', newline='') as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError(f'Expected exactly one row in summary CSV: {csv_path}')
    return rows[0]


def read_rows(csv_path):
    with csv_path.open('r', encoding='utf-8', newline='') as file_obj:
        return list(csv.DictReader(file_obj))


def run_eval_for_traffic_level(args, traffic_level):
    artifact_stem = f'bestckpt_{args.road_scenario}_{traffic_level}_{int(args.eval_episodes)}ep_seed{int(args.seed):02d}'
    scenario_root = REPO_ROOT / 'training_runs' / 'ours' / f'{args.road_scenario}_{traffic_level}'
    summary_path = scenario_root / 'eval_reports' / f'eval_summary_{artifact_stem}.csv'
    detail_path = scenario_root / 'eval_reports' / f'eval_detail_{artifact_stem}.csv'
    failure_path = scenario_root / 'eval_reports' / f'eval_failure_{artifact_stem}.csv'
    if summary_path.exists() and detail_path.exists() and failure_path.exists():
        return artifact_stem, summary_path, detail_path, failure_path

    command = [
        sys.executable,
        str(RUNNER),
        '--model', args.model,
        '--road_scenario', args.road_scenario,
        '--traffic_level', traffic_level,
        '--cuda', str(args.cuda),
        '--seed', str(args.seed),
        '--lane_profile', args.lane_profile,
        '--highway_reward_stage', args.highway_reward_stage,
        '--eval_only',
        '--eval_episodes', str(args.eval_episodes),
        '--warm_start_prefix', args.checkpoint_prefix,
        '--eval_save_details',
        '--eval_artifact_stem', artifact_stem,
    ]
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not detail_path.exists():
        raise FileNotFoundError(detail_path)
    if not failure_path.exists():
        raise FileNotFoundError(failure_path)
    return artifact_stem, summary_path, detail_path, failure_path


def write_csv(output_path, fieldnames, rows):
    with output_path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric_comparison(output_dir, summary_rows, metric_key, title, filename, ylim=None):
    labels = [row['traffic_level'] for row in summary_rows]
    values = [float(row[metric_key]) for row in summary_rows]
    colors = ['#1f77b4', '#d62728']
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    ax.bar(labels, values, color=colors[:len(labels)], width=0.55)
    ax.set_title(title)
    ax.set_ylabel(metric_key)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for index, value in enumerate(values):
        ax.text(index, value, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)


def write_markdown_summary(output_path, summary_rows):
    lines = [
        '| traffic_level | avg_reward | success_rate | collision_rate | timeout_rate | avg_length | avg_speed | avg_progress |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary_rows:
        lines.append(
            '| {traffic_level} | {mean_return} | {success_rate} | {collision_rate} | {timeout_rate} | {mean_length} | {mean_speed} | {mean_progress} |'.format(
                **row
            )
        )
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = get_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_outputs = []
    for traffic_level in args.traffic_levels:
        artifact_stem, summary_path, detail_path, failure_path = run_eval_for_traffic_level(args, traffic_level)
        summary_row = read_single_row_csv(summary_path)
        summary_row['artifact_stem'] = artifact_stem
        summary_row['summary_file'] = str(summary_path)
        summary_row['detail_file'] = str(detail_path)
        summary_row['failure_file'] = str(failure_path)
        scenario_outputs.append((traffic_level, summary_row, detail_path, failure_path))

    summary_rows = [item[1] for item in scenario_outputs]
    summary_fieldnames = [
        'road_scenario',
        'traffic_level',
        'artifact_stem',
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
        'summary_file',
        'detail_file',
        'failure_file',
    ]
    combined_summary_path = output_dir / 'highway_best_checkpoint_summary.csv'
    write_csv(combined_summary_path, summary_fieldnames, summary_rows)

    combined_detail_rows = []
    combined_failure_rows = []
    for traffic_level, _summary_row, detail_path, failure_path in scenario_outputs:
        for row in read_rows(detail_path):
            row['traffic_level'] = traffic_level
            combined_detail_rows.append(row)
        for row in read_rows(failure_path):
            row['traffic_level'] = traffic_level
            combined_failure_rows.append(row)

    if combined_detail_rows:
        combined_detail_fieldnames = ['traffic_level'] + [key for key in combined_detail_rows[0].keys() if key != 'traffic_level']
        write_csv(output_dir / 'highway_best_checkpoint_details.csv', combined_detail_fieldnames, combined_detail_rows)
    if combined_failure_rows:
        combined_failure_fieldnames = ['traffic_level'] + [key for key in combined_failure_rows[0].keys() if key != 'traffic_level']
        write_csv(output_dir / 'highway_best_checkpoint_collision_details.csv', combined_failure_fieldnames, combined_failure_rows)

    failure_summary_rows = []
    for row in summary_rows:
        failure_summary_rows.append({
            'traffic_level': row['traffic_level'],
            'collision_count': row['collision_count'],
            'collision_rate': row['collision_rate'],
            'collision_mean_speed': row['collision_mean_speed'],
            'collision_mean_progress': row['collision_mean_progress'],
            'startup_count': row['startup_count'],
            'car_following_count': row['car_following_count'],
            'lane_change_count': row['lane_change_count'],
            'overtake_count': row['overtake_count'],
            'interaction_count': row['interaction_count'],
        })
    write_csv(
        output_dir / 'highway_best_checkpoint_collision_summary.csv',
        [
            'traffic_level',
            'collision_count',
            'collision_rate',
            'collision_mean_speed',
            'collision_mean_progress',
            'startup_count',
            'car_following_count',
            'lane_change_count',
            'overtake_count',
            'interaction_count',
        ],
        failure_summary_rows,
    )

    write_markdown_summary(output_dir / 'highway_best_checkpoint_summary.md', summary_rows)

    plot_metric_comparison(
        output_dir,
        summary_rows,
        metric_key='success_rate',
        title='Best Checkpoint Success Rate',
        filename='compare_success_rate.png',
        ylim=(0.0, 1.0),
    )
    plot_metric_comparison(
        output_dir,
        summary_rows,
        metric_key='collision_rate',
        title='Best Checkpoint Collision Rate',
        filename='compare_collision_rate.png',
        ylim=(0.0, 1.0),
    )
    plot_metric_comparison(
        output_dir,
        summary_rows,
        metric_key='mean_return',
        title='Best Checkpoint Average Reward',
        filename='compare_average_reward.png',
    )
    plot_metric_comparison(
        output_dir,
        summary_rows,
        metric_key='mean_progress',
        title='Best Checkpoint Average Progress',
        filename='compare_average_progress.png',
        ylim=(0.0, 1.0),
    )

    print(f'Summary CSV: {combined_summary_path}')
    print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
