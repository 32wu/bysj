#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a thesis-ready clean generalization evaluation bundle.')
    parser.add_argument('--merge-baseline-summary', type=Path, required=True)
    parser.add_argument('--merge-ours-summary', type=Path, required=True)
    parser.add_argument('--roundabout-baseline-summary', type=Path, required=True)
    parser.add_argument('--roundabout-ours-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def read_single_row(csv_path: Path):
    with csv_path.open('r', encoding='utf-8', newline='') as file_obj:
        rows = list(csv.DictReader(file_obj))
    if len(rows) != 1:
        raise RuntimeError(f'Expected exactly one row in {csv_path}')
    return rows[0]


def to_row(summary_row, method_label):
    return {
        'scenario': summary_row['road_scenario'],
        'method': method_label,
        'success_rate': float(summary_row['success_rate']),
        'collision_rate': float(summary_row['collision_rate']),
        'timeout_rate': float(summary_row['timeout_rate']),
        'average_reward': float(summary_row['mean_return']),
        'average_episode_length': float(summary_row['mean_length']),
        'average_speed': float(summary_row['mean_speed']),
        'average_progress': float(summary_row['mean_progress']),
        'checkpoint_prefix': summary_row['checkpoint_prefix'],
        'eval_episodes': int(summary_row['eval_episodes']),
        'summary_csv': str(Path(summary_row.get('summary_csv', '')).resolve()) if summary_row.get('summary_csv') else '',
    }


def write_csv(output_path: Path, rows) -> None:
    fieldnames = list(rows[0].keys())
    with output_path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output_path: Path, rows) -> None:
    lines = [
        '| Scenario | Method | Success Rate | Collision Rate | Timeout Rate | Average Reward | Average Episode Length | Average Speed | Average Progress |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            '| {scenario} | {method} | {success_rate:.4f} | {collision_rate:.4f} | {timeout_rate:.4f} | {average_reward:.4f} | {average_episode_length:.4f} | {average_speed:.4f} | {average_progress:.4f} |'.format(
                **row
            )
        )
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_paper_text(output_path: Path, rows) -> None:
    grouped = {}
    for row in rows:
        grouped.setdefault(row['scenario'], []).append(row)

    lines = []
    for scenario in ['merge', 'roundabout']:
        scenario_rows = grouped.get(scenario, [])
        if len(scenario_rows) != 2:
            continue
        baseline_row = next(row for row in scenario_rows if row['method'] == 'Baseline')
        ours_row = next(row for row in scenario_rows if row['method'] == 'Ours')
        lines.append(
            (
                f'{scenario} 场景 clean evaluation（{baseline_row["eval_episodes"]} episodes）：'
                f'Baseline 成功率 {baseline_row["success_rate"]:.2%}，碰撞率 {baseline_row["collision_rate"]:.2%}，'
                f'平均回报 {baseline_row["average_reward"]:.2f}；'
                f'Ours 成功率 {ours_row["success_rate"]:.2%}，碰撞率 {ours_row["collision_rate"]:.2%}，'
                f'平均回报 {ours_row["average_reward"]:.2f}。'
            )
        )
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_specs = [
        ('Baseline', args.merge_baseline_summary),
        ('Ours', args.merge_ours_summary),
        ('Baseline', args.roundabout_baseline_summary),
        ('Ours', args.roundabout_ours_summary),
    ]
    rows = []
    for method_label, summary_path in input_specs:
        summary_row = read_single_row(summary_path.resolve())
        summary_row['summary_csv'] = str(summary_path.resolve())
        rows.append(to_row(summary_row, method_label))

    rows.sort(key=lambda item: (item['scenario'], item['method']))

    csv_path = args.output_dir / 'generalization_clean_evaluation.csv'
    md_path = args.output_dir / 'generalization_clean_evaluation.md'
    txt_path = args.output_dir / 'generalization_clean_evaluation.txt'
    paper_md_path = args.output_dir / 'generalization_clean_evaluation_paper.md'
    paper_txt_path = args.output_dir / 'generalization_clean_evaluation_paper.txt'

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    txt_path.write_text(md_path.read_text(encoding='utf-8'), encoding='utf-8')
    write_paper_text(paper_md_path, rows)
    paper_txt_path.write_text(paper_md_path.read_text(encoding='utf-8'), encoding='utf-8')

    print(f'Wrote: {csv_path}')
    print(f'Wrote: {md_path}')
    print(f'Wrote: {txt_path}')
    print(f'Wrote: {paper_md_path}')
    print(f'Wrote: {paper_txt_path}')


if __name__ == '__main__':
    main()
