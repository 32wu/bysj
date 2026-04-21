#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = PROJECT_ROOT / 'training_runs'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'thesis_figures'
SMOOTH_WINDOW = 3
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S_%f'

SCENARIO_SPECS = [
    {
        'run_kind': 'baseline',
        'scenario_dir': 'highway_standard',
        'label': 'Baseline',
        'scenario_name': 'Highway Standard',
        'style': {'color': '#4C78A8', 'marker': 'o'},
    },
    {
        'run_kind': 'ours',
        'scenario_dir': 'highway_standard',
        'label': 'Ours',
        'scenario_name': 'Highway Standard',
        'style': {'color': '#E45756', 'marker': 's'},
    },
    {
        'run_kind': 'baseline',
        'scenario_dir': 'highway_dense',
        'label': 'Baseline',
        'scenario_name': 'Highway Dense',
        'style': {'color': '#4C78A8', 'marker': 'o'},
    },
    {
        'run_kind': 'ours',
        'scenario_dir': 'highway_dense',
        'label': 'Ours',
        'scenario_name': 'Highway Dense',
        'style': {'color': '#E45756', 'marker': 's'},
    },
    {
        'run_kind': 'baseline',
        'scenario_dir': 'merge',
        'label': 'Baseline',
        'scenario_name': 'Merge',
        'style': {'color': '#4C78A8', 'marker': 'o'},
    },
    {
        'run_kind': 'ours',
        'scenario_dir': 'merge',
        'label': 'Ours',
        'scenario_name': 'Merge',
        'style': {'color': '#E45756', 'marker': 's'},
    },
    {
        'run_kind': 'baseline',
        'scenario_dir': 'roundabout',
        'label': 'Baseline',
        'scenario_name': 'Roundabout',
        'style': {'color': '#4C78A8', 'marker': 'o'},
    },
    {
        'run_kind': 'ours',
        'scenario_dir': 'roundabout',
        'label': 'Ours',
        'scenario_name': 'Roundabout',
        'style': {'color': '#E45756', 'marker': 's'},
    },
]


@dataclass
class ValRecord:
    episode: int
    mean_return: float
    collision_rate: float
    mean_steps: float
    mean_speed: float
    success_rate: float
    aux_metric: float


@dataclass
class RunData:
    run_kind: str
    scenario_dir: str
    label: str
    scenario_name: str
    log_path: Path
    records: List[ValRecord]
    style: Dict[str, str]

    @property
    def best_steps(self) -> ValRecord:
        return max(self.records, key=lambda item: item.mean_steps)

    @property
    def best_return(self) -> ValRecord:
        return max(self.records, key=lambda item: item.mean_return)

    @property
    def final(self) -> ValRecord:
        return self.records[-1]

    def metric_values(self, metric_name: str) -> List[float]:
        return [getattr(record, metric_name) for record in self.records]

    def episodes(self) -> List[int]:
        return [record.episode for record in self.records]


def configure_plot_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['SimHei', 'Noto Sans CJK SC', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        }
    )


def moving_average(values: List[float], window: int = SMOOTH_WINDOW) -> List[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)
    smoothed: List[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        current = values[start:index + 1]
        smoothed.append(sum(current) / float(len(current)))
    return smoothed


def cumulative_best(values: List[float]) -> List[float]:
    best_values: List[float] = []
    current_best = None
    for value in values:
        if current_best is None or value > current_best:
            current_best = value
        best_values.append(current_best)
    return best_values


def safe_max(values: List[float], lower_bound: float = 1.0) -> float:
    return max(lower_bound, max(values) if values else lower_bound)


def find_latest_log(run_kind: str, scenario_dir: str) -> Path:
    log_dir = TRAINING_ROOT / run_kind / scenario_dir / 'logs'
    if not log_dir.exists():
        raise FileNotFoundError(f'Log directory not found: {log_dir}')
    candidates = sorted(log_dir.glob('*.txt'), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No log files found in: {log_dir}')
    return candidates[0]


def parse_val_records(log_path: Path) -> List[ValRecord]:
    records: List[ValRecord] = []
    with log_path.open('r', encoding='utf-8') as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line.startswith('val,'):
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 8:
                records.append(
                    ValRecord(
                        episode=int(parts[1]),
                        mean_return=float(parts[2]),
                        collision_rate=float(parts[3]),
                        mean_steps=float(parts[4]),
                        mean_speed=float(parts[5]),
                        success_rate=float(parts[6]),
                        aux_metric=float(parts[7]),
                    )
                )
            elif len(parts) >= 4:
                records.append(
                    ValRecord(
                        episode=int(parts[1]),
                        mean_return=float(parts[2]),
                        collision_rate=math.nan,
                        mean_steps=float(parts[3]),
                        mean_speed=math.nan,
                        success_rate=math.nan,
                        aux_metric=math.nan,
                    )
                )
    if not records:
        raise ValueError(f'No validation records found in: {log_path}')
    return records


def load_default_runs() -> List[RunData]:
    runs: List[RunData] = []
    for spec in SCENARIO_SPECS:
        log_path = find_latest_log(spec['run_kind'], spec['scenario_dir'])
        runs.append(
            RunData(
                run_kind=spec['run_kind'],
                scenario_dir=spec['scenario_dir'],
                label=spec['label'],
                scenario_name=spec['scenario_name'],
                log_path=log_path,
                records=parse_val_records(log_path),
                style=spec['style'],
            )
        )
    return runs


def create_timestamp() -> str:
    return datetime.utcnow().strftime(TIMESTAMP_FORMAT)


def make_output_dir(output_root: Path, scenario_dir: str, timestamp: str) -> Path:
    output_dir = output_root / scenario_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_scenario_reward_curve(scenario_runs: List[RunData], output_dir: Path, timestamp: str) -> List[Path]:
    scenario_name = scenario_runs[0].scenario_name

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), constrained_layout=True)
    ymax_raw: List[float] = []
    ymax_best: List[float] = []

    for run in scenario_runs:
        episodes = run.episodes()
        raw_returns = run.metric_values('mean_return')
        smoothed_returns = moving_average(raw_returns)
        best_so_far_returns = cumulative_best(raw_returns)
        best_record = run.best_return
        ymax_raw.extend(raw_returns)
        ymax_best.extend(best_so_far_returns)

        axes[0].plot(
            episodes,
            raw_returns,
            color=run.style['color'],
            linewidth=1.0,
            alpha=0.20,
        )
        axes[0].plot(
            episodes,
            smoothed_returns,
            color=run.style['color'],
            linewidth=2.6,
            marker=run.style['marker'],
            markersize=4.0,
            alpha=0.98,
            label=f'{run.label} smoothed reward',
        )
        axes[0].scatter(
            [best_record.episode],
            [best_record.mean_return],
            color=run.style['color'],
            s=62,
            zorder=5,
            edgecolor='black',
            linewidth=0.55,
        )
        axes[0].annotate(
            f'{run.label}: {best_record.mean_return:.1f}',
            xy=(best_record.episode, best_record.mean_return),
            xytext=(6, 8),
            textcoords='offset points',
            fontsize=9,
            color=run.style['color'],
        )

        axes[1].plot(
            episodes,
            best_so_far_returns,
            color=run.style['color'],
            linewidth=2.7,
            alpha=0.98,
            label=f'{run.label} best-so-far reward',
        )
        axes[1].axvline(best_record.episode, color=run.style['color'], linestyle='--', linewidth=1.1, alpha=0.55)

    axes[0].set_title(f'{scenario_name}: Validation Reward Curve')
    axes[0].set_xlabel('Training Episodes')
    axes[0].set_ylabel('Validation Mean Reward')
    axes[0].legend(frameon=True)
    axes[0].set_ylim(min(0.0, min(ymax_raw) * 1.15), max(20.0, safe_max(ymax_raw, lower_bound=20.0) * 1.18))

    axes[1].set_title(f'{scenario_name}: Best-so-far Reward Envelope')
    axes[1].set_xlabel('Training Episodes')
    axes[1].set_ylabel('Best Validation Reward')
    axes[1].legend(frameon=True)
    axes[1].set_ylim(min(0.0, min(ymax_best) * 1.15), max(20.0, safe_max(ymax_best, lower_bound=20.0) * 1.18))

    stem = f'training_reward_compare_{scenario_runs[0].scenario_dir}_{timestamp}'
    png_path = output_dir / f'{stem}.png'
    pdf_path = output_dir / f'{stem}.pdf'
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return [png_path, pdf_path]


def write_scenario_summary_csv(scenario_runs: List[RunData], output_dir: Path, timestamp: str) -> Path:
    scenario_dir = scenario_runs[0].scenario_dir
    csv_path = output_dir / f'training_reward_summary_{scenario_dir}_{timestamp}.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            'run_kind',
            'scenario_dir',
            'label',
            'log_file',
            'num_val_points',
            'best_return_episode',
            'best_mean_return',
            'best_return_steps',
            'final_episode',
            'final_mean_return',
            'final_mean_steps',
            'smoothed_final_return',
        ])
        for run in scenario_runs:
            smoothed_returns = moving_average(run.metric_values('mean_return'))
            writer.writerow([
                run.run_kind,
                run.scenario_dir,
                run.label,
                run.log_path.name,
                len(run.records),
                run.best_return.episode,
                f'{run.best_return.mean_return:.4f}',
                f'{run.best_return.mean_steps:.4f}',
                run.final.episode,
                f'{run.final.mean_return:.4f}',
                f'{run.final.mean_steps:.4f}',
                f'{smoothed_returns[-1]:.4f}',
            ])
    return csv_path


def print_scenario_summary(scenario_runs: List[RunData], output_dir: Path) -> None:
    print(f"\nScenario: {scenario_runs[0].scenario_dir}")
    for run in scenario_runs:
        print(f"  {run.label:8s} | {run.log_path.name}")
        print(
            '    '
            f'best_return ep={run.best_return.episode}, reward={run.best_return.mean_return:.1f}, '
            f'best_steps_at_best_return={run.best_return.mean_steps:.1f}; '
            f'final raw reward={run.final.mean_return:.1f}, '
            f'smoothed_final_reward={moving_average(run.metric_values("mean_return"))[-1]:.1f}'
        )
    print(f'  output_dir: {output_dir}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate thesis-ready reward figures from the latest training logs.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory to store generated thesis figures.')
    parser.add_argument(
        '--scenario',
        type=str,
        default=None,
        choices=['highway_standard', 'highway_dense', 'merge', 'roundabout'],
        help='Generate figures only for the selected scenario.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    runs = load_default_runs()
    timestamp = create_timestamp()

    scenario_dirs = []
    for run in runs:
        if run.scenario_dir not in scenario_dirs:
            scenario_dirs.append(run.scenario_dir)
    if args.scenario is not None:
        scenario_dirs = [scenario_dir for scenario_dir in scenario_dirs if scenario_dir == args.scenario]

    for scenario_dir in scenario_dirs:
        scenario_runs = [run for run in runs if run.scenario_dir == scenario_dir]
        run_output_dir = make_output_dir(output_dir, scenario_dir, timestamp)
        figure_paths = plot_scenario_reward_curve(scenario_runs, run_output_dir, timestamp)
        csv_path = write_scenario_summary_csv(scenario_runs, run_output_dir, timestamp)
        print_scenario_summary(scenario_runs, run_output_dir)
        for figure_path in figure_paths:
            print(f'Generated figure: {figure_path}')
        print(f'Generated summary csv: {csv_path}')


if __name__ == '__main__':
    main()
