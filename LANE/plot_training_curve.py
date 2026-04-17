#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = PROJECT_ROOT / 'training_runs'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'thesis_figures'

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
            if len(parts) < 8:
                continue
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


def plot_metric_panel(ax, runs: List[RunData], metric_name: str, ylabel: str, title: str, annotate_best: bool = False) -> None:
    for run in runs:
        episodes = [record.episode for record in run.records]
        metric_values = [getattr(record, metric_name) for record in run.records]
        ax.plot(
            episodes,
            metric_values,
            label=run.label,
            color=run.style['color'],
            marker=run.style['marker'],
            linewidth=2.0,
            markersize=4.5,
            alpha=0.95,
        )
        if annotate_best:
            best_record = max(run.records, key=lambda item: getattr(item, metric_name))
            ax.scatter(
                [best_record.episode],
                [getattr(best_record, metric_name)],
                color=run.style['color'],
                s=55,
                zorder=5,
                edgecolor='black',
                linewidth=0.5,
            )
            ax.annotate(
                f"{run.label}: {getattr(best_record, metric_name):.1f}",
                xy=(best_record.episode, getattr(best_record, metric_name)),
                xytext=(6, 8),
                textcoords='offset points',
                fontsize=9,
                color=run.style['color'],
            )
    ax.set_title(title)
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)


def generate_training_curve_figure(runs: List[RunData], output_dir: Path) -> Path:
    grouped: Dict[str, List[RunData]] = {'highway_standard': [], 'highway_dense': []}
    for run in runs:
        grouped[run.scenario_dir].append(run)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    standard_runs = grouped['highway_standard']
    dense_runs = grouped['highway_dense']

    plot_metric_panel(
        axes[0, 0],
        standard_runs,
        metric_name='mean_steps',
        ylabel='Validation Mean Steps',
        title='(a) Highway Standard: Validation Mean Steps',
        annotate_best=True,
    )
    axes[0, 0].set_ylim(0, 210)

    plot_metric_panel(
        axes[0, 1],
        dense_runs,
        metric_name='mean_steps',
        ylabel='Validation Mean Steps',
        title='(b) Highway Dense: Validation Mean Steps',
        annotate_best=True,
    )
    axes[0, 1].set_ylim(0, 210)

    plot_metric_panel(
        axes[1, 0],
        standard_runs,
        metric_name='mean_return',
        ylabel='Validation Mean Return',
        title='(c) Highway Standard: Validation Mean Return',
    )

    plot_metric_panel(
        axes[1, 1],
        dense_runs,
        metric_name='mean_return',
        ylabel='Validation Mean Return',
        title='(d) Highway Dense: Validation Mean Return',
    )

    figure_path = output_dir / 'highway_training_curves_latest.png'
    fig.savefig(figure_path, bbox_inches='tight')
    fig.savefig(figure_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    return figure_path


def generate_summary_bar_figure(runs: List[RunData], output_dir: Path) -> Path:
    labels = []
    best_step_values = []
    best_return_values = []
    colors = []
    for run in runs:
        labels.append(f"{run.label}\n{run.scenario_name.replace('Highway ', '')}")
        best_step_values.append(run.best_steps.mean_steps)
        best_return_values.append(run.best_return.mean_return)
        colors.append(run.style['color'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    bars_steps = axes[0].bar(labels, best_step_values, color=colors, alpha=0.9)
    axes[0].set_title('(a) Best Validation Mean Steps')
    axes[0].set_ylabel('Mean Steps')
    axes[0].set_ylim(0, 210)
    for bar, value in zip(bars_steps, best_step_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 3, f'{value:.1f}', ha='center', va='bottom', fontsize=9)

    bars_return = axes[1].bar(labels, best_return_values, color=colors, alpha=0.9)
    axes[1].set_title('(b) Best Validation Mean Return')
    axes[1].set_ylabel('Mean Return')
    for bar, value in zip(bars_return, best_return_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 3, f'{value:.1f}', ha='center', va='bottom', fontsize=9)

    figure_path = output_dir / 'highway_training_summary_latest.png'
    fig.savefig(figure_path, bbox_inches='tight')
    fig.savefig(figure_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    return figure_path


def write_summary_csv(runs: List[RunData], output_dir: Path) -> Path:
    csv_path = output_dir / 'highway_training_summary_latest.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                'run_kind',
                'scenario_dir',
                'label',
                'log_file',
                'num_val_points',
                'best_steps_episode',
                'best_mean_steps',
                'best_steps_return',
                'best_steps_collision_rate',
                'best_return_episode',
                'best_mean_return',
                'best_return_steps',
                'final_episode',
                'final_mean_steps',
                'final_mean_return',
                'final_collision_rate',
                'final_success_rate',
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.run_kind,
                    run.scenario_dir,
                    run.label,
                    run.log_path.name,
                    len(run.records),
                    run.best_steps.episode,
                    f'{run.best_steps.mean_steps:.4f}',
                    f'{run.best_steps.mean_return:.4f}',
                    f'{run.best_steps.collision_rate:.4f}',
                    run.best_return.episode,
                    f'{run.best_return.mean_return:.4f}',
                    f'{run.best_return.mean_steps:.4f}',
                    run.final.episode,
                    f'{run.final.mean_steps:.4f}',
                    f'{run.final.mean_return:.4f}',
                    f'{run.final.collision_rate:.4f}',
                    f'{run.final.success_rate:.4f}',
                ]
            )
    return csv_path


def print_run_summary(runs: List[RunData]) -> None:
    print('Selected latest highway logs:')
    for run in runs:
        print(f'- {run.label:8s} | {run.scenario_dir:16s} | {run.log_path.name}')
        print(
            '  '
            f'best_steps ep={run.best_steps.episode}, steps={run.best_steps.mean_steps:.1f}, '
            f'return={run.best_steps.mean_return:.1f}; '
            f'final ep={run.final.episode}, steps={run.final.mean_steps:.1f}, '
            f'return={run.final.mean_return:.1f}'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate thesis-ready highway training figures from the latest logs.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory to store the generated thesis figures.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    runs = load_default_runs()
    curve_path = generate_training_curve_figure(runs, output_dir)
    summary_path = generate_summary_bar_figure(runs, output_dir)
    csv_path = write_summary_csv(runs, output_dir)
    print_run_summary(runs)
    print(f'Generated curve figure: {curve_path}')
    print(f'Generated summary figure: {summary_path}')
    print(f'Generated summary csv:   {csv_path}')


if __name__ == '__main__':
    main()
