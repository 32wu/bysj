#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_CSV = Path(
    '/root/autodl-tmp/SVPG2023/LANE/thesis_figures/highway_clean_eval/highway_clean_evaluation_baseline_vs_ours.csv'
)
DEFAULT_OUTPUT_STEM = Path(
    '/root/autodl-tmp/SVPG2023/LANE/thesis_figures/highway_clean_eval/highway_clean_evaluation_baseline_vs_ours_bar'
)

METHOD_STYLES = {
    'Baseline': '#4C78A8',
    'Ours': '#E45756',
}

SCENARIO_LABELS = {
    'highway-standard': 'Highway-Standard',
    'highway-dense': 'Highway-Dense',
    'merge': 'Merge',
    'roundabout': 'Roundabout',
}

METRICS = [
    ('success_rate', 'Success Rate (%)', True),
    ('collision_rate', 'Collision Rate (%)', True),
    ('timeout_rate', 'Timeout Rate (%)', True),
    ('average_reward', 'Average Reward', False),
    ('average_episode_length', 'Average Episode Length', False),
    ('average_speed', 'Average Speed', False),
    ('average_progress', 'Average Progress (%)', True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render thesis-ready clean-evaluation bar charts.')
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument('--output-stem', type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'font.family': 'DejaVu Sans',
        }
    )


def read_rows(csv_path: Path):
    with csv_path.open('r', encoding='utf-8', newline='') as file_obj:
        return list(csv.DictReader(file_obj))


def scaled_value(raw_value: str, as_percent: bool) -> float:
    value = float(raw_value)
    return value * 100.0 if as_percent else value


def format_value(value: float, as_percent: bool) -> str:
    if as_percent:
        return f'{value:.1f}%'
    return f'{value:.2f}'


def index_rows(rows):
    indexed = {}
    for row in rows:
        indexed[(row['scenario'], row['method'])] = row
    return indexed


def ordered_unique_scenarios(rows):
    scenario_keys = []
    for row in rows:
        scenario_key = row['scenario']
        if scenario_key not in scenario_keys:
            scenario_keys.append(scenario_key)
    return scenario_keys


def scenario_display_label(scenario_key: str) -> str:
    if scenario_key in SCENARIO_LABELS:
        return SCENARIO_LABELS[scenario_key]
    return scenario_key.replace('-', ' ').title()


def render_chart(rows, output_stem: Path) -> None:
    configure_plot_style()
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    indexed = index_rows(rows)
    scenario_keys = ordered_unique_scenarios(rows)
    method_keys = ['Baseline', 'Ours']
    x_positions = np.arange(len(scenario_keys), dtype=np.float64)
    bar_width = 0.32

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.6), constrained_layout=True)
    axes = axes.flatten()

    for axis, (metric_key, ylabel, as_percent) in zip(axes, METRICS):
        all_values = []
        for method_index, method_key in enumerate(method_keys):
            offsets = x_positions + (method_index - 0.5) * bar_width
            values = [
                scaled_value(indexed[(scenario_key, method_key)][metric_key], as_percent)
                for scenario_key in scenario_keys
            ]
            all_values.extend(values)

            bars = axis.bar(
                offsets,
                values,
                width=bar_width,
                color=METHOD_STYLES[method_key],
                label=method_key,
            )
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + (max(values) if max(values) > 0 else 1.0) * 0.03,
                    format_value(value, as_percent),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

        max_value = max(all_values) if all_values else 1.0
        if as_percent:
            axis.set_ylim(0.0, max(100.0, max_value * 1.18))
        else:
            axis.set_ylim(0.0, max_value * 1.18 if max_value > 0 else 1.0)

        axis.set_title(ylabel)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([scenario_display_label(key) for key in scenario_keys], rotation=0)
        axis.grid(axis='y', linestyle='--', alpha=0.35)
        axis.set_axisbelow(True)

    axes[-1].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Clean Evaluation Results', fontsize=14)

    fig.savefig(output_stem.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv.resolve())
    render_chart(rows, args.output_stem.resolve())
    print(f'Rendered: {args.output_stem.with_suffix(".png")}')
    print(f'Rendered: {args.output_stem.with_suffix(".pdf")}')


if __name__ == '__main__':
    main()
