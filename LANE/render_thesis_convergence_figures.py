#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ('mean_return', 'Average Reward', None),
    ('success_rate', 'Success Rate', (0.0, 1.0)),
    ('collision_rate', 'Collision Rate', (0.0, 1.0)),
    ('mean_speed', 'Average Speed', None),
]

METHOD_STYLES = {
    'baseline': {
        'label': 'Baseline',
        'color': '#4C78A8',
        'fill_alpha': 0.14,
        'line_alpha': 0.30,
    },
    'ours': {
        'label': 'Ours',
        'color': '#E45756',
        'fill_alpha': 0.14,
        'line_alpha': 0.30,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render thesis-ready convergence figures from an aggregated comparison CSV.'
    )
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output-stem', type=Path, required=True)
    parser.add_argument('--title', type=str, required=True)
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'font.family': 'DejaVu Sans',
        }
    )


def read_rows(csv_path: Path):
    with csv_path.open('r', encoding='utf-8', newline='') as file_obj:
        return list(csv.DictReader(file_obj))


def maybe_float(value: str):
    if value is None or value == '':
        return np.nan
    return float(value)


def build_series(rows, method_key: str, metric_key: str):
    episodes = np.asarray([int(row['episode']) for row in rows], dtype=np.int64)
    mean_values = np.asarray(
        [maybe_float(row[f'{method_key}_{metric_key}_mean']) for row in rows],
        dtype=np.float64,
    )
    std_values = np.asarray(
        [maybe_float(row[f'{method_key}_{metric_key}_std']) for row in rows],
        dtype=np.float64,
    )
    smooth_values = np.asarray(
        [maybe_float(row[f'{method_key}_{metric_key}_smooth_mean']) for row in rows],
        dtype=np.float64,
    )
    valid_mask = ~np.isnan(mean_values)
    return episodes[valid_mask], mean_values[valid_mask], std_values[valid_mask], smooth_values[valid_mask]


def render_figure(rows, output_stem: Path, title: str) -> None:
    configure_plot_style()
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = list(axes.flatten())

    for axis, (metric_key, ylabel, ylim) in zip(axes, METRICS):
        for method_key in ['baseline', 'ours']:
            style = METHOD_STYLES[method_key]
            episodes, mean_values, std_values, smooth_values = build_series(rows, method_key, metric_key)
            if episodes.size == 0:
                continue

            axis.plot(
                episodes,
                mean_values,
                color=style['color'],
                linewidth=1.4,
                alpha=style['line_alpha'],
                label=f"{style['label']} Mean",
            )
            axis.fill_between(
                episodes,
                mean_values - std_values,
                mean_values + std_values,
                color=style['color'],
                alpha=style['fill_alpha'],
                label=f"{style['label']} ±1 SD",
            )
            axis.plot(
                episodes,
                smooth_values,
                color=style['color'],
                linewidth=2.4,
                label=f"{style['label']} Smoothed",
            )

        axis.set_xlabel('Training Episode')
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.legend(loc='best')

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_stem.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv.resolve())
    render_figure(rows, args.output_stem.resolve(), args.title)
    print(f'Rendered: {args.output_stem.with_suffix(".png")}')
    print(f'Rendered: {args.output_stem.with_suffix(".pdf")}')


if __name__ == '__main__':
    main()
