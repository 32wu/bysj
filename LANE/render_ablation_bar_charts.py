#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


REPO_ROOT = Path('/root/autodl-tmp/SVPG2023/LANE')
DEFAULT_STANDARD_CSV = REPO_ROOT / 'thesis_tables' / 'ablation_highway_standard.csv'
DEFAULT_DENSE_CSV = REPO_ROOT / 'thesis_tables' / 'ablation_highway_dense.csv'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'thesis_figures' / 'ablation'

COLORS = ['#9AA5B1', '#5B8FF9', '#1F7A8C']
SCENARIO_STYLES = {
    'Highway-Standard': '#4C78A8',
    'Highway-Dense': '#E45756',
}
METRIC_SPECS = [
    ('Success Rate ↑', 'Success Rate (%)', True),
    ('Collision Rate ↓', 'Collision Rate (%)', True),
    ('Avg Reward ↑', 'Avg Reward', False),
    ('Avg Speed ↑', 'Avg Speed', False),
    ('Avg Progress ↑', 'Avg Progress (%)', True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render thesis ablation bar charts.')
    parser.add_argument('--standard-csv', type=Path, default=DEFAULT_STANDARD_CSV)
    parser.add_argument('--dense-csv', type=Path, default=DEFAULT_DENSE_CSV)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_plot_style() -> None:
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
    ]
    for font_path in font_candidates:
        path_obj = Path(font_path)
        if path_obj.exists():
            font_manager.fontManager.addfont(str(path_obj))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
        }
    )


def read_rows(csv_path: Path):
    with csv_path.open('r', encoding='utf-8', newline='') as file_obj:
        return list(csv.DictReader(file_obj))


def short_method_label(method_name: str) -> str:
    if method_name == 'Ours w/o Domain Randomization / Curriculum':
        return 'Ours w/o\nDR/Curr.'
    return method_name


def format_value(value: float, as_percent: bool) -> str:
    if as_percent:
        return f'{value:.1f}%'
    return f'{value:.2f}'


def scaled_value(value: float, as_percent: bool) -> float:
    return value * 100.0 if as_percent else value


def render_single_chart(rows, scenario_title: str, output_stem: Path) -> None:
    method_labels = [short_method_label(row['Method']) for row in rows]
    fig, axes = plt.subplots(1, len(METRIC_SPECS), figsize=(16, 3.8), constrained_layout=True)

    for ax, (metric_key, ylabel, as_percent) in zip(axes, METRIC_SPECS):
        values = [scaled_value(float(row[metric_key]), as_percent) for row in rows]
        bars = ax.bar(method_labels, values, color=COLORS[:len(rows)], width=0.62)
        ax.set_title(ylabel)
        ax.tick_params(axis='x', rotation=16)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)

        if as_percent:
            max_value = max(values)
            upper = max(100.0, max_value * 1.18)
            ax.set_ylim(0.0, upper)
        else:
            max_value = max(values)
            upper = max_value * 1.18 if max_value > 0 else 1.0
            ax.set_ylim(0.0, upper)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ax.get_ylim()[1] * 0.02,
                format_value(value, as_percent),
                ha='center',
                va='bottom',
                fontsize=8,
            )

    fig.suptitle(scenario_title, fontsize=14, fontweight='bold')
    fig.savefig(output_stem.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def render_combined_chart(standard_rows, dense_rows, output_stem: Path) -> None:
    method_labels = [short_method_label(row['Method']) for row in standard_rows]
    x_positions = list(range(len(method_labels)))
    bar_width = 0.34

    fig, axes = plt.subplots(1, len(METRIC_SPECS), figsize=(16, 3.9), constrained_layout=True)

    for ax, (metric_key, ylabel, as_percent) in zip(axes, METRIC_SPECS):
        standard_values = [scaled_value(float(row[metric_key]), as_percent) for row in standard_rows]
        dense_values = [scaled_value(float(row[metric_key]), as_percent) for row in dense_rows]

        standard_bars = ax.bar(
            [x - bar_width / 2.0 for x in x_positions],
            standard_values,
            width=bar_width,
            color=SCENARIO_STYLES['Highway-Standard'],
            label='Highway-Standard',
        )
        dense_bars = ax.bar(
            [x + bar_width / 2.0 for x in x_positions],
            dense_values,
            width=bar_width,
            color=SCENARIO_STYLES['Highway-Dense'],
            label='Highway-Dense',
        )

        ax.set_title(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(method_labels, rotation=16)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)

        all_values = standard_values + dense_values
        max_value = max(all_values)
        if as_percent:
            upper = max(100.0, max_value * 1.18)
        else:
            upper = max_value * 1.18 if max_value > 0 else 1.0
        ax.set_ylim(0.0, upper)

        for bars, values in [(standard_bars, standard_values), (dense_bars, dense_values)]:
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + ax.get_ylim()[1] * 0.02,
                    format_value(value, as_percent),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle('Highway Ablation Comparison', fontsize=14, fontweight='bold')
    fig.text(
        0.5,
        0.985,
        'Grouped bars compare Highway-Standard and Highway-Dense under the same method for each metric.',
        ha='center',
        va='top',
        fontsize=9,
        color='#444444',
    )
    fig.savefig(output_stem.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    standard_rows = read_rows(args.standard_csv.resolve())
    dense_rows = read_rows(args.dense_csv.resolve())

    render_single_chart(
        standard_rows,
        scenario_title='Highway-Standard Ablation',
        output_stem=args.output_dir / 'highway_standard_ablation_bar',
    )
    render_single_chart(
        dense_rows,
        scenario_title='Highway-Dense Ablation',
        output_stem=args.output_dir / 'highway_dense_ablation_bar',
    )
    render_combined_chart(
        standard_rows,
        dense_rows,
        output_stem=args.output_dir / 'highway_ablation_combined_bar',
    )

    print(f'Wrote: {args.output_dir / "highway_standard_ablation_bar.png"}')
    print(f'Wrote: {args.output_dir / "highway_standard_ablation_bar.pdf"}')
    print(f'Wrote: {args.output_dir / "highway_dense_ablation_bar.png"}')
    print(f'Wrote: {args.output_dir / "highway_dense_ablation_bar.pdf"}')
    print(f'Wrote: {args.output_dir / "highway_ablation_combined_bar.png"}')
    print(f'Wrote: {args.output_dir / "highway_ablation_combined_bar.pdf"}')


if __name__ == '__main__':
    main()
