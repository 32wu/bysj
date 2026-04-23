#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


DEFAULT_OUTPUT_DIR = Path('/root/autodl-tmp/SVPG2023/LANE/thesis_figures/highway_standard')
METRIC_SPECS = [
    ('mean_return', '验证平均回报', None),
    ('success_rate', '验证成功率', (-0.05, 1.05)),
    ('collision_rate', '验证碰撞率', (-0.05, 1.05)),
    ('mean_speed', '验证平均速度', None),
]

METHOD_STYLES = {
    'baseline': {
        'label': 'base',
        'mean_color': '#4C78A8',
        'seed_color': '#9EC3E6',
    },
    'ours': {
        'label': 'ours',
        'mean_color': '#E45756',
        'seed_color': '#F3A7A0',
    },
}


@dataclass
class ValRecord:
    episode: int
    timesteps: int
    mean_return: float
    collision_rate: float
    mean_length: float
    mean_lane_change: float
    success_rate: float
    mean_speed: float
    mean_progress: float


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
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
        }
    )


def ema_smooth(values: List[float], alpha: float) -> List[float]:
    if not values:
        return []
    alpha = min(1.0, max(1e-6, float(alpha)))
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def sma_smooth(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, int(window))
    smoothed: List[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        current = values[start:index + 1]
        smoothed.append(float(sum(current) / max(1, len(current))))
    return smoothed


def smooth_values(values: List[float], method: str, ema_alpha: float, sma_window: int) -> List[float]:
    if method == 'sma':
        return sma_smooth(values, sma_window)
    return ema_smooth(values, ema_alpha)


def parse_labeled_float(parts: List[str], label: str, default: float = 0.0) -> float:
    prefix = f'{label} '
    for part in parts:
        item = part.strip()
        if item.startswith(prefix):
            try:
                return float(item[len(prefix):].split()[0])
            except ValueError:
                return default
    return default


def parse_validation_log(log_path: Path) -> List[ValRecord]:
    records_by_episode: Dict[int, ValRecord] = {}
    with log_path.open('r', encoding='utf-8') as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line.startswith('val'):
                continue
            parts = [part.strip() for part in line.split(',')]
            tag = parts[0]
            if tag in {'val_t', 'val_save_t'} and len(parts) >= 9:
                episode = int(parts[1])
                records_by_episode[episode] = ValRecord(
                    episode=episode,
                    timesteps=int(parts[2]),
                    mean_return=float(parts[3]),
                    collision_rate=float(parts[4]),
                    mean_length=float(parts[5]),
                    mean_lane_change=float(parts[6]),
                    success_rate=float(parts[7]),
                    mean_speed=float(parts[8]),
                    mean_progress=parse_labeled_float(parts[9:], 'progress', default=0.0),
                )
            elif tag in {'val', 'val_save'} and len(parts) >= 8:
                episode = int(parts[1])
                existing = records_by_episode.get(episode)
                if existing is not None:
                    continue
                records_by_episode[episode] = ValRecord(
                    episode=episode,
                    timesteps=episode,
                    mean_return=float(parts[2]),
                    collision_rate=float(parts[3]),
                    mean_length=float(parts[4]),
                    mean_lane_change=float(parts[5]),
                    success_rate=float(parts[6]),
                    mean_speed=float(parts[7]),
                    mean_progress=parse_labeled_float(parts[8:], 'progress', default=0.0),
                )
    records = [records_by_episode[key] for key in sorted(records_by_episode.keys())]
    if not records:
        raise ValueError(f'No validation records found in: {log_path}')
    return records


def interpolate_series(common_x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    output = np.full(common_x.shape, np.nan, dtype=np.float64)
    if xs.size == 0:
        return output
    valid_mask = (common_x >= xs.min()) & (common_x <= xs.max())
    if not np.any(valid_mask):
        return output
    output[valid_mask] = np.interp(common_x[valid_mask], xs, ys)
    return output


def aggregate_record_groups(record_groups: List[List[ValRecord]]) -> Dict[str, np.ndarray]:
    common_x = np.array(sorted({record.episode for group in record_groups for record in group}), dtype=np.int64)
    aggregated: Dict[str, np.ndarray] = {'episode': common_x}
    for metric_name, _label, _ylim in METRIC_SPECS:
        stacks = []
        for records in record_groups:
            xs = np.array([record.episode for record in records], dtype=np.float64)
            ys = np.array([getattr(record, metric_name) for record in records], dtype=np.float64)
            stacks.append(interpolate_series(common_x.astype(np.float64), xs, ys))
        stack_array = np.vstack(stacks)
        aggregated[f'{metric_name}_mean'] = np.nanmean(stack_array, axis=0)
        aggregated[f'{metric_name}_std'] = np.nanstd(stack_array, axis=0)
    return aggregated


def save_comparison_csv(
    baseline_agg: Dict[str, np.ndarray],
    ours_agg: Dict[str, np.ndarray],
    output_csv: Path,
    smooth_method: str,
    ema_alpha: float,
    sma_window: int,
) -> None:
    common_x = np.array(sorted(set(baseline_agg['episode'].tolist()) | set(ours_agg['episode'].tolist())), dtype=np.int64)
    with output_csv.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)
        header = ['episode']
        for method_key in ['baseline', 'ours']:
            for metric_name, _label, _ylim in METRIC_SPECS:
                header.extend([
                    f'{method_key}_{metric_name}_mean',
                    f'{method_key}_{metric_name}_std',
                    f'{method_key}_{metric_name}_smooth_mean',
                ])
        writer.writerow(header)

        baseline_map = {int(x): idx for idx, x in enumerate(baseline_agg['episode'])}
        ours_map = {int(x): idx for idx, x in enumerate(ours_agg['episode'])}
        baseline_smoothed = {
            metric_name: smooth_values(baseline_agg[f'{metric_name}_mean'].tolist(), smooth_method, ema_alpha, sma_window)
            for metric_name, _label, _ylim in METRIC_SPECS
        }
        ours_smoothed = {
            metric_name: smooth_values(ours_agg[f'{metric_name}_mean'].tolist(), smooth_method, ema_alpha, sma_window)
            for metric_name, _label, _ylim in METRIC_SPECS
        }

        for episode in common_x:
            row = [int(episode)]
            for method_key, agg, agg_map, smoothed in [
                ('baseline', baseline_agg, baseline_map, baseline_smoothed),
                ('ours', ours_agg, ours_map, ours_smoothed),
            ]:
                index = agg_map.get(int(episode))
                for metric_name, _label, _ylim in METRIC_SPECS:
                    if index is None:
                        row.extend(['', '', ''])
                    else:
                        row.extend([
                            f'{agg[f"{metric_name}_mean"][index]:.6f}',
                            f'{agg[f"{metric_name}_std"][index]:.6f}',
                            f'{smoothed[metric_name][index]:.6f}',
                        ])
            writer.writerow(row)


def plot_method_comparison(
    baseline_groups: List[List[ValRecord]],
    ours_groups: List[List[ValRecord]],
    baseline_agg: Dict[str, np.ndarray],
    ours_agg: Dict[str, np.ndarray],
    output_png: Path,
    output_pdf: Path,
    smooth_method: str,
    ema_alpha: float,
    sma_window: int,
    title: str,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = list(axes.flatten())

    for axis, (metric_name, ylabel, ylim) in zip(axes, METRIC_SPECS):
        for method_key, record_groups, aggregated in [
            ('baseline', baseline_groups, baseline_agg),
            ('ours', ours_groups, ours_agg),
        ]:
            style = METHOD_STYLES[method_key]
            for records in record_groups:
                axis.plot(
                    [record.episode for record in records],
                    [getattr(record, metric_name) for record in records],
                    color=style['seed_color'],
                    linewidth=1.0,
                    alpha=0.18,
                )

            mean_values = aggregated[f'{metric_name}_mean']
            std_values = aggregated[f'{metric_name}_std']
            episodes = aggregated['episode']
            smooth_curve = smooth_values(mean_values.tolist(), smooth_method, ema_alpha, sma_window)
            axis.plot(
                episodes,
                mean_values,
                color=style['mean_color'],
                linewidth=1.6,
                alpha=0.35,
                label=f"{style['label']} 均值",
            )
            axis.fill_between(
                episodes,
                mean_values - std_values,
                mean_values + std_values,
                color=style['mean_color'],
                alpha=0.12,
                label=f"{style['label']} ±1σ",
            )
            axis.plot(
                episodes,
                smooth_curve,
                color=style['mean_color'],
                linewidth=2.4,
                label=f"{style['label']} 平滑",
            )

        axis.set_xlabel('时间步数')
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.legend(loc='best')

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot baseline-vs-ours multi-seed comparison for highway_standard.')
    parser.add_argument('--base-logs', nargs='+', type=Path, required=True, help='Baseline log files.')
    parser.add_argument('--ours-logs', nargs='+', type=Path, required=True, help='Ours log files.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory.')
    parser.add_argument('--stem', type=str, default='highway_standard_base_vs_ours_5seed_8000', help='Output file stem.')
    parser.add_argument('--title', type=str, default='Highway Standard：base 与 ours 五组多种子对比', help='Figure title.')
    parser.add_argument('--smooth', type=str, default='ema', choices=['ema', 'sma'], help='Smoothing method.')
    parser.add_argument('--ema-alpha', type=float, default=0.25, help='EMA alpha.')
    parser.add_argument('--sma-window', type=int, default=3, help='SMA window size.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_groups = [parse_validation_log(path.resolve()) for path in args.base_logs]
    ours_groups = [parse_validation_log(path.resolve()) for path in args.ours_logs]
    baseline_agg = aggregate_record_groups(baseline_groups)
    ours_agg = aggregate_record_groups(ours_groups)

    output_png = args.output_dir / f'{args.stem}.png'
    output_pdf = args.output_dir / f'{args.stem}.pdf'
    output_csv = args.output_dir / f'{args.stem}.csv'

    plot_method_comparison(
        baseline_groups,
        ours_groups,
        baseline_agg,
        ours_agg,
        output_png,
        output_pdf,
        args.smooth,
        args.ema_alpha,
        args.sma_window,
        args.title,
    )
    save_comparison_csv(
        baseline_agg,
        ours_agg,
        output_csv,
        args.smooth,
        args.ema_alpha,
        args.sma_window,
    )

    print(f'Generated plot: {output_png}')
    print(f'Generated plot: {output_pdf}')
    print(f'Generated metrics CSV: {output_csv}')


if __name__ == '__main__':
    main()
