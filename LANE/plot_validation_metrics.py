#!/usr/bin/env python3
import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


DEFAULT_LOG_PATH = Path(
    '/root/autodl-tmp/SVPG2023/LANE/training_runs/ours/highway_standard/logs/'
    'log_ppo_gymip_rwtaspk_h8-8-40_none_rmsprop_0.000200_0.02_0.99800_4_0.2000_'
    'ro512_mb128_lam0.95_rs0.80_gc0.50_adaptive_roadhighway_tfstandard_hrsc_success_'
    'ew1.15e120_te4_seed11.txt'
)
DEFAULT_OUTPUT_DIR = Path('/root/autodl-tmp/SVPG2023/LANE/thesis_figures/highway_standard')
METRIC_SPECS = [
    ('mean_return', 'Validation Average Reward', None),
    ('success_rate', 'Validation Success Rate', (-0.05, 1.05)),
    ('collision_rate', 'Validation Collision Rate', (-0.05, 1.05)),
    ('mean_progress', 'Validation Average Progress', (-0.05, 1.05)),
    ('mean_speed', 'Validation Average Speed', None),
]


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
    is_best: bool = False


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
        window_values = values[start:index + 1]
        smoothed.append(float(sum(window_values) / max(1, len(window_values))))
    return smoothed


def smooth_values(values: List[float], method: str, ema_alpha: float, sma_window: int) -> List[float]:
    if method == 'sma':
        return sma_smooth(values, sma_window)
    return ema_smooth(values, ema_alpha)


def resolve_logs(args: argparse.Namespace) -> List[Path]:
    log_paths: List[Path] = []
    if args.logs:
        log_paths.extend(Path(item) for item in args.logs)
    if args.log_glob:
        for pattern in args.log_glob:
            log_paths.extend(sorted(Path(item) for item in glob.glob(pattern)))
    if not log_paths:
        log_paths = [args.log]
    unique_paths: List[Path] = []
    seen = set()
    for path in log_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


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
    records: List[ValRecord] = []
    best_keys = set()
    with log_path.open('r', encoding='utf-8') as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(',')]
            tag = parts[0]
            if tag == 'val_save_t' and len(parts) >= 9:
                best_keys.add((int(parts[1]), int(parts[2])))
            elif tag == 'val_t' and len(parts) >= 9:
                records.append(
                    ValRecord(
                        episode=int(parts[1]),
                        timesteps=int(parts[2]),
                        mean_return=float(parts[3]),
                        collision_rate=float(parts[4]),
                        mean_length=float(parts[5]),
                        mean_lane_change=float(parts[6]),
                        success_rate=float(parts[7]),
                        mean_speed=float(parts[8]),
                        mean_progress=parse_labeled_float(parts[9:], 'progress', default=0.0),
                    )
                )
            elif tag == 'val' and len(parts) >= 8:
                episode_index = int(parts[1])
                records.append(
                    ValRecord(
                        episode=episode_index,
                        timesteps=episode_index,
                        mean_return=float(parts[2]),
                        collision_rate=float(parts[3]),
                        mean_length=float(parts[4]),
                        mean_lane_change=float(parts[5]),
                        success_rate=float(parts[6]),
                        mean_speed=float(parts[7]),
                        mean_progress=parse_labeled_float(parts[8:], 'progress', default=0.0),
                    )
                )
    for record in records:
        record.is_best = (record.episode, record.timesteps) in best_keys
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


def aggregate_records(record_groups: List[List[ValRecord]]) -> Dict[str, np.ndarray]:
    common_x = np.array(sorted({record.timesteps for records in record_groups for record in records}), dtype=np.int64)
    aggregated: Dict[str, np.ndarray] = {'timesteps': common_x}
    for metric_name, _label, _ylim in METRIC_SPECS:
        stacks = []
        for records in record_groups:
            xs = np.array([record.timesteps for record in records], dtype=np.float64)
            ys = np.array([getattr(record, metric_name) for record in records], dtype=np.float64)
            stacks.append(interpolate_series(common_x.astype(np.float64), xs, ys))
        stack_array = np.vstack(stacks)
        aggregated[f'{metric_name}_mean'] = np.nanmean(stack_array, axis=0)
        aggregated[f'{metric_name}_std'] = np.nanstd(stack_array, axis=0)
    return aggregated


def save_single_seed_csv(records: List[ValRecord], csv_path: Path, smooth_method: str, ema_alpha: float, sma_window: int) -> None:
    metric_columns = {}
    for metric_name, _label, _ylim in METRIC_SPECS:
        raw_values = [getattr(record, metric_name) for record in records]
        metric_columns[metric_name] = raw_values
        metric_columns[f'{metric_name}_smooth'] = smooth_values(raw_values, smooth_method, ema_alpha, sma_window)
    with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)
        header = ['episode', 'timesteps', 'is_best']
        for metric_name, _label, _ylim in METRIC_SPECS:
            header.extend([metric_name, f'{metric_name}_smooth'])
        writer.writerow(header)
        for index, record in enumerate(records):
            row = [record.episode, record.timesteps, int(record.is_best)]
            for metric_name, _label, _ylim in METRIC_SPECS:
                row.extend([
                    f'{metric_columns[metric_name][index]:.6f}',
                    f'{metric_columns[f"{metric_name}_smooth"][index]:.6f}',
                ])
            writer.writerow(row)


def save_multi_seed_csv(aggregated: Dict[str, np.ndarray], csv_path: Path, smooth_method: str, ema_alpha: float, sma_window: int) -> None:
    smoothed_columns = {}
    for metric_name, _label, _ylim in METRIC_SPECS:
        smoothed_columns[f'{metric_name}_smooth_mean'] = smooth_values(
            aggregated[f'{metric_name}_mean'].tolist(),
            smooth_method,
            ema_alpha,
            sma_window,
        )
    with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)
        header = ['timesteps']
        for metric_name, _label, _ylim in METRIC_SPECS:
            header.extend([
                f'{metric_name}_mean',
                f'{metric_name}_std',
                f'{metric_name}_smooth_mean',
            ])
        writer.writerow(header)
        for index, timestep in enumerate(aggregated['timesteps']):
            row = [int(timestep)]
            for metric_name, _label, _ylim in METRIC_SPECS:
                row.extend([
                    f'{aggregated[f"{metric_name}_mean"][index]:.6f}',
                    f'{aggregated[f"{metric_name}_std"][index]:.6f}',
                    f'{smoothed_columns[f"{metric_name}_smooth_mean"][index]:.6f}',
                ])
            writer.writerow(row)


def plot_single_seed(records: List[ValRecord], output_png: Path, output_pdf: Path, smooth_method: str, ema_alpha: float, sma_window: int) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    axes = list(axes.flatten())
    timesteps = [record.timesteps for record in records]
    for axis, (metric_name, ylabel, ylim) in zip(axes, METRIC_SPECS):
        raw_values = [getattr(record, metric_name) for record in records]
        smooth_curve = smooth_values(raw_values, smooth_method, ema_alpha, sma_window)
        best_points = [(record.timesteps, getattr(record, metric_name)) for record in records if record.is_best]
        axis.plot(timesteps, raw_values, color='#4C78A8', linewidth=1.6, alpha=0.35, marker='o', label='Raw')
        axis.plot(timesteps, smooth_curve, color='#E45756', linewidth=2.2, marker='o', label=smooth_method.upper())
        if best_points:
            axis.scatter(
                [item[0] for item in best_points],
                [item[1] for item in best_points],
                color='#2E8B57',
                marker='*',
                s=90,
                zorder=5,
                label='Best Checkpoint',
            )
        axis.set_xlabel('Total Environment Timesteps')
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.legend(loc='best')
    axes[-1].axis('off')
    fig.suptitle('Single-Seed Validation Convergence Curves', fontsize=14)
    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def plot_multi_seed(record_groups: List[List[ValRecord]], aggregated: Dict[str, np.ndarray], output_png: Path, output_pdf: Path, smooth_method: str, ema_alpha: float, sma_window: int) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    axes = list(axes.flatten())
    common_x = aggregated['timesteps']
    for axis, (metric_name, ylabel, ylim) in zip(axes, METRIC_SPECS):
        for records in record_groups:
            axis.plot(
                [record.timesteps for record in records],
                [getattr(record, metric_name) for record in records],
                color='#9AA5B1',
                linewidth=1.0,
                alpha=0.20,
            )
        mean_values = aggregated[f'{metric_name}_mean']
        std_values = aggregated[f'{metric_name}_std']
        smooth_curve = smooth_values(mean_values.tolist(), smooth_method, ema_alpha, sma_window)
        axis.plot(common_x, mean_values, color='#4C78A8', linewidth=1.8, label='Mean Raw')
        axis.fill_between(common_x, mean_values - std_values, mean_values + std_values, color='#4C78A8', alpha=0.15, label='Mean ± Std')
        axis.plot(common_x, smooth_curve, color='#E45756', linewidth=2.3, label=f'{smooth_method.upper()} Mean')
        axis.set_xlabel('Total Environment Timesteps')
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.legend(loc='best')
    axes[-1].axis('off')
    fig.suptitle('Multi-Seed Validation Convergence Curves', fontsize=14)
    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot paper-style validation convergence curves from training logs.')
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG_PATH, help='Single input log file.')
    parser.add_argument('--logs', nargs='*', default=None, help='Multiple input log files.')
    parser.add_argument('--log-glob', nargs='*', default=None, help='Glob pattern(s) for multi-seed logs.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory.')
    parser.add_argument('--stem', type=str, default='validation_convergence_timesteps', help='Output filename stem.')
    parser.add_argument('--smooth', type=str, default='ema', choices=['ema', 'sma'], help='Smoothing method.')
    parser.add_argument('--ema-alpha', type=float, default=0.25, help='EMA alpha.')
    parser.add_argument('--sma-window', type=int, default=3, help='SMA window size.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_paths = resolve_logs(args)
    record_groups = [parse_validation_log(path) for path in log_paths]
    output_png = args.output_dir / f'{args.stem}.png'
    output_pdf = args.output_dir / f'{args.stem}.pdf'
    output_csv = args.output_dir / f'{args.stem}.csv'
    if len(record_groups) == 1:
        plot_single_seed(record_groups[0], output_png, output_pdf, args.smooth, args.ema_alpha, args.sma_window)
        save_single_seed_csv(record_groups[0], output_csv, args.smooth, args.ema_alpha, args.sma_window)
    else:
        aggregated = aggregate_records(record_groups)
        plot_multi_seed(record_groups, aggregated, output_png, output_pdf, args.smooth, args.ema_alpha, args.sma_window)
        save_multi_seed_csv(aggregated, output_csv, args.smooth, args.ema_alpha, args.sma_window)
    print(f'Generated plot: {output_png}')
    print(f'Generated plot: {output_pdf}')
    print(f'Generated metrics CSV: {output_csv}')


if __name__ == '__main__':
    main()
