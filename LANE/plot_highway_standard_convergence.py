#!/usr/bin/env python3
import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


DEFAULT_OUTPUT_DIR = Path('/root/autodl-tmp/SVPG2023/LANE/thesis_figures/highway_standard')
METRIC_SPECS = [
    ('mean_return', 'Average Reward', None),
    ('success_rate', 'Success Rate', (-0.05, 1.05)),
    ('collision_rate', 'Collision Rate', (-0.05, 1.05)),
    ('mean_length', 'Episode Length', None),
]
GROUP_STYLES = {
    'baseline': {
        'raw': '#4C78A8',
        'smooth': '#1F4E79',
        'fill': '#4C78A8',
        'label': 'Baseline',
    },
    'ours': {
        'raw': '#E45756',
        'smooth': '#9C2F2F',
        'fill': '#E45756',
        'label': 'Ours',
    },
}


@dataclass
class ValRecord:
    episode: int
    timesteps: int
    mean_return: float
    collision_rate: float
    mean_length: float
    success_rate: float


def configure_plot_style() -> None:
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
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
            'legend.fontsize': 8,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
        }
    )


def parse_labeled_float(parts: Sequence[str], label: str, default: float = 0.0) -> float:
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
    with log_path.open('r', encoding='utf-8') as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(',')]
            tag = parts[0]
            if tag == 'val_t' and len(parts) >= 8:
                records.append(
                    ValRecord(
                        episode=int(parts[1]),
                        timesteps=int(parts[2]),
                        mean_return=float(parts[3]),
                        collision_rate=float(parts[4]),
                        mean_length=float(parts[5]),
                        success_rate=float(parts[7]),
                    )
                )
            elif tag == 'val' and len(parts) >= 7:
                records.append(
                    ValRecord(
                        episode=int(parts[1]),
                        timesteps=int(parts[1]),
                        mean_return=float(parts[2]),
                        collision_rate=float(parts[3]),
                        mean_length=float(parts[4]),
                        success_rate=float(parts[6]),
                    )
                )
    if not records:
        raise ValueError(f'No validation records found in {log_path}')
    return records


def resolve_paths(logs: Sequence[str], log_globs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in logs or []:
        paths.append(Path(item).resolve())
    for pattern in log_globs or []:
        paths.extend(Path(item).resolve() for item in sorted(glob.glob(pattern)))
    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def interpolate_series(common_x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    output = np.full(common_x.shape, np.nan, dtype=np.float64)
    if xs.size == 0:
        return output
    valid_mask = (common_x >= xs.min()) & (common_x <= xs.max())
    if np.any(valid_mask):
        output[valid_mask] = np.interp(common_x[valid_mask], xs, ys)
    return output


def aggregate_group(record_groups: Sequence[Sequence[ValRecord]]) -> Dict[str, np.ndarray]:
    common_x = np.array(
        sorted({record.timesteps for records in record_groups for record in records}),
        dtype=np.int64,
    )
    aggregated: Dict[str, np.ndarray] = {'timesteps': common_x}
    for metric_name, _label, _ylim in METRIC_SPECS:
        stack_rows = []
        for records in record_groups:
            xs = np.array([record.timesteps for record in records], dtype=np.float64)
            ys = np.array([getattr(record, metric_name) for record in records], dtype=np.float64)
            stack_rows.append(interpolate_series(common_x.astype(np.float64), xs, ys))
        stack_array = np.vstack(stack_rows)
        aggregated[f'{metric_name}_mean'] = np.nanmean(stack_array, axis=0)
        aggregated[f'{metric_name}_std'] = np.nanstd(stack_array, axis=0)
    return aggregated


def ema_smooth(values: Sequence[float], alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    alpha = min(1.0, max(1e-6, float(alpha)))
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for index in range(1, values.size):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def sma_smooth(values: Sequence[float], window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    window = max(1, int(window))
    smoothed = np.zeros_like(values)
    for index in range(values.size):
        start = max(0, index - window + 1)
        smoothed[index] = np.mean(values[start:index + 1])
    return smoothed


def smooth_values(values: Sequence[float], method: str, ema_alpha: float, sma_window: int) -> np.ndarray:
    if method == 'sma':
        return sma_smooth(values, sma_window)
    return ema_smooth(values, ema_alpha)


def build_export_rows(
    baseline_data: Dict[str, np.ndarray],
    ours_data: Dict[str, np.ndarray],
    smooth_method: str,
    ema_alpha: float,
    sma_window: int,
) -> List[Dict[str, str]]:
    common_x = np.array(
        sorted(set(baseline_data['timesteps'].tolist()) | set(ours_data['timesteps'].tolist())),
        dtype=np.int64,
    )
    export_rows: List[Dict[str, str]] = []
    group_data = {
        'baseline': baseline_data,
        'ours': ours_data,
    }
    smoothed_cache: Dict[str, np.ndarray] = {}
    for group_name, group_values in group_data.items():
        for metric_name, _label, _ylim in METRIC_SPECS:
            key = f'{group_name}_{metric_name}_smooth'
            smoothed_cache[key] = smooth_values(
                group_values[f'{metric_name}_mean'],
                smooth_method,
                ema_alpha,
                sma_window,
            )
    for timestep in common_x:
        row: Dict[str, str] = {'timesteps': str(int(timestep))}
        for group_name, group_values in group_data.items():
            group_timesteps = group_values['timesteps']
            for metric_name, _label, _ylim in METRIC_SPECS:
                raw_col = f'{group_name}_{metric_name}_raw'
                smooth_col = f'{group_name}_{metric_name}_smooth'
                std_col = f'{group_name}_{metric_name}_std'
                matched = np.where(group_timesteps == timestep)[0]
                if matched.size == 0:
                    row[raw_col] = ''
                    row[smooth_col] = ''
                    row[std_col] = ''
                else:
                    index = int(matched[0])
                    row[raw_col] = f'{group_values[f"{metric_name}_mean"][index]:.6f}'
                    row[smooth_col] = f'{smoothed_cache[f"{group_name}_{metric_name}_smooth"][index]:.6f}'
                    row[std_col] = f'{group_values[f"{metric_name}_std"][index]:.6f}'
        export_rows.append(row)
    return export_rows


def save_csv(
    csv_path: Path,
    baseline_data: Dict[str, np.ndarray],
    ours_data: Dict[str, np.ndarray],
    smooth_method: str,
    ema_alpha: float,
    sma_window: int,
) -> None:
    fieldnames = ['timesteps']
    for group_name in ['baseline', 'ours']:
        for metric_name, _label, _ylim in METRIC_SPECS:
            fieldnames.extend(
                [
                    f'{group_name}_{metric_name}_raw',
                    f'{group_name}_{metric_name}_smooth',
                    f'{group_name}_{metric_name}_std',
                ]
            )
    rows = build_export_rows(baseline_data, ours_data, smooth_method, ema_alpha, sma_window)
    with csv_path.open('w', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_group_metric(axis, group_name: str, group_data: Dict[str, np.ndarray], metric_name: str, smooth_method: str, ema_alpha: float, sma_window: int) -> None:
    style = GROUP_STYLES[group_name]
    timesteps = group_data['timesteps']
    raw_mean = group_data[f'{metric_name}_mean']
    raw_std = group_data[f'{metric_name}_std']
    smooth_mean = smooth_values(raw_mean, smooth_method, ema_alpha, sma_window)
    axis.plot(
        timesteps,
        raw_mean,
        color=style['raw'],
        linewidth=1.4,
        alpha=0.45,
        label=f"{style['label']} Raw",
    )
    axis.fill_between(
        timesteps,
        raw_mean - raw_std,
        raw_mean + raw_std,
        color=style['fill'],
        alpha=0.10,
    )
    axis.plot(
        timesteps,
        smooth_mean,
        color=style['smooth'],
        linewidth=2.3,
        label=f"{style['label']} {smooth_method.upper()}",
    )


def plot_compare(
    baseline_data: Dict[str, np.ndarray],
    ours_data: Dict[str, np.ndarray],
    output_png: Path,
    output_pdf: Path,
    smooth_method: str,
    ema_alpha: float,
    sma_window: int,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = list(axes.flatten())
    for axis, (metric_name, ylabel, ylim) in zip(axes, METRIC_SPECS):
        plot_group_metric(axis, 'baseline', baseline_data, metric_name, smooth_method, ema_alpha, sma_window)
        plot_group_metric(axis, 'ours', ours_data, metric_name, smooth_method, ema_alpha, sma_window)
        axis.set_xlabel('Total Environment Timesteps')
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.legend(loc='best')
    fig.suptitle('Highway-Standard Validation Convergence', fontsize=14)
    fig.savefig(output_png, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot highway-standard baseline-vs-ours convergence curves.')
    parser.add_argument('--baseline-logs', nargs='*', default=None, help='Explicit baseline log paths.')
    parser.add_argument('--baseline-log-glob', nargs='*', default=None, help='Glob pattern(s) for baseline logs.')
    parser.add_argument('--ours-logs', nargs='*', default=None, help='Explicit ours log paths.')
    parser.add_argument('--ours-log-glob', nargs='*', default=None, help='Glob pattern(s) for ours logs.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory.')
    parser.add_argument('--stem', type=str, default='highway_standard_baseline_vs_ours_convergence', help='Output filename stem.')
    parser.add_argument('--smooth', type=str, default='ema', choices=['ema', 'sma'], help='Smoothing method.')
    parser.add_argument('--ema-alpha', type=float, default=0.25, help='EMA alpha.')
    parser.add_argument('--sma-window', type=int, default=3, help='SMA window size.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_paths = resolve_paths(args.baseline_logs, args.baseline_log_glob)
    ours_paths = resolve_paths(args.ours_logs, args.ours_log_glob)
    if not baseline_paths:
        raise SystemExit('No baseline logs provided.')
    if not ours_paths:
        raise SystemExit('No ours logs provided.')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_records = [parse_validation_log(path) for path in baseline_paths]
    ours_records = [parse_validation_log(path) for path in ours_paths]
    baseline_data = aggregate_group(baseline_records)
    ours_data = aggregate_group(ours_records)

    output_png = args.output_dir / f'{args.stem}.png'
    output_pdf = args.output_dir / f'{args.stem}.pdf'
    output_csv = args.output_dir / f'{args.stem}.csv'
    plot_compare(
        baseline_data,
        ours_data,
        output_png,
        output_pdf,
        args.smooth,
        args.ema_alpha,
        args.sma_window,
    )
    save_csv(output_csv, baseline_data, ours_data, args.smooth, args.ema_alpha, args.sma_window)
    print(f'Generated plot: {output_png}')
    print(f'Generated plot: {output_pdf}')
    print(f'Generated metrics CSV: {output_csv}')


if __name__ == '__main__':
    main()
