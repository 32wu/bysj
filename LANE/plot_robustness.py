import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

import checkpoint_utils


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.style.use('seaborn-v0_8-whitegrid')

MODEL_LABELS = {
    'baseline': 'Baseline',
    'optimized': 'Ours',
    'ours': 'Ours',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Plot LANE robustness curves from comparison report CSV files')
    parser.add_argument('--report-dir', type=str, default=None,
                        help='Directory containing action_failure_eval.csv / input_noise_eval.csv. Defaults to latest report under comparison_reports.')
    parser.add_argument('--report-dirs', type=str, nargs='*', default=None,
                        help='Optional list of robustness report directories. When two or more are provided, generate paper-style multi-scenario figures.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for generated figures. Defaults to the report directory for single-report mode.')
    parser.add_argument('--metric', type=str, default='success_rate',
                        choices=[
                            'success_rate', 'collision_rate', 'average_reward', 'average_episode_length',
                            'success_mean', 'collision_mean', 'return_mean', 'length_mean',
                        ])
    return parser.parse_args()


def find_latest_report_dir():
    report_root = os.path.join(checkpoint_utils.LANE_DIR, 'comparison_reports')
    if not os.path.isdir(report_root):
        raise FileNotFoundError('comparison_reports directory does not exist.')
    candidates = []
    for name in os.listdir(report_root):
        path = os.path.join(report_root, name)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, 'action_failure_eval.csv')):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError('No comparison report directory with robustness CSV files was found.')
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_metric_column(df, metric_name):
    aliases = {
        'success_rate': ['success_rate', 'success_mean'],
        'collision_rate': ['collision_rate', 'collision_mean'],
        'average_reward': ['average_reward', 'return_mean'],
        'average_episode_length': ['average_episode_length', 'length_mean'],
        'success_mean': ['success_mean', 'success_rate'],
        'collision_mean': ['collision_mean', 'collision_rate'],
        'return_mean': ['return_mean', 'average_reward'],
        'length_mean': ['length_mean', 'average_episode_length'],
    }
    for candidate in aliases.get(metric_name, [metric_name]):
        if candidate in df.columns:
            return candidate
    raise KeyError(f'Cannot find metric column for {metric_name}. Available columns: {list(df.columns)}')


def report_scene_label(df):
    if 'scenario' not in df.columns or 'traffic_level' not in df.columns or df.empty:
        return 'Scenario'
    scenario = str(df['scenario'].iloc[0]).strip().capitalize()
    traffic = str(df['traffic_level'].iloc[0]).strip().capitalize()
    return f'{scenario}-{traffic}'


def save_current_figure(output_base):
    paths = []
    for ext in ['png', 'pdf']:
        path = f'{output_base}.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        paths.append(path)
    return paths


def plot_metric(report_dir, output_dir, csv_name, x_column, y_column, title, xlabel, output_name):
    csv_path = os.path.join(report_dir, csv_name)
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    metric_column = resolve_metric_column(df, y_column)

    plt.figure(figsize=(8, 5), dpi=300)
    for model_name in ['baseline', 'ours', 'optimized']:
        model_df = df[df['model'] == model_name].sort_values(x_column)
        if model_df.empty:
            continue
        linestyle = '--' if model_name == 'baseline' else '-'
        marker = 'o' if model_name == 'baseline' else 's'
        plt.plot(
            model_df[x_column],
            model_df[metric_column],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=MODEL_LABELS.get(model_name, model_name),
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(metric_column, fontsize=12)
    plt.title(title, fontsize=14)
    if metric_column in ['success_rate', 'success_mean', 'collision_rate', 'collision_mean']:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.legend()
    plt.tight_layout()
    output_base = os.path.join(output_dir, output_name)
    output_paths = save_current_figure(output_base)
    plt.close()
    return output_paths


def plot_paper_figure(report_dirs, output_dir, csv_name, x_column, xlabel, figure_index, figure_stem, title_prefix):
    report_dfs = []
    for report_dir in report_dirs:
        csv_path = os.path.join(report_dir, csv_name)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if not df.empty:
            report_dfs.append((report_dir, df))
    if not report_dfs:
        return []

    fig, axes = plt.subplots(len(report_dfs), 2, figsize=(11, 4.2 * len(report_dfs)), dpi=300, squeeze=False)
    metric_specs = [
        ('success_rate', 'Success Rate'),
        ('collision_rate', 'Collision Rate'),
    ]

    for row_index, (_, df) in enumerate(report_dfs):
        scene_label = report_scene_label(df)
        for col_index, (metric_name, metric_title) in enumerate(metric_specs):
            ax = axes[row_index][col_index]
            metric_column = resolve_metric_column(df, metric_name)
            for model_name in ['baseline', 'ours', 'optimized']:
                model_df = df[df['model'] == model_name].sort_values(x_column)
                if model_df.empty:
                    continue
                linestyle = '--' if model_name == 'baseline' else '-'
                marker = 'o' if model_name == 'baseline' else 's'
                ax.plot(
                    model_df[x_column],
                    model_df[metric_column],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    label=MODEL_LABELS.get(model_name, model_name),
                )
            ax.set_title(f'{scene_label} / {metric_title}', fontsize=12)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(metric_title, fontsize=11)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, alpha=0.35)
            ax.set_ylim(0.0, 1.0)
            if row_index == 0 and col_index == 1:
                ax.legend(loc='best')

    fig.suptitle(f'Figure {figure_index}. {title_prefix}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_base = os.path.join(output_dir, figure_stem)
    output_paths = save_current_figure(output_base)
    plt.close(fig)
    return output_paths


def main():
    args = parse_args()
    report_dir = args.report_dir or find_latest_report_dir()
    report_dirs = [path for path in (args.report_dirs or []) if path]
    output_dir = args.output_dir or report_dir
    os.makedirs(output_dir, exist_ok=True)

    metric_title = {
        'success_rate': '成功率',
        'average_episode_length': '平均存活步数',
        'collision_rate': '碰撞率',
        'average_reward': '平均回报',
        'success_mean': '成功率',
        'length_mean': '平均存活步数',
        'collision_mean': '碰撞率',
        'return_mean': '平均回报',
    }[args.metric]

    generated_paths = []
    if len(report_dirs) >= 2:
        generated_paths.extend(plot_paper_figure(
            report_dirs=report_dirs,
            output_dir=output_dir,
            csv_name='action_failure_eval.csv',
            x_column='failure_rate',
            xlabel='Action Failure Rate',
            figure_index=3,
            figure_stem='fig3_action_failure_robustness',
            title_prefix='Action Failure Robustness',
        ))
        generated_paths.extend(plot_paper_figure(
            report_dirs=report_dirs,
            output_dir=output_dir,
            csv_name='input_noise_eval.csv',
            x_column='input_noise',
            xlabel='Input Gaussian Noise Std',
            figure_index=4,
            figure_stem='fig4_input_noise_robustness',
            title_prefix='Input Noise Robustness',
        ))
        generated_paths.extend(plot_paper_figure(
            report_dirs=report_dirs,
            output_dir=output_dir,
            csv_name='weight_noise_eval.csv',
            x_column='weight_noise',
            xlabel='Weight Gaussian Noise Std',
            figure_index=5,
            figure_stem='fig5_weight_noise_robustness',
            title_prefix='Weight Noise Robustness',
        ))
    else:
        generated_paths.extend(plot_metric(
            report_dir=report_dir,
            output_dir=output_dir,
            csv_name='action_failure_eval.csv',
            x_column='failure_rate',
            y_column=args.metric,
            title=f'执行器故障鲁棒性对比 ({metric_title})',
            xlabel='执行器故障率',
            output_name=f'action_failure_{args.metric}',
        ))
        generated_paths.extend(plot_metric(
            report_dir=report_dir,
            output_dir=output_dir,
            csv_name='input_noise_eval.csv',
            x_column='input_noise',
            y_column=args.metric,
            title=f'输入噪声鲁棒性对比 ({metric_title})',
            xlabel='输入高斯噪声标准差',
            output_name=f'input_noise_{args.metric}',
        ))
        generated_paths.extend(plot_metric(
            report_dir=report_dir,
            output_dir=output_dir,
            csv_name='weight_noise_eval.csv',
            x_column='weight_noise',
            y_column=args.metric,
            title=f'权重噪声鲁棒性对比 ({metric_title})',
            xlabel='权重高斯噪声标准差',
            output_name=f'weight_noise_{args.metric}',
        ))

    if not generated_paths:
        raise FileNotFoundError(f'No robustness CSV files were found in {report_dir}')

    print('Generated robustness plots:')
    for path in generated_paths:
        print(' -', path)


if __name__ == '__main__':
    main()
