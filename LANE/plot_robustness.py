import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

import checkpoint_utils


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.style.use('seaborn-v0_8-whitegrid')

MODEL_LABELS = {
    'baseline': 'Baseline',
    'optimized': 'Ours',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Plot LANE robustness curves from comparison report CSV files')
    parser.add_argument('--report-dir', type=str, default=None,
                        help='Directory containing action_failure_eval.csv / input_noise_eval.csv. Defaults to latest report under comparison_reports.')
    parser.add_argument('--metric', type=str, default='success_mean',
                        choices=['success_mean', 'length_mean', 'collision_mean', 'return_mean'])
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


def plot_metric(report_dir, csv_name, x_column, y_column, title, xlabel, output_name):
    csv_path = os.path.join(report_dir, csv_name)
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    plt.figure(figsize=(8, 5), dpi=300)
    for model_name in ['baseline', 'optimized']:
        model_df = df[df['model'] == model_name].sort_values(x_column)
        if model_df.empty:
            continue
        linestyle = '--' if model_name == 'baseline' else '-'
        marker = 'o' if model_name == 'baseline' else 's'
        plt.plot(
            model_df[x_column],
            model_df[y_column],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=MODEL_LABELS.get(model_name, model_name),
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(report_dir, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def main():
    args = parse_args()
    report_dir = args.report_dir or find_latest_report_dir()

    metric_title = {
        'success_mean': '成功率',
        'length_mean': '平均存活步数',
        'collision_mean': '碰撞率',
        'return_mean': '平均回报',
    }[args.metric]

    generated_paths = []
    action_path = plot_metric(
        report_dir=report_dir,
        csv_name='action_failure_eval.csv',
        x_column='failure_rate',
        y_column=args.metric,
        title=f'执行器故障鲁棒性对比 ({metric_title})',
        xlabel='执行器故障率',
        output_name=f'action_failure_{args.metric}.png',
    )
    if action_path is not None:
        generated_paths.append(action_path)

    input_path = plot_metric(
        report_dir=report_dir,
        csv_name='input_noise_eval.csv',
        x_column='input_noise',
        y_column=args.metric,
        title=f'输入噪声鲁棒性对比 ({metric_title})',
        xlabel='输入高斯噪声标准差',
        output_name=f'input_noise_{args.metric}.png',
    )
    if input_path is not None:
        generated_paths.append(input_path)

    weight_path = plot_metric(
        report_dir=report_dir,
        csv_name='weight_noise_eval.csv',
        x_column='weight_noise',
        y_column=args.metric,
        title=f'权重噪声鲁棒性对比 ({metric_title})',
        xlabel='权重高斯噪声标准差',
        output_name=f'weight_noise_{args.metric}.png',
    )
    if weight_path is not None:
        generated_paths.append(weight_path)

    if not generated_paths:
        raise FileNotFoundError(f'No robustness CSV files were found in {report_dir}')

    print('Generated robustness plots:')
    for path in generated_paths:
        print(' -', path)


if __name__ == '__main__':
    main()
