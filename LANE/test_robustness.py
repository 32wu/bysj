import argparse
import csv
import os
import re
from datetime import datetime

import numpy as np
import torch

import checkpoint_utils
import env_lane
import model_mlp
import model_rwta
from run_RL_ours import build_lane_episode_seed


def parse_args():
    parser = argparse.ArgumentParser(description='LANE robustness evaluation')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--road-scenario', type=str, default='highway', choices=['highway', 'merge', 'roundabout'])
    parser.add_argument('--traffic-level', type=str, default='dense', choices=['light', 'standard', 'dense'])
    parser.add_argument('--baseline-prefix', type=str, default=None,
                        help='Checkpoint prefix without _w_1.pt, or the checkpoint filename/path itself.')
    parser.add_argument('--ours-prefix', type=str, default=None,
                        help='Checkpoint prefix without _w_1.pt, or the checkpoint filename/path itself.')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--failure-rates', type=float, nargs='*', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--input-noise-levels', type=float, nargs='*', default=[0.0, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument('--weight-noise-levels', type=float, nargs='*', default=[0.0, 0.02, 0.05, 0.10])
    parser.add_argument('--output-dir', type=str, default=os.path.join(checkpoint_utils.LANE_DIR, 'comparison_reports'))
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional fixed subdirectory name under output-dir. When omitted, uses robustness_<timestamp>.')
    parser.add_argument('--model-kinds', type=str, nargs='+', default=['baseline', 'ours'],
                        choices=['baseline', 'ours'],
                        help='Select which model groups to evaluate.')
    parser.add_argument('--skip-clean', action='store_true',
                        help='Skip the extra clean-evaluation pass and only run perturbation sweeps.')
    return parser.parse_args()


def normalize_prefix(prefix_value):
    return checkpoint_utils.normalize_prefix(prefix_value) if prefix_value else None


def auto_detect_prefix(kind, road_scenario, traffic_level):
    return checkpoint_utils.find_latest_checkpoint_prefix(
        kind=kind,
        road_scenario=road_scenario,
        traffic_level=traffic_level,
        best_only=True,
    )


def resolve_prefixes(args):
    return {
        'baseline': normalize_prefix(args.baseline_prefix) or auto_detect_prefix(
            'baseline',
            args.road_scenario,
            args.traffic_level,
        ),
        'ours': normalize_prefix(args.ours_prefix) or auto_detect_prefix(
            'ours',
            args.road_scenario,
            args.traffic_level,
        ),
    }


def _parse_hidden_num(prefix_basename):
    match = re.search(r'_h(\d+)_-', prefix_basename)
    return int(match.group(1)) if match else 64


def _parse_rwta_dims(prefix_basename):
    match = re.search(r'_h(\d+)-(\d+)', prefix_basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 8, 8


def _parse_response_window(prefix_basename):
    match = re.search(r'_h\d+-\d+-(\d+)_', prefix_basename)
    return int(match.group(1)) if match else 40


def _parse_snn_num_steps(prefix_basename):
    match = re.search(r'_h(\d+)_(\d+)_', prefix_basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 64, 15


def build_model_from_prefix(prefix, device):
    basename = os.path.basename(prefix)
    if 'mlp3soft' in basename:
        hidden_num = _parse_hidden_num(basename)
        model = model_mlp.MLP_3(
            layer_sizes=[25, hidden_num, 5],
            hid_activate='softmax',
            hid_group_size=8,
            out_activate='softmax',
            optimizer_name='rmsprop',
            optimizer_learning_rate=0.001,
            entropy_ratio=0.1,
            dev=device,
        )
    elif 'mlp3relu' in basename:
        hidden_num = _parse_hidden_num(basename)
        model = model_mlp.MLP_3(
            layer_sizes=[25, hidden_num, 5],
            hid_activate='relu',
            hid_group_size=8,
            out_activate='softmax',
            optimizer_name='rmsprop',
            optimizer_learning_rate=0.001,
            entropy_ratio=0.1,
            dev=device,
        )
    elif 'snnbptt' in basename:
        import model_snnbptt

        hidden_num, snn_num_steps = _parse_snn_num_steps(basename)
        model = model_snnbptt.SNNBPTT3(
            layer_sizes=[25, hidden_num, 5],
            snn_num_steps=snn_num_steps,
            optimizer_name='rmsprop',
            optimizer_learning_rate=0.001,
            entropy_ratio=0.1,
            dev=device,
        )
    elif 'rwtaprob' in basename:
        hid_num, hid_size = _parse_rwta_dims(basename)
        model = model_rwta.RWTAprob(
            input_size=25,
            output_size=5,
            hid_num=hid_num,
            hid_size=hid_size,
            remove_connection_pattern='none',
            optimizer_name='rmsprop',
            optimizer_learning_rate=0.001,
            entropy_ratio=5.0,
            device=device,
        )
    else:
        hid_num, hid_size = _parse_rwta_dims(basename)
        response_window = _parse_response_window(basename)
        model = model_rwta.RWTAspike(
            input_size=25,
            output_size=5,
            hid_num=hid_num,
            hid_size=hid_size,
            spk_response_window='uni',
            spk_full_time=42,
            spk_resp_time=response_window,
            remove_connection_pattern='none',
            optimizer_name='rmsprop',
            optimizer_learning_rate=0.001,
            entropy_ratio=5.0,
            device=device,
        )
    model.load_model(prefix)
    return model


def set_all_seeds(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def collect_episode_metrics(env):
    episode_summary = env.get_episode_summary()
    return {
        'return': float(env.episode_return),
        'length': float(env.step_num),
        'collision': float(env.collision_count > 0),
        'success': float(env.episode_success()),
        'timeout': float(episode_summary.get('timeout_rate', 0.0)),
        'mean_speed': float(episode_summary.get('mean_speed', env.episode_mean_speed())),
        'final_progress': float(episode_summary.get('final_progress', env.episode_progress())),
        'lane_change': float(env.lane_change_count),
    }


def summarize_metrics(metric_list):
    summary = {}
    for key in ['return', 'length', 'collision', 'success', 'timeout', 'mean_speed', 'final_progress', 'lane_change']:
        values = np.asarray([item[key] for item in metric_list], dtype=np.float32)
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))
    summary['average_reward'] = summary['return_mean']
    summary['average_episode_length'] = summary['length_mean']
    summary['success_rate'] = summary['success_mean']
    summary['collision_rate'] = summary['collision_mean']
    summary['timeout_rate'] = summary['timeout_mean']
    summary['average_speed'] = summary['mean_speed_mean']
    summary['average_progress'] = summary['final_progress_mean']
    return summary


def onehot_action(action_index, action_num, device):
    return torch.nn.functional.one_hot(
        torch.tensor([int(action_index)], device=device),
        num_classes=action_num,
    ).float()


def evaluate_model(prefix, args, device, failure_rate=0.0, input_noise=0.0, weight_noise=0.0):
    env = env_lane.GymLane(
        dev=device,
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
    )
    env.suppress_test_progress = True
    metrics = []
    for episode_idx in range(args.episodes):
        episode_seed = build_lane_episode_seed(args.seed, episode_idx)
        model = build_model_from_prefix(prefix, device)
        if weight_noise > 0 and hasattr(model, 'add_noise_abs'):
            model.add_noise_abs('gaussian', weight_noise)
        with torch.no_grad():
            env.init_test(record_video=False, seed=episode_seed)
            while True:
                observation = env.get_test_observation(
                    noise_type='gaussian' if input_noise > 0 else 'none',
                    noise_param=input_noise,
                )
                model_output, _ = model(observation)
                action_index = int(torch.argmax(model_output, dim=1).item())
                if failure_rate > 0 and np.random.rand() < failure_rate:
                    action_index = int(np.random.randint(0, env.action_num))
                env.make_action(onehot_action(action_index, env.action_num, device))
                if env.done_signal:
                    break
        metrics.append(collect_episode_metrics(env))
    env.close()
    return summarize_metrics(metrics)


def print_summary(title, value, summary):
    print(
        f'{title:<14} {value:>6.3f} | '
        f'success {summary["success_rate"]:>6.3f} | '
        f'collision {summary["collision_rate"]:>6.3f} | '
        f'timeout {summary["timeout_rate"]:>6.3f} | '
        f'reward {summary["average_reward"]:>7.3f} | '
        f'len {summary["average_episode_length"]:>7.3f} | '
        f'speed {summary["average_speed"]:>6.3f} | '
        f'progress {summary["average_progress"]:>6.3f}'
    )


def ensure_output_dir(output_root, run_name=None):
    os.makedirs(output_root, exist_ok=True)
    run_dir = os.path.join(
        output_root,
        run_name or f'robustness_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = torch.device(args.device)
    prefixes = {
        model_name: prefix
        for model_name, prefix in resolve_prefixes(args).items()
        if model_name in set(args.model_kinds)
    }
    if not prefixes:
        raise ValueError('No model prefixes selected for evaluation.')
    run_dir = ensure_output_dir(args.output_dir, args.run_name)

    clean_rows = []
    failure_rows = []
    noise_rows = []
    weight_rows = []

    print('Robustness evaluation start')
    print(f'Scenario: {args.road_scenario} / {args.traffic_level}')
    for model_name, prefix in prefixes.items():
        print(f'  - {model_name}: {prefix}')

    for model_name, prefix in prefixes.items():
        print('\n======================================')
        print('Model:', model_name, prefix)

        if not args.skip_clean:
            clean_summary = evaluate_model(prefix, args, device)
            clean_rows.append({
                'model': model_name,
                'scenario': args.road_scenario,
                'traffic_level': args.traffic_level,
                'eval_episodes': args.episodes,
                'checkpoint_prefix': prefix,
                'checkpoint': os.path.basename(prefix),
                **clean_summary,
            })
            print_summary('clean_eval', 0.0, clean_summary)

        print('\n[Action Failure]')
        for failure_rate in args.failure_rates:
            set_all_seeds(args.seed + int(failure_rate * 1000), device)
            summary = evaluate_model(prefix, args, device, failure_rate=failure_rate)
            print_summary('failure_rate', failure_rate, summary)
            failure_rows.append({
                'model': model_name,
                'scenario': args.road_scenario,
                'traffic_level': args.traffic_level,
                'eval_episodes': args.episodes,
                'failure_rate': failure_rate,
                'checkpoint_prefix': prefix,
                'checkpoint': os.path.basename(prefix),
                **summary,
            })

        print('\n[Input Gaussian Noise]')
        for noise_level in args.input_noise_levels:
            set_all_seeds(args.seed + int(noise_level * 1000) + 1000, device)
            summary = evaluate_model(prefix, args, device, input_noise=noise_level)
            print_summary('input_noise', noise_level, summary)
            noise_rows.append({
                'model': model_name,
                'scenario': args.road_scenario,
                'traffic_level': args.traffic_level,
                'eval_episodes': args.episodes,
                'input_noise': noise_level,
                'checkpoint_prefix': prefix,
                'checkpoint': os.path.basename(prefix),
                **summary,
            })

        print('\n[Weight Gaussian Noise]')
        for noise_level in args.weight_noise_levels:
            set_all_seeds(args.seed + int(noise_level * 1000) + 2000, device)
            summary = evaluate_model(prefix, args, device, weight_noise=noise_level)
            print_summary('weight_noise', noise_level, summary)
            weight_rows.append({
                'model': model_name,
                'scenario': args.road_scenario,
                'traffic_level': args.traffic_level,
                'eval_episodes': args.episodes,
                'weight_noise': noise_level,
                'checkpoint_prefix': prefix,
                'checkpoint': os.path.basename(prefix),
                **summary,
            })

    write_csv(os.path.join(run_dir, 'clean_eval.csv'), clean_rows)
    write_csv(os.path.join(run_dir, 'action_failure_eval.csv'), failure_rows)
    write_csv(os.path.join(run_dir, 'input_noise_eval.csv'), noise_rows)
    write_csv(os.path.join(run_dir, 'weight_noise_eval.csv'), weight_rows)
    print('\nSaved robustness tables to:', run_dir)


if __name__ == '__main__':
    main()
