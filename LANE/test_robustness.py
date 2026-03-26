import argparse
import glob
import os
import re

import numpy as np
import torch

import env_lane
import model_rwta


def parse_args():
    parser = argparse.ArgumentParser(description='LANE robustness evaluation')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoints', nargs='*', default=None)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_models', type=int, default=2)
    parser.add_argument('--failure_rates', type=float, nargs='*', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--input_noise_levels', type=float, nargs='*', default=[0.0, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument('--weight_noise_levels', type=float, nargs='*', default=[0.0, 0.02, 0.05, 0.10])
    return parser.parse_args()


def auto_detect_checkpoints(max_models):
    prefixes = {}
    for path in glob.glob('./log_model/*_w_1.pt'):
        prefix = path[:-len('_w_1.pt')]
        prefixes[prefix] = os.path.getmtime(path)
    ordered = [item[0] for item in sorted(prefixes.items(), key=lambda item: item[1], reverse=True)]
    if not ordered:
        raise FileNotFoundError('No RWTA checkpoints were found under ./log_model.')
    return ordered[:max_models]


def build_model_from_prefix(prefix, device):
    basename = os.path.basename(prefix)
    if 'rwtaprob' in basename:
        model = model_rwta.RWTAprob(
            input_size=25,
            output_size=5,
            hid_num=8,
            hid_size=8,
            remove_connection_pattern='none',
            optimizer_name='rmsprop',
            optimizer_learning_rate=0.001,
            entropy_ratio=5.0,
            device=device,
        )
    else:
        response_window_match = re.search(r'h\d+-\d+-(\d+)_', basename)
        response_window = int(response_window_match.group(1)) if response_window_match else 40
        model = model_rwta.RWTAspike(
            input_size=25,
            output_size=5,
            hid_num=8,
            hid_size=8,
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


def collect_episode_metrics(env):
    return {
        'return': env.episode_return,
        'length': env.step_num,
        'collision': 1.0 if env.collision_count > 0 else 0.0,
    }


def summarize_metrics(metric_list):
    return {
        'return_mean': float(np.mean([item['return'] for item in metric_list])),
        'return_std': float(np.std([item['return'] for item in metric_list])),
        'length_mean': float(np.mean([item['length'] for item in metric_list])),
        'length_std': float(np.std([item['length'] for item in metric_list])),
        'collision_mean': float(np.mean([item['collision'] for item in metric_list])),
        'collision_std': float(np.std([item['collision'] for item in metric_list])),
    }


def run_action_failure_eval(model, env, failure_rate, episodes):
    metrics = []
    with torch.no_grad():
        for _ in range(episodes):
            env.init_test(record_video=False)
            observation = env.get_test_observation()
            while True:
                model_output, _ = model(observation)
                action_index = int(torch.argmax(model_output, dim=1).item())
                if np.random.rand() < failure_rate:
                    action_index = np.random.randint(0, env.action_num)
                action_onehot = torch.nn.functional.one_hot(torch.tensor([action_index]), num_classes=env.action_num).float()
                _, _, _ = env.make_action(action_onehot)
                if env.done_signal:
                    break
                observation = env.get_test_observation()
            metrics.append(collect_episode_metrics(env))
    return summarize_metrics(metrics)


def run_input_noise_eval(model, env, noise_level, episodes):
    metrics = []
    with torch.no_grad():
        for _ in range(episodes):
            env.init_test(record_video=False)
            while True:
                observation = env.get_test_observation(noise_type='gaussian', noise_param=noise_level)
                model_output, _ = model(observation)
                action_index = torch.argmax(model_output, dim=1)
                action_onehot = torch.nn.functional.one_hot(action_index, num_classes=env.action_num).float()
                _, _, _ = env.make_action(action_onehot)
                if env.done_signal:
                    break
            metrics.append(collect_episode_metrics(env))
    return summarize_metrics(metrics)


def run_weight_noise_eval(prefix, device, env, noise_level, episodes):
    metrics = []
    for _ in range(episodes):
        model = build_model_from_prefix(prefix, device)
        model.add_noise_abs('gaussian', noise_level)
        with torch.no_grad():
            env.init_test(record_video=False)
            observation = env.get_test_observation()
            while True:
                model_output, _ = model(observation)
                action_index = torch.argmax(model_output, dim=1)
                action_onehot = torch.nn.functional.one_hot(action_index, num_classes=env.action_num).float()
                _, _, _ = env.make_action(action_onehot)
                if env.done_signal:
                    break
                observation = env.get_test_observation()
        metrics.append(collect_episode_metrics(env))
    return summarize_metrics(metrics)


def print_summary(title, value, summary):
    print(
        f'{title:<14} {value:>6.3f} | '
        f'return {summary["return_mean"]:>7.3f} +/- {summary["return_std"]:>6.3f} | '
        f'length {summary["length_mean"]:>7.3f} +/- {summary["length_std"]:>6.3f} | '
        f'collision {summary["collision_mean"]:>6.3f} +/- {summary["collision_std"]:>6.3f}'
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoints = args.checkpoints if args.checkpoints else auto_detect_checkpoints(args.max_models)
    env = env_lane.GymLane(dev=device)

    print('Robustness evaluation start')
    print('Checkpoints:')
    for checkpoint in checkpoints:
        print('  -', checkpoint)

    for checkpoint in checkpoints:
        print('\n======================================')
        print('Model:', checkpoint)
        model = build_model_from_prefix(checkpoint, device)

        print('\n[Action Failure]')
        for failure_rate in args.failure_rates:
            summary = run_action_failure_eval(model, env, failure_rate, args.episodes)
            print_summary('failure_rate', failure_rate, summary)

        print('\n[Input Gaussian Noise]')
        for noise_level in args.input_noise_levels:
            summary = run_input_noise_eval(model, env, noise_level, args.episodes)
            print_summary('input_noise', noise_level, summary)

        print('\n[Weight Gaussian Noise]')
        for noise_level in args.weight_noise_levels:
            summary = run_weight_noise_eval(checkpoint, device, env, noise_level, args.episodes)
            print_summary('weight_noise', noise_level, summary)


if __name__ == '__main__':
    main()
