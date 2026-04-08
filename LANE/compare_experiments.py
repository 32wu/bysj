# -*- coding: utf-8 -*-
import argparse
import contextlib
import csv
import io
import os
import random
import re
from datetime import datetime

import numpy as np
import torch

import checkpoint_utils
import env_lane
import model_rwta


DEFAULT_SCENARIO_SUITE = ['highway:standard', 'highway:dense', 'merge', 'roundabout']


def parse_args():
    parser = argparse.ArgumentParser(description='Compare baseline and optimized LANE experiments')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--baseline-prefix', type=str, default=None,
                        help='Checkpoint prefix without _w_1.pt, or the checkpoint filename/path itself.')
    parser.add_argument('--ours-prefix', type=str, default=None,
                        help='Checkpoint prefix without _w_1.pt, or the checkpoint filename/path itself.')
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--failure-rates', type=float, nargs='*', default=[0.0, 0.2, 0.4])
    parser.add_argument('--input-noise-levels', type=float, nargs='*', default=[0.0, 0.05, 0.10])
    parser.add_argument('--scenario-suite', nargs='*', default=DEFAULT_SCENARIO_SUITE,
                        help='Examples: highway:standard highway:dense merge roundabout')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=os.path.join(checkpoint_utils.LANE_DIR, 'comparison_reports'))
    return parser.parse_args()


def normalize_prefix(prefix_value):
    return checkpoint_utils.normalize_prefix(prefix_value)


def auto_detect_prefix(kind, road_scenario=None, traffic_level='standard'):
    return checkpoint_utils.find_latest_checkpoint_prefix(
        kind=kind,
        road_scenario=road_scenario,
        traffic_level=traffic_level,
        best_only=True,
    )


def infer_log_path(prefix):
    return checkpoint_utils.infer_log_path(prefix)


def parse_log_summary(log_path):
    summary = {
        'log_path': log_path,
        'log_type': 'unknown',
        'best_val_episode': None,
        'best_val_score': None,
        'best_val_length': None,
        'best_val_collision': None,
        'best_val_lane_change': None,
        'final_train_episode': None,
        'final_train_score': None,
        'final_train_length': None,
        'final_train_collision': None,
        'final_train_lane_change': None,
    }
    if log_path is None or not os.path.exists(log_path):
        return summary

    with open(log_path, 'r', encoding='utf-8') as file:
        for raw_line in file:
            parts = [item for item in re.sub(',', ' ', raw_line).split()]
            if not parts:
                continue
            tag = parts[0]
            if tag == 'train' and len(parts) >= 3:
                summary['final_train_episode'] = int(parts[1])
                summary['final_train_score'] = float(parts[2])
                if len(parts) >= 6:
                    summary['log_type'] = 'improved'
                    summary['final_train_length'] = float(parts[3])
                    summary['final_train_collision'] = float(parts[4])
                    summary['final_train_lane_change'] = float(parts[5])
                elif len(parts) >= 4:
                    summary['log_type'] = 'baseline'
                    summary['final_train_length'] = float(parts[3])
                elif summary['log_type'] == 'unknown':
                    summary['log_type'] = 'baseline'
            elif tag == 'val_save' and len(parts) >= 3:
                summary['best_val_episode'] = int(parts[1])
                summary['best_val_score'] = float(parts[2])
                if len(parts) >= 6:
                    summary['log_type'] = 'improved'
                    summary['best_val_collision'] = float(parts[3])
                    summary['best_val_length'] = float(parts[4])
                    summary['best_val_lane_change'] = float(parts[5])
                elif len(parts) >= 4:
                    summary['log_type'] = 'baseline'
                    summary['best_val_length'] = float(parts[3])
                elif summary['log_type'] == 'unknown':
                    summary['log_type'] = 'baseline'
    return summary


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


def set_all_seeds(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def parse_scenario_spec(spec):
    if ':' in spec:
        scenario_name, traffic_level = spec.split(':', 1)
    else:
        scenario_name, traffic_level = spec, 'standard'
    return scenario_name, traffic_level


def resolve_prefix_for_model(model_name, prefix_override, road_scenario, traffic_level):
    if prefix_override is not None:
        return prefix_override
    kind = 'baseline' if model_name == 'baseline' else 'ours'
    return auto_detect_prefix(kind, road_scenario, traffic_level)


def onehot_action(action_index, action_num, device):
    return torch.nn.functional.one_hot(
        torch.tensor([int(action_index)], device=device),
        num_classes=action_num,
    ).float()


def collect_episode_metrics(env):
    return {
        'return': float(env.episode_return),
        'length': float(env.step_num),
        'collision': float(env.collision_count > 0),
        'success': float(env.episode_success()),
        'lane_change': float(env.lane_change_count),
    }


def summarize_metrics(metric_list):
    keys = ['return', 'length', 'collision', 'success', 'lane_change']
    summary = {}
    for key in keys:
        values = np.array([item[key] for item in metric_list], dtype=np.float32)
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))
    return summary


def evaluate_condition(prefix, device, road_scenario, traffic_level, episodes, base_seed,
                       failure_rate=0.0, input_noise=0.0):
    set_all_seeds(base_seed, device)
    env = env_lane.GymLane(dev=device, road_scenario=road_scenario, traffic_level=traffic_level)
    model = build_model_from_prefix(prefix, device)
    episode_metrics = []
    with torch.no_grad():
        for episode_idx in range(episodes):
            episode_seed = base_seed + episode_idx
            with contextlib.redirect_stdout(io.StringIO()):
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
                    action_onehot = onehot_action(action_index, env.action_num, device)
                    env.make_action(action_onehot)
                    if env.done_signal:
                        break
            episode_metrics.append(collect_episode_metrics(env))
    env.close()
    return summarize_metrics(episode_metrics)


def evaluate_models(model_prefix_overrides, args, device):
    clean_rows = []
    failure_rows = []
    noise_rows = []

    for scenario_spec in args.scenario_suite:
        road_scenario, traffic_level = parse_scenario_spec(scenario_spec)
        for model_name, prefix_override in model_prefix_overrides.items():
            prefix = resolve_prefix_for_model(model_name, prefix_override, road_scenario, traffic_level)
            base_seed = args.seed + abs(hash((model_name, scenario_spec, 'clean'))) % 10000
            summary = evaluate_condition(
                prefix=prefix,
                device=device,
                road_scenario=road_scenario,
                traffic_level=traffic_level,
                episodes=args.episodes,
                base_seed=base_seed,
            )
            clean_rows.append({
                'model': model_name,
                'scenario': road_scenario,
                'traffic_level': traffic_level,
                'checkpoint': os.path.basename(prefix),
                **summary,
            })

    benchmark_scenario, benchmark_traffic = parse_scenario_spec(args.scenario_suite[0])
    benchmark_prefixes = {
        model_name: resolve_prefix_for_model(model_name, prefix_override, benchmark_scenario, benchmark_traffic)
        for model_name, prefix_override in model_prefix_overrides.items()
    }

    for failure_rate in args.failure_rates:
        for model_name, prefix in benchmark_prefixes.items():
            base_seed = args.seed + abs(hash((model_name, failure_rate, 'failure'))) % 10000
            summary = evaluate_condition(
                prefix=prefix,
                device=device,
                road_scenario=benchmark_scenario,
                traffic_level=benchmark_traffic,
                episodes=args.episodes,
                base_seed=base_seed,
                failure_rate=failure_rate,
            )
            failure_rows.append({
                'model': model_name,
                'scenario': benchmark_scenario,
                'traffic_level': benchmark_traffic,
                'failure_rate': failure_rate,
                'checkpoint': os.path.basename(prefix),
                **summary,
            })

    for input_noise in args.input_noise_levels:
        for model_name, prefix in benchmark_prefixes.items():
            base_seed = args.seed + abs(hash((model_name, input_noise, 'noise'))) % 10000
            summary = evaluate_condition(
                prefix=prefix,
                device=device,
                road_scenario=benchmark_scenario,
                traffic_level=benchmark_traffic,
                episodes=args.episodes,
                base_seed=base_seed,
                input_noise=input_noise,
            )
            noise_rows.append({
                'model': model_name,
                'scenario': benchmark_scenario,
                'traffic_level': benchmark_traffic,
                'input_noise': input_noise,
                'checkpoint': os.path.basename(prefix),
                **summary,
            })
    return clean_rows, failure_rows, noise_rows, benchmark_prefixes


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f'compare_{timestamp}')
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


def format_float(value, digits=3):
    if value is None:
        return '-'
    return f'{value:.{digits}f}'


def build_markdown_report(run_dir, model_prefixes, log_summaries, clean_rows, failure_rows, noise_rows):
    report_lines = []
    report_lines.append('# LANE Baseline vs Optimized Comparison')
    report_lines.append('')
    report_lines.append('## Experiment Mapping')
    for model_name, prefix in model_prefixes.items():
        report_lines.append(f'- {model_name}: `{prefix}`')
    report_lines.append('')
    report_lines.append('## Training Log Summary')
    report_lines.append('| Model | Log Type | Best Val Episode | Best Val Score | Best Val Length | Best Collision | Final Train Episode | Final Train Score |')
    report_lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for model_name in ['baseline', 'optimized']:
        summary = log_summaries[model_name]
        report_lines.append(
            '| {model} | {log_type} | {best_ep} | {best_score} | {best_len} | {best_col} | {final_ep} | {final_score} |'.format(
                model=model_name,
                log_type=summary['log_type'],
                best_ep=summary['best_val_episode'] if summary['best_val_episode'] is not None else '-',
                best_score=format_float(summary['best_val_score']),
                best_len=format_float(summary['best_val_length']),
                best_col=format_float(summary['best_val_collision']),
                final_ep=summary['final_train_episode'] if summary['final_train_episode'] is not None else '-',
                final_score=format_float(summary['final_train_score']),
            )
        )
    report_lines.append('')
    report_lines.append('## Clean Evaluation')
    report_lines.append('| Scenario | Traffic | Model | Return | Length | Collision | Success | Lane Changes | Checkpoint |')
    report_lines.append('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |')
    for row in clean_rows:
        report_lines.append(
            '| {scenario} | {traffic} | {model} | {ret} | {length} | {collision} | {success} | {lane} | `{checkpoint}` |'.format(
                scenario=row['scenario'],
                traffic=row['traffic_level'],
                model=row['model'],
                ret=format_float(row['return_mean']),
                length=format_float(row['length_mean']),
                collision=format_float(row['collision_mean']),
                success=format_float(row['success_mean']),
                lane=format_float(row['lane_change_mean']),
                checkpoint=row['checkpoint'],
            )
        )
    report_lines.append('')
    report_lines.append('## Action Failure Robustness')
    report_lines.append('| Failure Rate | Model | Return | Length | Collision | Success | Checkpoint |')
    report_lines.append('| --- | --- | ---: | ---: | ---: | ---: | --- |')
    for row in failure_rows:
        report_lines.append(
            '| {failure} | {model} | {ret} | {length} | {collision} | {success} | `{checkpoint}` |'.format(
                failure=format_float(row['failure_rate']),
                model=row['model'],
                ret=format_float(row['return_mean']),
                length=format_float(row['length_mean']),
                collision=format_float(row['collision_mean']),
                success=format_float(row['success_mean']),
                checkpoint=row['checkpoint'],
            )
        )
    report_lines.append('')
    report_lines.append('## Input Noise Robustness')
    report_lines.append('| Noise Std | Model | Return | Length | Collision | Success | Checkpoint |')
    report_lines.append('| --- | --- | ---: | ---: | ---: | ---: | --- |')
    for row in noise_rows:
        report_lines.append(
            '| {noise} | {model} | {ret} | {length} | {collision} | {success} | `{checkpoint}` |'.format(
                noise=format_float(row['input_noise']),
                model=row['model'],
                ret=format_float(row['return_mean']),
                length=format_float(row['length_mean']),
                collision=format_float(row['collision_mean']),
                success=format_float(row['success_mean']),
                checkpoint=row['checkpoint'],
            )
        )
    report_path = os.path.join(run_dir, 'summary.md')
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(report_lines) + "\n")
    return report_path


def print_console_summary(log_summaries, clean_rows):
    print("\n[Training Logs]")
    for model_name in ['baseline', 'optimized']:
        summary = log_summaries[model_name]
        print(
            f"{model_name:<10} | log={summary['log_type']:<8} | best_ep={summary['best_val_episode']} | "
            f"best_score={format_float(summary['best_val_score'])} | best_len={format_float(summary['best_val_length'])}"
        )
    print("\n[Clean Evaluation]")
    for row in clean_rows:
        print(
            f"{row['scenario']:<10} {row['traffic_level']:<8} {row['model']:<10} | "
            f"return={format_float(row['return_mean'])} | length={format_float(row['length_mean'])} | "
            f"collision={format_float(row['collision_mean'])} | success={format_float(row['success_mean'])}"
        )


def main():
    args = parse_args()
    device = torch.device(args.device)
    benchmark_scenario, benchmark_traffic = parse_scenario_spec(args.scenario_suite[0])
    model_prefix_overrides = {
        'baseline': normalize_prefix(args.baseline_prefix),
        'optimized': normalize_prefix(args.ours_prefix),
    }
    clean_rows, failure_rows, noise_rows, benchmark_prefixes = evaluate_models(model_prefix_overrides, args, device)
    log_summaries = {
        model_name: parse_log_summary(infer_log_path(prefix))
        for model_name, prefix in benchmark_prefixes.items()
    }
    run_dir = ensure_output_dir(args.output_dir)
    write_csv(os.path.join(run_dir, 'clean_eval.csv'), clean_rows)
    write_csv(os.path.join(run_dir, 'action_failure_eval.csv'), failure_rows)
    write_csv(os.path.join(run_dir, 'input_noise_eval.csv'), noise_rows)
    report_path = build_markdown_report(run_dir, benchmark_prefixes, log_summaries, clean_rows, failure_rows, noise_rows)
    print_console_summary(log_summaries, clean_rows)
    print(f"\nSaved report to: {report_path}")


if __name__ == '__main__':
    main()
