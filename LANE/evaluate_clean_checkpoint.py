#!/usr/bin/env python3
import argparse
import csv
import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import env_lane
from run_RL_ours import (
    build_eval_episode_record,
    build_lane_episode_seed,
    format_validation_metrics,
    save_eval_artifacts,
    summarize_collision_records,
)
from test_robustness import build_model_from_prefix, onehot_action, normalize_prefix


def parse_args():
    parser = argparse.ArgumentParser(description='Deterministic clean evaluation for a checkpoint prefix.')
    parser.add_argument('--checkpoint-prefix', type=str, required=True)
    parser.add_argument('--road-scenario', type=str, required=True, choices=['highway', 'merge', 'roundabout'])
    parser.add_argument('--traffic-level', type=str, required=True, choices=['light', 'standard', 'dense'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--episode-offset', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--artifact-stem', type=str, required=True)
    parser.add_argument('--run-kind', type=str, default='baseline')
    return parser.parse_args()


def evaluate_checkpoint(prefix, args, device):
    env = env_lane.GymLane(
        dev=device,
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
    )
    model = build_model_from_prefix(prefix, device)

    metrics = {
        'mean_return': 0.0,
        'mean_length': 0.0,
        'mean_speed': 0.0,
        'collision_rate': 0.0,
        'mean_lane_change': 0.0,
        'success_rate': 0.0,
        'timeout_rate': 0.0,
        'offroad_rate': 0.0,
        'low_speed_abort_rate': 0.0,
        'scenario_complete_low_speed_rate': 0.0,
        'other_terminal_rate': 0.0,
        'mean_progress': 0.0,
        'mean_final_progress': 0.0,
        'termination_reason_distribution': '',
    }
    returns = []
    lengths = []
    speeds = []
    collisions = []
    lane_changes = []
    successes = []
    timeouts = []
    offroads = []
    low_speed_aborts = []
    scenario_complete_low_speed = []
    other_terminals = []
    final_progresses = []
    termination_reason_counter = {}
    episode_records = []

    with torch.no_grad():
        for episode_index in range(int(args.episodes)):
            absolute_episode_index = int(args.episode_offset) + int(episode_index)
            episode_seed = build_lane_episode_seed(int(args.seed), absolute_episode_index)
            env.init_test(record_video=False, seed=episode_seed)
            while True:
                observation = env.get_test_observation()
                model_output, _ = model(observation)
                action_index = int(torch.argmax(model_output, dim=1).item())
                env.make_action(onehot_action(action_index, env.action_num, device))
                if env.done_signal:
                    break

            returns.append(float(env.episode_return))
            lengths.append(float(env.step_num))
            speeds.append(float(env.episode_mean_speed()))
            collisions.append(float(env.collision_count > 0))
            lane_changes.append(float(env.lane_change_count))
            successes.append(float(env.episode_success()))
            episode_summary = env.get_episode_summary()
            timeouts.append(float(episode_summary.get('timeout_rate', 0.0)))
            offroads.append(float(episode_summary.get('offroad_rate', 0.0)))
            low_speed_aborts.append(float(episode_summary.get('low_speed_abort_rate', 0.0)))
            scenario_complete_low_speed.append(float(episode_summary.get('scenario_complete_low_speed_rate', 0.0)))
            other_terminals.append(float(episode_summary.get('other_terminal_rate', 0.0)))
            final_progresses.append(float(episode_summary.get('final_progress', 0.0)))
            termination_reason = str(episode_summary.get('termination_reason', 'other'))
            termination_reason_counter[termination_reason] = termination_reason_counter.get(termination_reason, 0) + 1
            episode_records.append(build_eval_episode_record(env, absolute_episode_index, episode_summary))

    env.close()

    metrics['mean_return'] = float(np.mean(returns))
    metrics['mean_length'] = float(np.mean(lengths))
    metrics['mean_speed'] = float(np.mean(speeds))
    metrics['collision_rate'] = float(np.mean(collisions))
    metrics['mean_lane_change'] = float(np.mean(lane_changes))
    metrics['success_rate'] = float(np.mean(successes))
    metrics['timeout_rate'] = float(np.mean(timeouts))
    metrics['offroad_rate'] = float(np.mean(offroads))
    metrics['low_speed_abort_rate'] = float(np.mean(low_speed_aborts))
    metrics['scenario_complete_low_speed_rate'] = float(np.mean(scenario_complete_low_speed))
    metrics['other_terminal_rate'] = float(np.mean(other_terminals))
    metrics['mean_progress'] = float(np.mean(final_progresses))
    metrics['mean_final_progress'] = float(np.mean(final_progresses))
    metrics['termination_reason_distribution'] = '|'.join(
        f'{reason}:{count}/{int(args.episodes)}'
        for reason, count in sorted(termination_reason_counter.items())
    )
    metrics['episode_records'] = episode_records
    metrics['collision_analysis'] = summarize_collision_records(episode_records)
    return metrics


def main():
    args = parse_args()
    checkpoint_prefix = normalize_prefix(args.checkpoint_prefix)
    device = torch.device(args.device)
    metrics = evaluate_checkpoint(checkpoint_prefix, args, device)

    prefix_path = Path(checkpoint_prefix)
    scenario_root = prefix_path.parents[1]
    log_dir = scenario_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    args_for_save = SimpleNamespace(
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
    )
    summary_path, detail_path, failure_path = save_eval_artifacts(
        str(log_dir),
        args.artifact_stem,
        metrics,
        args_for_save,
        checkpoint_prefix,
    )

    log_path = log_dir / f'eval_{args.artifact_stem}.txt'
    with log_path.open('a', encoding='utf-8') as log_file:
        log_file.write(f'eval_init,{datetime.datetime.now()}\n')
        log_file.write(f'eval_checkpoint,{checkpoint_prefix}\n')
        log_file.write(f'eval,{format_validation_metrics(metrics, 0)}\n')
        log_file.write(f'eval_summary_file,{summary_path}\n')
        log_file.write(f'eval_detail_file,{detail_path}\n')
        log_file.write(f'eval_failure_file,{failure_path}\n')
        log_file.write(f'finish,{datetime.datetime.now()}\n')

    print(f'Summary: {summary_path}')
    print(f'Detail: {detail_path}')
    print(f'Failure: {failure_path}')


if __name__ == '__main__':
    main()
