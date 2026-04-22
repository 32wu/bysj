# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import datetime
import re
import time
import torch
import torch.nn as nn
import random

import memory_lib
import checkpoint_utils
from ppo_stability import STABLE_HIGHWAY_STANDARD_PPO, scheduled_entropy, should_restore_best_checkpoint


def get_arguments():
    parser = argparse.ArgumentParser(description='Description: run_RL')
    # RL
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--entropy', type=float, default=0.10)
    parser.add_argument('--PPO_epochs', type=int, default=5)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--alg', type=str, default='ppo',
                        choices=['ppo', 'reinforce'])
    # Program
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--thread', type=int, default=-1)
    parser.add_argument('--train_num', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--val_interval_timesteps', type=int, default=0)
    parser.add_argument('--val_episodes', type=int, default=20)
    parser.add_argument('--rep', type=int, default=11)
    parser.add_argument('--ignore_checkpoint', default=False, action='store_true')
    parser.add_argument('--monitor_time', default=False, action='store_true')
    parser.add_argument('--allow_long_train', default=False, action='store_true')
    # Task amd model
    parser.add_argument('--task', type=str, default='gymip',
                        choices=['gymip'])
    parser.add_argument('--model', type=str, default='rwtaprob',
                        choices=['mlp3soft', 'mlp3relu', 'rwtaprob', 'rwtaspk', 'snnbptt', 'ann2snn'])
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                        choices=['sgd', 'adam', 'rmsprop'])
    # ---------------------------------------------------
    # for Gym IP
    parser.add_argument('--gymip_train_xml', type=str, default='inverted_pendulum_ChangeThk_0.050000.xml')
    # for mlp3, snnbptt
    parser.add_argument('--hidden_num', type=int, default=64)
    # for rwta
    parser.add_argument('--hid_group_num', type=int, default=8)
    parser.add_argument('--hid_group_size', type=int, default=8)
    parser.add_argument('--rwta_del_connection', type=str, default='none',
                        choices=['none', 'hh', 'sa', 'hhsa', 'ha', 'sh'])
    # for rwtaspk
    parser.add_argument('--response_window', type=int, default=40)
    # for snnbptt
    parser.add_argument('--snn_num_steps', type=int, default=15)
    parser.add_argument('--road_scenario', type=str, default='highway', choices=['highway', 'merge', 'roundabout'])
    parser.add_argument('--traffic_level', type=str, default='standard', choices=['light', 'standard', 'dense'])
    parser.add_argument('--highway_reward_stage', type=str, default='c_success', choices=['baseline', 'a_timeout', 'b_progress_speed', 'c_success'])
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--adv_norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--critic_lr', type=float, default=0.0)
    parser.add_argument('--entropy_min', type=float, default=0.001)
    parser.add_argument('--entropy_decay', type=float, default=1.0)
    parser.add_argument('--entropy_warmup_scale', type=float, default=1.0)
    parser.add_argument('--entropy_warmup_episodes', type=int, default=0)
    parser.add_argument('--skip_post_tests', default=False, action='store_true')
    # ---------------------------------------------------
    return parser.parse_args()


HIGHWAY_STANDARD_BASE_PROFILE = dict(STABLE_HIGHWAY_STANDARD_PPO)
BASE_PROFILE_OPTIONAL_EXPLORATION_KEYS = {
    'entropy_warmup_scale',
    'entropy_warmup_episodes',
}


def reload_log_file(filename):
    train_epi_num = 0
    val_best_return = -10000.0
    val_best_collision = float('inf')
    val_best_length = 0.0
    val_best_success = 0.0
    val_best_speed = 0.0
    val_best_lane_change = 0.0
    total_env_steps = 0
    with open(filename) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = [item.strip() for item in line.split(',')]
            if not parts:
                continue
            tag = parts[0]
            if tag == 'train' and len(parts) > 1:
                train_epi_num = max(train_epi_num, int(parts[1]))
            elif tag == 'train_t' and len(parts) > 2:
                train_epi_num = max(train_epi_num, int(parts[1]))
                total_env_steps = max(total_env_steps, int(parts[2]))
            elif tag == 'summary' and len(parts) > 2 and parts[1] == 'total_env_steps':
                total_env_steps = max(total_env_steps, int(parts[2]))
            elif tag in ['val_save', 'val_save_t']:
                numeric_offset = 2
                if tag == 'val_save_t' and len(parts) > 3:
                    total_env_steps = max(total_env_steps, int(parts[2]))
                    numeric_offset = 3
                if len(parts) > numeric_offset:
                    val_best_return = float(parts[numeric_offset])
                if len(parts) > numeric_offset + 1:
                    val_best_collision = float(parts[numeric_offset + 1])
                if len(parts) > numeric_offset + 2:
                    val_best_length = float(parts[numeric_offset + 2])
                if len(parts) > numeric_offset + 4:
                    val_best_success = float(parts[numeric_offset + 4])
                if len(parts) > numeric_offset + 5:
                    val_best_speed = float(parts[numeric_offset + 5])
                if len(parts) > numeric_offset + 3:
                    val_best_lane_change = float(parts[numeric_offset + 3])
            elif tag in ['val', 'val_t'] and val_best_return <= -9999.0:
                numeric_offset = 2
                if tag == 'val_t' and len(parts) > 3:
                    total_env_steps = max(total_env_steps, int(parts[2]))
                    numeric_offset = 3
                if len(parts) > numeric_offset:
                    val_best_return = float(parts[numeric_offset])
    return (
        train_epi_num,
        val_best_return,
        val_best_collision,
        val_best_length,
        val_best_success,
        val_best_speed,
        val_best_lane_change,
        total_env_steps,
    )


def log_text(file_handle, type_str, record_text, onscreen=True):
    global log_text_flush_time
    if onscreen:
        print('\033[92m%s\033[0m' % type_str.ljust(10), record_text)
    file_handle.write((type_str+',').ljust(10) + record_text + '\n')
    if time.time() - log_text_flush_time > 10:
        log_text_flush_time = time.time()
        file_handle.flush()
        os.fsync(file_handle.fileno())


class TimeMonitor:
    def __init__(self):
        self.size = 100
        self.time_pointer_inference = 0
        self.time_pointer_optimize = 0
        self.time_inference = np.ones([self.size], dtype=np.float32) * (-1)
        self.time_optimize = np.ones([self.size], dtype=np.float32) * (-1)

    def record_time(self, rec_type=1, value=0.0):
        if rec_type == 1:
            self.time_inference[self.time_pointer_inference] = value * 1000
            self.time_pointer_inference = (self.time_pointer_inference + 1) % self.size
            if self.time_pointer_inference == 0:
                print('timer inf: %7.3f %7.3f %7.3f %7.3f' % (
                        float(np.mean(self.time_inference)),
                        float(np.std(self.time_inference)),
                        np.min(self.time_inference),
                        np.max(self.time_inference), ))
        if rec_type == 2:
            self.time_optimize[self.time_pointer_optimize] = value * 1000
            self.time_pointer_optimize = (self.time_pointer_optimize + 1) % self.size
            if self.time_pointer_optimize == 0:
                print('timer opt: %7.3f %7.3f %7.3f %7.3f' % (
                        float(np.mean(self.time_optimize)),
                        float(np.std(self.time_optimize)),
                        np.min(self.time_optimize),
                        np.max(self.time_optimize), ))


def apply_lane_baseline_profile(args):
    adjustments = []

    def set_exact(attr_name, target_value):
        current_value = getattr(args, attr_name)
        if current_value != target_value:
            setattr(args, attr_name, target_value)
            adjustments.append(f'{attr_name}->{target_value}')

    def clamp_min(attr_name, target_value):
        current_value = getattr(args, attr_name)
        if current_value < target_value:
            setattr(args, attr_name, target_value)
            adjustments.append(f'{attr_name}->{target_value}')

    def clamp_max(attr_name, target_value):
        current_value = getattr(args, attr_name)
        if current_value > target_value:
            setattr(args, attr_name, target_value)
            adjustments.append(f'{attr_name}->{target_value}')

    if args.road_scenario == 'highway':
        traffic_level = getattr(args, 'traffic_level', 'standard')
        if traffic_level == 'standard':
            for attr_name, target_value in HIGHWAY_STANDARD_BASE_PROFILE.items():
                if attr_name in BASE_PROFILE_OPTIONAL_EXPLORATION_KEYS:
                    continue
                set_exact(attr_name, target_value)
            if not getattr(args, 'allow_long_train', False):
                clamp_max('train_num', 2000)
            return adjustments
        lr_cap = {
            'light': 3.0e-4,
            'standard': 2.5e-4,
            'dense': 2.0e-4,
        }
        gamma_floor = {
            'light': 0.995,
            'standard': 0.998,
            'dense': 0.998,
        }
        gae_lambda_floor = {
            'light': 0.95,
            'standard': 0.95,
            'dense': 0.95,
        }
        entropy_cap = {
            'light': 0.03,
            'standard': 0.02,
            'dense': 0.02,
        }
        train_num_floor = {
            'light': 2400,
        }
        clamp_max('lr', lr_cap.get(traffic_level, 0.0007))
        clamp_min('gamma', gamma_floor.get(traffic_level, 0.994))
        clamp_min('gae_lambda', gae_lambda_floor.get(traffic_level, 0.97))
        clamp_max('entropy', entropy_cap.get(traffic_level, 0.02))
        clamp_max('PPO_epochs', 4)
        clamp_max('eps_clip', 0.20)
        clamp_max('reward_scale', 0.80)
        clamp_max('grad_clip', 0.50)
        clamp_max('entropy_min', 0.005)
        if args.entropy_decay >= 1.0:
            args.entropy_decay = STABLE_HIGHWAY_STANDARD_PPO['entropy_decay']
            adjustments.append(f'entropy_decay->{args.entropy_decay}')
        if traffic_level in ['standard', 'dense'] and not getattr(args, 'allow_long_train', False):
            clamp_max('train_num', 2000)
        elif args.train_num < train_num_floor.get(traffic_level, 3200):
            args.train_num = train_num_floor.get(traffic_level, 3200)
            adjustments.append(f'train_num->{args.train_num}')
    elif args.road_scenario in ['merge', 'roundabout']:
        if not getattr(args, 'allow_long_train', False):
            clamp_max('train_num', 2000)
    return adjustments


def select_lane_vehicle_count(args, train_epi_i):
    if getattr(args, 'road_scenario', None) != 'highway':
        return None
    traffic_target = {
        'light': 28,
        'standard': 10,
        'dense': 84,
    }
    traffic_start = {
        'light': 16,
        'standard': 10,
        'dense': 46,
    }
    curriculum_span = {
        'light': 120,
        'standard': 1,
        'dense': 480,
    }
    traffic_level = getattr(args, 'traffic_level', 'standard')
    start_vehicle_num = traffic_start.get(traffic_level, 30)
    end_vehicle_num = traffic_target.get(traffic_level, 50)
    span = curriculum_span.get(traffic_level, 280)
    progress = min(1.0, train_epi_i / max(1, span))
    return int(round(start_vehicle_num + (end_vehicle_num - start_vehicle_num) * progress))


def build_lane_episode_seed(base_seed, episode_index, slot_index=0):
    if int(base_seed) < 0:
        return None
    modulus = 2147483647
    composite = (int(base_seed) + 1) * 1000003 + int(episode_index) * 1009 + int(slot_index) * 97
    return int(composite % modulus)


def set_random_seed(seed, device):
    if int(seed) < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def resolve_validation_schedule(args):
    if int(getattr(args, 'val_interval_timesteps', 0)) > 0:
        return int(args.val_interval_timesteps), max(1, int(args.val_episodes))
    if args.road_scenario == 'merge':
        return 1500, max(1, int(args.val_episodes))
    if args.road_scenario == 'roundabout':
        return 2250, max(1, int(args.val_episodes))
    if args.road_scenario == 'highway' and getattr(args, 'traffic_level', 'standard') == 'dense':
        return 4500, max(1, int(args.val_episodes))
    return 3000, max(1, int(args.val_episodes))


def evaluate_lane_policy(env, model, episode_num, vehicles_count=None):
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
    with torch.no_grad():
        for episode_index in range(episode_num):
            episode_seed = build_lane_episode_seed(getattr(env, 'seed_base', -1), episode_index)
            env.init_val(vehicles_count=vehicles_count, seed=episode_seed)
            observation = env.get_val_observation()
            for _step_i in range(env.max_step_num):
                model_output, _ = model(observation)
                action_index = torch.argmax(model_output, dim=1)
                action_onehot = torch.nn.functional.one_hot(action_index, num_classes=env.action_num).float()
                next_state, reward, done, info, step_record = env.make_action(action_onehot)
                if env.done_signal:
                    break
                observation = next_state
            returns.append(env.episode_return)
            lengths.append(env.step_num)
            speeds.append(env.episode_mean_speed())
            collisions.append(1.0 if env.collision_count > 0 else 0.0)
            lane_changes.append(env.lane_change_count)
            successes.append(env.episode_success())
            episode_summary = env.get_episode_summary()
            timeouts.append(float(episode_summary['timeout_rate']))
            offroads.append(float(episode_summary['offroad_rate']))
            low_speed_aborts.append(float(episode_summary['low_speed_abort_rate']))
            scenario_complete_low_speed.append(float(episode_summary['scenario_complete_low_speed_rate']))
            other_terminals.append(float(episode_summary['other_terminal_rate']))
            final_progresses.append(float(episode_summary['final_progress']))
            termination_reason = episode_summary['termination_reason']
            termination_reason_counter[termination_reason] = termination_reason_counter.get(termination_reason, 0) + 1
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
    reason_parts = []
    for reason_name in sorted(termination_reason_counter.keys()):
        reason_parts.append(f'{reason_name}:{termination_reason_counter[reason_name]}/{episode_num}')
    metrics['termination_reason_distribution'] = '|'.join(reason_parts)
    return metrics


def format_validation_metrics(metrics, episode_index, total_env_steps=None):
    if total_env_steps is None:
        base_record = (
            '%d, %8.6f, %6.4f, %8.4f, %8.4f, %6.4f, %8.4f' % (
                episode_index,
                metrics['mean_return'],
                metrics['collision_rate'],
                metrics['mean_length'],
                metrics['mean_lane_change'],
                metrics['success_rate'],
                metrics['mean_speed'],
            )
        )
    else:
        base_record = (
            '%d, %d, %8.6f, %6.4f, %8.4f, %8.4f, %6.4f, %8.4f' % (
                episode_index,
                int(total_env_steps),
                metrics['mean_return'],
                metrics['collision_rate'],
                metrics['mean_length'],
                metrics['mean_lane_change'],
                metrics['success_rate'],
                metrics['mean_speed'],
            )
        )
    extra_record = (
        ', timeout %6.4f, progress %6.4f, offroad %6.4f, low_speed_abort %6.4f, '
        'complete_low_speed %6.4f, other_terminal %6.4f, term %s'
    ) % (
        metrics.get('timeout_rate', 0.0),
        metrics.get('mean_final_progress', metrics.get('mean_progress', 0.0)),
        metrics.get('offroad_rate', 0.0),
        metrics.get('low_speed_abort_rate', 0.0),
        metrics.get('scenario_complete_low_speed_rate', 0.0),
        metrics.get('other_terminal_rate', 0.0),
        metrics.get('termination_reason_distribution', ''),
    )
    return base_record + extra_record


def format_train_episode_metrics(train_epi_i, env, episode_vehicle_count, entropy_value, total_env_steps=None):
    episode_summary = env.get_episode_summary()
    if total_env_steps is None:
        base_record = (
            '%d, %8.6f, %4d, %3d, %6.4f' % (
                train_epi_i,
                env.episode_return,
                env.step_num,
                -1 if episode_vehicle_count is None else int(episode_vehicle_count),
                entropy_value,
            )
        )
    else:
        base_record = (
            '%d, %d, %8.6f, %4d, %3d, %6.4f' % (
                train_epi_i,
                int(total_env_steps),
                env.episode_return,
                env.step_num,
                -1 if episode_vehicle_count is None else int(episode_vehicle_count),
                entropy_value,
            )
        )
    extra_record = (
        ', success %4.2f, collision %4.2f, timeout %4.2f, progress %5.3f, speed %6.3f, term %s'
    ) % (
        float(episode_summary.get('success_rate', 0.0)),
        float(episode_summary.get('collision_rate', 0.0)),
        float(episode_summary.get('timeout_rate', 0.0)),
        float(episode_summary.get('final_progress', episode_summary.get('route_progress', 0.0))),
        float(episode_summary.get('mean_speed', 0.0)),
        episode_summary.get('termination_reason', 'unknown'),
    )
    return base_record + extra_record


def lane_validation_quality(metrics):
    excessive_lane_changes = max(0.0, float(metrics['mean_lane_change']) - 1.5)
    return (
        float(metrics['mean_return'])
        + 0.55 * float(metrics.get('mean_length', 0.0))
        + 1.5 * float(metrics['mean_speed'])
        - 30.0 * float(metrics['collision_rate'])
        - 4.0 * excessive_lane_changes
    )


def align_action_to_env_execution(model_output, sampled_action_onehot, step_info, action_num):
    executed_action_index = None
    if isinstance(step_info, dict):
        executed_action_index = step_info.get('executed_action')
    if executed_action_index is None:
        return sampled_action_onehot, torch.distributions.OneHotCategorical(model_output).log_prob(sampled_action_onehot)
    executed_action_index = int(executed_action_index)
    executed_action_onehot = torch.nn.functional.one_hot(
        torch.tensor([executed_action_index], device=model_output.device),
        num_classes=action_num,
    ).float()
    executed_action_logprob = torch.distributions.OneHotCategorical(model_output).log_prob(executed_action_onehot)
    return executed_action_onehot, executed_action_logprob


def set_model_learning_rates(model, model_c, actor_lr, critic_lr=None):
    actor_lr = float(actor_lr)
    critic_lr = float(actor_lr if critic_lr is None else critic_lr)
    if hasattr(model, 'optimizer_learning_rate'):
        model.optimizer_learning_rate = actor_lr
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = actor_lr
    if hasattr(model_c, 'set_learning_rate'):
        model_c.set_learning_rate(critic_lr)
    return actor_lr, critic_lr


def maybe_update_entropy(model, entropy_value):
    if hasattr(model, 'update_entropy'):
        model.update_entropy(entropy_value)


def select_entropy_value(args, train_epi_i):
    return scheduled_entropy(
        initial_entropy=args.entropy,
        entropy_min=args.entropy_min,
        entropy_decay=args.entropy_decay,
        episode_index=train_epi_i,
        entropy_warmup_scale=getattr(args, 'entropy_warmup_scale', 1.0),
        entropy_warmup_episodes=getattr(args, 'entropy_warmup_episodes', 0),
    )


def compute_gae(rewards, dones, state_values, next_state_values, gamma, gae_lambda, reward_scale, adv_norm):
    scaled_rewards = rewards * reward_scale
    advantages = torch.zeros_like(scaled_rewards)
    gae = torch.zeros(1, device=rewards.device, dtype=torch.float32)
    for index in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[index]
        delta = scaled_rewards[index] + gamma * next_state_values[index] * mask - state_values[index]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[index] = gae
    returns = advantages + state_values
    if adv_norm:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return advantages.detach(), returns.detach()


def is_better_lane_checkpoint(val_metrics, best_return, best_collision, best_length, best_success, best_speed, best_lane_change):
    if val_metrics['success_rate'] > best_success + 1e-6:
        return True
    if (
        abs(val_metrics['success_rate'] - best_success) <= 1e-6 and
        val_metrics['collision_rate'] <= best_collision + 0.02 and
        val_metrics['mean_length'] > best_length + 1.0
    ):
        return True
    if best_success <= 1e-6 and val_metrics['success_rate'] <= 1e-6:
        candidate_quality = lane_validation_quality(val_metrics)
        best_quality = lane_validation_quality({
            'mean_return': best_return,
            'mean_length': best_length,
            'mean_speed': best_speed,
            'collision_rate': best_collision,
            'mean_lane_change': best_lane_change,
        })
        return candidate_quality > best_quality + 1e-6
    if abs(val_metrics['success_rate'] - best_success) <= 1e-6 and val_metrics['collision_rate'] < best_collision - 1e-6:
        return True
    if (
        abs(val_metrics['success_rate'] - best_success) <= 1e-6 and
        abs(val_metrics['collision_rate'] - best_collision) <= 1e-6 and
        val_metrics['mean_speed'] > best_speed + 1e-6
    ):
        return True
    if (
        abs(val_metrics['success_rate'] - best_success) <= 1e-6 and
        abs(val_metrics['collision_rate'] - best_collision) <= 1e-6 and
        abs(val_metrics['mean_speed'] - best_speed) <= 1e-6 and
        val_metrics['mean_return'] > best_return + 1e-6
    ):
        return True
    return False


if __name__ == "__main__":
    # Arguments
    args = get_arguments()
    if args.model in ['mlp3soft', 'mlp3relu']:
        model_str = 'h%d_-' % args.hidden_num
    elif args.model in ['snnbptt']:
        model_str = 'h%d_%d' % (args.hidden_num, args.snn_num_steps)
    elif args.model in ['rwtaprob']:
        model_str = 'h%d-%d_%s' % (args.hid_group_num, args.hid_group_size, args.rwta_del_connection)
    elif args.model in ['rwtaspk']:
        model_str = 'h%d-%d-%d_%s' % (args.hid_group_num, args.hid_group_size,
                                      args.response_window, args.rwta_del_connection)
    elif args.model in ['ann2snn']:
        model_str = 'h%d_-' % (args.hidden_num)
    else:
        model_str = 'error'
        print('\033[91mError in arguments\033[0m')
    # Pre-defined Parameters
    args.hidden_num, args.hid_group_num, args.hid_group_size = 64, 8, 8
    if args.alg == 'ppo':
        args.train_num = 2000 if args.train_num == 20000 else args.train_num
    else:
        args.train_num = 5000 if args.train_num == 20000 else args.train_num
    baseline_profile_adjustments = apply_lane_baseline_profile(args)
    if args.critic_lr <= 0:
        args.critic_lr = args.lr
    if args.entropy_decay <= 0:
        args.entropy_decay = 1.0
    run_kind = 'baseline'
    reward_stage_suffix = ''
    if args.road_scenario == 'highway':
        reward_stage_suffix = '_hrs%s' % args.highway_reward_stage
    entropy_warmup_suffix = ''
    if args.entropy_warmup_scale > 1.0 + 1e-8 and args.entropy_warmup_episodes > 0:
        entropy_warmup_suffix = '_ew%4.2fe%d' % (args.entropy_warmup_scale, args.entropy_warmup_episodes)
    run_id_suffix = '_rep%02d' % args.rep
    if args.seed >= 0:
        run_id_suffix = '_seed%02d' % args.seed
    EXP_NAME = '%s_%s_%s_%s_%s_%8.6f_%4.2f_%6.5f_%d_%5.4f_road%s_tf%s%s%s%s' % (
            args.alg, args.task, args.model, model_str, args.optimizer,
            args.lr, args.entropy, args.gamma,
            args.PPO_epochs, args.eps_clip,
            args.road_scenario, args.traffic_level, reward_stage_suffix, entropy_warmup_suffix, run_id_suffix)
    active_model_dir, active_log_dir = checkpoint_utils.activate_scenario_output_dirs(
            run_kind=run_kind, road_scenario=args.road_scenario, traffic_level=args.traffic_level, create=True)
    # Task specified variables
    if args.road_scenario == 'highway' and args.alg == 'ppo':
        test_num = 10
        train_frequency = 32
    else:
        test_num = 10
        train_frequency = 10
    val_interval_timesteps, val_num = resolve_validation_schedule(args)
    # Device
    if args.cuda < 0:
        torch_device = torch.device('cpu')
        if args.thread == -1:
            pass
        else:
            torch.set_num_threads(args.thread)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%1d' % args.cuda
        torch_device = torch.device('cuda:0')
    set_random_seed(args.seed, torch_device)
    # Environment Setup
    import env_lane
    env = env_lane.GymLane(
        dev=torch_device,
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
        highway_reward_stage=args.highway_reward_stage,
    )
    env.seed_base = args.seed
    input_dimension, output_dimension = env.state_dimension, env.action_num
    mem = memory_lib.MemoryBuffer(s_size=input_dimension, a_size=output_dimension, dev=torch_device)
    # Model Setup
    if args.model == 'mlp3soft':
        import model_mlp
        model = model_mlp.MLP_3(layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                                hid_activate='softmax', hid_group_size=args.hid_group_size,
                                out_activate='softmax',
                                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                                entropy_ratio=args.entropy,)
    elif args.model == 'mlp3relu':
        import model_mlp
        model = model_mlp.MLP_3(layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                                hid_activate='relu', hid_group_size=args.hid_group_size, 
                                out_activate='softmax',
                                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                                entropy_ratio=args.entropy, dev=torch_device)
    elif args.model == 'snnbptt':
        import model_snnbptt
        model = model_snnbptt.SNNBPTT3(
                layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                snn_num_steps = args.snn_num_steps,
                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy, dev=torch_device)
    elif args.model == 'rwtaprob':
        import model_rwta
        model = model_rwta.RWTAprob(input_size=input_dimension, output_size=output_dimension,
                                    hid_num=args.hid_group_num, hid_size=args.hid_group_size,
                                    remove_connection_pattern=args.rwta_del_connection,
                                    optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                                    entropy_ratio=args.entropy, device=torch_device)
        mem.init_for_rwta(q_size=model.dim_has, v_size=model.dim_ha)
    elif args.model == 'rwtaspk':
        import model_rwta
        model = model_rwta.RWTAspike(
                input_size=input_dimension, output_size=output_dimension,
                hid_num=args.hid_group_num, hid_size=args.hid_group_size,
                spk_response_window='uni', spk_full_time=42, spk_resp_time=args.response_window,         # special for spiking version
                remove_connection_pattern=args.rwta_del_connection,
                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy,
                device=torch_device)
        mem.init_for_rwta(q_size=model.dim_has, v_size=model.dim_ha)
    elif args.model == 'ann2snn':
        import model_convert
        model = model_convert.MLP_3(
                layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                hid_activate='relu', hid_group_size=args.hid_group_size,
                out_activate='softmax', optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                snn_num_steps = args.snn_num_steps,
                entropy_ratio=args.entropy, device=torch_device)
    import model_critic
    model_c = model_critic.Critic(
        input_size=input_dimension,
        output_size=output_dimension,
        dev=torch_device,
        small=True,
        optimizer_learning_rate=args.critic_lr,
    )
    if hasattr(model, 'set_grad_clip'):
        model.set_grad_clip(args.grad_clip)
    if hasattr(model_c, 'set_grad_clip'):
        model_c.set_grad_clip(args.grad_clip)
    set_model_learning_rates(model, model_c, args.lr, args.critic_lr)
    # Storage Folders
    checkpoint_utils.get_model_root(create=True, run_kind=run_kind)
    checkpoint_utils.get_log_root(create=True, run_kind=run_kind)
    model_current_save_time = time.time()
    log_text_flush_time = time.time()
    # Reload
    reload_data = True
    log_filename = os.path.join(active_log_dir, 'log_' + EXP_NAME + '.txt')
    if not os.path.exists(log_filename):
        reload_data = False
    if not os.path.exists(checkpoint_utils.resolve_checkpoint_file(EXP_NAME + '_current_1')):
        if not os.path.exists(checkpoint_utils.resolve_checkpoint_file(EXP_NAME + '_current_b_1')):
            reload_data = False
    if args.ignore_checkpoint == True:
        reload_data = False
    if args.model == 'ann2snn':
        reload_data = False
    if reload_data:
        (
            last_train_epi_num,
            last_val_best,
            last_val_best_collision,
            last_val_best_length,
            last_val_best_success,
            last_val_best_speed,
            last_val_best_lane_change,
            total_env_steps,
        ) = reload_log_file(log_filename)
        File = open(log_filename, 'a')
        log_text(File, 'resume', str(datetime.datetime.now()))
        model.load_model(EXP_NAME + '_current')
        model_c.load_model(EXP_NAME + 'critic' + '_current')
    else:           # Initialize training
        last_train_epi_num = 0
        last_val_best = -10000.0
        last_val_best_collision = float('inf')
        last_val_best_length = 0.0
        last_val_best_success = 0.0
        last_val_best_speed = 0.0
        last_val_best_lane_change = 0.0
        total_env_steps = 0
        File = open(log_filename, 'w')
        log_text(File, 'init', str(datetime.datetime.now()))
        log_text(File, 'arguments', str(args))
        log_text(File, 'seed', str(args.seed))
        if baseline_profile_adjustments:
            log_text(File, 'profile', '; '.join(baseline_profile_adjustments))
        log_text(File, 'model_dir', active_model_dir)
        log_text(File, 'log_dir', active_log_dir, onscreen=False)
        if args.model == 'ann2snn':
            model.load_model_ann(EXP_NAME + '_best')
    # Time Monitor
    calculation_time_monitor = TimeMonitor()
    use_short_horizon_replay = bool(args.road_scenario == 'highway' and args.alg == 'ppo')
    # >>>>  Main Loop
    mem.reset()         # memory buffer is shared across episodes
    train_step_num_total = 0
    next_validation_timestep = int(((max(0, total_env_steps) // val_interval_timesteps) + 1) * val_interval_timesteps)
    for train_epi_i in range((last_train_epi_num + 1), args.train_num):
        if args.model == 'ann2snn':
            break
        entropy_value = select_entropy_value(args, train_epi_i)
        maybe_update_entropy(model, entropy_value)
        episode_vehicle_count = select_lane_vehicle_count(args, train_epi_i)
        episode_seed = build_lane_episode_seed(args.seed, train_epi_i)
        env.init_train(vehicles_count=episode_vehicle_count, seed=episode_seed)
        observation = env.get_train_observation()
        for train_step_i in range(env.max_step_num):
            # Inference
            if args.monitor_time:
                start_time = time.time()
            model_output, model_other_output = model(observation)
            if args.monitor_time:
                calculation_time_monitor.record_time(rec_type=1, value=(time.time()-start_time))
            # Process Output
            if args.model in ['rwtaprob', 'rwtaspk']:
                action_chosen_onehot = model_other_output[0]
            else:
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_chosen_onehot = action_distribution.sample()
            observation_next, reward, _, step_info, step_record = env.make_action(action_chosen_onehot)
            action_executed_onehot, action_logprob = align_action_to_env_execution(
                model_output,
                action_chosen_onehot,
                step_info,
                env.action_num,
            )
            if args.model in ['rwtaprob', 'rwtaspk']:
                mem.add_transition(s1=observation, model_output=model_output,
                                   a=action_executed_onehot, a_log=action_logprob,
                                   r=reward, s2=observation_next, done=env.done_signal,
                                   q_has=model_other_output[2], v_ha=model_other_output[3])
            else:
                mem.add_transition(s1=observation, model_output=model_output.detach(),
                                   a=action_executed_onehot, a_log=action_logprob.detach(),
                                   r=reward, s2=observation_next, done=env.done_signal)
            total_env_steps += 1
            # >>>> Train
            train_step_num_total = (train_step_num_total + 1) % train_frequency     # every number of steps
            if train_step_num_total == 0:
                if args.model in ['rwtaprob', 'rwtaspk']:
                    s1, s2, model_output_1, a_1, a_logprob_1, r, done, q_has_1, v_ha_1 = mem.get_batch()
                else:
                    s1, s2, model_output_1, a_1, a_logprob_1, r, done = mem.get_batch()
                batch_size = s1.shape[0]
                with torch.no_grad():
                    s1_value = model_c(s1)
                    s2_value = model_c(s2)
                    next_action_prob = model(s2)[0]
                    state_values = torch.sum(model_output_1 * s1_value, dim=1)
                    next_state_values = torch.sum(next_action_prob * s2_value, dim=1)
                    advantage, returns = compute_gae(
                        r,
                        done,
                        state_values,
                        next_state_values,
                        args.gamma,
                        args.gae_lambda,
                        args.reward_scale,
                        bool(args.adv_norm),
                    )
                    state_value_target = s1_value.detach().clone()
                    a1_index = torch.argmax(a_1, dim=1)
                    state_value_target[torch.arange(batch_size), a1_index] = returns
                model_c.learn(model_c(s1), state_value_target)
                if args.alg == 'ppo':
                    old_logprob = torch.clone(a_logprob_1)
                    for _ in range(args.PPO_epochs):
                        model_output_ppo, model_other_output_ppo = model(s1)
                        if args.model in ['rwtaprob', 'rwtaspk']:
                            model_output_ppo.requires_grad_()
                            action_distribution = torch.distributions.OneHotCategorical(model_output_ppo)
                            action_logprob_ppo = action_distribution.log_prob(a_1)
                        else:
                            action_distribution = torch.distributions.OneHotCategorical(model_output_ppo)
                            action_logprob_ppo = action_distribution.log_prob(a_1)
                        action_entropy = action_distribution.entropy()
                        # Optimization
                        if args.monitor_time:
                            start_time2 = time.time()
                        if args.model in ['rwtaprob', 'rwtaspk']:
                            model.learn_ppo(action_logprob_ppo, old_logprob, advantage,
                                    args.eps_clip, action_entropy,
                                    old_vha=v_ha_1, old_qhas=q_has_1, model_output=model_output_ppo,
                                    current_other=model_other_output_ppo,)
                        else:
                            model.learn_ppo(action_logprob_ppo, old_logprob, advantage,
                                    args.eps_clip, action_entropy,)
                        if args.monitor_time:
                            calculation_time_monitor.record_time(rec_type=2, value=(time.time()-start_time2))
                else:           # 'reinforce'
                    if args.model in ['rwtaprob', 'rwtaspk']:
                        model_output_rei = torch.clone(model_output_1)
                        model_output_rei.requires_grad_()
                        action_distribution = torch.distributions.OneHotCategorical(model_output_rei)
                        action_logprob_rei = action_distribution.log_prob(a_1)
                    else:
                        model_output_rei, model_other_output_rei = model(s1)
                        action_distribution = torch.distributions.OneHotCategorical(model_output_rei)
                        action_logprob_rei = action_distribution.log_prob(a_1)
                    action_entropy = action_distribution.entropy()
                    if args.monitor_time:
                        start_time2 = time.time()
                    if args.model in ['rwtaprob', 'rwtaspk']:
                        model.learn_reinforce(action_logprob_rei, advantage, action_entropy,
                                              v_ha=v_ha_1, q_has=q_has_1,
                                              model_output=model_output_rei)
                    else:
                        model.learn_reinforce(action_logprob_rei, advantage, action_entropy,)
                    if args.monitor_time:
                        calculation_time_monitor.record_time(rec_type=2, value=(time.time()-start_time2))
                if use_short_horizon_replay:
                    mem.reset()
            # Episode End
            if env.done_signal == True:
                break
            observation = observation_next
        # Checkpoint
        if time.time() - model_current_save_time > 10:
            model.save_model(EXP_NAME + '_current')
            model_c.save_model(EXP_NAME + 'critic' + '_current')
            model_current_save_time = time.time()
        log_text(
            File,
            'train',
            format_train_episode_metrics(
                train_epi_i,
                env,
                episode_vehicle_count,
                entropy_value,
            ),
            onscreen=False,
        )
        log_text(
            File,
            'train_t',
            format_train_episode_metrics(
                train_epi_i,
                env,
                episode_vehicle_count,
                entropy_value,
                total_env_steps=total_env_steps,
            ),
            onscreen=False,
        )
        # Validation
        while total_env_steps >= next_validation_timestep:
            val_vehicle_count = None
            if args.road_scenario == 'highway':
                val_vehicle_count = select_lane_vehicle_count(args, train_epi_i)
            val_metrics = evaluate_lane_policy(env, model, val_num, vehicles_count=val_vehicle_count)
            better_model = is_better_lane_checkpoint(
                val_metrics,
                last_val_best,
                last_val_best_collision,
                last_val_best_length,
                last_val_best_success,
                last_val_best_speed,
                last_val_best_lane_change,
            )
            if better_model:
                model.save_model(EXP_NAME + '_best')
                model_c.save_model(EXP_NAME + 'critic' + '_best')
                last_val_best = val_metrics['mean_return']
                last_val_best_collision = val_metrics['collision_rate']
                last_val_best_length = val_metrics['mean_length']
                last_val_best_success = val_metrics['success_rate']
                last_val_best_speed = val_metrics['mean_speed']
                last_val_best_lane_change = val_metrics['mean_lane_change']
                log_text(
                    File,
                    'val_save',
                    format_validation_metrics(val_metrics, train_epi_i),
                )
                log_text(
                    File,
                    'val_save_t',
                    format_validation_metrics(val_metrics, train_epi_i, total_env_steps=total_env_steps),
                    onscreen=False,
                )
            log_text(
                File,
                'val',
                format_validation_metrics(val_metrics, train_epi_i),
            )
            log_text(
                File,
                'val_t',
                format_validation_metrics(val_metrics, train_epi_i, total_env_steps=total_env_steps),
                onscreen=False,
            )
            if (
                not better_model and
                should_restore_best_checkpoint(
                    val_metrics,
                    last_val_best_length,
                    last_val_best_collision,
                )
            ):
                best_actor_suffix = '_best_w_1' if args.model in ['rwtaprob', 'rwtaspk'] else '_best_1'
                best_actor_file = checkpoint_utils.resolve_checkpoint_file(EXP_NAME + best_actor_suffix)
                best_critic_file = checkpoint_utils.resolve_checkpoint_file(EXP_NAME + 'critic_best_1')
                if os.path.exists(best_actor_file) and os.path.exists(best_critic_file):
                    model.load_model(EXP_NAME + '_best')
                    model_c.load_model(EXP_NAME + 'critic_best')
                    log_text(
                        File,
                        'val_restore',
                        '%d, best_length %8.4f, current_length %8.4f, collision %6.4f' % (
                            train_epi_i,
                            last_val_best_length,
                            val_metrics['mean_length'],
                            val_metrics['collision_rate'],
                        ),
                    )
            next_validation_timestep += val_interval_timesteps


    # ANN2SNN Implementation
    if args.model == 'ann2snn':
        # Collect data
        print('ANN2SNN -> collect data for conversion')
        while True:
            if model.model_collect_full is True:
                break
            env.init_train()
            for train_step_i in range(env.max_step_num):
                observation = env.get_train_observation()
                model_output = model.ANN_model.get_prediction(observation)
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_chosen_onehot = action_distribution.sample()
                action_logprob = action_distribution.log_prob(action_chosen_onehot)
                observation_next, reward, _, _, step_record = env.make_action(action_chosen_onehot)
                model.add_s_list(observation)
                if env.done_signal is True:
                    break
        model.convert_model()


    model.save_model(EXP_NAME + '_current')
    model_c.save_model(EXP_NAME + 'critic' + '_current')
    log_text(
        File,
        'summary',
        'total_env_steps,%d,val_interval_timesteps,%d,val_episodes,%d' % (
            int(total_env_steps),
            int(val_interval_timesteps),
            int(val_num),
        ),
        onscreen=False,
    )
    if args.skip_post_tests:
        cleanup_summary = checkpoint_utils.cleanup_final_best_checkpoints(
            actor_best_prefix=EXP_NAME + '_best',
            actor_current_prefix=EXP_NAME + '_current',
            critic_best_prefix=EXP_NAME + 'critic_best',
            critic_current_prefix=EXP_NAME + 'critic_current',
        )
        log_text(File, 'checkpoint_cleanup', checkpoint_utils.summarize_checkpoint_cleanup(cleanup_summary), onscreen=False)
        log_text(File, 'finish', str(datetime.datetime.now()))
        File.flush()
        File.close()
        raise SystemExit(0)


    # ~~~~~~~~~~~~~~~~~~~~TEST~~~~~~~~~~~~~~~~~~~~~~~~~~
    log_text(File, 'test', str(datetime.datetime.now()))

    # >>>> Test Adversarial
    import model_adversarial
    ad_mem_size = 1000
    ad_train_epi_num = 100
    ad_mem_s = torch.zeros([ad_mem_size, input_dimension]).to(torch_device)
    ad_mem_a = torch.zeros([ad_mem_size, output_dimension]).to(torch_device)
    ad_model = model_adversarial.Adversarial(input_dimension, output_dimension)
    ad_model = ad_model.to(torch_device)
    pointer, total_num = 0, 0
    model.load_model(EXP_NAME + '_best')
    for collect_epi_i in range(ad_train_epi_num):
        env.init_train()
        for collect_step_i in range(env.max_step_num):
            observation = env.get_train_observation()
            model_output, model_other_output = model(observation)
            action_chosen_index = torch.argmax(model_output, dim=1)
            action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
            env.make_action(action_chosen_onehot)
            for sample_i in range(observation.shape[0]):
                ad_mem_s[pointer, :] = observation[sample_i, :]
                ad_mem_a[pointer, :] = action_chosen_onehot[sample_i, :]
                pointer = (pointer + 1) % ad_mem_size
                total_num = min(ad_mem_size, total_num + 1)
            if env.done_signal == True:
                break
        sample_list = random.sample(range(0, total_num), min(total_num, 200))
        ad_s_batch = ad_mem_s[sample_list]
        ad_a_batch = ad_mem_a[sample_list]
        ad_predict = ad_model(ad_s_batch)
        ad_loss = ad_model.learn(ad_s_batch, ad_predict, ad_a_batch)
        log_text(File, 'ADtrain', '%8d,   %8.6f' % (collect_epi_i, ad_loss), onscreen=False)
    perturb_loss = nn.CrossEntropyLoss()
    model.load_model(EXP_NAME + '_best')
    for epsilon in np.arange(0, 0.2, 0.01):
        test_preformance_list = []
        for test_epi_i in range(test_num):
            env.init_test()
            for test_step_i in range(env.max_step_num):
                observation = env.get_test_observation()
                observation.requires_grad = True            # for FGSM
                ad_output = ad_model(observation)
                ad_argmax = torch.argmax(ad_output, dim=1)
                perturb_loss_value = perturb_loss(ad_output, ad_argmax).mean()
                ad_model.zero_grad()
                perturb_loss_value.backward()
                observation_grad = observation.grad.data
                sign_grad = observation_grad.sign()
                observation_perturb = observation + epsilon * sign_grad
    
                model_output, model_other_output = model(observation_perturb)
                action_chosen_index = torch.argmax(model_output, dim=1)
                action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                _, _, _, _, test_other_step_record = env.make_action(action_chosen_onehot)
                if env.done_signal == True:
                    break
            test_preformance_list.append(test_other_step_record[2])
        test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
        log_text(File, 'FGSM', '%8.6f,   %8.6f' % (epsilon, test_performance_mean))
        File.flush()
    
    
    # >>>> Test Weight ABS
    noise_type_list = ['gaussian', 'uniform']
    for noise_type in noise_type_list:
        # Noise Parameters
        if noise_type in ['gaussian']:
            noise_param_list = np.arange(0, 1.0, 0.02)
        else:    # if noise_type in ['uniform']:
            noise_param_list = np.arange(0, 4.0, 0.05)
        for noise_param in noise_param_list:
            test_preformance_list = []
            for test_epi_i in range(test_num):
                model.load_model(EXP_NAME + '_best')
                model.add_noise_abs(noise_type, noise_param)
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation()
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    _, _, _, _, test_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[2])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'w_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()


    # >>>> Test Weight REL
    noise_type_list = ['gaussian', 'uniform']
    for noise_type in noise_type_list:
        # Noise Parameters
        if noise_type in ['gaussian']:
            noise_param_list = np.arange(0, 5, 0.1)
        else:    # if noise_type in ['uniform']:
            noise_param_list = np.arange(0, 5, 0.1)
        for noise_param in noise_param_list:
            test_preformance_list = []
            for test_epi_i in range(test_num):
                model.load_model(EXP_NAME + '_best')
                model.add_noise_relative(noise_type, noise_param)
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation()
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    _, _, _, _, test_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[2])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'w_rel_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()

    # >>>> Test Input
    model.load_model(EXP_NAME + '_best')
    noise_type_list = ['gaussian', 'pepper', 'salt', 's&p', 'gaussian&salt']
    for noise_type in noise_type_list:
        if noise_type in ['gaussian']:
            noise_param_list = np.arange(0, 1.6, 0.05)
        if noise_type in ['pepper', 'salt', 's&p']:
            noise_param_list = np.arange(0, 0.51, 0.02)
        if noise_type in ['gaussian&salt']:
            noise_param_list = np.arange(0, 0.505, 0.005)
        for noise_param in noise_param_list:
            test_preformance_list = []
            for test_epi_i in range(test_num):
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation(noise_type=noise_type, noise_param=noise_param)
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    _, _, _, _, test_other_step_record = env.make_action(action_chosen_onehot)

                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[2])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'i_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()

    # >>>> Test RWTA Connection Knock Out
    if args.model in ['rwtaprob', 'rwtaspk']:
        noise_type_list = ['hh', 'sh', 'sa', 'ha', 'all']
        noise_param_list = np.arange(0, 0.5, 0.02)
        for noise_type in noise_type_list:
            for noise_param in noise_param_list:
                test_preformance_list = []
                for test_epi_i in range(test_num):
                    model.load_model(EXP_NAME + '_best')
                    if noise_type == 'all':
                        model.random_remove_weight('sh', noise_param)
                        model.random_remove_weight('sa', noise_param)
                        model.random_remove_weight('hh', noise_param)
                        model.random_remove_weight('ha', noise_param)
                    else:
                        model.random_remove_weight(noise_type, noise_param)
                    env.init_test()
                    for test_step_i in range(env.max_step_num):
                        observation = env.get_test_observation()
                        model_output, model_other_output = model(observation)
                        action_chosen_index = torch.argmax(model_output, dim=1)
                        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                        _, _, _, _, test_other_step_record = env.make_action(action_chosen_onehot)
                        if env.done_signal == True:
                            break
                    test_preformance_list.append(test_other_step_record[2])
                test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
                log_text(File, 'RWTA_conn', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
                File.flush()

    cleanup_summary = checkpoint_utils.cleanup_final_best_checkpoints(
        actor_best_prefix=EXP_NAME + '_best',
        actor_current_prefix=EXP_NAME + '_current',
        critic_best_prefix=EXP_NAME + 'critic_best',
        critic_current_prefix=EXP_NAME + 'critic_current',
    )
    log_text(File, 'checkpoint_cleanup', checkpoint_utils.summarize_checkpoint_cleanup(cleanup_summary), onscreen=False)
    File.flush()
    File.close()
