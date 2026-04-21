# -*- coding: utf-8 -*-
import argparse
import datetime
import multiprocessing as mp
import os
import random
import re
import time


def _sanitize_omp_threads():
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads is None:
        return
    try:
        if int(omp_threads) > 0:
            return
    except (TypeError, ValueError):
        pass
    os.environ["OMP_NUM_THREADS"] = "1"


_sanitize_omp_threads()

import numpy as np
import torch

import checkpoint_utils
from ppo_stability import STABLE_HIGHWAY_STANDARD_PPO, scheduled_entropy


def get_arguments():
    parser = argparse.ArgumentParser(description='Description: run_RL_ours')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--entropy', type=float, default=0.10)
    parser.add_argument('--PPO_epochs', type=int, default=5)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--alg', type=str, default='ppo', choices=['ppo', 'reinforce'])

    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--thread', type=int, default=-1)
    parser.add_argument('--train_num', type=int, default=500)
    parser.add_argument('--train_envs', type=int, default=4)
    parser.add_argument('--val_interval_timesteps', type=int, default=0)
    parser.add_argument('--val_episodes', type=int, default=20)
    parser.add_argument('--eval_episodes', type=int, default=30)
    parser.add_argument('--final_eval_episodes', type=int, default=30)
    parser.add_argument('--rep', type=int, default=11)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--ignore_checkpoint', default=False, action='store_true')
    parser.add_argument('--monitor_time', default=False, action='store_true')
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--eval_checkpoint', type=str, default='best', choices=['best', 'current'])

    parser.add_argument('--task', type=str, default='gymip', choices=['gymip'])
    parser.add_argument('--model', type=str, default='rwtaprob', choices=['mlp3soft', 'mlp3relu', 'rwtaprob', 'rwtaspk', 'snnbptt', 'ann2snn'])
    parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--gymip_train_xml', type=str, default='inverted_pendulum_ChangeThk_0.050000.xml')
    parser.add_argument('--hidden_num', type=int, default=64)
    parser.add_argument('--hid_group_num', type=int, default=8)
    parser.add_argument('--hid_group_size', type=int, default=8)
    parser.add_argument('--rwta_del_connection', type=str, default='none', choices=['none', 'hh', 'sa', 'hhsa', 'ha', 'sh'])
    parser.add_argument('--response_window', type=int, default=40)
    parser.add_argument('--snn_num_steps', type=int, default=15)

    parser.add_argument('--rollout_steps', type=int, default=256)
    parser.add_argument('--mini_batch_size', type=int, default=64)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--adv_norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--critic_lr', type=float, default=0.0)
    parser.add_argument('--curriculum_mode', type=str, default='fixed', choices=['fixed', 'adaptive'])
    parser.add_argument('--curriculum_clean_ratio', type=float, default=0.7)
    parser.add_argument('--max_noise', type=float, default=0.15)
    parser.add_argument('--entropy_min', type=float, default=0.1)
    parser.add_argument('--entropy_decay', type=float, default=1.0)
    parser.add_argument('--entropy_warmup_scale', type=float, default=1.10)
    parser.add_argument('--entropy_warmup_episodes', type=int, default=50)
    parser.add_argument('--curriculum_patience', type=int, default=2)
    parser.add_argument('--curriculum_noise_step', type=float, default=0.03)
    parser.add_argument('--curriculum_entropy_decay', type=float, default=0.9)
    parser.add_argument('--lane_profile', type=str, default='auto', choices=['auto', 'legacy'])
    parser.add_argument('--road_scenario', type=str, default='highway', choices=['highway', 'merge', 'roundabout'])
    parser.add_argument('--traffic_level', type=str, default='standard', choices=['light', 'standard', 'dense'])
    parser.add_argument('--highway_reward_stage', type=str, default='c_success', choices=['baseline', 'a_timeout', 'b_progress_speed', 'c_success'])
    parser.add_argument('--warm_start_kind', type=str, default='none', choices=['none', 'baseline', 'ours', 'any'])
    parser.add_argument('--warm_start_prefix', type=str, default='')
    return parser.parse_args()


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
                if len(parts) > numeric_offset + 3:
                    val_best_lane_change = float(parts[numeric_offset + 3])
                if len(parts) > numeric_offset + 5:
                    val_best_speed = float(parts[numeric_offset + 5])
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
    file_handle.write((type_str + ',').ljust(10) + record_text + '\n')
    if time.time() - log_text_flush_time > 10:
        log_text_flush_time = time.time()
        file_handle.flush()
        os.fsync(file_handle.fileno())


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
                    np.max(self.time_inference),
                ))
        if rec_type == 2:
            self.time_optimize[self.time_pointer_optimize] = value * 1000
            self.time_pointer_optimize = (self.time_pointer_optimize + 1) % self.size
            if self.time_pointer_optimize == 0:
                print('timer opt: %7.3f %7.3f %7.3f %7.3f' % (
                    float(np.mean(self.time_optimize)),
                    float(np.std(self.time_optimize)),
                    np.min(self.time_optimize),
                    np.max(self.time_optimize),
                ))


class RolloutBuffer:
    def __init__(self, for_rwta=False):
        self.for_rwta = for_rwta
        self.reset()

    def reset(self):
        self.s1 = []
        self.s2 = []
        self.model_output = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.done = []
        self.q_has = []
        self.v_ha = []

    def add_transition(self, s1, s2, model_output, a, a_logprob, reward, done, q_has=None, v_ha=None):
        self.s1.append(s1.detach().clone())
        self.s2.append(s2.detach().clone())
        self.model_output.append(model_output.detach().clone())
        self.a.append(a.detach().clone().float())
        self.a_logprob.append(a_logprob.detach().clone())
        self.r.append(float(reward.detach().item()))
        self.done.append(float(done))
        if self.for_rwta:
            self.q_has.append(q_has.detach().clone())
            self.v_ha.append(v_ha.detach().clone())

    def size(self):
        return len(self.r)

    def stack(self, device):
        if self.size() == 0:
            return None
        data = {
            's1': torch.cat(self.s1, dim=0).to(device),
            's2': torch.cat(self.s2, dim=0).to(device),
            'model_output': torch.cat(self.model_output, dim=0).to(device),
            'a': torch.cat(self.a, dim=0).to(device),
            'a_logprob': torch.stack(self.a_logprob).to(device),
            'r': torch.tensor(self.r, dtype=torch.float32, device=device),
            'done': torch.tensor(self.done, dtype=torch.float32, device=device),
        }
        if self.for_rwta:
            data['q_has'] = torch.cat(self.q_has, dim=0).to(device)
            data['v_ha'] = torch.cat(self.v_ha, dim=0).to(device)
        return data



def build_lane_episode_seed(base_seed, episode_index, slot_index=0):
    modulus = 2147483647
    composite = (int(base_seed) + 1) * 1000003 + int(episode_index) * 1009 + int(slot_index) * 97
    return int(composite % modulus)


def stack_lane_observation_batch(observation_list, device):
    if not observation_list:
        return None
    return torch.as_tensor(np.stack(observation_list, axis=0), dtype=torch.float32, device=device)


def apply_lane_batch_observation_noise(observation_batch, noise_levels):
    if observation_batch is None:
        return None
    if not noise_levels or max(float(level) for level in noise_levels) <= 0.0:
        return observation_batch
    noise_tensor = torch.as_tensor(noise_levels, dtype=torch.float32, device=observation_batch.device).unsqueeze(1)
    return torch.clamp(observation_batch + torch.randn_like(observation_batch) * noise_tensor, 0.0, 1.0)


def lane_parallel_worker(connection, road_scenario, traffic_level, highway_reward_stage):
    import env_lane
    worker_device = torch.device('cpu')
    env = env_lane.GymLane(
        dev=worker_device,
        road_scenario=road_scenario,
        traffic_level=traffic_level,
        highway_reward_stage=highway_reward_stage,
    )
    try:
        while True:
            message = connection.recv()
            command = message.get('cmd')
            if command == 'reset':
                env.init_train(
                    vehicles_count=int(message['vehicles_count']),
                    seed=message.get('episode_seed'),
                )
                observation = env.get_observation().squeeze(0).cpu().numpy().astype(np.float32, copy=True)
                connection.send({'observation': observation})
            elif command == 'step':
                action_index = int(message['action_index'])
                action_onehot = torch.nn.functional.one_hot(
                    torch.tensor([action_index], dtype=torch.long),
                    num_classes=env.action_num,
                ).float()
                next_state, reward, done, info, _step_record = env.make_action(action_onehot)
                observation = next_state.squeeze(0).cpu().numpy().astype(np.float32, copy=True)
                episode_summary = None
                if done:
                    episode_summary = env.get_episode_summary()
                    episode_summary.update({
                        'episode_return': float(env.episode_return),
                        'step_num': int(env.step_num),
                        'collision_flag': float(env.collision_count > 0),
                        'lane_change_count': int(env.lane_change_count),
                    })
                connection.send({
                    'observation': observation,
                    'reward': float(reward.item()),
                    'done': bool(done),
                    'executed_action': int(info.get('executed_action', action_index)) if isinstance(info, dict) else int(action_index),
                    'episode_summary': episode_summary,
                })
            elif command == 'close':
                break
            else:
                raise ValueError(f'Unsupported worker command: {command}')
    except EOFError:
        pass
    finally:
        env.close()
        connection.close()


class ParallelLaneCollector:
    def __init__(self, num_envs, road_scenario, traffic_level, highway_reward_stage):
        self.num_envs = max(1, int(num_envs))
        self.ctx = mp.get_context('spawn')
        self.parents = []
        self.processes = []
        for _ in range(self.num_envs):
            parent_conn, child_conn = self.ctx.Pipe()
            process = self.ctx.Process(
                target=lane_parallel_worker,
                args=(child_conn, road_scenario, traffic_level, highway_reward_stage),
            )
            process.daemon = True
            process.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.processes.append(process)

    def reset(self, slot_index, vehicles_count, episode_seed=None):
        self.parents[slot_index].send({
            'cmd': 'reset',
            'vehicles_count': int(vehicles_count),
            'episode_seed': episode_seed,
        })
        return self.parents[slot_index].recv()

    def step(self, slot_action_map):
        for slot_index, action_index in slot_action_map.items():
            self.parents[slot_index].send({
                'cmd': 'step',
                'action_index': int(action_index),
            })
        results = {}
        for slot_index in slot_action_map.keys():
            results[slot_index] = self.parents[slot_index].recv()
        return results

    def close(self):
        for parent in self.parents:
            try:
                parent.send({'cmd': 'close'})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for parent in self.parents:
            try:
                parent.close()
            except OSError:
                pass
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def maybe_apply_rollout_update(file_handle, train_epi_i, model, model_c, rollout, args, calculation_time_monitor, update_count):
    if rollout.size() < args.rollout_steps:
        return update_count
    update_stats = update_policy(model, model_c, rollout, args, calculation_time_monitor)
    rollout.reset()
    update_count += 1
    log_text(
        file_handle,
        'update',
        '%d, %4d, %4d, %8.6f, %8.6f' % (
            train_epi_i,
            update_count,
            update_stats['rollout_size'],
            update_stats['mean_return_target'],
            update_stats['mean_advantage'],
        ),
        onscreen=False,
    )
    return update_count


def maybe_save_current_models(model, model_c, exp_name, model_current_save_time):
    if time.time() - model_current_save_time > 10:
        model.save_model(exp_name + '_current')
        model_c.save_model(exp_name + 'critic_current')
        return time.time()
    return model_current_save_time


def set_model_learning_rates(model, model_c, actor_lr, critic_lr=None):
    actor_lr = float(actor_lr)
    critic_lr = float(actor_lr if critic_lr is None else critic_lr)
    if hasattr(model, 'optimizer_learning_rate'):
        model.optimizer_learning_rate = actor_lr
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = actor_lr
    critic_optimizer = getattr(model_c, 'optimizer', None)
    if critic_optimizer is not None:
        for param_group in critic_optimizer.param_groups:
            param_group['lr'] = critic_lr
    return actor_lr, critic_lr


def apply_curriculum_optimizer_state(model, model_c, curriculum_state, args):
    actor_lr = float(curriculum_state.get('actor_lr', args.lr))
    critic_lr = float(curriculum_state.get('critic_lr', actor_lr))
    return set_model_learning_rates(model, model_c, actor_lr, critic_lr)


def seed_curriculum_from_resume(args, curriculum_state, best_length, best_collision, best_success):
    if getattr(args, 'road_scenario', None) != 'highway':
        return None
    if getattr(args, 'traffic_level', 'standard') != 'standard':
        return None

    best_length = float(best_length)
    best_collision = float(best_collision)
    best_success = float(best_success)
    if best_length < 70.0 and best_collision > 0.45 and best_success <= 0.0:
        return None

    old_entropy = float(curriculum_state.get('entropy', args.entropy))
    old_actor_lr = float(curriculum_state.get('actor_lr', args.lr))
    old_critic_lr = float(curriculum_state.get('critic_lr', args.critic_lr))

    curriculum_state['best_mean_length'] = max(curriculum_state.get('best_mean_length', 0.0), best_length)
    curriculum_state['best_collision'] = min(curriculum_state.get('best_collision', 1.0), best_collision)
    curriculum_state['stability_phase'] = True
    curriculum_state['reanchor_pending'] = False
    curriculum_state['noise_cap'] = 0.0
    curriculum_state['improvement_streak'] = 0
    curriculum_state['entropy'] = max(args.entropy_min, min(old_entropy, 0.03))
    curriculum_state['actor_lr'] = max(6.0e-5, min(old_actor_lr, args.lr * 0.45))
    curriculum_state['critic_lr'] = max(8.0e-5, min(old_critic_lr, args.critic_lr * 0.65))

    return (
        'resume_stability best_length %.4f, best_collision %.4f, '
        'entropy %.4f -> %.4f, actor_lr %.6f -> %.6f, critic_lr %.6f -> %.6f'
    ) % (
        best_length,
        best_collision,
        old_entropy,
        curriculum_state['entropy'],
        old_actor_lr,
        curriculum_state['actor_lr'],
        old_critic_lr,
        curriculum_state['critic_lr'],
    )


def run_validation_cycle(
    file_handle,
    train_epi_i,
    total_env_steps,
    env,
    model,
    model_c,
    args,
    val_num,
    curriculum_state,
    exp_name,
    rollout,
    last_val_best,
    last_val_best_collision,
    last_val_best_length,
    last_val_best_success,
    last_val_best_speed,
    last_val_best_lane_change,
    model_current_save_time,
):
    val_vehicle_count = None
    if args.road_scenario == 'highway':
        val_vehicle_count = select_lane_vehicle_count(
            args,
            train_epi_i,
            curriculum_state=curriculum_state,
            allow_stage_mix=False,
        )
    val_metrics = evaluate_policy(
        env,
        model,
        val_num,
        mode='val',
        vehicles_count=val_vehicle_count,
    )
    dense_train_val_metrics = None
    if args.road_scenario == 'highway' and args.traffic_level == 'dense':
        dense_train_vehicle_count = select_lane_vehicle_count(
            args,
            train_epi_i,
            curriculum_state=curriculum_state,
            allow_stage_mix=False,
        )
        dense_train_val_metrics = evaluate_policy(
            env,
            model,
            max(4, min(6, val_num)),
            mode='val',
            vehicles_count=dense_train_vehicle_count,
        )
    better_model = is_better_lane_checkpoint(
        val_metrics,
        last_val_best,
        last_val_best_collision,
        last_val_best_length,
        last_val_best_success,
        last_val_best_speed,
        last_val_best_lane_change,
        traffic_level=args.traffic_level,
    )
    if better_model:
        model.save_model(exp_name + '_best')
        model_c.save_model(exp_name + 'critic_best')
        last_val_best = val_metrics['mean_return']
        last_val_best_collision = val_metrics['collision_rate']
        last_val_best_length = val_metrics['mean_length']
        last_val_best_success = val_metrics['success_rate']
        last_val_best_speed = val_metrics['mean_speed']
        last_val_best_lane_change = val_metrics['mean_lane_change']
        log_text(
            file_handle,
            'val_save',
            format_validation_metrics(val_metrics, train_epi_i),
        )
        log_text(
            file_handle,
            'val_save_t',
            format_validation_metrics(val_metrics, train_epi_i, total_env_steps=total_env_steps),
            onscreen=False,
        )
    log_text(
        file_handle,
        'val',
        format_validation_metrics(val_metrics, train_epi_i),
    )
    log_text(
        file_handle,
        'val_t',
        format_validation_metrics(val_metrics, train_epi_i, total_env_steps=total_env_steps),
        onscreen=False,
    )
    if dense_train_val_metrics is not None:
        log_text(
            file_handle,
            'traffic_val',
            '%d, %2d, %s' % (
                train_epi_i,
                dense_train_vehicle_count,
                format_validation_metrics(dense_train_val_metrics, train_epi_i),
            ),
            onscreen=False,
        )
        traffic_message = update_dense_highway_vehicle_curriculum(
            curriculum_state,
            dense_train_val_metrics,
            full_val_metrics=val_metrics,
        )
        if traffic_message is not None:
            log_text(file_handle, 'traffic_curriculum', traffic_message)
    if args.road_scenario == 'highway':
        curriculum_signal = (
            120.0 * val_metrics['success_rate']
            - 50.0 * val_metrics['collision_rate']
            + val_metrics['mean_speed']
            + 0.55 * val_metrics['mean_length']
            + 0.05 * val_metrics['mean_return']
        )
        if args.traffic_level == 'dense':
            curriculum_signal += 0.25 * val_metrics['mean_length']
    else:
        curriculum_signal = (
            100.0 * val_metrics['success_rate']
            - 40.0 * val_metrics['collision_rate']
            + val_metrics['mean_speed']
            + 0.1 * val_metrics['mean_return']
        )
    curriculum_message = update_curriculum(args, curriculum_state, curriculum_signal, val_metrics=val_metrics)
    if curriculum_message is not None:
        log_text(file_handle, 'curriculum', curriculum_message)
    apply_curriculum_optimizer_state(model, model_c, curriculum_state, args)
    if (
        args.road_scenario == 'highway' and
        args.traffic_level in ['standard', 'dense'] and
        curriculum_state.get('reanchor_pending', False)
    ):
        best_actor_file = checkpoint_utils.resolve_checkpoint_file(exp_name + '_best_w_1')
        best_critic_file = checkpoint_utils.resolve_checkpoint_file(exp_name + 'critic_best_1')
        if os.path.exists(best_actor_file) and os.path.exists(best_critic_file):
            model.load_model(exp_name + '_best')
            model_c.load_model(exp_name + 'critic_best')
            model.save_model(exp_name + '_current')
            model_c.save_model(exp_name + 'critic_current')
            model_current_save_time = time.time()
            rollout.reset()
            curriculum_state['reanchor_pending'] = False
            curriculum_state['noise_cap'] = 0.0
            if args.traffic_level == 'standard':
                curriculum_state['entropy'] = max(args.entropy_min, min(curriculum_state['entropy'], 0.03))
                curriculum_state['actor_lr'] = max(6.0e-5, min(curriculum_state.get('actor_lr', args.lr), args.lr * 0.45))
                curriculum_state['critic_lr'] = max(8.0e-5, min(curriculum_state.get('critic_lr', args.critic_lr), args.critic_lr * 0.65))
            else:
                curriculum_state['entropy'] = max(args.entropy_min, min(curriculum_state['entropy'], 0.12))
            apply_curriculum_optimizer_state(model, model_c, curriculum_state, args)
            log_text(
                file_handle,
                'reanchor',
                '%d, best_length %8.4f, current_length %8.4f, collision %6.4f' % (
                    train_epi_i,
                    last_val_best_length,
                    val_metrics['mean_length'],
                    val_metrics['collision_rate'],
                ),
            )
    return (
        last_val_best,
        last_val_best_collision,
        last_val_best_length,
        last_val_best_success,
        last_val_best_speed,
        last_val_best_lane_change,
        model_current_save_time,
    )


def set_random_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def build_experiment_name(args, model_str, seed):
    train_env_suffix = ''
    if int(getattr(args, 'train_envs', 1)) > 1:
        train_env_suffix = f'_te{int(args.train_envs)}'
    reward_stage_suffix = ''
    if getattr(args, 'road_scenario', None) == 'highway':
        reward_stage_suffix = f'_hrs{getattr(args, "highway_reward_stage", "c_success")}'
    entropy_warmup_suffix = ''
    if (
        float(getattr(args, 'entropy_warmup_scale', 1.0)) > 1.0 + 1e-8 and
        int(getattr(args, 'entropy_warmup_episodes', 0)) > 0
    ):
        entropy_warmup_suffix = (
            f'_ew{float(args.entropy_warmup_scale):.2f}'
            f'e{int(args.entropy_warmup_episodes)}'
        )
    return (
        f'{args.alg}_{args.task}_{args.model}_{model_str}_{args.optimizer}_'
        f'{args.lr:.6f}_{args.entropy:.2f}_{args.gamma:.5f}_{args.PPO_epochs}_{args.eps_clip:.4f}_'
        f'ro{args.rollout_steps}_mb{args.mini_batch_size}_lam{args.gae_lambda:.2f}_'
        f'rs{args.reward_scale:.2f}_gc{args.grad_clip:.2f}_{args.curriculum_mode}_'
        f'road{args.road_scenario}_tf{args.traffic_level}{reward_stage_suffix}{entropy_warmup_suffix}'
        f'{train_env_suffix}_seed{seed:02d}'
    )


def actor_prefix_to_critic_prefix(actor_prefix):
    actor_prefix = checkpoint_utils.normalize_prefix(actor_prefix)
    if actor_prefix.endswith('_best'):
        return actor_prefix[:-len('_best')] + 'critic_best'
    if actor_prefix.endswith('_current'):
        return actor_prefix[:-len('_current')] + 'critic_current'
    raise ValueError(f'Unsupported warm-start actor prefix: {actor_prefix}')


def resolve_warm_start_prefix(args):
    if args.warm_start_prefix:
        return checkpoint_utils.normalize_prefix(args.warm_start_prefix)
    if args.warm_start_kind == 'none':
        return None
    checkpoint_kind = None if args.warm_start_kind == 'any' else args.warm_start_kind
    return checkpoint_utils.find_latest_checkpoint_prefix(
        kind=checkpoint_kind,
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
        best_only=True,
    )


def maybe_load_warm_start(args, model, model_c):
    actor_prefix = resolve_warm_start_prefix(args)
    if actor_prefix is None:
        return None, None
    critic_prefix = actor_prefix_to_critic_prefix(actor_prefix)
    model.load_model(actor_prefix)
    model_c.load_model(critic_prefix)
    return actor_prefix, critic_prefix


HIGHWAY_STANDARD_PROFILE = {
    **STABLE_HIGHWAY_STANDARD_PPO,
    'train_envs': 4,
    'rollout_steps': 512,
    'mini_batch_size': 128,
    'curriculum_mode': 'adaptive',
    'max_noise': 0.0,
}

PROFILE_OPTIONAL_EXPLORATION_KEYS = {
    'entropy_warmup_scale',
    'entropy_warmup_episodes',
}


def apply_lane_stability_profile(args):
    if args.task != 'gymip' or args.model != 'rwtaspk' or args.lane_profile != 'auto':
        return []
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

    traffic_level = getattr(args, 'traffic_level', 'standard')
    if args.road_scenario == 'highway' and traffic_level == 'standard':
        # Keep the standard-highway profile explicit so it is not re-overridden
        # by multiple clamp passes below.
        for attr_name, target_value in HIGHWAY_STANDARD_PROFILE.items():
            if attr_name in PROFILE_OPTIONAL_EXPLORATION_KEYS:
                continue
            set_exact(attr_name, target_value)
        clamp_max('train_num', 2000)
        clamp_max('reward_scale', 0.90)
        set_exact('curriculum_clean_ratio', 1.0)
        set_exact('curriculum_noise_step', 0.0)
        set_exact('curriculum_patience', 1)
        set_exact('curriculum_entropy_decay', 0.97)
        return adjustments

    clamp_max('lr', 3.0e-4)
    clamp_min('gamma', 0.995)
    clamp_max('entropy', 0.03)
    clamp_max('PPO_epochs', 4)
    clamp_max('eps_clip', 0.20)
    clamp_min('rollout_steps', 512)
    clamp_min('mini_batch_size', 128)
    clamp_min('gae_lambda', 0.95)
    clamp_max('grad_clip', 0.5)
    clamp_min('curriculum_clean_ratio', 0.8)
    clamp_max('max_noise', 0.08)
    clamp_max('curriculum_noise_step', 0.02)

    if args.road_scenario == 'highway':
        lr_cap = {
            'light': 2.5e-4,
            'dense': 2.0e-4,
        }
        gamma_floor = {
            'light': 0.996,
            'dense': 0.998,
        }
        entropy_cap = {
            'light': 0.025,
            'dense': 0.020,
        }
        entropy_min_cap = {
            'light': 0.005,
            'dense': 0.004,
        }
        rollout_floor = {
            'light': 640,
            'dense': 896,
        }
        mini_batch_floor = {
            'light': 160,
            'dense': 224,
        }
        clean_ratio_floor = {
            'light': 0.86,
            'dense': 0.99,
        }
        max_noise_cap = {
            'light': 0.05,
            'dense': 0.008,
        }
        noise_step_cap = {
            'light': 0.010,
            'dense': 0.001,
        }
        train_num_floor = {
            'light': 2800,
        }

        if traffic_level == 'dense':
            clamp_max('lr', 2.0e-4)
        else:
            clamp_max('lr', lr_cap.get(traffic_level, 2.5e-4))
        clamp_min('gamma', gamma_floor.get(traffic_level, 0.996))
        clamp_max('entropy', entropy_cap.get(traffic_level, 0.02))
        clamp_max('entropy_min', entropy_min_cap.get(traffic_level, 0.005))
        clamp_max('PPO_epochs', 4)
        clamp_max('eps_clip', 0.20)
        clamp_min('rollout_steps', rollout_floor.get(traffic_level, 768))
        clamp_min('mini_batch_size', mini_batch_floor.get(traffic_level, 192))
        if traffic_level == 'dense':
            clamp_min('gae_lambda', 0.95)
        else:
            clamp_min('gae_lambda', 0.95)
        clamp_max('reward_scale', 0.80)
        clamp_max('grad_clip', 0.50)
        clamp_min('curriculum_clean_ratio', clean_ratio_floor.get(traffic_level, 0.90))
        clamp_max('max_noise', max_noise_cap.get(traffic_level, 0.04))
        clamp_max('curriculum_noise_step', noise_step_cap.get(traffic_level, 0.008))
        if args.curriculum_patience < 3:
            args.curriculum_patience = 3
            adjustments.append('curriculum_patience->3')
        if args.curriculum_entropy_decay < 0.97:
            args.curriculum_entropy_decay = 0.97
            adjustments.append('curriculum_entropy_decay->0.97')
        if traffic_level == 'dense' and args.curriculum_patience < 4:
            args.curriculum_patience = 4
            adjustments.append('curriculum_patience->4')
        if traffic_level == 'dense' and args.curriculum_entropy_decay < 0.98:
            args.curriculum_entropy_decay = 0.98
            adjustments.append('curriculum_entropy_decay->0.98')
        if traffic_level == 'dense':
            clamp_max('train_num', 2000)
        elif args.train_num < train_num_floor.get(traffic_level, 3600):
            args.train_num = train_num_floor.get(traffic_level, 3600)
            adjustments.append(f'train_num->{args.train_num}')
        if args.road_scenario == 'highway' and traffic_level == 'dense' and args.train_envs < 8:
            args.train_envs = 8
            adjustments.append('train_envs->8')

    if args.road_scenario == 'merge':
        clamp_max('train_num', 2000)
        clamp_max('entropy', 0.05)
        clamp_max('entropy_min', 0.05)
        clamp_min('curriculum_clean_ratio', 1.0)
        clamp_max('max_noise', 0.0)
        clamp_max('curriculum_noise_step', 0.0)
        if args.curriculum_patience > 1:
            args.curriculum_patience = 1
            adjustments.append('curriculum_patience->1')
        if args.curriculum_entropy_decay > 0.85:
            args.curriculum_entropy_decay = 0.85
            adjustments.append('curriculum_entropy_decay->0.85')
        # if args.train_num < 3000:
        #     args.train_num = 3000
        #     adjustments.append('train_num->3000')
    if args.road_scenario == 'roundabout':
        clamp_max('train_num', 2000)
    return adjustments


DENSE_HIGHWAY_TRAIN_VEHICLE_STAGES = [40, 48, 56, 64, 72]


def _sample_dense_highway_vehicle_count(curriculum_state):
    stages = curriculum_state.get('train_vehicle_stages')
    if not stages:
        return int(curriculum_state.get('train_vehicle_count', 40))

    stage_index = int(curriculum_state.get('train_vehicle_stage_index', 0))
    stage_index = max(0, min(stage_index, len(stages) - 1))
    current_vehicle_count = int(stages[stage_index])
    if stage_index >= len(stages) - 1:
        return current_vehicle_count

    next_prob = float(curriculum_state.get('train_vehicle_mix_next_prob', 0.0))
    full_prob = float(curriculum_state.get('train_vehicle_mix_full_prob', 0.0))
    next_prob = max(0.0, min(0.45, next_prob))
    full_prob = max(0.0, min(0.20, full_prob))
    next_vehicle_count = int(stages[min(stage_index + 1, len(stages) - 1)])
    full_vehicle_count = int(stages[-1])

    if next_vehicle_count == full_vehicle_count:
        next_prob = min(0.60, next_prob + full_prob)
        full_prob = 0.0

    draw = random.random()
    if full_prob > 0.0 and draw < full_prob:
        return full_vehicle_count
    if draw < full_prob + next_prob:
        return next_vehicle_count
    return current_vehicle_count


def select_lane_vehicle_count(args, train_epi_i, curriculum_state=None, allow_stage_mix=True):
    if args.task != 'gymip':
        return 40
    if args.model != 'rwtaspk' or args.lane_profile != 'auto':
        return 40
    if getattr(args, 'road_scenario', None) == 'highway':
        traffic_target = {
            'light': 28,
            'standard': 10,
            'dense': 72,
        }
        traffic_start = {
            'light': 16,
            'standard': 10,
            'dense': 48,
        }
        curriculum_span = {
            'light': 120,
            'standard': 1,
            'dense': 520,
        }
        traffic_level = getattr(args, 'traffic_level', 'standard')
        if traffic_level == 'dense' and curriculum_state is not None:
            staged_vehicle_count = curriculum_state.get('train_vehicle_count')
            if staged_vehicle_count is not None:
                if allow_stage_mix:
                    return _sample_dense_highway_vehicle_count(curriculum_state)
                return int(staged_vehicle_count)
        start_vehicle_num = traffic_start.get(traffic_level, 32)
        end_vehicle_num = traffic_target.get(traffic_level, 50)
        span = curriculum_span.get(traffic_level, 320)
        progress = min(1.0, train_epi_i / max(1, span))
        return int(round(start_vehicle_num + (end_vehicle_num - start_vehicle_num) * progress))

    scenario_target = {
        'merge': {
            'light': 6,
            'standard': 8,
            'dense': 10,
        },
        'roundabout': {
            'light': 8,
            'standard': 10,
            'dense': 12,
        },
    }
    scenario_name = getattr(args, 'road_scenario', 'merge')
    traffic_level = getattr(args, 'traffic_level', 'standard')
    return int(scenario_target.get(scenario_name, scenario_target['merge']).get(traffic_level, 8))


def init_curriculum(args):
    initial_curriculum_entropy = scheduled_entropy(
        initial_entropy=args.entropy,
        entropy_min=args.entropy_min,
        entropy_decay=args.entropy_decay,
        episode_index=0,
        entropy_warmup_scale=getattr(args, 'entropy_warmup_scale', 1.0),
        entropy_warmup_episodes=getattr(args, 'entropy_warmup_episodes', 0),
    )
    curriculum_state = {
        'noise_cap': 0.0,
        'entropy': initial_curriculum_entropy,
        'best_val_return': -10000.0,
        'improvement_streak': 0,
        'actor_lr': args.lr,
        'critic_lr': args.critic_lr,
    }
    if getattr(args, 'road_scenario', None) == 'highway' and getattr(args, 'traffic_level', 'standard') == 'standard':
        curriculum_state.update({
            'best_mean_length': 0.0,
            'best_collision': 1.0,
            'stability_phase': False,
            'reanchor_pending': False,
        })
    if getattr(args, 'road_scenario', None) == 'highway' and getattr(args, 'traffic_level', 'standard') == 'dense':
        curriculum_state.update({
            'best_mean_length': 0.0,
            'best_collision': 1.0,
            'stability_phase': False,
            'reanchor_pending': False,
            'train_vehicle_stages': list(DENSE_HIGHWAY_TRAIN_VEHICLE_STAGES),
            'train_vehicle_stage_index': 0,
            'train_vehicle_count': int(DENSE_HIGHWAY_TRAIN_VEHICLE_STAGES[0]),
            'train_vehicle_promote_streak': 0,
            'train_vehicle_collapse_streak': 0,
            'train_vehicle_mix_next_prob': 0.0,
            'train_vehicle_mix_full_prob': 0.0,
        })
    return curriculum_state


def select_curriculum_settings(args, curriculum_state, train_epi_i):
    scheduled_entropy_value = scheduled_entropy(
        initial_entropy=args.entropy,
        entropy_min=args.entropy_min,
        entropy_decay=args.entropy_decay,
        episode_index=train_epi_i,
        entropy_warmup_scale=getattr(args, 'entropy_warmup_scale', 1.0),
        entropy_warmup_episodes=getattr(args, 'entropy_warmup_episodes', 0),
    )
    if args.curriculum_mode == 'fixed':
        progress = train_epi_i / max(1, args.train_num - 1)
        noise_cap = min(args.max_noise, args.max_noise * progress)
        entropy_value = scheduled_entropy_value
    else:
        noise_cap = curriculum_state['noise_cap']
        entropy_value = scheduled_entropy(
            initial_entropy=args.entropy,
            entropy_min=args.entropy_min,
            entropy_decay=args.entropy_decay,
            episode_index=train_epi_i,
            adaptive_entropy=curriculum_state['entropy'],
            entropy_warmup_scale=getattr(args, 'entropy_warmup_scale', 1.0),
            entropy_warmup_episodes=getattr(args, 'entropy_warmup_episodes', 0),
        )
    use_noise = noise_cap > 0 and random.random() > args.curriculum_clean_ratio
    episode_noise = noise_cap if use_noise else 0.0
    return entropy_value, episode_noise, noise_cap


def update_dense_highway_vehicle_curriculum(curriculum_state, train_metrics, full_val_metrics=None):
    stages = curriculum_state.get('train_vehicle_stages')
    if not stages:
        return None

    stage_index = int(curriculum_state.get('train_vehicle_stage_index', 0))
    stage_index = max(0, min(stage_index, len(stages) - 1))
    current_vehicle_count = int(stages[stage_index])
    mean_length = float(train_metrics.get('mean_length', 0.0))
    collision_rate = float(train_metrics.get('collision_rate', 1.0))
    success_rate = float(train_metrics.get('success_rate', 0.0))
    full_dense_length = float(full_val_metrics.get('mean_length', 0.0)) if full_val_metrics is not None else 0.0
    full_dense_collision = float(full_val_metrics.get('collision_rate', 1.0)) if full_val_metrics is not None else 1.0
    previous_next_prob = round(float(curriculum_state.get('train_vehicle_mix_next_prob', 0.0)), 2)
    previous_full_prob = round(float(curriculum_state.get('train_vehicle_mix_full_prob', 0.0)), 2)

    # If full dense validation is already strong, sync the staged curriculum upward
    # so training no longer lingers on easier traffic densities.
    full_dense_target_stage = None
    if full_val_metrics is not None:
        if full_dense_length >= 145.0 and full_dense_collision <= 0.10:
            full_dense_target_stage = len(stages) - 1
        elif stage_index < len(stages) - 2 and full_dense_length >= 115.0 and full_dense_collision <= 0.28:
            full_dense_target_stage = len(stages) - 2
    if full_dense_target_stage is not None and full_dense_target_stage > stage_index:
        next_vehicle_count = int(stages[full_dense_target_stage])
        curriculum_state['train_vehicle_stage_index'] = int(full_dense_target_stage)
        curriculum_state['train_vehicle_count'] = int(next_vehicle_count)
        curriculum_state['train_vehicle_promote_streak'] = 0
        curriculum_state['train_vehicle_collapse_streak'] = 0
        curriculum_state['train_vehicle_mix_next_prob'] = 0.0
        curriculum_state['train_vehicle_mix_full_prob'] = 0.0
        return (
            'dense_traffic_sync %d -> %d, full_length %.1f, full_collision %.3f'
        ) % (
            current_vehicle_count,
            next_vehicle_count,
            full_dense_length,
            full_dense_collision,
        )

    next_mix_prob = 0.0
    full_mix_prob = 0.0
    if stage_index < len(stages) - 1:
        light_bridge = mean_length >= 95.0 and collision_rate <= 0.72
        medium_bridge = mean_length >= 120.0 and collision_rate <= 0.58
        strong_bridge = (
            success_rate >= 0.18 or
            (mean_length >= 145.0 and collision_rate <= 0.45)
        )
        if strong_bridge:
            next_mix_prob = 0.30
            full_mix_prob = 0.08 if (
                full_dense_length >= 80.0 or
                stage_index >= 1 or
                full_dense_collision <= 0.70
            ) else 0.04
        elif medium_bridge:
            next_mix_prob = 0.20
            if full_dense_length >= 80.0 or stage_index >= 2:
                full_mix_prob = 0.04
            elif stage_index >= 1:
                full_mix_prob = 0.02
        elif light_bridge:
            next_mix_prob = 0.10
            if stage_index >= 1 and (full_dense_length >= 60.0 or full_dense_collision <= 0.80):
                full_mix_prob = 0.02

    curriculum_state['train_vehicle_mix_next_prob'] = float(next_mix_prob)
    curriculum_state['train_vehicle_mix_full_prob'] = float(full_mix_prob)

    strong_ready = (
        success_rate >= 0.55 or
        (mean_length >= 180.0 and collision_rate <= 0.25) or
        (mean_length >= 172.0 and collision_rate <= 0.10)
    )
    ready = mean_length >= 165.0 and collision_rate <= 0.35
    collapse = stage_index > 0 and mean_length < 55.0 and collision_rate >= 0.85

    if stage_index == len(stages) - 2 and full_val_metrics is not None:
        ready = ready and (full_dense_length >= 75.0 or full_dense_collision <= 0.55)
        strong_ready = strong_ready and (full_dense_length >= 105.0 or full_dense_collision <= 0.35)

    if strong_ready:
        promote_streak = 2
    elif ready:
        promote_streak = int(curriculum_state.get('train_vehicle_promote_streak', 0)) + 1
    else:
        promote_streak = 0

    if collapse:
        collapse_streak = int(curriculum_state.get('train_vehicle_collapse_streak', 0)) + 1
    else:
        collapse_streak = 0

    curriculum_state['train_vehicle_promote_streak'] = int(promote_streak)
    curriculum_state['train_vehicle_collapse_streak'] = int(collapse_streak)

    if stage_index < len(stages) - 1 and promote_streak >= 2:
        next_vehicle_count = int(stages[stage_index + 1])
        curriculum_state['train_vehicle_stage_index'] = int(stage_index + 1)
        curriculum_state['train_vehicle_count'] = int(next_vehicle_count)
        curriculum_state['train_vehicle_promote_streak'] = 0
        curriculum_state['train_vehicle_collapse_streak'] = 0
        curriculum_state['train_vehicle_mix_next_prob'] = 0.0
        curriculum_state['train_vehicle_mix_full_prob'] = 0.0
        return (
            'dense_traffic_up %d -> %d, stage_length %.1f, stage_collision %.3f, '
            'full_length %.1f, full_collision %.3f'
        ) % (
            current_vehicle_count,
            next_vehicle_count,
            mean_length,
            collision_rate,
            full_dense_length,
            full_dense_collision,
        )

    if stage_index > 0 and collapse_streak >= 2:
        previous_vehicle_count = int(stages[stage_index - 1])
        curriculum_state['train_vehicle_stage_index'] = int(stage_index - 1)
        curriculum_state['train_vehicle_count'] = int(previous_vehicle_count)
        curriculum_state['train_vehicle_promote_streak'] = 0
        curriculum_state['train_vehicle_collapse_streak'] = 0
        curriculum_state['train_vehicle_mix_next_prob'] = 0.0
        curriculum_state['train_vehicle_mix_full_prob'] = 0.0
        return (
            'dense_traffic_down %d -> %d, stage_length %.1f, stage_collision %.3f'
        ) % (
            current_vehicle_count,
            previous_vehicle_count,
            mean_length,
            collision_rate,
        )

    mix_tuple = (round(float(next_mix_prob), 2), round(float(full_mix_prob), 2))
    if mix_tuple != (previous_next_prob, previous_full_prob):
        next_vehicle_count = int(stages[min(stage_index + 1, len(stages) - 1)])
        if next_vehicle_count == int(stages[-1]):
            return (
                'dense_traffic_mix keep %d @ %.2f, full %d @ %.2f'
            ) % (
                current_vehicle_count,
                max(0.0, 1.0 - next_mix_prob - full_mix_prob),
                int(stages[-1]),
                next_mix_prob + full_mix_prob,
            )
        return (
            'dense_traffic_mix keep %d @ %.2f, next %d @ %.2f, full %d @ %.2f'
        ) % (
            current_vehicle_count,
            max(0.0, 1.0 - next_mix_prob - full_mix_prob),
            next_vehicle_count,
            next_mix_prob,
            int(stages[-1]),
            full_mix_prob,
        )

    return None


def update_curriculum(args, curriculum_state, val_return, val_metrics=None):
    if args.curriculum_mode != 'adaptive':
        return None

    standard_highway = (
        getattr(args, 'road_scenario', None) == 'highway' and
        getattr(args, 'traffic_level', 'standard') == 'standard'
    )
    dense_highway = (
        getattr(args, 'road_scenario', None) == 'highway' and
        getattr(args, 'traffic_level', 'standard') == 'dense'
    )
    if standard_highway and val_metrics is not None:
        mean_length = float(val_metrics.get('mean_length', 0.0))
        collision_rate = float(val_metrics.get('collision_rate', 1.0))
        success_rate = float(val_metrics.get('success_rate', 0.0))
        current_entropy = float(curriculum_state.get('entropy', args.entropy))
        current_actor_lr = float(curriculum_state.get('actor_lr', args.lr))
        current_critic_lr = float(curriculum_state.get('critic_lr', args.critic_lr))
        curriculum_state['best_mean_length'] = max(curriculum_state.get('best_mean_length', 0.0), mean_length)
        curriculum_state['best_collision'] = min(curriculum_state.get('best_collision', 1.0), collision_rate)

        if (
            not curriculum_state.get('stability_phase', False) and
            (
                mean_length >= 70.0 or
                collision_rate <= 0.45 or
                success_rate >= 0.05
            )
        ):
            curriculum_state['stability_phase'] = True
            curriculum_state['noise_cap'] = 0.0
            curriculum_state['entropy'] = max(args.entropy_min, min(current_entropy, 0.035))
            curriculum_state['actor_lr'] = max(7.5e-5, min(current_actor_lr, args.lr * 0.55))
            curriculum_state['critic_lr'] = max(1.0e-4, min(current_critic_lr, args.critic_lr * 0.75))
            curriculum_state['improvement_streak'] = 0
            curriculum_state['reanchor_pending'] = False
            return (
                'standard_stability_on length %.4f, collision %.4f, '
                'entropy %.4f -> %.4f, actor_lr %.6f -> %.6f, critic_lr %.6f -> %.6f'
            ) % (
                mean_length,
                collision_rate,
                current_entropy,
                curriculum_state['entropy'],
                current_actor_lr,
                curriculum_state['actor_lr'],
                current_critic_lr,
                curriculum_state['critic_lr'],
            )

        if curriculum_state.get('stability_phase', False):
            old_entropy = float(curriculum_state['entropy'])
            old_actor_lr = float(curriculum_state['actor_lr'])
            old_critic_lr = float(curriculum_state['critic_lr'])
            curriculum_state['noise_cap'] = 0.0
            healthy = (
                mean_length >= max(48.0, 0.68 * curriculum_state['best_mean_length']) or
                collision_rate <= min(0.55, curriculum_state['best_collision'] + 0.18) or
                success_rate >= 0.05
            )
            if healthy:
                curriculum_state['entropy'] = max(args.entropy_min, curriculum_state['entropy'] * 0.90)
                curriculum_state['actor_lr'] = max(6.0e-5, curriculum_state['actor_lr'] * 0.92)
                curriculum_state['critic_lr'] = max(8.0e-5, curriculum_state['critic_lr'] * 0.94)
            else:
                curriculum_state['entropy'] = max(args.entropy_min, min(curriculum_state['entropy'], 0.03))
                curriculum_state['actor_lr'] = max(6.0e-5, min(curriculum_state['actor_lr'], args.lr * 0.45))
                curriculum_state['critic_lr'] = max(8.0e-5, min(curriculum_state['critic_lr'], args.critic_lr * 0.65))
            collapse = (
                curriculum_state['best_mean_length'] >= 80.0 and
                mean_length < max(22.0, 0.42 * curriculum_state['best_mean_length']) and
                collision_rate >= min(1.0, curriculum_state['best_collision'] + 0.30)
            )
            curriculum_state['reanchor_pending'] = bool(collapse)
            curriculum_state['improvement_streak'] = 0
            if collapse:
                return (
                    'standard_reanchor_pending best_length %.4f current_length %.4f collision %.4f'
                ) % (
                    curriculum_state['best_mean_length'],
                    mean_length,
                    collision_rate,
                )
            if (
                abs(old_entropy - curriculum_state['entropy']) > 1e-8 or
                abs(old_actor_lr - curriculum_state['actor_lr']) > 1e-8 or
                abs(old_critic_lr - curriculum_state['critic_lr']) > 1e-8
            ):
                return (
                    'standard_stability entropy %.4f -> %.4f, actor_lr %.6f -> %.6f, critic_lr %.6f -> %.6f'
                ) % (
                    old_entropy,
                    curriculum_state['entropy'],
                    old_actor_lr,
                    curriculum_state['actor_lr'],
                    old_critic_lr,
                    curriculum_state['critic_lr'],
                )
            return None

    if dense_highway and val_metrics is not None:
        mean_length = float(val_metrics.get('mean_length', 0.0))
        collision_rate = float(val_metrics.get('collision_rate', 1.0))
        success_rate = float(val_metrics.get('success_rate', 0.0))
        curriculum_state['best_mean_length'] = max(curriculum_state.get('best_mean_length', 0.0), mean_length)
        curriculum_state['best_collision'] = min(curriculum_state.get('best_collision', 1.0), collision_rate)

        if (
            not curriculum_state.get('stability_phase', False) and
            (curriculum_state['best_mean_length'] >= 95.0 or success_rate >= 0.10)
        ):
            old_noise = curriculum_state['noise_cap']
            old_entropy = curriculum_state['entropy']
            curriculum_state['stability_phase'] = True
            curriculum_state['noise_cap'] = 0.0
            curriculum_state['entropy'] = max(args.entropy_min, min(curriculum_state['entropy'], 0.12))
            curriculum_state['improvement_streak'] = 0
            curriculum_state['reanchor_pending'] = False
            return 'dense_stability_on noise_cap %.4f -> %.4f, entropy %.4f -> %.4f, best_length %.4f' % (
                old_noise,
                curriculum_state['noise_cap'],
                old_entropy,
                curriculum_state['entropy'],
                curriculum_state['best_mean_length'],
            )

        if curriculum_state.get('stability_phase', False):
            old_noise = curriculum_state['noise_cap']
            old_entropy = curriculum_state['entropy']
            curriculum_state['noise_cap'] = 0.0
            if mean_length >= max(80.0, 0.85 * curriculum_state['best_mean_length']) or success_rate >= 0.10:
                curriculum_state['entropy'] = max(args.entropy_min, curriculum_state['entropy'] * 0.96)
            else:
                curriculum_state['entropy'] = max(args.entropy_min, min(curriculum_state['entropy'], 0.12))
            collapse = (
                curriculum_state['best_mean_length'] >= 95.0 and
                mean_length < max(28.0, 0.45 * curriculum_state['best_mean_length']) and
                collision_rate >= min(1.0, curriculum_state['best_collision'] + 0.25)
            )
            curriculum_state['reanchor_pending'] = bool(collapse)
            curriculum_state['improvement_streak'] = 0
            if collapse:
                return 'dense_reanchor_pending best_length %.4f current_length %.4f collision %.4f' % (
                    curriculum_state['best_mean_length'],
                    mean_length,
                    collision_rate,
                )
            if abs(old_noise - curriculum_state['noise_cap']) > 1e-8 or abs(old_entropy - curriculum_state['entropy']) > 1e-8:
                return 'dense_stability noise_cap %.4f -> %.4f, entropy %.4f -> %.4f' % (
                    old_noise,
                    curriculum_state['noise_cap'],
                    old_entropy,
                    curriculum_state['entropy'],
                )
            return None

    if val_return > curriculum_state['best_val_return'] + 1e-6:
        curriculum_state['best_val_return'] = val_return
        curriculum_state['improvement_streak'] += 1
    else:
        curriculum_state['improvement_streak'] = 0
    if curriculum_state['improvement_streak'] < args.curriculum_patience:
        return None
    old_noise = curriculum_state['noise_cap']
    old_entropy = curriculum_state['entropy']
    curriculum_state['noise_cap'] = min(args.max_noise, curriculum_state['noise_cap'] + args.curriculum_noise_step)
    curriculum_state['entropy'] = max(args.entropy_min, curriculum_state['entropy'] * args.curriculum_entropy_decay)
    curriculum_state['improvement_streak'] = 0
    return 'noise_cap %.4f -> %.4f, entropy %.4f -> %.4f' % (
        old_noise,
        curriculum_state['noise_cap'],
        old_entropy,
        curriculum_state['entropy'],
    )


def maybe_update_entropy(model, entropy_value):
    if hasattr(model, 'update_entropy'):
        model.update_entropy(entropy_value)


def align_action_to_env_execution(model_output, sampled_action_onehot, info, action_num):
    executed_action_index = None
    if isinstance(info, dict):
        executed_action_index = info.get('executed_action')
    if executed_action_index is None:
        return sampled_action_onehot, torch.distributions.OneHotCategorical(model_output).log_prob(sampled_action_onehot)
    executed_action_index = int(executed_action_index)
    executed_action_onehot = torch.nn.functional.one_hot(
        torch.tensor([executed_action_index], device=model_output.device),
        num_classes=action_num,
    ).float()
    executed_action_logprob = torch.distributions.OneHotCategorical(model_output).log_prob(executed_action_onehot)
    return executed_action_onehot, executed_action_logprob


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


def evaluate_policy(env, model, episode_num, mode='val', vehicles_count=None):
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
        for _ in range(episode_num):
            if mode == 'val':
                env.init_val(vehicles_count=vehicles_count)
                observation = env.get_val_observation()
            else:
                env.init_test(record_video=False, vehicles_count=vehicles_count)
                observation = env.get_test_observation()
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


def format_train_episode_metrics(
    episode_index,
    total_env_steps,
    episode_return,
    step_num,
    collision_flag,
    lane_change_count,
    episode_noise,
    noise_cap,
    entropy_value,
    episode_vehicle_count,
    episode_summary,
):
    base_record = (
        '%d, %d, %8.6f, %4d, %4.2f, %4d, %5.3f, %5.3f, %5.3f, %2d' % (
            episode_index,
            int(total_env_steps),
            episode_return,
            step_num,
            collision_flag,
            lane_change_count,
            episode_noise,
            noise_cap,
            entropy_value,
            episode_vehicle_count,
        )
    )
    extra_record = (
        ', success %4.2f, timeout %4.2f, progress %5.3f, speed %6.3f, term %s'
    ) % (
        float(episode_summary.get('success_rate', 0.0)),
        float(episode_summary.get('timeout_rate', 0.0)),
        float(episode_summary.get('final_progress', episode_summary.get('route_progress', 0.0))),
        float(episode_summary.get('mean_speed', 0.0)),
        episode_summary.get('termination_reason', 'unknown'),
    )
    return base_record + extra_record


def lane_validation_quality(metrics, traffic_level='standard'):
    dense_mode = traffic_level == 'dense'
    excessive_lane_threshold = 4.5 if dense_mode else 4.0
    excessive_lane_changes = max(0.0, float(metrics['mean_lane_change']) - excessive_lane_threshold)
    length_weight = 1.05 if dense_mode else 0.45
    collision_penalty = 38.0 if dense_mode else 30.0
    short_length_penalty = 0.0
    if dense_mode:
        short_length_penalty = 0.80 * max(0.0, 55.0 - float(metrics.get('mean_length', 0.0)))
    return (
        float(metrics['mean_return'])
        + length_weight * float(metrics.get('mean_length', 0.0))
        + 1.5 * float(metrics['mean_speed'])
        - collision_penalty * float(metrics['collision_rate'])
        - 2.5 * excessive_lane_changes
        - short_length_penalty
    )


def is_better_lane_checkpoint(
    val_metrics,
    best_return,
    best_collision,
    best_length,
    best_success,
    best_speed,
    best_lane_change,
    traffic_level='standard',
):
    if val_metrics['success_rate'] > best_success + 1e-6:
        return True
    if (
        abs(val_metrics['success_rate'] - best_success) <= 1e-6 and
        val_metrics['collision_rate'] <= best_collision + 0.02 and
        val_metrics['mean_length'] > best_length + 1.0
    ):
        return True
    if (
        traffic_level == 'dense' and
        best_success <= 1e-6 and
        val_metrics['success_rate'] <= 1e-6 and
        val_metrics['collision_rate'] <= best_collision + 0.05 and
        val_metrics['mean_length'] > best_length + 6.0
    ):
        return True
    if best_success <= 1e-6 and val_metrics['success_rate'] <= 1e-6:
        candidate_quality = lane_validation_quality(val_metrics, traffic_level=traffic_level)
        best_quality = lane_validation_quality({
            'mean_return': best_return,
            'mean_length': best_length,
            'mean_speed': best_speed,
            'collision_rate': best_collision,
            'mean_lane_change': best_lane_change,
        }, traffic_level=traffic_level)
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


def update_policy(model, model_c, rollout, args, calculation_time_monitor=None):
    if rollout.size() == 0:
        return None
    torch.set_grad_enabled(True)
    if hasattr(model, 'weight') and not model.weight.requires_grad:
        model.weight.requires_grad = True
    if hasattr(model, 'bias') and not model.bias.requires_grad:
        model.bias.requires_grad = True
    critic_device = next(model_c.parameters()).device
    rollout_data = rollout.stack(critic_device)
    s1 = rollout_data['s1']
    s2 = rollout_data['s2']
    old_model_output = rollout_data['model_output']
    actions = rollout_data['a']
    old_logprob = rollout_data['a_logprob']
    rewards = rollout_data['r']
    dones = rollout_data['done']
    batch_size = rewards.shape[0]

    with torch.no_grad():
        s1_value = model_c(s1)
        s2_value = model_c(s2)
        next_action_prob = model(s2)[0]
        state_values = torch.sum(old_model_output * s1_value, dim=1)
        next_state_values = torch.sum(next_action_prob * s2_value, dim=1)
        advantages, returns = compute_gae(
            rewards,
            dones,
            state_values,
            next_state_values,
            args.gamma,
            args.gae_lambda,
            args.reward_scale,
            bool(args.adv_norm),
        )
        critic_target = s1_value.detach().clone()
        action_index = torch.argmax(actions, dim=1)
        critic_target[torch.arange(batch_size, device=s1.device), action_index] = returns

    mini_batch_size = min(args.mini_batch_size, batch_size)
    for _epoch_i in range(args.PPO_epochs if args.alg == 'ppo' else 1):
        permutation = torch.randperm(batch_size, device=s1.device)
        for start_index in range(0, batch_size, mini_batch_size):
            batch_index = permutation[start_index:start_index + mini_batch_size]
            critic_prediction = model_c(s1[batch_index])
            model_c.learn(critic_prediction, critic_target[batch_index])

            if args.monitor_time and calculation_time_monitor is not None:
                start_time2 = time.time()
            model_output_batch, model_other_output_batch = model(s1[batch_index])
            action_distribution = torch.distributions.OneHotCategorical(model_output_batch)
            action_logprob_batch = action_distribution.log_prob(actions[batch_index])
            action_entropy = action_distribution.entropy()
            if args.model in ['rwtaprob', 'rwtaspk']:
                model_output_batch.requires_grad_()
                if args.alg == 'ppo':
                    model.learn_ppo(
                        action_logprob_batch,
                        old_logprob[batch_index],
                        advantages[batch_index],
                        args.eps_clip,
                        action_entropy,
                        old_vha=rollout_data['v_ha'][batch_index],
                        old_qhas=rollout_data['q_has'][batch_index],
                        model_output=model_output_batch,
                        current_other=model_other_output_batch,
                    )
                else:
                    model.learn_reinforce(
                        action_logprob_batch,
                        advantages[batch_index],
                        action_entropy,
                        v_ha=rollout_data['v_ha'][batch_index],
                        q_has=rollout_data['q_has'][batch_index],
                        model_output=model_output_batch,
                    )
            else:
                if args.alg == 'ppo':
                    model.learn_ppo(
                        action_logprob_batch,
                        old_logprob[batch_index],
                        advantages[batch_index],
                        args.eps_clip,
                        action_entropy,
                    )
                else:
                    model.learn_reinforce(action_logprob_batch, advantages[batch_index], action_entropy)
            if args.monitor_time and calculation_time_monitor is not None:
                calculation_time_monitor.record_time(rec_type=2, value=(time.time() - start_time2))
    return {
        'rollout_size': batch_size,
        'mean_return_target': float(returns.mean().item()),
        'mean_advantage': float(advantages.mean().item()),
    }


if __name__ == '__main__':
    args = get_arguments()
    if args.model in ['mlp3soft', 'mlp3relu']:
        model_str = 'h%d_-' % args.hidden_num
    elif args.model in ['snnbptt']:
        model_str = 'h%d_%d' % (args.hidden_num, args.snn_num_steps)
    elif args.model in ['rwtaprob']:
        model_str = 'h%d-%d_%s' % (args.hid_group_num, args.hid_group_size, args.rwta_del_connection)
    elif args.model in ['rwtaspk']:
        model_str = 'h%d-%d-%d_%s' % (args.hid_group_num, args.hid_group_size, args.response_window, args.rwta_del_connection)
    elif args.model in ['ann2snn']:
        model_str = 'h%d_-' % args.hidden_num
    else:
        raise ValueError('Error in arguments')

    lane_profile_adjustments = apply_lane_stability_profile(args)
    if args.critic_lr <= 0:
        args.critic_lr = args.lr
    if args.entropy_decay <= 0:
        args.entropy_decay = 1.0
    val_interval_timesteps, val_num = resolve_validation_schedule(args)

    if args.cuda < 0:
        torch_device = torch.device('cpu')
        if args.thread != -1:
            torch.set_num_threads(args.thread)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%1d' % args.cuda
        torch_device = torch.device('cuda:0')

    seed = args.seed if args.seed >= 0 else args.rep
    set_random_seed(seed, torch_device)
    run_kind = 'ours'
    EXP_NAME = build_experiment_name(args, model_str, seed)
    active_model_dir, active_log_dir = checkpoint_utils.activate_scenario_output_dirs(
        run_kind=run_kind,
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
        create=True,
    )

    if args.task == 'gymip':
        import env_lane
        env = env_lane.GymLane(
            dev=torch_device,
            road_scenario=args.road_scenario,
            traffic_level=args.traffic_level,
            highway_reward_stage=args.highway_reward_stage,
        )
        input_dimension, output_dimension = env.state_dimension, env.action_num
    else:
        raise ValueError('Only gymip/LANE is supported in this runner.')

    if args.model == 'mlp3soft':
        import model_mlp
        model = model_mlp.MLP_3(
            layer_sizes=[input_dimension, args.hidden_num, output_dimension],
            hid_activate='softmax',
            hid_group_size=args.hid_group_size,
            out_activate='softmax',
            optimizer_name=args.optimizer,
            optimizer_learning_rate=args.lr,
            entropy_ratio=args.entropy,
            dev=torch_device,
        )
    elif args.model == 'mlp3relu':
        import model_mlp
        model = model_mlp.MLP_3(
            layer_sizes=[input_dimension, args.hidden_num, output_dimension],
            hid_activate='relu',
            hid_group_size=args.hid_group_size,
            out_activate='softmax',
            optimizer_name=args.optimizer,
            optimizer_learning_rate=args.lr,
            entropy_ratio=args.entropy,
            dev=torch_device,
        )
    elif args.model == 'snnbptt':
        import model_snnbptt
        model = model_snnbptt.SNNBPTT3(
            layer_sizes=[input_dimension, args.hidden_num, output_dimension],
            snn_num_steps=args.snn_num_steps,
            optimizer_name=args.optimizer,
            optimizer_learning_rate=args.lr,
            entropy_ratio=args.entropy,
            dev=torch_device,
        )
    elif args.model == 'rwtaprob':
        import model_rwta
        model = model_rwta.RWTAprob(
            input_size=input_dimension,
            output_size=output_dimension,
            hid_num=args.hid_group_num,
            hid_size=args.hid_group_size,
            remove_connection_pattern=args.rwta_del_connection,
            optimizer_name=args.optimizer,
            optimizer_learning_rate=args.lr,
            entropy_ratio=args.entropy,
            device=torch_device,
        )
    elif args.model == 'rwtaspk':
        import model_rwta
        model = model_rwta.RWTAspike(
            input_size=input_dimension,
            output_size=output_dimension,
            hid_num=args.hid_group_num,
            hid_size=args.hid_group_size,
            spk_response_window='uni',
            spk_full_time=42,
            spk_resp_time=args.response_window,
            remove_connection_pattern=args.rwta_del_connection,
            optimizer_name=args.optimizer,
            optimizer_learning_rate=args.lr,
            entropy_ratio=args.entropy,
            device=torch_device,
        )
    elif args.model == 'ann2snn':
        raise SystemExit('ANN2SNN is not supported in the improved LANE runner yet.')
    else:
        raise ValueError('Error in model name.')

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

    checkpoint_utils.get_model_root(create=True, run_kind=run_kind)
    checkpoint_utils.get_log_root(create=True, run_kind=run_kind)

    model_current_save_time = time.time()
    log_text_flush_time = time.time()
    globals()['log_text_flush_time'] = log_text_flush_time

    if args.eval_only:
        eval_episode_index = max(0, args.train_num - 1)
        eval_vehicle_count = None
        if args.road_scenario == 'highway':
            eval_vehicle_count = select_lane_vehicle_count(
                args,
                eval_episode_index,
                curriculum_state=init_curriculum(args),
                allow_stage_mix=False,
            )
        actor_eval_prefix = None
        critic_eval_prefix = None
        if args.warm_start_prefix:
            actor_eval_prefix = checkpoint_utils.normalize_prefix(args.warm_start_prefix)
            critic_eval_prefix = actor_prefix_to_critic_prefix(actor_eval_prefix)
        elif args.eval_checkpoint == 'best':
            actor_eval_prefix = EXP_NAME + '_best'
            critic_eval_prefix = EXP_NAME + 'critic_best'
        else:
            actor_eval_prefix = EXP_NAME + '_current'
            critic_eval_prefix = EXP_NAME + 'critic_current'
        model.load_model(actor_eval_prefix)
        model_c.load_model(critic_eval_prefix)
        eval_metrics = evaluate_policy(
            env,
            model,
            max(1, int(args.eval_episodes)),
            mode='val',
            vehicles_count=eval_vehicle_count,
        )
        eval_log_filename = os.path.join(active_log_dir, 'eval_' + EXP_NAME + '.txt')
        EvalFile = open(eval_log_filename, 'a')
        log_text(EvalFile, 'eval_init', str(datetime.datetime.now()))
        log_text(EvalFile, 'eval_args', str(args), onscreen=False)
        log_text(EvalFile, 'eval_checkpoint', actor_eval_prefix)
        log_text(EvalFile, 'eval', format_validation_metrics(eval_metrics, eval_episode_index))
        log_text(EvalFile, 'eval_t', format_validation_metrics(eval_metrics, eval_episode_index, total_env_steps=0), onscreen=False)
        log_text(EvalFile, 'finish', str(datetime.datetime.now()))
        EvalFile.flush()
        EvalFile.close()
        raise SystemExit(0)

    log_filename = os.path.join(active_log_dir, 'log_' + EXP_NAME + '.txt')
    reload_data = os.path.exists(log_filename)
    if args.model in ['rwtaprob', 'rwtaspk']:
        reload_data = reload_data and os.path.exists(checkpoint_utils.resolve_checkpoint_file(EXP_NAME + '_current_b_1'))
    else:
        reload_data = reload_data and os.path.exists(checkpoint_utils.resolve_checkpoint_file(EXP_NAME + '_current_1'))
    if args.ignore_checkpoint:
        reload_data = False

    last_train_epi_num, last_val_best, last_val_best_collision, last_val_best_length, last_val_best_success, last_val_best_speed, last_val_best_lane_change, total_env_steps = 0, -10000.0, float('inf'), 0.0, 0.0, 0.0, 0.0, 0
    warm_start_actor_prefix, warm_start_critic_prefix = None, None
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
        resume_from_best = (
            args.road_scenario == 'highway' and
            getattr(args, 'traffic_level', 'standard') == 'standard' and
            last_val_best_length >= 100.0 and
            last_val_best_collision <= 0.10 and
            os.path.exists(checkpoint_utils.resolve_checkpoint_file(EXP_NAME + '_best_w_1')) and
            os.path.exists(checkpoint_utils.resolve_checkpoint_file(EXP_NAME + 'critic_best_1'))
        )
        if resume_from_best:
            model.load_model(EXP_NAME + '_best')
            model_c.load_model(EXP_NAME + 'critic_best')
            log_text(File, 'resume_model', 'best')
        else:
            model.load_model(EXP_NAME + '_current')
            model_c.load_model(EXP_NAME + 'critic_current')
            log_text(File, 'resume_model', 'current')
    else:
        File = open(log_filename, 'w')
        log_text(File, 'init', str(datetime.datetime.now()))
        log_text(File, 'arguments', str(args))
        log_text(File, 'seed', str(seed))
        log_text(File, 'model_dir', active_model_dir)
        log_text(File, 'log_dir', active_log_dir, onscreen=False)
        if lane_profile_adjustments:
            log_text(File, 'profile', 'lane auto-stabilize: ' + ', '.join(lane_profile_adjustments))
        if args.warm_start_kind != 'none' or args.warm_start_prefix:
            try:
                warm_start_actor_prefix, warm_start_critic_prefix = maybe_load_warm_start(args, model, model_c)
            except (FileNotFoundError, ValueError) as exc:
                File.flush()
                File.close()
                raise SystemExit(f'Warm-start failed: {exc}')
            if warm_start_actor_prefix is not None:
                log_text(File, 'warm_start', warm_start_actor_prefix)
                log_text(File, 'warm_critic', warm_start_critic_prefix, onscreen=False)


    calculation_time_monitor = TimeMonitor()
    curriculum_state = init_curriculum(args)
    if reload_data:
        resume_curriculum_message = seed_curriculum_from_resume(
            args,
            curriculum_state,
            last_val_best_length,
            last_val_best_collision,
            last_val_best_success,
        )
        if resume_curriculum_message is not None:
            log_text(File, 'curriculum', resume_curriculum_message)
    apply_curriculum_optimizer_state(model, model_c, curriculum_state, args)
    rollout = RolloutBuffer(for_rwta=(args.model in ['rwtaprob', 'rwtaspk']))
    update_count = 0
    final_episode_index = args.train_num - 1
    train_env_count = max(1, int(getattr(args, 'train_envs', 1)))
    next_validation_timestep = int(((max(0, total_env_steps) // val_interval_timesteps) + 1) * val_interval_timesteps)

    if train_env_count <= 1:
        for train_epi_i in range(last_train_epi_num + 1, args.train_num):
            torch.set_grad_enabled(True)
            entropy_value, episode_noise, noise_cap = select_curriculum_settings(args, curriculum_state, train_epi_i)
            maybe_update_entropy(model, entropy_value)

            episode_vehicle_count = select_lane_vehicle_count(args, train_epi_i, curriculum_state=curriculum_state)
            env.init_train(vehicles_count=episode_vehicle_count)
            observation = env.get_train_observation(noise_level=episode_noise)
            for _train_step_i in range(env.max_step_num):
                if args.monitor_time:
                    start_time = time.time()
                with torch.no_grad():
                    model_output, model_other_output = model(observation)
                    if args.model in ['rwtaprob', 'rwtaspk']:
                        action_chosen_onehot = model_other_output[0]
                    else:
                        action_distribution = torch.distributions.OneHotCategorical(model_output)
                        action_chosen_onehot = action_distribution.sample()
                    action_executed_onehot, action_logprob = align_action_to_env_execution(
                        model_output,
                        action_chosen_onehot,
                        info=None,
                        action_num=env.action_num,
                    )
                if args.monitor_time:
                    calculation_time_monitor.record_time(rec_type=1, value=(time.time() - start_time))

                next_state_clean, reward, done, info, _step_record = env.make_action(action_chosen_onehot)
                if info is not None and info.get('executed_action') is not None:
                    with torch.no_grad():
                        action_executed_onehot, action_logprob = align_action_to_env_execution(
                            model_output,
                            action_chosen_onehot,
                            info,
                            env.action_num,
                        )

                if env.done_signal or episode_noise <= 0:
                    observation_next = next_state_clean
                else:
                    observation_next = env._apply_observation_noise(next_state_clean, noise_level=episode_noise)
                total_env_steps += 1

                if args.model in ['rwtaprob', 'rwtaspk']:
                    rollout.add_transition(
                        s1=observation,
                        s2=observation_next,
                        model_output=model_output,
                        a=action_executed_onehot,
                        a_logprob=action_logprob,
                        reward=reward,
                        done=env.done_signal,
                        q_has=model_other_output[2],
                        v_ha=model_other_output[3],
                    )
                else:
                    rollout.add_transition(
                        s1=observation,
                        s2=observation_next,
                        model_output=model_output,
                        a=action_executed_onehot,
                        a_logprob=action_logprob,
                        reward=reward,
                        done=env.done_signal,
                    )
                observation = observation_next
                if env.done_signal:
                    break

            log_text(
                File,
                'train',
                format_train_episode_metrics(
                    train_epi_i,
                    total_env_steps,
                    env.episode_return,
                    env.step_num,
                    float(env.collision_count > 0),
                    env.lane_change_count,
                    episode_noise,
                    noise_cap,
                    entropy_value,
                    episode_vehicle_count,
                    env.get_episode_summary(),
                ),
                onscreen=False,
            )
            log_text(
                File,
                'train_t',
                format_train_episode_metrics(
                    train_epi_i,
                    total_env_steps,
                    env.episode_return,
                    env.step_num,
                    float(env.collision_count > 0),
                    env.lane_change_count,
                    episode_noise,
                    noise_cap,
                    entropy_value,
                    episode_vehicle_count,
                    env.get_episode_summary(),
                ),
                onscreen=False,
            )

            update_count = maybe_apply_rollout_update(
                File,
                train_epi_i,
                model,
                model_c,
                rollout,
                args,
                calculation_time_monitor,
                update_count,
            )
            model_current_save_time = maybe_save_current_models(
                model,
                model_c,
                EXP_NAME,
                model_current_save_time,
            )
            while total_env_steps >= next_validation_timestep:
                (
                    last_val_best,
                    last_val_best_collision,
                    last_val_best_length,
                    last_val_best_success,
                    last_val_best_speed,
                    last_val_best_lane_change,
                    model_current_save_time,
                ) = run_validation_cycle(
                    File,
                    train_epi_i,
                    total_env_steps,
                    env,
                    model,
                    model_c,
                    args,
                    val_num,
                    curriculum_state,
                    EXP_NAME,
                    rollout,
                    last_val_best,
                    last_val_best_collision,
                    last_val_best_length,
                    last_val_best_success,
                    last_val_best_speed,
                    last_val_best_lane_change,
                    model_current_save_time,
                )
                next_validation_timestep += val_interval_timesteps
    else:
        log_text(File, 'parallel', f'train_envs {train_env_count}')
        train_collector = ParallelLaneCollector(
            num_envs=train_env_count,
            road_scenario=args.road_scenario,
            traffic_level=args.traffic_level,
            highway_reward_stage=args.highway_reward_stage,
        )
        next_train_episode_index = last_train_epi_num + 1
        last_completed_episode_index = last_train_epi_num
        slot_states = {}

        def start_parallel_slot(slot_index, episode_index):
            entropy_value_slot, episode_noise_slot, noise_cap_slot = select_curriculum_settings(args, curriculum_state, episode_index)
            maybe_update_entropy(model, entropy_value_slot)
            episode_vehicle_count_slot = select_lane_vehicle_count(
                args,
                episode_index,
                curriculum_state=curriculum_state,
            )
            episode_seed_slot = build_lane_episode_seed(seed, episode_index, slot_index)
            reset_result = train_collector.reset(
                slot_index,
                vehicles_count=episode_vehicle_count_slot,
                episode_seed=episode_seed_slot,
            )
            slot_states[slot_index] = {
                'episode_index': int(episode_index),
                'episode_noise': float(episode_noise_slot),
                'noise_cap': float(noise_cap_slot),
                'entropy_value': float(entropy_value_slot),
                'episode_vehicle_count': int(episode_vehicle_count_slot),
                'observation': np.array(reset_result['observation'], copy=True),
            }

        try:
            while next_train_episode_index <= final_episode_index and len(slot_states) < train_env_count:
                start_parallel_slot(len(slot_states), next_train_episode_index)
                next_train_episode_index += 1

            while slot_states:
                ordered_slot_indices = sorted(slot_states.keys())
                active_slots = [slot_states[slot_index] for slot_index in ordered_slot_indices]
                maybe_update_entropy(
                    model,
                    float(np.mean([slot['entropy_value'] for slot in active_slots])),
                )
                clean_observation_batch = stack_lane_observation_batch(
                    [slot['observation'] for slot in active_slots],
                    torch_device,
                )
                observation_batch = apply_lane_batch_observation_noise(
                    clean_observation_batch,
                    [slot['episode_noise'] for slot in active_slots],
                )

                if args.monitor_time:
                    start_time = time.time()
                with torch.no_grad():
                    model_output_batch, model_other_output_batch = model(observation_batch)
                    if args.model in ['rwtaprob', 'rwtaspk']:
                        sampled_action_batch = model_other_output_batch[0]
                    else:
                        action_distribution = torch.distributions.OneHotCategorical(model_output_batch)
                        sampled_action_batch = action_distribution.sample()
                    action_index_batch = torch.argmax(sampled_action_batch, dim=1)
                if args.monitor_time:
                    calculation_time_monitor.record_time(rec_type=1, value=(time.time() - start_time))

                step_results = train_collector.step({
                    slot_index: int(action_index_batch[batch_index].item())
                    for batch_index, slot_index in enumerate(ordered_slot_indices)
                })
                total_env_steps += len(ordered_slot_indices)

                completed_episodes = []
                for batch_index, slot_index in enumerate(ordered_slot_indices):
                    slot_state = slot_states[slot_index]
                    result = step_results[slot_index]
                    model_output_row = model_output_batch[batch_index:batch_index + 1]
                    sampled_action_row = sampled_action_batch[batch_index:batch_index + 1]
                    with torch.no_grad():
                        action_executed_onehot, action_logprob = align_action_to_env_execution(
                            model_output_row,
                            sampled_action_row,
                            {'executed_action': int(result['executed_action'])},
                            env.action_num,
                        )
                    next_state_clean = torch.as_tensor(
                        result['observation'],
                        dtype=torch.float32,
                        device=torch_device,
                    ).unsqueeze(0)
                    if bool(result['done']) or slot_state['episode_noise'] <= 0:
                        observation_next = next_state_clean
                    else:
                        observation_next = env._apply_observation_noise(
                            next_state_clean,
                            noise_level=slot_state['episode_noise'],
                        )
                    reward_tensor = torch.tensor([float(result['reward'])], dtype=torch.float32, device=torch_device)
                    if args.model in ['rwtaprob', 'rwtaspk']:
                        rollout.add_transition(
                            s1=observation_batch[batch_index:batch_index + 1],
                            s2=observation_next,
                            model_output=model_output_row,
                            a=action_executed_onehot,
                            a_logprob=action_logprob,
                            reward=reward_tensor,
                            done=bool(result['done']),
                            q_has=model_other_output_batch[2][batch_index:batch_index + 1],
                            v_ha=model_other_output_batch[3][batch_index:batch_index + 1],
                        )
                    else:
                        rollout.add_transition(
                            s1=observation_batch[batch_index:batch_index + 1],
                            s2=observation_next,
                            model_output=model_output_row,
                            a=action_executed_onehot,
                            a_logprob=action_logprob,
                            reward=reward_tensor,
                            done=bool(result['done']),
                        )
                    slot_state['observation'] = np.array(result['observation'], copy=True)

                    if bool(result['done']):
                        episode_summary = result.get('episode_summary') or {}
                        train_epi_i = int(slot_state['episode_index'])
                        last_completed_episode_index = train_epi_i
                        log_text(
                            File,
                            'train',
                            format_train_episode_metrics(
                                train_epi_i,
                                total_env_steps,
                                float(episode_summary.get('episode_return', 0.0)),
                                int(episode_summary.get('step_num', 0)),
                                float(episode_summary.get('collision_flag', 0.0)),
                                int(episode_summary.get('lane_change_count', 0)),
                                slot_state['episode_noise'],
                                slot_state['noise_cap'],
                                slot_state['entropy_value'],
                                slot_state['episode_vehicle_count'],
                                episode_summary,
                            ),
                            onscreen=False,
                        )
                        log_text(
                            File,
                            'train_t',
                            format_train_episode_metrics(
                                train_epi_i,
                                total_env_steps,
                                float(episode_summary.get('episode_return', 0.0)),
                                int(episode_summary.get('step_num', 0)),
                                float(episode_summary.get('collision_flag', 0.0)),
                                int(episode_summary.get('lane_change_count', 0)),
                                slot_state['episode_noise'],
                                slot_state['noise_cap'],
                                slot_state['entropy_value'],
                                slot_state['episode_vehicle_count'],
                                episode_summary,
                            ),
                            onscreen=False,
                        )
                        completed_episodes.append(train_epi_i)
                        del slot_states[slot_index]
                        if next_train_episode_index <= final_episode_index:
                            start_parallel_slot(slot_index, next_train_episode_index)
                            next_train_episode_index += 1

                update_episode_index = max(last_completed_episode_index, min(final_episode_index, next_train_episode_index - 1))
                update_count = maybe_apply_rollout_update(
                    File,
                    update_episode_index,
                    model,
                    model_c,
                    rollout,
                    args,
                    calculation_time_monitor,
                    update_count,
                )
                model_current_save_time = maybe_save_current_models(
                    model,
                    model_c,
                    EXP_NAME,
                    model_current_save_time,
                )

                while completed_episodes and total_env_steps >= next_validation_timestep:
                    (
                        last_val_best,
                        last_val_best_collision,
                        last_val_best_length,
                        last_val_best_success,
                        last_val_best_speed,
                        last_val_best_lane_change,
                        model_current_save_time,
                    ) = run_validation_cycle(
                        File,
                        max(completed_episodes),
                        total_env_steps,
                        env,
                        model,
                        model_c,
                        args,
                        val_num,
                        curriculum_state,
                        EXP_NAME,
                        rollout,
                        last_val_best,
                        last_val_best_collision,
                        last_val_best_length,
                        last_val_best_success,
                        last_val_best_speed,
                        last_val_best_lane_change,
                        model_current_save_time,
                    )
                    next_validation_timestep += val_interval_timesteps
        finally:
            train_collector.close()

    if rollout.size() > 0:
        update_stats = update_policy(model, model_c, rollout, args, calculation_time_monitor)
        update_count += 1
        log_text(
            File,
            'update',
            '%d, %4d, %4d, %8.6f, %8.6f' % (
                args.train_num,
                update_count,
                update_stats['rollout_size'],
                update_stats['mean_return_target'],
                update_stats['mean_advantage'],
            ),
            onscreen=False,
        )

    model.save_model(EXP_NAME + '_current')
    model_c.save_model(EXP_NAME + 'critic_current')
    best_actor_file = checkpoint_utils.resolve_checkpoint_file(EXP_NAME + '_best_w_1')
    best_critic_file = checkpoint_utils.resolve_checkpoint_file(EXP_NAME + 'critic_best_1')
    if os.path.exists(best_actor_file) and os.path.exists(best_critic_file):
        model.load_model(EXP_NAME + '_best')
        model_c.load_model(EXP_NAME + 'critic_best')
        final_eval_vehicle_count = None
        if args.road_scenario == 'highway':
            final_eval_vehicle_count = select_lane_vehicle_count(
                args,
                final_episode_index,
                curriculum_state=curriculum_state,
                allow_stage_mix=False,
            )
        final_best_metrics = evaluate_policy(
            env,
            model,
            max(1, int(args.final_eval_episodes)),
            mode='val',
            vehicles_count=final_eval_vehicle_count,
        )
        log_text(File, 'final_best_eval', format_validation_metrics(final_best_metrics, final_episode_index))
        log_text(
            File,
            'final_best_eval_t',
            format_validation_metrics(final_best_metrics, final_episode_index, total_env_steps=total_env_steps),
            onscreen=False,
        )
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
    cleanup_summary = checkpoint_utils.cleanup_final_best_checkpoints(
        actor_best_prefix=EXP_NAME + '_best',
        actor_current_prefix=EXP_NAME + '_current',
        critic_best_prefix=EXP_NAME + 'critic_best',
        critic_current_prefix=EXP_NAME + 'critic_current',
        keep_current=(args.road_scenario == 'highway'),
    )
    log_text(File, 'checkpoint_cleanup', checkpoint_utils.summarize_checkpoint_cleanup(cleanup_summary), onscreen=False)
    log_text(File, 'finish', str(datetime.datetime.now()))
    File.flush()
    File.close()
