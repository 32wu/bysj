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
    parser.add_argument('--rep', type=int, default=11)
    parser.add_argument('--ignore_checkpoint', default=False, action='store_true')
    parser.add_argument('--monitor_time', default=False, action='store_true')
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
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--skip_post_tests', default=False, action='store_true')
    # ---------------------------------------------------
    return parser.parse_args()


def reload_log_file(filename):
    train_epi_num = 0
    val_best_return = -10000.0
    val_best_collision = float('inf')
    val_best_length = 0.0
    val_best_success = 0.0
    val_best_speed = 0.0
    val_best_lane_change = 0.0
    with open(filename) as file:
        for line in file:
            str_list = [i for i in re.sub(',', ' ', line).split()]
            if not str_list:
                continue
            if str_list[0] == 'train':
                train_epi_num = int(str_list[1])
            if str_list[0] == 'val_save':
                val_best_return = float(str_list[2])
                if len(str_list) > 3:
                    val_best_collision = float(str_list[3])
                if len(str_list) > 4:
                    val_best_length = float(str_list[4])
                if len(str_list) > 6:
                    val_best_success = float(str_list[6])
                if len(str_list) > 7:
                    val_best_speed = float(str_list[7])
                if len(str_list) > 5:
                    val_best_lane_change = float(str_list[5])
            elif str_list[0] == 'val' and val_best_return <= -9999.0 and len(str_list) > 2:
                val_best_return = float(str_list[2])
    return (
        train_epi_num,
        val_best_return,
        val_best_collision,
        val_best_length,
        val_best_success,
        val_best_speed,
        val_best_lane_change,
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
    if args.task != 'gymip':
        return []
    adjustments = []

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
        lr_cap = {
            'light': 0.0008,
            'standard': 0.0007,
            'dense': 0.0006,
        }
        gamma_floor = {
            'light': 0.992,
            # 【方案C-base】standard/dense场景对齐ours的长视野折扣，使base也重视长期存活
            'standard': 0.998,
            'dense': 0.998,
        }
        gae_lambda_floor = {
            'light': 0.97,
            # 【方案C-base】standard/dense提升GAE lambda，让优势估计更精确反映长期影响
            'standard': 0.990,
            'dense': 0.990,
        }
        entropy_floor = {
            'light': 0.25,
            'standard': 0.30,
            'dense': 0.35,
        }
        train_num_floor = {
            'light': 2400,
            'standard': 3200,
            'dense': 4200,
        }
        clamp_max('lr', lr_cap.get(traffic_level, 0.0007))
        clamp_min('gamma', gamma_floor.get(traffic_level, 0.994))
        clamp_min('gae_lambda', gae_lambda_floor.get(traffic_level, 0.97))
        clamp_min('entropy', entropy_floor.get(traffic_level, 0.30))
        clamp_max('PPO_epochs', 4)
        clamp_max('eps_clip', 0.15)
        if args.train_num < train_num_floor.get(traffic_level, 3200):
            args.train_num = train_num_floor.get(traffic_level, 3200)
            adjustments.append(f'train_num->{args.train_num}')
    return adjustments


def select_lane_vehicle_count(args, train_epi_i):
    if args.task != 'gymip' or getattr(args, 'road_scenario', None) != 'highway':
        return None
    traffic_target = {
        'light': 40,
        'standard': 60,
        'dense': 84,
    }
    traffic_start = {
        'light': 28,
        'standard': 36,
        'dense': 46,
    }
    curriculum_span = {
        'light': 160,
        'standard': 320,
        'dense': 480,
    }
    traffic_level = getattr(args, 'traffic_level', 'standard')
    start_vehicle_num = traffic_start.get(traffic_level, 30)
    end_vehicle_num = traffic_target.get(traffic_level, 50)
    span = curriculum_span.get(traffic_level, 280)
    progress = min(1.0, train_epi_i / max(1, span))
    return int(round(start_vehicle_num + (end_vehicle_num - start_vehicle_num) * progress))


def evaluate_lane_policy(env, model, episode_num):
    metrics = {
        'mean_return': 0.0,
        'mean_length': 0.0,
        'mean_speed': 0.0,
        'collision_rate': 0.0,
        'mean_lane_change': 0.0,
        'success_rate': 0.0,
    }
    returns = []
    lengths = []
    speeds = []
    collisions = []
    lane_changes = []
    successes = []
    with torch.no_grad():
        for _ in range(episode_num):
            env.init_val()
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
    metrics['mean_return'] = float(np.mean(returns))
    metrics['mean_length'] = float(np.mean(lengths))
    metrics['mean_speed'] = float(np.mean(speeds))
    metrics['collision_rate'] = float(np.mean(collisions))
    metrics['mean_lane_change'] = float(np.mean(lane_changes))
    metrics['success_rate'] = float(np.mean(successes))
    return metrics


def lane_validation_quality(metrics):
    excessive_lane_changes = max(0.0, float(metrics['mean_lane_change']) - 4.0)
    return (
        float(metrics['mean_return'])
        + 0.45 * float(metrics.get('mean_length', 0.0))
        + 1.5 * float(metrics['mean_speed'])
        - 30.0 * float(metrics['collision_rate'])
        - 2.5 * excessive_lane_changes
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
    if args.task in ['gymip']:
        args.hidden_num, args.hid_group_num, args.hid_group_size = 64, 8, 8
        if args.alg == 'ppo':
            args.train_num = 2000 if args.train_num == 20000 else args.train_num
        else:
            args.train_num = 5000 if args.train_num == 20000 else args.train_num
    if args.task in ['mnist']:
        args.hidden_num, args.hid_group_num, args.hid_group_size = 200, 20, 10
        args.train_num = 10000
    if args.task in ['vizdoom']:
        args.hidden_num, args.hid_group_num, args.hid_group_size = 500, 50, 10
        if args.alg == 'ppo':
            args.train_num = 2000 if args.train_num == 20000 else args.train_num
        else:
            args.train_num = 5000 if args.train_num == 20000 else args.train_num
    baseline_profile_adjustments = apply_lane_baseline_profile(args)
    run_kind = 'baseline'
    EXP_NAME = '%s_%s_%s_%s_%s_%8.6f_%4.2f_%6.5f_%d_%5.4f_road%s_tf%s_rep%02d' % (
            args.alg, args.task, args.model, model_str, args.optimizer,
            args.lr, args.entropy, args.gamma,
            args.PPO_epochs, args.eps_clip,
            args.road_scenario, args.traffic_level, args.rep)
    active_model_dir, active_log_dir = checkpoint_utils.activate_scenario_output_dirs(
            run_kind=run_kind, road_scenario=args.road_scenario, traffic_level=args.traffic_level, create=True)
    # Task specified variables
    if args.task in ['gymip',]:
        if args.road_scenario == 'highway' and args.alg == 'ppo':
            val_freq, val_num, test_num = 25, 5, 10
            train_frequency = 32
        else:
            val_freq, val_num, test_num = 100, 10, 10
            train_frequency = 10
    elif args.task in ['vizdoom']:
        val_freq, val_num, test_num = 100, 10, 10
        train_frequency = 30
    elif args.task in ['mnist', 'cifar10']:
        val_freq, val_num, test_num = 100, 1, 1
        train_frequency = 5
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
    # Environment Setup
    if args.task == 'mnist':
        import env_mnist
        env = env_mnist.MnistDataset(dev=torch_device)
        input_dimension, output_dimension = env.state_dim, env.action_num
        mem = memory_lib.MemoryBuffer(s_size=input_dimension, a_size=output_dimension, dev=torch_device)
    elif args.task == 'gymip':      # 名字继续用 gymip 骗过系统，但里面已经是小车了！
        import env_lane
        env = env_lane.GymLane(dev=torch_device, road_scenario=args.road_scenario, traffic_level=args.traffic_level)  # 删掉了多余的 xml 参数
        input_dimension, output_dimension = env.state_dimension, env.action_num
        mem = memory_lib.MemoryBuffer(s_size=input_dimension, a_size=output_dimension, dev=torch_device)
    elif args.task == 'vizdoom':    # ViZDoom Health Gathering
        import env_vizdoom
        env = env_vizdoom.DoomHealthGathering(dev=torch_device)
        input_dimension, output_dimension = env.state_dimension, env.action_num
        mem = memory_lib.MemoryBuffer(s_size=input_dimension, a_size=output_dimension,
                                      memory_size=2000, batch_size=100, dev=torch_device)
    else:
        input_dimension, output_dimension = None, None
        env, mem = None, None
        print('Error in model name.')
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
    if args.task in ['gymip', 'gymdip']:
        model_c = model_critic.Critic(input_size=input_dimension, output_size=output_dimension,
                                      dev=torch_device, small=True)
    else:
        model_c = model_critic.Critic(input_size=input_dimension, output_size=output_dimension,
                                      dev=torch_device)
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
        File = open(log_filename, 'w')
        log_text(File, 'init', str(datetime.datetime.now()))
        log_text(File, 'arguments', str(args))
        if baseline_profile_adjustments:
            log_text(File, 'profile', '; '.join(baseline_profile_adjustments))
        log_text(File, 'model_dir', active_model_dir)
        log_text(File, 'log_dir', active_log_dir, onscreen=False)
        if args.model == 'ann2snn':
            model.load_model_ann(EXP_NAME + '_best')
    # Time Monitor
    calculation_time_monitor = TimeMonitor()
    use_short_horizon_replay = bool(args.task == 'gymip' and args.road_scenario == 'highway' and args.alg == 'ppo')
    # >>>>  Main Loop
    mem.reset()         # memory buffer is shared across episodes
    train_step_num_total = 0
    for train_epi_i in range((last_train_epi_num + 1), args.train_num):
        if args.model == 'ann2snn':
            break
        episode_vehicle_count = select_lane_vehicle_count(args, train_epi_i)
        env.init_train(vehicles_count=episode_vehicle_count)
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
            # >>>> Train
            train_step_num_total = (train_step_num_total + 1) % train_frequency     # every number of steps
            if train_step_num_total == 0:
                if args.model in ['rwtaprob', 'rwtaspk']:
                    s1, s2, model_output_1, a_1, a_logprob_1, r, done, q_has_1, v_ha_1 = mem.get_batch()
                else:
                    s1, s2, model_output_1, a_1, a_logprob_1, r, done = mem.get_batch()
                batch_size = s1.shape[0]
                model_output_2, model_other_output_2 = model(s2)
                s1_value = model_c(s1)
                s2_value = model_c(s2)
                a1_prob = model_output_1
                a2_prob = model_output_2
                s1_value_ave = torch.sum(a1_prob * s1_value, dim=1).detach()
                s2_value_ave = torch.sum(a2_prob * s2_value, dim=1).detach()
                # Update Critic
                state_value_target = s1_value.clone().detach()
                a1_index = torch.argmax(a_1, dim=1)     # onehot to index
                state_value_target[torch.arange(batch_size), a1_index] = \
                    r + (args.gamma * s2_value_ave) * (1 - done)
                model_c.learn(s1_value, state_value_target)
                # Update Agent
                advantage = (s1_value[torch.arange(batch_size), a1_index] - s1_value_ave).detach()
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
            '%d, %8.6f, %4d, %3d' % (
                train_epi_i,
                env.episode_return,
                env.step_num,
                -1 if episode_vehicle_count is None else int(episode_vehicle_count),
            ),
            onscreen=False,
        )
        # Validation
        if train_epi_i % val_freq == (val_freq - 1):
            if args.task == 'gymip':
                val_metrics = evaluate_lane_policy(env, model, val_num)
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
                        '%d, %8.6f, %6.4f, %8.4f, %8.4f, %6.4f, %8.4f' % (
                            train_epi_i,
                            val_metrics['mean_return'],
                            val_metrics['collision_rate'],
                            val_metrics['mean_length'],
                            val_metrics['mean_lane_change'],
                            val_metrics['success_rate'],
                            val_metrics['mean_speed'],
                        ),
                    )
                log_text(
                    File,
                    'val',
                    '%d, %8.6f, %6.4f, %8.4f, %8.4f, %6.4f, %8.4f' % (
                        train_epi_i,
                        val_metrics['mean_return'],
                        val_metrics['collision_rate'],
                        val_metrics['mean_length'],
                        val_metrics['mean_lane_change'],
                        val_metrics['success_rate'],
                        val_metrics['mean_speed'],
                    ),
                )
            else:
                val_preformance_list = []
                val_step_num_list = []
                for val_epi_i in range(val_num):
                    env.init_val()
                    observation = env.get_val_observation()
                    for val_step_i in range(env.max_step_num):
                        model_output, model_other_output = model(observation)
                        action_chosen_index = torch.argmax(model_output, dim=1)
                        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                        observation_next, reward, _, _, step_record_val = env.make_action(action_chosen_onehot)
                        if env.done_signal == True:
                            break
                        observation = observation_next
                    val_preformance_list.append(step_record_val[2])
                    val_step_num_list.append(env.step_num)
                val_performance_mean = sum(val_preformance_list) / len(val_preformance_list)
                val_step_num_mean = sum(val_step_num_list) / len(val_step_num_list)
                if last_val_best <= val_performance_mean:
                    model.save_model(EXP_NAME + '_best')
                    model_c.save_model(EXP_NAME + 'critic' + '_best')
                    log_text(
                        File,
                        'val_save',
                        '%d,   %8.6f,   %8.4f' % (
                            train_epi_i,
                            val_performance_mean,
                            val_step_num_mean,
                        ),
                    )
                    last_val_best = val_performance_mean
                log_text(
                    File,
                    'val',
                    '%d,   %8.6f,   %8.4f' % (
                        train_epi_i,
                        val_performance_mean,
                        val_step_num_mean,
                    ),
                )


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
    if args.task in ['mnist', 'cifar10']:
        ad_train_epi_num = 1000
    else:               # 'vizdoom', 'gymip'
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

    # >>>> Test GYMIP / GYMDIP Env
    if args.task in ['gymip', 'gymdip']:
        model.load_model(EXP_NAME + '_best')
        noise_type_list = ['length', 'thick', 'union']
        if args.task == 'gymdip':
            noise_type_list = ['thick']
        for noise_type in noise_type_list:
            if noise_type == 'length':
                noise_param_list = np.arange(0.16, 4.88, 0.08)
            elif noise_type == 'thick':
                noise_param_list = np.arange(0.01, 0.305, 0.005)
            else:
                noise_param_list = np.arange(0.02, 0.305, 0.005)
            for noise_param in noise_param_list:
                test_preformance_list = []
                for test_epi_i in range(test_num):
                    env.init_test(variation_type=noise_type, variation_param=noise_param)
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
                log_text(File, 'e_gymip', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
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







