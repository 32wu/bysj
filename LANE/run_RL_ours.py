# -*- coding: utf-8 -*-
import argparse
import datetime
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
    parser.add_argument('--train_num', type=int, default=20000)
    parser.add_argument('--rep', type=int, default=11)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--ignore_checkpoint', default=False, action='store_true')
    parser.add_argument('--monitor_time', default=False, action='store_true')

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
    parser.add_argument('--curriculum_mode', type=str, default='adaptive', choices=['fixed', 'adaptive'])
    parser.add_argument('--curriculum_clean_ratio', type=float, default=0.7)
    parser.add_argument('--max_noise', type=float, default=0.15)
    parser.add_argument('--entropy_min', type=float, default=0.1)
    parser.add_argument('--curriculum_patience', type=int, default=2)
    parser.add_argument('--curriculum_noise_step', type=float, default=0.03)
    parser.add_argument('--curriculum_entropy_decay', type=float, default=0.9)
    parser.add_argument('--lane_profile', type=str, default='auto', choices=['auto', 'legacy'])
    parser.add_argument('--road_scenario', type=str, default='highway', choices=['highway', 'merge', 'roundabout'])
    parser.add_argument('--traffic_level', type=str, default='standard', choices=['light', 'standard', 'dense'])
    return parser.parse_args()


def reload_log_file(filename):
    train_epi_num = 0
    val_best_return = -10000.0
    val_best_collision = float('inf')
    val_best_length = 0.0
    val_best_success = 0.0
    with open(filename) as file:
        for line in file:
            str_list = [item for item in re.sub(',', ' ', line).split()]
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
    return train_epi_num, val_best_return, val_best_collision, val_best_length, val_best_success


def log_text(file_handle, type_str, record_text, onscreen=True):
    global log_text_flush_time
    if onscreen:
        print('\033[92m%s\033[0m' % type_str.ljust(10), record_text)
    file_handle.write((type_str + ',').ljust(10) + record_text + '\n')
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


def set_random_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def build_experiment_name(args, model_str, seed):
    return (
        f'{args.alg}_{args.task}_{args.model}_{model_str}_{args.optimizer}_'
        f'{args.lr:.6f}_{args.entropy:.2f}_{args.gamma:.5f}_{args.PPO_epochs}_{args.eps_clip:.4f}_'
        f'ro{args.rollout_steps}_mb{args.mini_batch_size}_lam{args.gae_lambda:.2f}_'
        f'rs{args.reward_scale:.2f}_gc{args.grad_clip:.2f}_{args.curriculum_mode}_'
        f'road{args.road_scenario}_tf{args.traffic_level}_seed{seed:02d}'
    )


def apply_lane_stability_profile(args):
    if args.task != 'gymip' or args.model != 'rwtaspk' or args.lane_profile != 'auto':
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

    clamp_max('lr', 0.0005)
    clamp_min('gamma', 0.995)
    clamp_max('entropy', 2.0)
    clamp_max('PPO_epochs', 4)
    clamp_max('eps_clip', 0.15)
    clamp_min('rollout_steps', 512)
    clamp_min('mini_batch_size', 128)
    clamp_min('gae_lambda', 0.97)
    clamp_max('grad_clip', 0.5)
    clamp_min('curriculum_clean_ratio', 0.8)
    clamp_max('max_noise', 0.08)
    clamp_max('curriculum_noise_step', 0.02)
    if args.road_scenario == 'merge':
        clamp_max('entropy', 0.8)
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
        if args.train_num < 3000:
            args.train_num = 3000
            adjustments.append('train_num->3000')
    return adjustments


def select_lane_vehicle_count(args, train_epi_i):
    if args.task != 'gymip':
        return 40
    if args.model != 'rwtaspk' or args.lane_profile != 'auto':
        return 40
    start_vehicle_num = 18
    if getattr(args, 'road_scenario', None) == 'highway':
        traffic_target = {
            'light': 24,
            'standard': 40,
            'dense': 60,
        }
        end_vehicle_num = traffic_target.get(getattr(args, 'traffic_level', 'standard'), 40)
        curriculum_span = 120 if getattr(args, 'traffic_level', 'standard') == 'dense' else 300
        progress = min(1.0, train_epi_i / max(1, curriculum_span))
    else:
        end_vehicle_num = 40
        progress = min(1.0, train_epi_i / max(1, args.train_num - 1))
    return int(round(start_vehicle_num + (end_vehicle_num - start_vehicle_num) * progress))


def init_curriculum(args):
    return {
        'noise_cap': 0.0,
        'entropy': args.entropy,
        'best_val_return': -10000.0,
        'improvement_streak': 0,
    }


def select_curriculum_settings(args, curriculum_state, train_epi_i):
    if args.curriculum_mode == 'fixed':
        progress = train_epi_i / max(1, args.train_num - 1)
        noise_cap = min(args.max_noise, args.max_noise * progress)
        entropy_value = max(args.entropy_min, args.entropy * (0.998 ** train_epi_i))
    else:
        noise_cap = curriculum_state['noise_cap']
        entropy_value = curriculum_state['entropy']
    use_noise = noise_cap > 0 and random.random() > args.curriculum_clean_ratio
    episode_noise = noise_cap if use_noise else 0.0
    return entropy_value, episode_noise, noise_cap


def update_curriculum(args, curriculum_state, val_return):
    if args.curriculum_mode != 'adaptive':
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


def evaluate_policy(env, model, episode_num, mode='val'):
    metrics = {
        'mean_return': 0.0,
        'mean_length': 0.0,
        'collision_rate': 0.0,
        'mean_lane_change': 0.0,
        'success_rate': 0.0,
    }
    returns = []
    lengths = []
    collisions = []
    lane_changes = []
    successes = []
    with torch.no_grad():
        for _ in range(episode_num):
            if mode == 'val':
                env.init_val()
                observation = env.get_val_observation()
            else:
                env.init_test(record_video=False)
                observation = env.get_test_observation()
            for _step_i in range(env.max_step_num):
                model_output, _ = model(observation)
                action_index = torch.argmax(model_output, dim=1)
                action_onehot = torch.nn.functional.one_hot(action_index, num_classes=env.action_num).float()
                next_state, reward, done, info, step_record = env.make_action(action_onehot)
                if env.done_signal:
                    break
                if mode == 'val':
                    observation = env.get_val_observation()
                else:
                    observation = env.get_test_observation()
            returns.append(env.episode_return)
            lengths.append(env.step_num)
            collisions.append(1.0 if env.collision_count > 0 else 0.0)
            lane_changes.append(env.lane_change_count)
            successes.append(env.episode_success())
    metrics['mean_return'] = float(np.mean(returns))
    metrics['mean_length'] = float(np.mean(lengths))
    metrics['collision_rate'] = float(np.mean(collisions))
    metrics['mean_lane_change'] = float(np.mean(lane_changes))
    metrics['success_rate'] = float(np.mean(successes))
    return metrics


def update_policy(model, model_c, rollout, args, calculation_time_monitor=None):
    if rollout.size() == 0:
        return None
        # ========================================================
    # 🌟 终极防线 1：无论前面谁关了梯度，更新前强行开启全局计算图！
    # ========================================================
    torch.set_grad_enabled(True)
    
    # ========================================================
    # 🌟 终极防线 2：强制唤醒底层 SNN 权重的可导属性（防断链）
    # ========================================================
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

    if args.task in ['gymip']:
        args.hidden_num, args.hid_group_num, args.hid_group_size = 64, 8, 8
        if args.alg == 'ppo':
            args.train_num = 2000 if args.train_num == 20000 else args.train_num
        else:
            args.train_num = 5000 if args.train_num == 20000 else args.train_num

    lane_profile_adjustments = apply_lane_stability_profile(args)
    if args.task == 'gymip' and args.model == 'rwtaspk' and args.lane_profile == 'auto':
        if args.road_scenario == 'merge':
            val_freq, val_num = 10, 10
        else:
            val_freq, val_num = 25, 5
    else:
        val_freq, val_num = 100, 10

    if args.cuda < 0:
        torch_device = torch.device('cpu')
        if args.thread != -1:
            torch.set_num_threads(args.thread)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%1d' % args.cuda
        torch_device = torch.device('cuda:0')

    seed = args.seed if args.seed >= 0 else args.rep
    set_random_seed(seed, torch_device)
    EXP_NAME = build_experiment_name(args, model_str, seed)
    active_model_dir = checkpoint_utils.activate_scenario_model_dir(
        args.road_scenario,
        args.traffic_level,
        create=True,
    )

    if args.task == 'gymip':
        import env_lane
        env = env_lane.GymLane(dev=torch_device, road_scenario=args.road_scenario, traffic_level=args.traffic_level)
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
    model_c = model_critic.Critic(input_size=input_dimension, output_size=output_dimension, dev=torch_device, small=True)
    if hasattr(model, 'set_grad_clip'):
        model.set_grad_clip(args.grad_clip)
    if hasattr(model_c, 'set_grad_clip'):
        model_c.set_grad_clip(args.grad_clip)

    checkpoint_utils.get_model_root(create=True)
    checkpoint_utils.get_log_root(create=True)

    model_current_save_time = time.time()
    log_text_flush_time = time.time()
    globals()['log_text_flush_time'] = log_text_flush_time

    log_filename = os.path.join(checkpoint_utils.get_log_root(create=True), 'log_' + EXP_NAME + '.txt')
    reload_data = os.path.exists(log_filename)
    if args.model in ['rwtaprob', 'rwtaspk']:
        reload_data = reload_data and os.path.exists(checkpoint_utils.resolve_checkpoint_file(EXP_NAME + '_current_b_1'))
    else:
        reload_data = reload_data and os.path.exists(checkpoint_utils.resolve_checkpoint_file(EXP_NAME + '_current_1'))
    if args.ignore_checkpoint:
        reload_data = False

    last_train_epi_num, last_val_best, last_val_best_collision, last_val_best_length, last_val_best_success = 0, -10000.0, float('inf'), 0.0, 0.0
    if reload_data:
        last_train_epi_num, last_val_best, last_val_best_collision, last_val_best_length, last_val_best_success = reload_log_file(log_filename)
        File = open(log_filename, 'a')
        log_text(File, 'resume', str(datetime.datetime.now()))
        model.load_model(EXP_NAME + '_current')
        model_c.load_model(EXP_NAME + 'critic_current')
    else:
        File = open(log_filename, 'w')
        log_text(File, 'init', str(datetime.datetime.now()))
        log_text(File, 'arguments', str(args))
        log_text(File, 'seed', str(seed))
        log_text(File, 'model_dir', active_model_dir)
        if lane_profile_adjustments:
            log_text(File, 'profile', 'lane auto-stabilize: ' + ', '.join(lane_profile_adjustments))

    calculation_time_monitor = TimeMonitor()
    curriculum_state = init_curriculum(args)
    rollout = RolloutBuffer(for_rwta=(args.model in ['rwtaprob', 'rwtaspk']))
    update_count = 0

    for train_epi_i in range(last_train_epi_num + 1, args.train_num):
        torch.set_grad_enabled(True)
        entropy_value, episode_noise, noise_cap = select_curriculum_settings(args, curriculum_state, train_epi_i)
        maybe_update_entropy(model, entropy_value)

        episode_vehicle_count = select_lane_vehicle_count(args, train_epi_i)
        env.init_train(vehicles_count=episode_vehicle_count)
        observation = env.get_train_observation(noise_level=episode_noise)
        for _train_step_i in range(env.max_step_num):
            if args.monitor_time:
                start_time = time.time()
            model_output, model_other_output = model(observation)
            if args.monitor_time:
                calculation_time_monitor.record_time(rec_type=1, value=(time.time() - start_time))

            if args.model in ['rwtaprob', 'rwtaspk']:
                action_chosen_onehot = model_other_output[0]
                action_logprob = model_other_output[1]
            else:
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_chosen_onehot = action_distribution.sample()
                action_logprob = action_distribution.log_prob(action_chosen_onehot)

            # 🌟 把它向左退格！让它无论是不是 SNN 都会执行！
            next_state_clean, reward, done, info, _step_record = env.make_action(action_chosen_onehot)
            
            if env.done_signal:
                observation_next = next_state_clean.detach().clone()
            else:
                observation_next = env.get_train_observation(noise_level=episode_noise)

            if args.model in ['rwtaprob', 'rwtaspk']:
                rollout.add_transition(
                    s1=observation,
                    s2=observation_next,
                    model_output=model_output,
                    a=action_chosen_onehot,
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
                    a=action_chosen_onehot,
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
            '%d, %8.6f, %4d, %4.2f, %4d, %5.3f, %5.3f, %5.3f, %2d' % (
                train_epi_i,
                env.episode_return,
                env.step_num,
                float(env.collision_count > 0),
                env.lane_change_count,
                episode_noise,
                noise_cap,
                entropy_value,
                episode_vehicle_count,
            ),
            onscreen=False,
        )

        if rollout.size() >= args.rollout_steps:
            update_stats = update_policy(model, model_c, rollout, args, calculation_time_monitor)
            rollout.reset()
            update_count += 1
            log_text(
                File,
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

        if time.time() - model_current_save_time > 10:
            model.save_model(EXP_NAME + '_current')
            model_c.save_model(EXP_NAME + 'critic_current')
            model_current_save_time = time.time()

        if train_epi_i % val_freq == (val_freq - 1):
            val_metrics = evaluate_policy(env, model, val_num, mode='val')
            if args.road_scenario == 'merge':
                better_model = (
                    val_metrics['success_rate'] > last_val_best_success + 1e-6 or
                    (
                        abs(val_metrics['success_rate'] - last_val_best_success) <= 1e-6 and
                        val_metrics['mean_length'] > last_val_best_length + 1e-6
                    ) or
                    (
                        abs(val_metrics['success_rate'] - last_val_best_success) <= 1e-6 and
                        abs(val_metrics['mean_length'] - last_val_best_length) <= 1e-6 and
                        val_metrics['collision_rate'] < last_val_best_collision
                    ) or
                    (
                        abs(val_metrics['success_rate'] - last_val_best_success) <= 1e-6 and
                        abs(val_metrics['mean_length'] - last_val_best_length) <= 1e-6 and
                        abs(val_metrics['collision_rate'] - last_val_best_collision) <= 1e-6 and
                        val_metrics['mean_return'] > last_val_best + 1e-6
                    )
                )
            else:
                better_model = (
                    val_metrics['mean_length'] > last_val_best_length + 1e-6 or
                    (
                        abs(val_metrics['mean_length'] - last_val_best_length) <= 1e-6 and
                        val_metrics['collision_rate'] < last_val_best_collision
                    ) or
                    (
                        abs(val_metrics['mean_length'] - last_val_best_length) <= 1e-6 and
                        abs(val_metrics['collision_rate'] - last_val_best_collision) <= 1e-6 and
                        val_metrics['mean_return'] > last_val_best + 1e-6
                    )
                )
            if better_model:
                model.save_model(EXP_NAME + '_best')
                model_c.save_model(EXP_NAME + 'critic_best')
                last_val_best = val_metrics['mean_return']
                last_val_best_collision = val_metrics['collision_rate']
                last_val_best_length = val_metrics['mean_length']
                last_val_best_success = val_metrics['success_rate']
                log_text(
                    File,
                    'val_save',
                    '%d, %8.6f, %6.4f, %8.4f, %8.4f, %6.4f' % (
                        train_epi_i,
                        val_metrics['mean_return'],
                        val_metrics['collision_rate'],
                        val_metrics['mean_length'],
                        val_metrics['mean_lane_change'],
                        val_metrics['success_rate'],
                    ),
                )
            log_text(
                File,
                'val',
                '%d, %8.6f, %6.4f, %8.4f, %8.4f, %6.4f' % (
                    train_epi_i,
                    val_metrics['mean_return'],
                    val_metrics['collision_rate'],
                    val_metrics['mean_length'],
                    val_metrics['mean_lane_change'],
                    val_metrics['success_rate'],
                ),
            )
            curriculum_signal = val_metrics['mean_length']
            if args.road_scenario == 'merge':
                curriculum_signal = val_metrics['mean_length'] + env.target_step_num * val_metrics['success_rate']
            curriculum_message = update_curriculum(args, curriculum_state, curriculum_signal)
            if curriculum_message is not None:
                log_text(File, 'curriculum', curriculum_message)

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
    log_text(File, 'finish', str(datetime.datetime.now()))
    File.flush()
    File.close()
