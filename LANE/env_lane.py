# -*- coding: utf-8 -*-

import numpy as np
np.bool8 = np.bool_
import torch

import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo


def print_info(input_string):
    print('[96mLANE_ENV_INFO|[0m', input_string)


class GymLane:
    def __init__(self, dev=torch.device('cpu')):
        self.max_step_num = 150
        self.state_dimension = 25
        self.action_num = 5
        self.dev = dev
        self.env = None

        self.mode = None
        self.step_num = 0
        self.done_signal = 0
        self.state_original = None
        self.state_processed = None

        self.episode_return = 0.0
        self.episode_raw_return = 0.0
        self.collision_count = 0
        self.lane_change_count = 0
        self.last_step_info = {}
        self.lane_change_action_ids = {0, 2}
        self.previous_action_index = None
        self.steps_since_lane_change = 1000

        self.obs_min_seen = None
        self.obs_max_seen = None

        print_info('ACTION NUMBER: %6d' % self.action_num)
        print_info('STATE DIMENSION: %6d' % self.state_dimension)

    def _configure_env(self, vehicles_count=40):
        self.env.unwrapped.configure({
            'duration': self.max_step_num,
            'vehicles_count': int(vehicles_count),
            'high_speed_reward': 0.8,
            'reward_speed_range': [20, 30],
            'collision_reward': -1.0,
            'lane_change_reward': -0.05,
        })

    def _refresh_action_metadata(self):
        action_type = getattr(self.env.unwrapped, 'action_type', None)
        if action_type is None or not hasattr(action_type, 'actions_indexes'):
            self.lane_change_action_ids = {0, 2}
            return
        action_indexes = action_type.actions_indexes
        lane_change_ids = []
        for action_name in ['LANE_LEFT', 'LANE_RIGHT']:
            if action_name in action_indexes:
                lane_change_ids.append(action_indexes[action_name])
        self.lane_change_action_ids = set(lane_change_ids) if lane_change_ids else {0, 2}

    def _record_observation_range(self, state):
        state_min = float(np.min(state))
        state_max = float(np.max(state))
        if self.obs_min_seen is None or state_min < self.obs_min_seen or state_max > self.obs_max_seen:
            self.obs_min_seen = state_min if self.obs_min_seen is None else min(self.obs_min_seen, state_min)
            self.obs_max_seen = state_max if self.obs_max_seen is None else max(self.obs_max_seen, state_max)
            #print_info('OBS RANGE min=%7.4f max=%7.4f' % (self.obs_min_seen, self.obs_max_seen))

    def state_to_tensor(self, state):
        self._record_observation_range(state)
        state_handle = np.copy(state).flatten()
        state_handle = np.clip((state_handle + 1.0) / 2.0, 0.0, 1.0)
        state_tensor = torch.FloatTensor(np.expand_dims(state_handle, axis=0)).to(self.dev)
        return state_tensor

    def _reset_episode_trackers(self):
        self.step_num = 0
        self.done_signal = 0
        self.episode_return = 0.0
        self.episode_raw_return = 0.0
        self.collision_count = 0
        self.lane_change_count = 0
        self.last_step_info = {}
        self.previous_action_index = None
        self.steps_since_lane_change = 1000
        self.state_original = self.env.reset()[0]
        self.state_processed = self.state_to_tensor(self.state_original)

    def init_train(self, vehicles_count=40):
        self.mode = 'train'
        if self.env is not None:
            self.env.close()
        self.env = gym.make('highway-v0', render_mode=None)
        self._configure_env(vehicles_count=vehicles_count)
        self._refresh_action_metadata()
        self._reset_episode_trackers()

    def init_val(self, vehicles_count=40):
        self.mode = 'val'
        if self.env is not None:
            self.env.close()
        self.env = gym.make('highway-v0', render_mode=None)
        self._configure_env(vehicles_count=vehicles_count)
        self._refresh_action_metadata()
        self._reset_episode_trackers()

    def init_test(self, variation_type='none', variation_param=0, record_video=False, vehicles_count=40):
        del variation_type, variation_param
        self.mode = 'test'
        if self.env is not None:
            self.env.close()
        render_mode = 'rgb_array' if record_video else None
        self.env = gym.make('highway-v0', render_mode=render_mode)
        self._configure_env(vehicles_count=vehicles_count)
        if record_video:
            self.env = RecordVideo(self.env, video_folder='./video_logs_lane', name_prefix='highway_show')
        self._refresh_action_metadata()
        self._reset_episode_trackers()

    def get_observation(self):
        self.state_processed = self.state_to_tensor(self.state_original)
        return self.state_processed

    def get_train_observation(self, noise_level=0.0, **kwargs):
        del kwargs
        state_tensor = self.get_observation()
        if noise_level > 0:
            noise = torch.randn_like(state_tensor) * noise_level
            state_tensor = torch.clamp(state_tensor + noise, 0.0, 1.0)
        return state_tensor

    def get_val_observation(self, **kwargs):
        del kwargs
        return self.get_observation()

    def get_test_observation(self, noise_type='none', noise_param=0, **kwargs):
        del kwargs
        state_tensor = self.get_observation()
        if noise_type == 'none':
            return state_tensor
        if noise_type == 'gaussian':
            state_tensor = state_tensor + torch.randn_like(state_tensor) * noise_param
        else:
            state_tensor = state_tensor + (torch.rand_like(state_tensor) - 0.5) * 2 * noise_param
        return torch.clamp(state_tensor, 0.0, 1.0)

    def _shape_reward(self, raw_reward, info, action_index, done):
        reward_items = info.get('rewards', {})
        speed_reward = float(reward_items.get('high_speed_reward', 0.0))
        lane_reward = float(reward_items.get('right_lane_reward', 0.0))
        on_road_reward = float(reward_items.get('on_road_reward', 1.0))
        crashed = float(bool(info.get('crashed', False)))
        progress = min(1.0, max(0.0, self.step_num / max(1, self.max_step_num)))
        lane_change_action = action_index in self.lane_change_action_ids
        lane_change_penalty = 0.75 if lane_change_action else 0.0
        repeated_lane_change_penalty = 0.0
        if lane_change_action and self.steps_since_lane_change < 6:
            repeated_lane_change_penalty = 0.90 * (6 - self.steps_since_lane_change) / 6.0
        zigzag_penalty = 0.25 if (
            self.previous_action_index in self.lane_change_action_ids and
            lane_change_action and
            self.previous_action_index != action_index
        ) else 0.0
        steady_action_bonus = 0.18 if not lane_change_action else 0.0
        survival_bonus = 0.45 if not crashed else 0.0
        progress_bonus = 0.35 * progress if not crashed else 0.0
        crash_penalty = crashed * (6.0 + 4.0 * (1.0 - progress))
        completion_bonus = 8.0 if self.step_num >= self.max_step_num and not crashed else 0.0
        shaped_reward = (
            survival_bonus
            + steady_action_bonus
            + progress_bonus
            + 0.55 * speed_reward
            + 0.10 * lane_reward
            + 0.15 * float(raw_reward)
            - 0.35 * (1.0 - on_road_reward)
            - lane_change_penalty
            - repeated_lane_change_penalty
            - zigzag_penalty
            - crash_penalty
            + completion_bonus
        )
        info['raw_reward'] = float(raw_reward)
        info['shaped_reward'] = float(shaped_reward)
        info['reward_breakdown'] = {
            'survival_bonus': float(survival_bonus),
            'steady_action_bonus': float(steady_action_bonus),
            'progress_bonus': float(progress_bonus),
            'speed_bonus': float(0.55 * speed_reward),
            'lane_bonus': float(0.10 * lane_reward),
            'base_reward_bonus': float(0.15 * float(raw_reward)),
            'lane_change_penalty': float(lane_change_penalty),
            'repeated_lane_change_penalty': float(repeated_lane_change_penalty),
            'zigzag_penalty': float(zigzag_penalty),
            'offroad_penalty': float(0.35 * (1.0 - on_road_reward)),
            'crash_penalty': float(crash_penalty),
            'completion_bonus': float(completion_bonus),
        }
        return float(shaped_reward)

    def make_action(self, action):
        action_index = int(torch.argmax(action).item())
        
        # ==========================================
        # 🚗 恢复：小车心跳监视器（每 10 步刷新一次）
        # ==========================================
        if self.step_num % 10 == 0:
            print(f"\r🚗 正在马路上飞驰... 当前回合已开 {self.step_num} 步", end="", flush=True)

        next_state, reward_value, terminated, truncated, info = self.env.step(action_index)
        done = terminated or truncated

        self.state_original = np.copy(next_state)
        self.state_processed = self.state_to_tensor(self.state_original)
        self.step_num += 1
        self.episode_raw_return += float(reward_value)

        info_action = info.get('action', action_index)
        if info_action in self.lane_change_action_ids:
            self.lane_change_count += 1
        if bool(info.get('crashed', False)):
            self.collision_count += 1

        shaped_reward = self._shape_reward(reward_value, info, info_action, done)
        self.episode_return += shaped_reward
        self.last_step_info = info

        if done or self.step_num >= self.max_step_num:
            self.done_signal = 1
            # ==========================================
            # 💥 恢复：回合结束播报（带上你优化的累计得分）
            # ==========================================
            print(f" -> 💥 回合结束！步奖励: {shaped_reward:.2f} | 累计回报: {self.episode_return:.2f} | 原始回报: {self.episode_raw_return:.2f}")
        else:
            self.done_signal = 0

        reward_tensor = torch.FloatTensor(np.array([shaped_reward])).to(self.dev)
        step_record = [
            self.step_num,
            float(shaped_reward),
            float(self.episode_return),
            int(self.collision_count),
            int(self.lane_change_count),
            int(bool(info.get('crashed', False))),
            float(self.episode_raw_return)
        ]
        if info_action in self.lane_change_action_ids:
            self.steps_since_lane_change = 0
        else:
            self.steps_since_lane_change += 1
        self.previous_action_index = info_action
        return self.state_processed, reward_tensor, done, info, step_record


if __name__ == '__main__':
    env = GymLane(dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print('[91mFINISH: env_lane[0m')
