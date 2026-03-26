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
        self.collision_count = 0
        self.lane_change_count = 0
        self.last_step_info = {}
        self.lane_change_action_ids = {0, 2}

        self.obs_min_seen = None
        self.obs_max_seen = None

        print_info('ACTION NUMBER: %6d' % self.action_num)
        print_info('STATE DIMENSION: %6d' % self.state_dimension)

    def _configure_env(self):
        self.env.unwrapped.configure({
            'duration': self.max_step_num,
            'vehicles_count': 40,
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
            print_info('OBS RANGE min=%7.4f max=%7.4f' % (self.obs_min_seen, self.obs_max_seen))

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
        self.collision_count = 0
        self.lane_change_count = 0
        self.last_step_info = {}
        self.state_original = self.env.reset()[0]
        self.state_processed = self.state_to_tensor(self.state_original)

    def init_train(self):
        self.mode = 'train'
        if self.env is not None:
            self.env.close()
        self.env = gym.make('highway-v0', render_mode=None)
        self._configure_env()
        self._refresh_action_metadata()
        self._reset_episode_trackers()

    def init_val(self):
        self.mode = 'val'
        if self.env is not None:
            self.env.close()
        self.env = gym.make('highway-v0', render_mode=None)
        self._configure_env()
        self._refresh_action_metadata()
        self._reset_episode_trackers()

    def init_test(self, variation_type='none', variation_param=0, record_video=False):
        del variation_type, variation_param
        self.mode = 'test'
        if self.env is not None:
            self.env.close()
        render_mode = 'rgb_array' if record_video else None
        self.env = gym.make('highway-v0', render_mode=render_mode)
        self._configure_env()
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

    def make_action(self, action):
        action_index = int(torch.argmax(action).item())
        next_state, reward_value, terminated, truncated, info = self.env.step(action_index)
        done = terminated or truncated

        self.state_original = np.copy(next_state)
        self.state_processed = self.state_to_tensor(self.state_original)
        self.step_num += 1
        self.episode_return += float(reward_value)
        self.last_step_info = info

        info_action = info.get('action', action_index)
        if info_action in self.lane_change_action_ids:
            self.lane_change_count += 1
        if bool(info.get('crashed', False)):
            self.collision_count += 1

        if done or self.step_num >= self.max_step_num:
            self.done_signal = 1
        else:
            self.done_signal = 0

        reward_tensor = torch.FloatTensor(np.array([reward_value])).to(self.dev)
        step_record = [
            self.step_num,
            float(reward_value),
            float(self.episode_return),
            int(self.collision_count),
            int(self.lane_change_count),
            int(bool(info.get('crashed', False))),
        ]
        return reward_tensor, self.state_processed, step_record


if __name__ == '__main__':
    env = GymLane(dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print('[91mFINISH: env_lane[0m')
