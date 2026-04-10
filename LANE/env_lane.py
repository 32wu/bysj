# -*- coding: utf-8 -*-

import os
from datetime import datetime

import numpy as np
np.bool8 = np.bool_
import torch

import checkpoint_utils
import gymnasium as gym
import highway_env
from highway_env import utils as highway_utils
from gymnasium.wrappers import RecordVideo


def print_info(input_string):
    print('\033[96mLANE_ENV_INFO|\033[0m', input_string)


SCENARIO_PRESETS = {
    'highway': {
        'env_id': 'highway-v0',
        'max_step_num': 150,
        'target_step_num': 140,
        'config': {
            'duration': 150,
            'lanes_count': 4,
            'vehicles_count': 40,
            'high_speed_reward': 0.8,
            'reward_speed_range': [22, 30],
            'collision_reward': -1.0,
            'lane_change_reward': 0.0,
            'right_lane_reward': 0.0,
        },
        'supports_vehicles_count': True,
    },
    'merge': {
        'env_id': 'merge-v0',
        'max_step_num':45,
        'target_step_num': 36,
        'config': {
            'collision_reward': -1.0,
            'high_speed_reward': 0.3,
            'lane_change_reward': -0.08,
        },
        'supports_vehicles_count': False,
    },
    'roundabout': {
        'env_id': 'roundabout-v0',
        'max_step_num': 90,
        'target_step_num': 90,
        'config': {
            'duration': 90,
            'collision_reward': -1.0,
            'high_speed_reward': 0.3,
            'lane_change_reward': -0.08,
        },
        'supports_vehicles_count': False,
    },
}

TRAFFIC_VEHICLE_COUNT = {
    'light': 24,
    'standard': 40,
    'dense': 60,
}

HIGHWAY_TRAFFIC_ENV_PROFILE = {
    'light': {
        'vehicles_density': 0.75,
        'ego_spacing': 2.8,
    },
    'standard': {
        'vehicles_density': 1.00,
        'ego_spacing': 2.0,
    },
    'dense': {
        'vehicles_density': 1.55,
        'ego_spacing': 1.2,
    },
}

HIGHWAY_DESIRED_CRUISE_SPEED = {
    'light': 29.0,
    'standard': 28.0,
    'dense': 25.5,
}

HIGHWAY_SUCCESS_MIN_SPEED = {
    'light': 25.0,
    'standard': 24.0,
    'dense': 23.0,
}

HIGHWAY_OVERTAKE_PASS_MARGIN = 6.0
HIGHWAY_POST_OVERTAKE_SETTLE_STEPS = 4
HIGHWAY_LANE_CHANGE_EVAL_WINDOW = 4
HIGHWAY_MIN_EFFECTIVE_FRONT_GAP_GAIN = 8.0
HIGHWAY_MIN_EFFECTIVE_SPEED_GAIN = 1.0
HIGHWAY_MIN_EFFECTIVE_FRONT_SPEED_GAIN = 1.5

HIGHWAY_TRAINING_SIMULATION_FREQUENCY = {
    'light': 13,
    'standard': 12,
    'dense': 10,
}

VIDEO_LAYOUT_SCENARIOS = [
    ('highway', 'standard'),
    ('highway', 'dense'),
    ('merge', 'standard'),
    ('roundabout', 'standard'),
]
VIDEO_LAYOUT_RUN_KINDS = ('base', 'ours')

SCENARIO_TRAFFIC_TARGETS = {
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

# 【核心修改1】：清空匝道上的幽灵车队，只保留主路车辆作为参照物
MERGE_TRAFFIC_BLUEPRINT = [
    {'lane': ('a', 'b', 0), 'longitudinal': 18.0, 'speed': 28.0, 'target_speed': 29.0},
    {'lane': ('a', 'b', 0), 'longitudinal': 108.0, 'speed': 29.0, 'target_speed': 30.0},
    {'lane': ('a', 'b', 1), 'longitudinal': 142.0, 'speed': 29.0, 'target_speed': 30.0},
    {'lane': ('b', 'c', 0), 'longitudinal': 18.0, 'speed': 27.5, 'target_speed': 29.0},
    {'lane': ('b', 'c', 1), 'longitudinal': 46.0, 'speed': 28.5, 'target_speed': 29.5},
]

ROUNDABOUT_DESTINATIONS = ['exr', 'sxr', 'nxr']
ROUNDABOUT_TRAFFIC_BLUEPRINT = [
    {'lane': ('we', 'sx', 1), 'longitudinal': 34.0, 'speed': 14.0, 'destination': 'exr'},
    {'lane': ('we', 'sx', 0), 'longitudinal': 56.0, 'speed': 13.5, 'destination': 'sxr'},
    {'lane': ('se', 'ex', 0), 'longitudinal': 38.0, 'speed': 13.0, 'destination': 'exr'},
    {'lane': ('ex', 'ee', 1), 'longitudinal': 18.0, 'speed': 13.5, 'destination': 'sxr'},
    {'lane': ('ee', 'nx', 0), 'longitudinal': 12.0, 'speed': 13.0, 'destination': 'sxr'},
    {'lane': ('sx', 'se', 1), 'longitudinal': 24.0, 'speed': 13.0, 'destination': 'nxr'},
    {'lane': ('nx', 'ne', 1), 'longitudinal': 28.0, 'speed': 13.0, 'destination': 'exr'},
    {'lane': ('ne', 'wx', 0), 'longitudinal': 22.0, 'speed': 13.0, 'destination': 'nxr'},
    {'lane': ('wer', 'wes', 0), 'longitudinal': 58.0, 'speed': 12.0, 'destination': 'sxr'},
    {'lane': ('eer', 'ees', 0), 'longitudinal': 72.0, 'speed': 12.0, 'destination': 'exr'},
    {'lane': ('ner', 'nes', 0), 'longitudinal': 44.0, 'speed': 12.0, 'destination': 'nxr'},
]

ROUNDABOUT_ACTIVE_TRAFFIC_LANE_PAIRS = {
    ('we', 'sx'),
    ('sx', 'se'),
    ('se', 'ex'),
    ('ex', 'ee'),
    ('ee', 'nx'),
    ('nx', 'ne'),
    ('ne', 'wx'),
    ('wx', 'we'),
    ('wer', 'wes'),
    ('eer', 'ees'),
    ('ner', 'nes'),
}

SCENARIO_REPLENISH_INTERVAL = {
    'merge': 4,
    'roundabout': 2,
}

MERGE_EGO_START_LANE = ('j', 'k', 0)
MERGE_EGO_START_LONGITUDINAL = 30.0
MERGE_EGO_START_SPEED = 24.0
MERGE_RAMP_LANE_PAIRS = {
    ('j', 'k'),
    ('k', 'b'),
}
MERGE_TARGET_LANE_BY_SEGMENT = {
    ('j', 'k', 0): ('a', 'b', 1),
    ('k', 'b', 0): ('a', 'b', 1),
    ('b', 'c', 2): ('b', 'c', 1),
}
MERGE_SUCCESS_X_THRESHOLD = 360.0
MERGE_MAINLINE_MIN_SPEED = 16.0
MERGE_STABLE_MAINLINE_STEPS = 4
MERGE_SAFE_FRONT_GAP = 24.0
MERGE_SAFE_REAR_GAP = 16.0
MERGE_OBSTACLE_BUFFER = 32.0
ROUNDABOUT_SUCCESS_LANE_PAIR = ('nxs', 'nxr')
ROUNDABOUT_EXIT_PROGRESS_THRESHOLD = 24.0
ROUNDABOUT_MIN_ACTIVE_TRAFFIC = 3


class GymLane:
    def __init__(self, dev=torch.device('cpu'), road_scenario='highway', traffic_level='standard'):
        self.state_dimension = 25
        self.action_num = 5
        self.dev = dev
        self.env = None
        self.video_root_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_logs_lane')
        self.video_folder = self.video_root_folder
        self.video_session_folder = None
        self.video_name_prefix = None
        self.video_path_hint = None
        self.video_run_kind = None
        self.road_scenario = None
        self.env_id = None
        self.scenario_supports_vehicles_count = False
        self.base_env_config = {}
        self.traffic_level = traffic_level if traffic_level in TRAFFIC_VEHICLE_COUNT else 'standard'
        self._ensure_video_root_scaffold()

        self.mode = None
        self.step_num = 0
        self.done_signal = 0
        self.state_original = None
        self.state_processed = None

        self.episode_return = 0.0
        self.episode_raw_return = 0.0
        self.episode_speed_sum = 0.0
        self.episode_speed_count = 0
        self.collision_count = 0
        self.lane_change_count = 0
        self.last_step_info = {}
        self.lane_change_action_ids = {0, 2}
        self.left_lane_action_id = 0
        self.right_lane_action_id = 2
        self.faster_action_id = 3
        self.slower_action_id = 4
        self.previous_action_index = None
        self.steps_since_lane_change = 1000
        self.target_step_bonus_awarded = False
        self.scenario_completed = False
        self.scenario_completion_awarded = False
        self.merge_mainline_step_count = 0
        self.highway_overtake_active = False
        self.highway_overtake_origin_lane_id = None
        self.highway_overtake_target_lane_id = None
        self.highway_overtake_target_vehicle = None
        self.highway_overtake_completed = False
        self.highway_overtake_completion_count = 0
        self.highway_overtake_stable_steps = 0
        self.highway_post_overtake_settle_steps = 0
        self.highway_recent_overtake_completion_steps = 0
        self.highway_lane_change_eval_steps_remaining = 0
        self.highway_lane_change_eval_origin_lane_id = None
        self.highway_lane_change_eval_target_lane_id = None
        self.highway_lane_change_eval_reference_front_gap = None
        self.highway_lane_change_eval_reference_speed = 0.0
        self.highway_lane_change_eval_reference_front_speed = None
        self.current_vehicle_target = self._resolve_vehicle_count(None)

        self.obs_min_seen = None
        self.obs_max_seen = None

        self.set_scenario(road_scenario)
        print_info('ACTION NUMBER: %6d' % self.action_num)
        print_info('STATE DIMENSION: %6d' % self.state_dimension)
        print_info('ROAD SCENARIO: %s (%s)' % (self.road_scenario, self.env_id))

    def set_scenario(self, road_scenario):
        if road_scenario not in SCENARIO_PRESETS:
            raise ValueError('Unsupported road_scenario: %s' % road_scenario)
        preset = SCENARIO_PRESETS[road_scenario]
        self.road_scenario = road_scenario
        self.env_id = preset['env_id']
        self.max_step_num = int(preset['max_step_num'])
        self.target_step_num = int(preset['target_step_num'])
        self.scenario_supports_vehicles_count = bool(preset.get('supports_vehicles_count', False))
        self.base_env_config = dict(preset.get('config', {}))
        self.current_vehicle_target = self._resolve_scenario_vehicle_target(None)

    def _video_scenario_dirname(self):
        return checkpoint_utils.scenario_dirname(self.road_scenario, self.traffic_level)

    def _video_run_kind_dirname(self, video_run_kind=None):
        normalized_kind = checkpoint_utils.normalize_run_kind(video_run_kind)
        if normalized_kind == 'baseline':
            return 'base'
        if normalized_kind == 'ours':
            return 'ours'
        return 'misc'

    def _ensure_video_root_scaffold(self):
        os.makedirs(self.video_root_folder, exist_ok=True)
        for run_kind_dirname in VIDEO_LAYOUT_RUN_KINDS:
            run_kind_folder = os.path.join(self.video_root_folder, run_kind_dirname)
            os.makedirs(run_kind_folder, exist_ok=True)
            for road_scenario, traffic_level in VIDEO_LAYOUT_SCENARIOS:
                scenario_dirname = checkpoint_utils.scenario_dirname(road_scenario, traffic_level)
                os.makedirs(os.path.join(run_kind_folder, scenario_dirname), exist_ok=True)

    def _sanitize_video_token(self, value):
        if value is None:
            return ''
        token = str(value).strip().lower()
        if not token:
            return ''
        clean_chars = []
        last_sep = False
        for char in token:
            if char.isalnum():
                clean_chars.append(char)
                last_sep = False
            else:
                if not last_sep:
                    clean_chars.append('-')
                    last_sep = True
        return ''.join(clean_chars).strip('-')

    def _build_video_recording_target(self, video_tag=None, video_run_kind=None):
        self._ensure_video_root_scaffold()
        scenario_dirname = self._video_scenario_dirname()
        run_kind_dirname = self._video_run_kind_dirname(video_run_kind)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        scenario_folder = os.path.join(self.video_root_folder, run_kind_dirname, scenario_dirname)
        session_folder = os.path.join(scenario_folder, timestamp)
        os.makedirs(session_folder, exist_ok=True)
        prefix_tokens = [
            self._sanitize_video_token(run_kind_dirname),
            self._sanitize_video_token(scenario_dirname),
            self._sanitize_video_token(self.mode or 'test'),
        ]
        extra_token = self._sanitize_video_token(video_tag)
        if extra_token:
            prefix_tokens.append(extra_token)
        prefix_tokens.append(timestamp)
        name_prefix = '_'.join(token for token in prefix_tokens if token)
        self.video_run_kind = run_kind_dirname
        self.video_folder = session_folder
        self.video_session_folder = session_folder
        self.video_name_prefix = name_prefix
        self.video_path_hint = os.path.join(session_folder, f'{name_prefix}-episode-*')
        return session_folder, name_prefix

    def _resolve_vehicle_count(self, vehicles_count):
        if vehicles_count is not None:
            return int(vehicles_count)
        return int(TRAFFIC_VEHICLE_COUNT.get(self.traffic_level, TRAFFIC_VEHICLE_COUNT['standard']))

    def _resolve_scenario_vehicle_target(self, vehicles_count):
        if self.road_scenario == 'highway':
            return self._resolve_vehicle_count(vehicles_count)
        if vehicles_count is not None:
            return max(4, int(vehicles_count))
        scenario_targets = SCENARIO_TRAFFIC_TARGETS.get(self.road_scenario)
        if scenario_targets is None:
            return self._resolve_vehicle_count(None)
        return int(scenario_targets.get(self.traffic_level, scenario_targets['standard']))

    def _configure_env(self, vehicles_count=None):
        config = dict(self.base_env_config)
        config['duration'] = self.max_step_num
        self.current_vehicle_target = self._resolve_scenario_vehicle_target(vehicles_count)
        if self.road_scenario == 'highway':
            traffic_profile = HIGHWAY_TRAFFIC_ENV_PROFILE.get(
                self.traffic_level,
                HIGHWAY_TRAFFIC_ENV_PROFILE['standard'],
            )
            config['vehicles_density'] = float(traffic_profile.get('vehicles_density', 1.0))
            config['ego_spacing'] = float(traffic_profile.get('ego_spacing', 2.0))
            if self.mode in ['train', 'val']:
                config['simulation_frequency'] = int(
                    HIGHWAY_TRAINING_SIMULATION_FREQUENCY.get(
                        self.traffic_level,
                        HIGHWAY_TRAINING_SIMULATION_FREQUENCY['standard'],
                    )
                )
        if self.scenario_supports_vehicles_count:
            config['vehicles_count'] = self._resolve_vehicle_count(vehicles_count)
        self.env.unwrapped.configure(config)

    def _refresh_action_metadata(self):
        action_type = getattr(self.env.unwrapped, 'action_type', None)
        if action_type is None or not hasattr(action_type, 'actions_indexes'):
            self.lane_change_action_ids = {0, 2}
            self.left_lane_action_id = 0
            self.right_lane_action_id = 2
            self.faster_action_id = 3
            self.slower_action_id = 4
            return
        action_indexes = action_type.actions_indexes
        lane_change_ids = []
        self.left_lane_action_id = action_indexes.get('LANE_LEFT', 0)
        self.right_lane_action_id = action_indexes.get('LANE_RIGHT', 2)
        self.faster_action_id = action_indexes.get('FASTER', 3)
        self.slower_action_id = action_indexes.get('SLOWER', 4)
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

    def state_to_tensor(self, state):
        self._record_observation_range(state)
        state_handle = np.asarray(state, dtype=np.float32).reshape(-1)
        state_handle = np.clip((state_handle + 1.0) / 2.0, 0.0, 1.0)
        return torch.as_tensor(state_handle, dtype=torch.float32, device=self.dev).unsqueeze(0)

    def _get_ego_vehicle(self):
        if self.env is None:
            return None
        return getattr(self.env.unwrapped, 'vehicle', None)

    def _refresh_observation_from_env(self, fallback_state=None):
        if self.env is None:
            return np.copy(fallback_state) if fallback_state is not None else None
        observation_type = getattr(self.env.unwrapped, 'observation_type', None)
        if observation_type is not None:
            return np.copy(observation_type.observe())
        return np.copy(fallback_state) if fallback_state is not None else None

    def _current_speed(self, info=None):
        if info is not None and 'speed' in info:
            return float(info.get('speed', 0.0))
        ego_vehicle = self._get_ego_vehicle()
        return float(getattr(ego_vehicle, 'speed', 0.0)) if ego_vehicle is not None else 0.0

    def _lane_longitudinal(self, vehicle, lane_index=None):
        if self.env is None or vehicle is None:
            return None
        active_lane_index = lane_index if lane_index is not None else getattr(vehicle, 'lane_index', None)
        if active_lane_index is None:
            return None
        try:
            lane = self.env.unwrapped.road.network.get_lane(active_lane_index)
            longitudinal, _ = lane.local_coordinates(vehicle.position)
            return float(longitudinal)
        except Exception:
            return None

    def _lane_neighbor_profile(self, vehicle=None, lane_index=None):
        empty_profile = {
            'front_gap': None,
            'front_speed': None,
            'rear_gap': None,
            'rear_speed': None,
        }
        if self.env is None:
            return empty_profile
        ego_vehicle = vehicle if vehicle is not None else self._get_ego_vehicle()
        active_lane_index = lane_index if lane_index is not None else getattr(ego_vehicle, 'lane_index', None)
        if ego_vehicle is None or active_lane_index is None:
            return empty_profile
        try:
            road = self.env.unwrapped.road
            lane = road.network.get_lane(active_lane_index)
            ego_s, _ = lane.local_coordinates(ego_vehicle.position)
            front_vehicle, rear_vehicle = road.neighbour_vehicles(ego_vehicle, lane_index=active_lane_index)
            front_gap = None
            front_speed = None
            rear_gap = None
            rear_speed = None
            if front_vehicle is not None:
                front_s, _ = lane.local_coordinates(front_vehicle.position)
                front_gap = max(0.0, float(front_s - ego_s))
                front_speed = float(getattr(front_vehicle, 'speed', 0.0))
            if rear_vehicle is not None:
                rear_s, _ = lane.local_coordinates(rear_vehicle.position)
                rear_gap = max(0.0, float(ego_s - rear_s))
                rear_speed = float(getattr(rear_vehicle, 'speed', 0.0))
            return {
                'front_gap': front_gap,
                'front_speed': front_speed,
                'rear_gap': rear_gap,
                'rear_speed': rear_speed,
            }
        except Exception:
            return empty_profile

    def _lane_neighbor_snapshot(self, vehicle=None, lane_index=None):
        empty_snapshot = {
            'front_vehicle': None,
            'rear_vehicle': None,
            'front_gap': None,
            'front_speed': None,
            'rear_gap': None,
            'rear_speed': None,
            'ego_longitudinal': None,
        }
        if self.env is None:
            return empty_snapshot
        ego_vehicle = vehicle if vehicle is not None else self._get_ego_vehicle()
        active_lane_index = lane_index if lane_index is not None else getattr(ego_vehicle, 'lane_index', None)
        if ego_vehicle is None or active_lane_index is None:
            return empty_snapshot
        try:
            road = self.env.unwrapped.road
            lane = road.network.get_lane(active_lane_index)
            ego_s, _ = lane.local_coordinates(ego_vehicle.position)
            front_vehicle, rear_vehicle = road.neighbour_vehicles(ego_vehicle, lane_index=active_lane_index)
            front_gap = None
            front_speed = None
            rear_gap = None
            rear_speed = None
            if front_vehicle is not None:
                front_s, _ = lane.local_coordinates(front_vehicle.position)
                front_gap = max(0.0, float(front_s - ego_s))
                front_speed = float(getattr(front_vehicle, 'speed', 0.0))
            if rear_vehicle is not None:
                rear_s, _ = lane.local_coordinates(rear_vehicle.position)
                rear_gap = max(0.0, float(ego_s - rear_s))
                rear_speed = float(getattr(rear_vehicle, 'speed', 0.0))
            return {
                'front_vehicle': front_vehicle,
                'rear_vehicle': rear_vehicle,
                'front_gap': front_gap,
                'front_speed': front_speed,
                'rear_gap': rear_gap,
                'rear_speed': rear_speed,
                'ego_longitudinal': float(ego_s),
            }
        except Exception:
            return empty_snapshot

    def _lane_pair_from_index(self, lane_index):
        if lane_index is None or len(lane_index) < 2:
            return None
        return tuple(lane_index[:2])

    def _lane_length(self, lane_index):
        if self.env is None or lane_index is None:
            return None
        try:
            lane = self.env.unwrapped.road.network.get_lane(lane_index)
            return float(getattr(lane, 'length', 0.0))
        except Exception:
            return None

    def _highway_lane_context(self, vehicle=None):
        empty_context = {
            'current_front_gap': None,
            'current_front_speed': None,
            'left_front_gap': None,
            'left_rear_gap': None,
            'left_lane_clear': False,
            'right_front_gap': None,
            'right_rear_gap': None,
            'right_lane_clear': False,
        }
        if self.env is None or self.road_scenario != 'highway':
            return empty_context
        ego_vehicle = vehicle if vehicle is not None else self._get_ego_vehicle()
        lane_index = getattr(ego_vehicle, 'lane_index', None)
        if ego_vehicle is None or lane_index is None or len(lane_index) < 3:
            return empty_context

        current_profile = self._lane_neighbor_profile(ego_vehicle, lane_index=lane_index)
        road_network = self.env.unwrapped.road.network
        left_lane_index = None
        right_lane_index = None
        for candidate_lane_index in road_network.side_lanes(lane_index):
            if len(candidate_lane_index) < 3:
                continue
            if candidate_lane_index[2] < lane_index[2]:
                if left_lane_index is None or candidate_lane_index[2] > left_lane_index[2]:
                    left_lane_index = candidate_lane_index
            elif candidate_lane_index[2] > lane_index[2]:
                if right_lane_index is None or candidate_lane_index[2] < right_lane_index[2]:
                    right_lane_index = candidate_lane_index

        left_profile = self._lane_neighbor_profile(ego_vehicle, lane_index=left_lane_index) if left_lane_index is not None else {}
        right_profile = self._lane_neighbor_profile(ego_vehicle, lane_index=right_lane_index) if right_lane_index is not None else {}
        left_lane_clear = (
            left_lane_index is not None and
            (left_profile.get('front_gap') is None or left_profile.get('front_gap') >= 24.0) and
            (left_profile.get('rear_gap') is None or left_profile.get('rear_gap') >= 14.0)
        )
        right_lane_clear = (
            right_lane_index is not None and
            (right_profile.get('front_gap') is None or right_profile.get('front_gap') >= 20.0) and
            (right_profile.get('rear_gap') is None or right_profile.get('rear_gap') >= 12.0)
        )
        return {
            'current_front_gap': current_profile.get('front_gap'),
            'current_front_speed': current_profile.get('front_speed'),
            'left_front_gap': left_profile.get('front_gap'),
            'left_rear_gap': left_profile.get('rear_gap'),
            'left_lane_clear': bool(left_lane_clear),
            'right_front_gap': right_profile.get('front_gap'),
            'right_rear_gap': right_profile.get('rear_gap'),
            'right_lane_clear': bool(right_lane_clear),
        }

    def _capture_highway_step_context(self):
        if self.env is None or self.road_scenario != 'highway':
            return {}
        ego_vehicle = self._get_ego_vehicle()
        lane_index = getattr(ego_vehicle, 'lane_index', None) if ego_vehicle is not None else None
        lane_id = int(lane_index[2]) if lane_index is not None and len(lane_index) >= 3 else None
        cruise_speed = self._target_highway_cruise_speed()
        context = self._highway_lane_context(ego_vehicle)
        neighbor_snapshot = self._lane_neighbor_snapshot(ego_vehicle, lane_index=lane_index)
        front_gap = context.get('current_front_gap')
        front_speed = context.get('current_front_speed')
        blocked_by_slower_front = (
            front_gap is not None and
            front_gap < 28.0 and
            front_speed is not None and
            front_speed < (cruise_speed - 1.0)
        )
        clear_road_ahead = front_gap is None or front_gap >= 36.0
        return {
            'lane_id': lane_id,
            'lane_index': lane_index,
            'speed': self._current_speed(),
            'context': context,
            'front_vehicle': neighbor_snapshot.get('front_vehicle'),
            'clear_road_ahead': bool(clear_road_ahead),
            'blocked_by_slower_front': bool(blocked_by_slower_front),
        }

    def _reset_highway_overtake_state(self, recent_completion_steps=0):
        self.highway_overtake_active = False
        self.highway_overtake_origin_lane_id = None
        self.highway_overtake_target_lane_id = None
        self.highway_overtake_target_vehicle = None
        self.highway_overtake_completed = False
        self.highway_overtake_stable_steps = 0
        self.highway_post_overtake_settle_steps = 0
        if recent_completion_steps > 0:
            self.highway_recent_overtake_completion_steps = max(
                self.highway_recent_overtake_completion_steps,
                int(recent_completion_steps),
            )

    def _reset_highway_lane_change_eval(self):
        self.highway_lane_change_eval_steps_remaining = 0
        self.highway_lane_change_eval_origin_lane_id = None
        self.highway_lane_change_eval_target_lane_id = None
        self.highway_lane_change_eval_reference_front_gap = None
        self.highway_lane_change_eval_reference_speed = 0.0
        self.highway_lane_change_eval_reference_front_speed = None

    def _start_highway_lane_change_eval(self, pre_step_highway_meta, action_index):
        if self.road_scenario != 'highway' or action_index != self.left_lane_action_id:
            self._reset_highway_lane_change_eval()
            return
        pre_context = pre_step_highway_meta.get('context', {})
        origin_lane_id = pre_step_highway_meta.get('lane_id')
        if origin_lane_id is None:
            self._reset_highway_lane_change_eval()
            return
        self.highway_lane_change_eval_steps_remaining = HIGHWAY_LANE_CHANGE_EVAL_WINDOW
        self.highway_lane_change_eval_origin_lane_id = int(origin_lane_id)
        self.highway_lane_change_eval_target_lane_id = int(origin_lane_id) - 1
        self.highway_lane_change_eval_reference_front_gap = pre_context.get('current_front_gap')
        self.highway_lane_change_eval_reference_speed = float(pre_step_highway_meta.get('speed', 0.0))
        self.highway_lane_change_eval_reference_front_speed = pre_context.get('current_front_speed')

    def _has_completed_highway_overtake(self, ego_vehicle, lane_index=None):
        if self.env is None or ego_vehicle is None or not self.highway_overtake_active:
            return False
        if self.highway_overtake_target_vehicle is None:
            return False
        ego_longitudinal = self._lane_longitudinal(ego_vehicle, lane_index=lane_index)
        target_lane_index = getattr(self.highway_overtake_target_vehicle, 'lane_index', None)
        target_longitudinal = self._lane_longitudinal(
            self.highway_overtake_target_vehicle,
            lane_index=target_lane_index,
        )
        if ego_longitudinal is None or target_longitudinal is None:
            return False
        return float(ego_longitudinal) >= float(target_longitudinal) + HIGHWAY_OVERTAKE_PASS_MARGIN

    def _merge_target_lane_index(self, lane_index=None):
        ego_vehicle = self._get_ego_vehicle()
        active_lane_index = lane_index if lane_index is not None else getattr(ego_vehicle, 'lane_index', None)
        if active_lane_index is None or len(active_lane_index) < 3:
            return None
        return MERGE_TARGET_LANE_BY_SEGMENT.get(tuple(active_lane_index[:3]))

    def _merge_on_mainline(self, lane_index=None):
        ego_vehicle = self._get_ego_vehicle()
        active_lane_index = lane_index if lane_index is not None else getattr(ego_vehicle, 'lane_index', None)
        if active_lane_index is None or len(active_lane_index) < 3:
            return False
        lane_pair = self._lane_pair_from_index(active_lane_index)
        return lane_pair in [('b', 'c'), ('c', 'd')] and int(active_lane_index[2]) in [0, 1]

    def _merge_route_progress(self, vehicle=None, lane_index=None):
        ego_vehicle = vehicle if vehicle is not None else self._get_ego_vehicle()
        active_lane_index = lane_index if lane_index is not None else getattr(ego_vehicle, 'lane_index', None)
        if ego_vehicle is None or active_lane_index is None or len(active_lane_index) < 3:
            return 0.0
        lane_pair = self._lane_pair_from_index(active_lane_index)
        if self._merge_on_mainline(active_lane_index):
            x_position = float(getattr(ego_vehicle, 'position', [0.0, 0.0])[0])
            return float(np.clip((x_position - 150.0) / max(1.0, MERGE_SUCCESS_X_THRESHOLD - 150.0), 0.0, 1.0))
        longitudinal = self._lane_longitudinal(ego_vehicle, lane_index=active_lane_index)
        if longitudinal is None:
            return 0.0
        if lane_pair == ('j', 'k'):
            route_position = longitudinal
        elif lane_pair == ('k', 'b'):
            route_position = 150.0 + longitudinal
        elif tuple(active_lane_index[:3]) == ('b', 'c', 2):
            route_position = 230.0 + longitudinal
        else:
            return 0.0
        return float(np.clip(route_position / 310.0, 0.0, 1.0))

    def _merge_obstacle_distance(self, vehicle=None, lane_index=None):
        ego_vehicle = vehicle if vehicle is not None else self._get_ego_vehicle()
        active_lane_index = lane_index if lane_index is not None else getattr(ego_vehicle, 'lane_index', None)
        if ego_vehicle is None or active_lane_index is None or tuple(active_lane_index[:3]) != ('b', 'c', 2):
            return None
        lane_length = self._lane_length(active_lane_index)
        longitudinal = self._lane_longitudinal(ego_vehicle, lane_index=active_lane_index)
        if lane_length is None or longitudinal is None:
            return None
        return max(0.0, float(lane_length - longitudinal))

    def _merge_gap_profile(self, vehicle=None, lane_index=None):
        empty_profile = {
            'front_gap': None,
            'rear_gap': None,
            'gap_score': 0.0,
            'unsafe_front': 0.0,
            'unsafe_rear': 0.0,
            'safe_window': False,
        }
        if self.env is None:
            return empty_profile
        ego_vehicle = vehicle if vehicle is not None else self._get_ego_vehicle()
        active_lane_index = lane_index if lane_index is not None else getattr(ego_vehicle, 'lane_index', None)
        target_lane_index = self._merge_target_lane_index(active_lane_index)
        if ego_vehicle is None or target_lane_index is None:
            return empty_profile
        try:
            road = self.env.unwrapped.road
            target_lane = road.network.get_lane(target_lane_index)
            ego_s, _ = target_lane.local_coordinates(ego_vehicle.position)
            front_vehicle, rear_vehicle = road.neighbour_vehicles(ego_vehicle, lane_index=target_lane_index)
            front_gap = None
            rear_gap = None
            if front_vehicle is not None:
                front_s, _ = target_lane.local_coordinates(front_vehicle.position)
                front_gap = float(front_s - ego_s)
            if rear_vehicle is not None:
                rear_s, _ = target_lane.local_coordinates(rear_vehicle.position)
                rear_gap = float(ego_s - rear_s)
        except Exception:
            return empty_profile
        front_score = 1.0 if front_gap is None else float(np.clip(front_gap / MERGE_SAFE_FRONT_GAP, 0.0, 1.0))
        rear_score = 1.0 if rear_gap is None else float(np.clip(rear_gap / MERGE_SAFE_REAR_GAP, 0.0, 1.0))
        safe_window = (front_gap is None or front_gap >= MERGE_SAFE_FRONT_GAP) and (
            rear_gap is None or rear_gap >= MERGE_SAFE_REAR_GAP
        )
        unsafe_front = 0.0 if front_gap is None else max(0.0, (10.0 - front_gap) / 10.0)
        unsafe_rear = 0.0 if rear_gap is None else max(0.0, (8.0 - rear_gap) / 8.0)
        return {
            'front_gap': front_gap,
            'rear_gap': rear_gap,
            'gap_score': 0.5 * (front_score + rear_score),
            'unsafe_front': unsafe_front,
            'unsafe_rear': unsafe_rear,
            'safe_window': bool(safe_window),
        }
    def _clear_ego_surroundings(self):
        """无论什么场景，保证小车出生时，方圆 20 米内干干净净，没有其他车"""
        if self.env is None:
            return
        base_env = self.env.unwrapped
        road = getattr(base_env, 'road', None)
        ego_vehicle = self._get_ego_vehicle()
        if road is None or ego_vehicle is None:
            return
            
        for v in list(getattr(road, 'vehicles', [])):
            if v is not ego_vehicle:
                # 凡是距离小车 20 米以内的车，全部删掉
                if np.linalg.norm(np.array(v.position) - np.array(ego_vehicle.position)) < 20.0:
                    road.vehicles.remove(v)
    def _reposition_merge_ego_vehicle(self):
        if self.env is None or self.road_scenario != 'merge':
            return
        base_env = self.env.unwrapped
        road = getattr(base_env, 'road', None)
        ego_vehicle = self._get_ego_vehicle()
        if road is None or ego_vehicle is None:
            return
        
        # 【核心修改2】：清空出生点周围 25 米的车辆，防止0步撞车
        ramp_lane = road.network.get_lane(MERGE_EGO_START_LANE)
        target_pos = ramp_lane.position(MERGE_EGO_START_LONGITUDINAL, 0.0)
        for v in list(getattr(road, 'vehicles', [])):
            if v is not ego_vehicle:
                if np.linalg.norm(np.array(v.position) - np.array(target_pos)) < 25.0:
                    road.vehicles.remove(v)
                    
        if ego_vehicle in getattr(road, 'vehicles', []):
            road.vehicles.remove(ego_vehicle)
        if hasattr(base_env, 'controlled_vehicles'):
            base_env.controlled_vehicles = [
                vehicle for vehicle in getattr(base_env, 'controlled_vehicles', []) if vehicle is not ego_vehicle
            ]
        vehicle_ctor = getattr(getattr(base_env, 'action_type', None), 'vehicle_class', None)
        if vehicle_ctor is None:
            vehicle_ctor = type(ego_vehicle)
        new_ego = vehicle_ctor(
            road,
            ramp_lane.position(MERGE_EGO_START_LONGITUDINAL, 0.0),
            speed=MERGE_EGO_START_SPEED,
        )
        if hasattr(new_ego, 'target_speed'):
            new_ego.target_speed = 30.0
        if hasattr(new_ego, 'plan_route_to'):
            try:
                new_ego.plan_route_to('d')
            except Exception:
                pass
        road.vehicles.append(new_ego)
        base_env.vehicle = new_ego
        if hasattr(base_env, 'controlled_vehicles'):
            base_env.controlled_vehicles = [new_ego]

    def _reposition_highway_ego_vehicle(self):
        if self.env is None or self.road_scenario != 'highway':
            return
        base_env = self.env.unwrapped
        road = getattr(base_env, 'road', None)
        ego_vehicle = self._get_ego_vehicle()
        lane_index = getattr(ego_vehicle, 'lane_index', None) if ego_vehicle is not None else None
        if road is None or ego_vehicle is None or lane_index is None or len(lane_index) < 3:
            return

        lane_count = int(base_env.config.get('lanes_count', 1))
        if lane_count < 2:
            return
        target_lane_id = 1 if lane_count > 2 else lane_count - 1
        if int(lane_index[2]) == target_lane_id:
            return

        target_lane_index = (lane_index[0], lane_index[1], target_lane_id)
        target_lane = road.network.get_lane(target_lane_index)
        target_longitudinal = self._lane_longitudinal(ego_vehicle, lane_index=lane_index)
        if target_longitudinal is None:
            return
        target_position = target_lane.position(target_longitudinal, 0.0)

        for vehicle in list(getattr(road, 'vehicles', [])):
            if vehicle is not ego_vehicle:
                if np.linalg.norm(np.array(vehicle.position) - np.array(target_position)) < 24.0:
                    road.vehicles.remove(vehicle)

        if ego_vehicle in getattr(road, 'vehicles', []):
            road.vehicles.remove(ego_vehicle)
        if hasattr(base_env, 'controlled_vehicles'):
            base_env.controlled_vehicles = [
                vehicle for vehicle in getattr(base_env, 'controlled_vehicles', []) if vehicle is not ego_vehicle
            ]
        vehicle_ctor = getattr(getattr(base_env, 'action_type', None), 'vehicle_class', None)
        if vehicle_ctor is None:
            vehicle_ctor = type(ego_vehicle)

        ego_speed = float(getattr(ego_vehicle, 'speed', self._target_highway_cruise_speed()))
        new_ego = vehicle_ctor(road, target_position, speed=ego_speed)
        if hasattr(new_ego, 'target_speed'):
            new_ego.target_speed = max(ego_speed, min(30.0, self._target_highway_cruise_speed() + 2.0))
        road.vehicles.append(new_ego)
        base_env.vehicle = new_ego
        if hasattr(base_env, 'controlled_vehicles'):
            base_env.controlled_vehicles = [new_ego]

    def _other_vehicle_count(self):
        ego_vehicle = self._get_ego_vehicle()
        if self.env is None:
            return 0
        return sum(1 for vehicle in getattr(self.env.unwrapped.road, 'vehicles', []) if vehicle is not ego_vehicle)

    def _count_other_vehicles_on_lane_pairs(self, lane_pairs):
        ego_vehicle = self._get_ego_vehicle()
        if self.env is None:
            return 0
        count = 0
        for vehicle in getattr(self.env.unwrapped.road, 'vehicles', []):
            if vehicle is ego_vehicle:
                continue
            lane_index = getattr(vehicle, 'lane_index', None)
            if lane_index is None or len(lane_index) < 2:
                continue
            if tuple(lane_index[:2]) in lane_pairs:
                count += 1
        return count

    def _get_other_vehicle_class(self):
        if self.env is None:
            return None
        other_type_path = self.env.unwrapped.config.get(
            'other_vehicles_type',
            'highway_env.vehicle.behavior.IDMVehicle',
        )
        return highway_utils.class_from_path(other_type_path)

    def _lane_is_clear(self, lane_index, longitudinal, min_gap):
        base_env = self.env.unwrapped
        lane = base_env.road.network.get_lane(lane_index)
        candidate_position = np.asarray(lane.position(longitudinal, 0.0), dtype=np.float32)
        for vehicle in list(getattr(base_env.road, 'vehicles', [])):
            vehicle_position = np.asarray(getattr(vehicle, 'position', candidate_position), dtype=np.float32)
            if np.linalg.norm(vehicle_position - candidate_position) < (min_gap * 0.8):
                return False
            if getattr(vehicle, 'lane_index', None) == lane_index:
                try:
                    vehicle_longitudinal, _ = lane.local_coordinates(vehicle.position)
                except Exception:
                    continue
                if abs(vehicle_longitudinal - longitudinal) < min_gap:
                    return False
        return True

    def _spawn_vehicle_on_lane(self, lane_index, longitudinal, speed, target_speed=None, destination=None, min_gap=18.0):
        if self.env is None:
            return False
        if not self._lane_is_clear(lane_index, longitudinal, min_gap):
            return False
        other_vehicle_class = self._get_other_vehicle_class()
        if other_vehicle_class is None:
            return False
        base_env = self.env.unwrapped
        vehicle = other_vehicle_class.make_on_lane(
            base_env.road,
            lane_index,
            longitudinal=float(longitudinal),
            speed=float(speed),
        )
        if target_speed is not None and hasattr(vehicle, 'target_speed'):
            vehicle.target_speed = float(target_speed)
        if destination is not None and hasattr(vehicle, 'plan_route_to'):
            try:
                vehicle.plan_route_to(destination)
            except Exception:
                pass
        if hasattr(vehicle, 'randomize_behavior'):
            vehicle.randomize_behavior()
        base_env.road.vehicles.append(vehicle)
        return True

    def _remove_vehicle(self, vehicle):
        if self.env is None or vehicle is None:
            return
        road = self.env.unwrapped.road
        road.vehicles = [candidate for candidate in road.vehicles if candidate is not vehicle]

    def _nearest_side_lanes(self, lane_index):
        left_lane_index = None
        right_lane_index = None
        if self.env is None or lane_index is None or len(lane_index) < 3:
            return left_lane_index, right_lane_index
        for candidate_lane_index in self.env.unwrapped.road.network.side_lanes(lane_index):
            if len(candidate_lane_index) < 3:
                continue
            if candidate_lane_index[2] < lane_index[2]:
                if left_lane_index is None or candidate_lane_index[2] > left_lane_index[2]:
                    left_lane_index = candidate_lane_index
            elif candidate_lane_index[2] > lane_index[2]:
                if right_lane_index is None or candidate_lane_index[2] < right_lane_index[2]:
                    right_lane_index = candidate_lane_index
        return left_lane_index, right_lane_index

    def _clear_highway_lane_window(self, lane_index, ego_longitudinal, rear_gap, front_gap):
        if self.env is None or lane_index is None:
            return
        lane = self.env.unwrapped.road.network.get_lane(lane_index)
        ego_vehicle = self._get_ego_vehicle()
        removable = []
        for vehicle in list(self.env.unwrapped.road.vehicles):
            if vehicle is ego_vehicle or getattr(vehicle, 'lane_index', None) != lane_index:
                continue
            try:
                vehicle_longitudinal, _ = lane.local_coordinates(vehicle.position)
            except Exception:
                continue
            relative_longitudinal = float(vehicle_longitudinal - ego_longitudinal)
            if -rear_gap <= relative_longitudinal <= front_gap:
                removable.append(vehicle)
        for vehicle in removable:
            self._remove_vehicle(vehicle)

    def _retune_highway_vehicle_speeds(self, ego_vehicle, ego_lane_index, ego_longitudinal):
        if self.env is None or ego_vehicle is None or ego_lane_index is None:
            return
        cruise_speed = self._target_highway_cruise_speed()
        for vehicle in list(self.env.unwrapped.road.vehicles):
            if vehicle is ego_vehicle:
                continue
            lane_index = getattr(vehicle, 'lane_index', None)
            if lane_index is None:
                continue
            vehicle_longitudinal = self._lane_longitudinal(vehicle, lane_index=lane_index)
            if vehicle_longitudinal is None:
                continue
            relative_longitudinal = vehicle_longitudinal - ego_longitudinal
            lane_delta = 0 if len(lane_index) < 3 else int(lane_index[2] - ego_lane_index[2])
            target_speed = cruise_speed + np.random.uniform(-2.5, 0.8)
            if lane_delta == 0 and relative_longitudinal > 0.0:
                if relative_longitudinal <= 38.0:
                    target_speed = cruise_speed - np.random.uniform(5.0, 7.5)
                elif relative_longitudinal <= 82.0:
                    target_speed = cruise_speed - np.random.uniform(3.0, 5.5)
                else:
                    target_speed = cruise_speed - np.random.uniform(1.0, 3.0)
            elif lane_delta < 0:
                if -12.0 <= relative_longitudinal <= 78.0:
                    target_speed = cruise_speed - np.random.uniform(0.2, 1.4)
                else:
                    target_speed = cruise_speed + np.random.uniform(-1.0, 0.5)
            elif lane_delta > 0:
                target_speed = cruise_speed - np.random.uniform(2.5, 4.8)
            target_speed = float(np.clip(target_speed, 17.0, 30.0))
            if hasattr(vehicle, 'target_speed'):
                vehicle.target_speed = target_speed
            if hasattr(vehicle, 'speed'):
                vehicle.speed = target_speed

    def _setup_highway_overtake_window(self):
        if self.env is None or self.road_scenario != 'highway':
            return
        ego_vehicle = self._get_ego_vehicle()
        ego_lane_index = getattr(ego_vehicle, 'lane_index', None) if ego_vehicle is not None else None
        if ego_vehicle is None or ego_lane_index is None:
            return
        ego_longitudinal = self._lane_longitudinal(ego_vehicle, lane_index=ego_lane_index)
        if ego_longitudinal is None:
            return
        left_lane_index, _right_lane_index = self._nearest_side_lanes(ego_lane_index)
        cruise_speed = self._target_highway_cruise_speed()
        ego_target_speed = min(30.0, cruise_speed + 2.0)
        if hasattr(ego_vehicle, 'target_speed'):
            ego_vehicle.target_speed = ego_target_speed
        if hasattr(ego_vehicle, 'speed'):
            ego_vehicle.speed = max(float(getattr(ego_vehicle, 'speed', ego_target_speed)), ego_target_speed)

        self._clear_highway_lane_window(ego_lane_index, ego_longitudinal, rear_gap=16.0, front_gap=72.0)
        if left_lane_index is not None:
            self._clear_highway_lane_window(left_lane_index, ego_longitudinal, rear_gap=32.0, front_gap=96.0)

        slow_front_speed = max(16.5, cruise_speed - 8.5)
        self._spawn_vehicle_on_lane(
            lane_index=ego_lane_index,
            longitudinal=ego_longitudinal + 28.0,
            speed=slow_front_speed,
            target_speed=slow_front_speed + 0.1,
            min_gap=18.0,
        )
        self._spawn_vehicle_on_lane(
            lane_index=ego_lane_index,
            longitudinal=ego_longitudinal + 58.0,
            speed=max(17.5, cruise_speed - 5.5),
            target_speed=max(18.0, cruise_speed - 4.5),
            min_gap=16.0,
        )
        if left_lane_index is not None:
            self._spawn_vehicle_on_lane(
                lane_index=left_lane_index,
                longitudinal=ego_longitudinal + 92.0,
                speed=min(29.0, cruise_speed - 0.2),
                target_speed=min(29.5, cruise_speed + 0.4),
                min_gap=20.0,
            )
        self._retune_highway_vehicle_speeds(ego_vehicle, ego_lane_index, ego_longitudinal)

    def _top_up_scenario_traffic(self):
        if self.env is None or self.road_scenario not in ['merge', 'roundabout']:
            return
        if self._other_vehicle_count() >= self.current_vehicle_target:
            return
        blueprint = MERGE_TRAFFIC_BLUEPRINT if self.road_scenario == 'merge' else ROUNDABOUT_TRAFFIC_BLUEPRINT
        # 【核心修改 4】将环岛的 NPC 生成安全距离从 16.0 拉大到 25.0 米，防止 NPC 连环车祸
        min_gap = 18.0 if self.road_scenario == 'merge' else 25.0
        for spec in blueprint:
            if self._other_vehicle_count() >= self.current_vehicle_target:
                break
            self._spawn_vehicle_on_lane(
                lane_index=spec['lane'],
                longitudinal=spec['longitudinal'],
                speed=spec['speed'],
                target_speed=spec.get('target_speed'),
                destination=spec.get('destination'),
                min_gap=min_gap,
            )

    def _scenario_completed_now(self, info=None):
        if info is not None and bool(info.get('crashed', False)):
            return False
        ego_vehicle = self._get_ego_vehicle()
        if ego_vehicle is None:
            return False
        lane_index = getattr(ego_vehicle, 'lane_index', None)
        lane_pair = self._lane_pair_from_index(lane_index)
        if self.road_scenario == 'merge':
            if not self._merge_on_mainline(lane_index):
                return False
            if self.merge_mainline_step_count < MERGE_STABLE_MAINLINE_STEPS:
                return False
            return float(getattr(ego_vehicle, 'position', [0.0, 0.0])[0]) >= MERGE_SUCCESS_X_THRESHOLD
        if self.road_scenario == 'roundabout':
            if lane_pair != ROUNDABOUT_SUCCESS_LANE_PAIR:
                return False
            if self._count_other_vehicles_on_lane_pairs(ROUNDABOUT_ACTIVE_TRAFFIC_LANE_PAIRS) < min(
                ROUNDABOUT_MIN_ACTIVE_TRAFFIC,
                self.current_vehicle_target,
            ):
                return False
            exit_progress = self._lane_longitudinal(ego_vehicle, lane_index=lane_index)
            return exit_progress is not None and exit_progress >= ROUNDABOUT_EXIT_PROGRESS_THRESHOLD
        return False

    def _target_highway_cruise_speed(self):
        return float(
            HIGHWAY_DESIRED_CRUISE_SPEED.get(
                self.traffic_level,
                HIGHWAY_DESIRED_CRUISE_SPEED['standard'],
            )
        )

    def _minimum_success_speed(self):
        if self.road_scenario == 'highway':
            return float(
                HIGHWAY_SUCCESS_MIN_SPEED.get(
                    self.traffic_level,
                    HIGHWAY_SUCCESS_MIN_SPEED['standard'],
                )
            )
        if self.road_scenario == 'merge':
            return 12.0
        if self.road_scenario == 'roundabout':
            return 7.0
        return 18.0

    def _should_abort_low_speed_highway_episode(self, info=None):
        if self.road_scenario != 'highway' or self.mode not in ['train', 'val']:
            return False
        if self.collision_count > 0 or self.step_num < 36:
            return False
        mean_speed = self.episode_mean_speed()
        current_speed = self._current_speed(info)
        min_success_speed = self._minimum_success_speed()
        if current_speed < (min_success_speed - 5.0) and mean_speed < (min_success_speed - 3.5):
            return True
        if self.step_num >= 60 and mean_speed < (min_success_speed - 2.5):
            return True
        if self.step_num >= 90 and mean_speed < (min_success_speed - 1.0):
            return True
        return False

    def _reset_episode_trackers(self, episode_seed=None):
        self.step_num = 0
        self.done_signal = 0
        self.episode_return = 0.0
        self.episode_raw_return = 0.0
        self.episode_speed_sum = 0.0
        self.episode_speed_count = 0
        self.collision_count = 0
        self.lane_change_count = 0
        self.last_step_info = {}
        self.previous_action_index = None
        self.steps_since_lane_change = 1000
        self.target_step_bonus_awarded = False
        self.scenario_completed = False
        self.scenario_completion_awarded = False
        self.merge_mainline_step_count = 0
        self.highway_overtake_completion_count = 0
        self._reset_highway_overtake_state()
        self.highway_recent_overtake_completion_steps = 0
        self._reset_highway_lane_change_eval()
        if episode_seed is None:
            self.state_original = self.env.reset()[0]
        else:
            self.state_original = self.env.reset(seed=int(episode_seed))[0]
            
        self._reposition_merge_ego_vehicle()
        self._reposition_highway_ego_vehicle()
        # 【核心修改 5】在生成交通车之前，先清空小车周围的障碍物！
        self._clear_ego_surroundings() 
        self._top_up_scenario_traffic()
        if self.road_scenario == 'highway':
            self._setup_highway_overtake_window()
        
        self.state_original = self._refresh_observation_from_env(self.state_original)
        self.state_processed = self.state_to_tensor(self.state_original)

    def init_train(self, vehicles_count=None, seed=None):
        self.mode = 'train'
        if self.env is not None:
            self.env.close()
        self.env = gym.make(self.env_id, render_mode=None)
        self._configure_env(vehicles_count=vehicles_count)
        self._refresh_action_metadata()
        self._reset_episode_trackers(episode_seed=seed)

    def init_val(self, vehicles_count=None, seed=None):
        self.mode = 'val'
        if self.env is not None:
            self.env.close()
        self.env = gym.make(self.env_id, render_mode=None)
        self._configure_env(vehicles_count=vehicles_count)
        self._refresh_action_metadata()
        self._reset_episode_trackers(episode_seed=seed)

    def init_test(self, variation_type='none', variation_param=0, record_video=False, vehicles_count=None, seed=None, video_tag=None, video_run_kind=None):
        del variation_type, variation_param
        self.mode = 'test'
        if self.env is not None:
            self.env.close()
        self.video_run_kind = self._video_run_kind_dirname(video_run_kind)
        self.video_folder = os.path.join(self.video_root_folder, self.video_run_kind, self._video_scenario_dirname())
        self.video_session_folder = None
        self.video_name_prefix = None
        self.video_path_hint = None
        render_mode = 'rgb_array' if record_video else None
        self.env = gym.make(self.env_id, render_mode=render_mode)
        self._configure_env(vehicles_count=vehicles_count)
        if record_video:
            video_folder, video_name_prefix = self._build_video_recording_target(
                video_tag=video_tag,
                video_run_kind=video_run_kind,
            )
            self.env = RecordVideo(
                self.env,
                video_folder=video_folder,
                name_prefix=video_name_prefix,
                episode_trigger=lambda episode_id: True,
            )
        self._refresh_action_metadata()
        self._reset_episode_trackers(episode_seed=seed)

    def get_observation(self):
        if self.state_processed is None and self.state_original is not None:
            self.state_processed = self.state_to_tensor(self.state_original)
        return self.state_processed

    def _apply_observation_noise(self, state_tensor, noise_level=0.0):
        if noise_level <= 0:
            return state_tensor
        noise = torch.randn_like(state_tensor) * noise_level
        return torch.clamp(state_tensor + noise, 0.0, 1.0)

    def get_train_observation(self, noise_level=0.0, **kwargs):
        del kwargs
        return self._apply_observation_noise(self.get_observation(), noise_level=noise_level)

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

    def _shape_reward(self, raw_reward, info, action_index, done, pre_step_highway_meta=None):
        del done
        reward_items = info.get('rewards', {})
        speed_reward = float(reward_items.get('high_speed_reward', 0.0))
        lane_reward = float(reward_items.get('right_lane_reward', 0.0))
        merging_speed_reward = float(reward_items.get('merging_speed_reward', 0.0))
        on_road_reward = float(reward_items.get('on_road_reward', 1.0))
        crashed = float(bool(info.get('crashed', False)))
        ego_speed_value = self._current_speed(info)
        lane_change_action = action_index in self.lane_change_action_ids
        scenario_completed = self._scenario_completed_now(info)

        highway_clear_road_bonus = 0.0
        highway_cruise_bonus = 0.0
        highway_overtake_bonus = 0.0
        highway_overtake_completion_bonus = 0.0
        highway_safe_follow_bonus = 0.0
        highway_post_overtake_stability_bonus = 0.0
        highway_return_after_overtake_bonus = 0.0
        highway_blocking_penalty = 0.0
        highway_missed_overtake_penalty = 0.0
        highway_unnecessary_lane_change_penalty = 0.0
        highway_ineffective_lane_change_penalty = 0.0
        highway_aborted_overtake_penalty = 0.0
        highway_tailgating_penalty = 0.0
        highway_under_cruise_penalty = 0.0
        highway_unnecessary_slow_penalty = 0.0

        if self.road_scenario == 'merge':
            lane_change_penalty_scale = 0.08
            repeated_lane_change_scale = 0.20
            zigzag_penalty_scale = 0.15
            steady_action_bonus_scale = 0.00
            survival_bonus_scale = 0.10
            speed_bonus_scale = 0.50
            lane_bonus_scale = 0.00
            raw_reward_scale = 0.05
            merging_speed_penalty_scale = 0.0
            offroad_penalty_scale = 0.50
            low_speed_threshold = 15.0
            low_speed_penalty_scale = 1.50
            stop_penalty = 3.00 if (self.step_num > 4 and ego_speed_value < 2.0) else 0.0
            crash_penalty = crashed * 50.0
            completion_bonus_value = 30.0
        elif self.road_scenario == 'roundabout':
            lane_change_penalty_scale = 0.20
            repeated_lane_change_scale = 0.40
            zigzag_penalty_scale = 0.25
            steady_action_bonus_scale = 0.05
            survival_bonus_scale = 0.20
            speed_bonus_scale = 0.60
            lane_bonus_scale = 0.00
            raw_reward_scale = 0.25
            merging_speed_penalty_scale = 0.0
            offroad_penalty_scale = 0.50
            low_speed_threshold = 8.0
            low_speed_penalty_scale = 0.0
            stop_penalty = 0.0
            crash_penalty = crashed * 50.0
            completion_bonus_value = 50.0
        else:
            cruise_speed = self._target_highway_cruise_speed()
            success_speed = self._minimum_success_speed()
            lane_change_penalty_scale = 0.10
            repeated_lane_change_scale = 0.85
            zigzag_penalty_scale = 1.40
            steady_action_bonus_scale = 0.0
            survival_bonus_scale = 0.01
            speed_bonus_scale = 1.55
            lane_bonus_scale = 0.0
            raw_reward_scale = 0.18
            merging_speed_penalty_scale = 0.0
            offroad_penalty_scale = 0.60
            low_speed_threshold = max(success_speed + 1.0, cruise_speed - 1.5)
            low_speed_penalty_scale = 4.80
            stop_penalty = 3.20 if (self.step_num > 4 and ego_speed_value < 5.0) else 0.0
            crash_penalty = crashed * 14.0
            completion_bonus_value = 0.0

            ego_vehicle = self._get_ego_vehicle()
            lane_index = getattr(ego_vehicle, 'lane_index', None) if ego_vehicle is not None else None
            lane_id = int(lane_index[2]) if lane_index is not None and len(lane_index) >= 3 else None
            highway_context = self._highway_lane_context(ego_vehicle)
            pre_step_highway_meta = pre_step_highway_meta or {}
            pre_context = pre_step_highway_meta.get('context', {})
            pre_lane_id = pre_step_highway_meta.get('lane_id')
            pre_clear_road_ahead = bool(pre_step_highway_meta.get('clear_road_ahead', False))
            pre_blocked_by_slower_front = bool(pre_step_highway_meta.get('blocked_by_slower_front', False))
            front_gap = highway_context['current_front_gap']
            front_speed = highway_context['current_front_speed']
            clear_road_ahead = front_gap is None or front_gap >= 36.0
            speed_floor = max(18.0, success_speed - 1.0)
            speed_window = max(1.0, cruise_speed - speed_floor)
            highway_cruise_bonus = 0.95 * float(
                np.clip((ego_speed_value - speed_floor) / speed_window, 0.0, 1.0)
            )
            if clear_road_ahead:
                clear_speed_ratio = float(np.clip((ego_speed_value - (cruise_speed - 6.0)) / 6.0, 0.0, 1.0))
                highway_clear_road_bonus = 0.78 * clear_speed_ratio
                if action_index == self.faster_action_id and ego_speed_value < cruise_speed + 0.5:
                    highway_clear_road_bonus += 0.55 * float(
                        np.clip((cruise_speed - ego_speed_value) / max(1.0, cruise_speed), 0.0, 1.0)
                    )
                if ego_speed_value < success_speed:
                    highway_under_cruise_penalty = 1.10 * float(
                        np.clip((success_speed - ego_speed_value) / max(1.0, success_speed), 0.0, 1.0)
                    )
                if action_index == self.slower_action_id and ego_speed_value >= (success_speed - 0.5):
                    highway_unnecessary_slow_penalty = 0.90

            blocked_by_slower_front = (
                front_gap is not None and
                front_gap < 28.0 and
                front_speed is not None and
                front_speed < (cruise_speed - 1.0)
            )
            pre_front_gap = pre_context.get('current_front_gap')
            pre_block_ratio = 0.0
            if pre_front_gap is not None:
                pre_block_ratio = float(np.clip((28.0 - pre_front_gap) / 18.0, 0.0, 1.0))
            justified_left_overtake = False
            justified_right_settle = False
            recent_completion_started_this_step = False
            if blocked_by_slower_front:
                block_ratio = float(np.clip((28.0 - front_gap) / 18.0, 0.0, 1.0))
                if front_gap < 12.0:
                    highway_tailgating_penalty = 1.10 * float(np.clip((12.0 - front_gap) / 12.0, 0.0, 1.0))
                if highway_context['left_lane_clear']:
                    if action_index == self.left_lane_action_id:
                        highway_overtake_bonus = 2.30 * block_ratio
                    elif action_index == self.slower_action_id and front_gap > 14.0:
                        highway_missed_overtake_penalty = 1.20 * block_ratio
                    elif action_index != self.faster_action_id:
                        highway_blocking_penalty = 0.95 * block_ratio
                    else:
                        highway_blocking_penalty = 0.80 * block_ratio
                elif action_index == self.slower_action_id and front_gap > 18.0:
                    highway_unnecessary_slow_penalty = max(highway_unnecessary_slow_penalty, 0.30 * block_ratio)
                elif action_index == self.slower_action_id and front_gap <= 18.0:
                    highway_safe_follow_bonus = 0.25 * block_ratio
                if ego_speed_value < success_speed - 1.0:
                    highway_under_cruise_penalty *= 0.40

            if lane_change_action:
                justified_left_overtake = (
                    action_index == self.left_lane_action_id and
                    pre_blocked_by_slower_front and
                    bool(pre_context.get('left_lane_clear', False)) and
                    pre_lane_id is not None
                )
                justified_right_settle = (
                    action_index == self.right_lane_action_id and
                    self.highway_overtake_active and
                    self.highway_overtake_completed and
                    bool(pre_context.get('right_lane_clear', False)) and
                    ego_speed_value >= success_speed and
                    pre_lane_id is not None and
                    self.highway_overtake_target_lane_id is not None and
                    pre_lane_id == self.highway_overtake_target_lane_id
                )
                if justified_left_overtake:
                    highway_overtake_bonus = max(highway_overtake_bonus, 2.50 * pre_block_ratio)
                    self.highway_overtake_active = True
                    self.highway_overtake_origin_lane_id = pre_lane_id
                    self.highway_overtake_target_lane_id = pre_lane_id - 1 if pre_lane_id is not None else None
                    self.highway_overtake_target_vehicle = pre_step_highway_meta.get('front_vehicle')
                    self.highway_overtake_completed = False
                    self.highway_overtake_stable_steps = 0
                    self.highway_recent_overtake_completion_steps = 0
                    self.highway_post_overtake_settle_steps = 0
                    self._start_highway_lane_change_eval(pre_step_highway_meta, action_index)
                elif justified_right_settle:
                    highway_return_after_overtake_bonus = 1.60
                    recent_completion_started_this_step = True
                    self._reset_highway_overtake_state(
                        recent_completion_steps=HIGHWAY_POST_OVERTAKE_SETTLE_STEPS,
                    )
                    self._reset_highway_lane_change_eval()
                else:
                    highway_unnecessary_lane_change_penalty = 1.30
                    if pre_clear_road_ahead:
                        highway_unnecessary_lane_change_penalty = 1.80
                    elif action_index == self.right_lane_action_id and self.highway_overtake_active:
                        highway_unnecessary_lane_change_penalty = 1.60
                    elif not pre_blocked_by_slower_front:
                        highway_unnecessary_lane_change_penalty = 1.45
                    self._reset_highway_lane_change_eval()

            if self.highway_overtake_active:
                on_target_lane = lane_id is not None and lane_id == self.highway_overtake_target_lane_id
                back_on_origin_lane = (
                    lane_id is not None and
                    self.highway_overtake_origin_lane_id is not None and
                    lane_id == self.highway_overtake_origin_lane_id
                )
                if on_target_lane and not self.highway_overtake_completed:
                    if not lane_change_action and action_index == self.faster_action_id and ego_speed_value < min(30.0, cruise_speed + 1.2):
                        highway_overtake_bonus += 0.45
                    elif not lane_change_action and ego_speed_value < cruise_speed - 0.5:
                        highway_under_cruise_penalty += 0.35
                    if self._has_completed_highway_overtake(ego_vehicle, lane_index=lane_index):
                        self.highway_overtake_completed = True
                        self.highway_overtake_completion_count += 1
                        self.highway_overtake_stable_steps = 0
                        self.highway_post_overtake_settle_steps = 0
                        highway_overtake_completion_bonus = 2.20
                if on_target_lane and self.highway_overtake_completed and not lane_change_action and clear_road_ahead and ego_speed_value >= success_speed:
                    self.highway_overtake_stable_steps += 1
                    highway_post_overtake_stability_bonus = 0.30 + 0.18 * min(self.highway_overtake_stable_steps, 4)
                elif back_on_origin_lane and not lane_change_action and self.highway_lane_change_eval_steps_remaining <= 0:
                    if self.highway_overtake_completed and clear_road_ahead and ego_speed_value >= success_speed:
                        highway_return_after_overtake_bonus = max(highway_return_after_overtake_bonus, 0.55)
                        recent_completion_started_this_step = True
                        self._reset_highway_overtake_state(
                            recent_completion_steps=HIGHWAY_POST_OVERTAKE_SETTLE_STEPS,
                        )
                    else:
                        highway_aborted_overtake_penalty = 1.80
                        self._reset_highway_overtake_state()
                        self._reset_highway_lane_change_eval()
                elif lane_change_action and not justified_left_overtake and not justified_right_settle:
                    self.highway_overtake_stable_steps = 0
                elif not clear_road_ahead or ego_speed_value < success_speed - 1.0:
                    self.highway_overtake_stable_steps = 0

            if self.highway_lane_change_eval_steps_remaining > 0 and not lane_change_action:
                target_lane_reached = (
                    lane_id is not None and
                    lane_id == self.highway_lane_change_eval_target_lane_id
                )
                reference_front_gap = self.highway_lane_change_eval_reference_front_gap
                reference_front_speed = self.highway_lane_change_eval_reference_front_speed
                if reference_front_gap is None:
                    front_gap_gain = HIGHWAY_MIN_EFFECTIVE_FRONT_GAP_GAIN if clear_road_ahead else 0.0
                elif front_gap is None:
                    front_gap_gain = HIGHWAY_MIN_EFFECTIVE_FRONT_GAP_GAIN + 1.0
                else:
                    front_gap_gain = float(front_gap - reference_front_gap)
                front_speed_gain = 0.0
                if reference_front_speed is not None and front_speed is not None:
                    front_speed_gain = float(front_speed - reference_front_speed)
                speed_gain = float(ego_speed_value - self.highway_lane_change_eval_reference_speed)
                effective_lane_change = target_lane_reached and (
                    self.highway_overtake_completed or
                    clear_road_ahead or
                    front_gap_gain >= HIGHWAY_MIN_EFFECTIVE_FRONT_GAP_GAIN or
                    speed_gain >= HIGHWAY_MIN_EFFECTIVE_SPEED_GAIN or
                    front_speed_gain >= HIGHWAY_MIN_EFFECTIVE_FRONT_SPEED_GAIN
                )
                if effective_lane_change:
                    self._reset_highway_lane_change_eval()
                else:
                    self.highway_lane_change_eval_steps_remaining -= 1
                    if self.highway_lane_change_eval_steps_remaining <= 0:
                        highway_ineffective_lane_change_penalty = 2.00
                        self._reset_highway_lane_change_eval()

            if self.highway_recent_overtake_completion_steps > 0 and not recent_completion_started_this_step:
                if not lane_change_action and clear_road_ahead and ego_speed_value >= success_speed:
                    self.highway_post_overtake_settle_steps += 1
                    highway_post_overtake_stability_bonus += 0.20 + 0.08 * min(
                        self.highway_post_overtake_settle_steps,
                        3,
                    )
                elif lane_change_action:
                    highway_unnecessary_lane_change_penalty = max(highway_unnecessary_lane_change_penalty, 1.00)
                    self.highway_post_overtake_settle_steps = 0
                else:
                    self.highway_post_overtake_settle_steps = 0
                self.highway_recent_overtake_completion_steps = max(0, self.highway_recent_overtake_completion_steps - 1)

        lane_change_penalty = lane_change_penalty_scale if lane_change_action else 0.0
        repeated_lane_change_penalty = 0.0
        if lane_change_action and self.steps_since_lane_change < 6:
            repeated_lane_change_penalty = repeated_lane_change_scale * (6 - self.steps_since_lane_change) / 6.0
        zigzag_penalty = zigzag_penalty_scale if (
            self.previous_action_index in self.lane_change_action_ids and
            lane_change_action and
            self.previous_action_index != action_index
        ) else 0.0
        steady_action_bonus = steady_action_bonus_scale if not lane_change_action else 0.0
        survival_bonus = survival_bonus_scale if not crashed else 0.0
        merge_mainline_bonus = 0.0
        merge_progress_bonus = 0.0
        merge_window_bonus = 0.0
        merge_commit_bonus = 0.0
        merge_wait_penalty = 0.0
        merge_deadline_penalty = 0.0
        merge_gap_penalty = 0.0

        if self.road_scenario == 'merge' and self.merge_mainline_step_count > 0 and not crashed:
            speed_factor = min(1.0, max(0.0, (ego_speed_value - MERGE_MAINLINE_MIN_SPEED) / 10.0))
            merge_mainline_bonus = 1.50 * (min(self.merge_mainline_step_count, MERGE_STABLE_MAINLINE_STEPS) / MERGE_STABLE_MAINLINE_STEPS) * speed_factor

        if self.road_scenario == 'merge':
            ego_vehicle = self._get_ego_vehicle()
            lane_index = getattr(ego_vehicle, 'lane_index', None) if ego_vehicle is not None else None
            lane_pair = self._lane_pair_from_index(lane_index)
            on_merge_lane = tuple(lane_index[:3]) == ('b', 'c', 2) if lane_index is not None and len(lane_index) >= 3 else False
            on_mainline = self._merge_on_mainline(lane_index)
            route_progress = self._merge_route_progress(ego_vehicle, lane_index)

            if on_mainline:
                merge_progress_bonus = 0.80 * route_progress
            elif on_merge_lane or lane_pair in MERGE_RAMP_LANE_PAIRS:
                merge_progress_bonus = 0.0
                survival_bonus = 0.0

            gap_profile = self._merge_gap_profile(ego_vehicle, lane_index)

            if on_merge_lane or lane_pair in MERGE_RAMP_LANE_PAIRS:
                merge_wait_penalty = 1.00

                if on_merge_lane:
                    merge_window_bonus = 0.55 * gap_profile['gap_score']
                    if gap_profile['safe_window'] and action_index != self.left_lane_action_id:
                        merge_wait_penalty += 4.00
                    merge_commit_bonus = 15.0 if (gap_profile['safe_window'] and action_index == self.left_lane_action_id) else 0.0

            elif on_mainline:
                merge_window_bonus = 0.10 * gap_profile['gap_score']

            obstacle_distance = self._merge_obstacle_distance(ego_vehicle, lane_index)
            if on_merge_lane and obstacle_distance is not None:
                merge_deadline_penalty = 4.00 * max(0.0, (120.0 - obstacle_distance) / 120.0)

        low_speed_penalty = 0.0
        if self.step_num > 2 and ego_speed_value < low_speed_threshold:
            low_speed_penalty = low_speed_penalty_scale * (low_speed_threshold - ego_speed_value) / max(1.0, low_speed_threshold)
        merging_speed_penalty = merging_speed_penalty_scale * max(0.0, merging_speed_reward)
        completion_bonus = completion_bonus_value if (scenario_completed and not crashed and not self.scenario_completion_awarded) else 0.0

        shaped_reward = (
            survival_bonus
            + steady_action_bonus
            + merge_mainline_bonus
            + merge_progress_bonus
            + merge_window_bonus
            + merge_commit_bonus
            + highway_clear_road_bonus
            + highway_cruise_bonus
            + highway_overtake_bonus
            + highway_overtake_completion_bonus
            + highway_safe_follow_bonus
            + highway_post_overtake_stability_bonus
            + highway_return_after_overtake_bonus
            + speed_bonus_scale * speed_reward
            + lane_bonus_scale * lane_reward
            + raw_reward_scale * float(raw_reward)
            + completion_bonus
            - offroad_penalty_scale * (1.0 - on_road_reward)
            - lane_change_penalty
            - repeated_lane_change_penalty
            - zigzag_penalty
            - merging_speed_penalty
            - low_speed_penalty
            - stop_penalty
            - merge_wait_penalty
            - merge_deadline_penalty
            - merge_gap_penalty
            - highway_blocking_penalty
            - highway_missed_overtake_penalty
            - highway_unnecessary_lane_change_penalty
            - highway_ineffective_lane_change_penalty
            - highway_aborted_overtake_penalty
            - highway_tailgating_penalty
            - highway_under_cruise_penalty
            - highway_unnecessary_slow_penalty
            - crash_penalty
        )

        info['raw_reward'] = float(raw_reward)
        info['shaped_reward'] = float(shaped_reward)
        info['scenario_completed'] = bool(scenario_completed)
        info['reward_breakdown'] = {
            'survival_bonus': float(survival_bonus),
            'steady_action_bonus': float(steady_action_bonus),
            'merge_mainline_bonus': float(merge_mainline_bonus),
            'merge_progress_bonus': float(merge_progress_bonus),
            'merge_window_bonus': float(merge_window_bonus),
            'merge_commit_bonus': float(merge_commit_bonus),
            'merge_wait_penalty': float(merge_wait_penalty),
            'highway_clear_road_bonus': float(highway_clear_road_bonus),
            'highway_cruise_bonus': float(highway_cruise_bonus),
            'highway_overtake_bonus': float(highway_overtake_bonus),
            'highway_overtake_completion_bonus': float(highway_overtake_completion_bonus),
            'highway_safe_follow_bonus': float(highway_safe_follow_bonus),
            'highway_post_overtake_stability_bonus': float(highway_post_overtake_stability_bonus),
            'highway_return_after_overtake_bonus': float(highway_return_after_overtake_bonus),
            'speed_bonus': float(speed_bonus_scale * speed_reward),
            'lane_bonus': float(lane_bonus_scale * lane_reward),
            'base_reward_bonus': float(raw_reward_scale * float(raw_reward)),
            'lane_change_penalty': float(lane_change_penalty),
            'repeated_lane_change_penalty': float(repeated_lane_change_penalty),
            'zigzag_penalty': float(zigzag_penalty),
            'merging_speed_penalty': float(merging_speed_penalty),
            'low_speed_penalty': float(low_speed_penalty),
            'stop_penalty': float(stop_penalty),
            'merge_deadline_penalty': float(merge_deadline_penalty),
            'merge_gap_penalty': float(merge_gap_penalty),
            'highway_blocking_penalty': float(highway_blocking_penalty),
            'highway_missed_overtake_penalty': float(highway_missed_overtake_penalty),
            'highway_unnecessary_lane_change_penalty': float(highway_unnecessary_lane_change_penalty),
            'highway_ineffective_lane_change_penalty': float(highway_ineffective_lane_change_penalty),
            'highway_aborted_overtake_penalty': float(highway_aborted_overtake_penalty),
            'highway_tailgating_penalty': float(highway_tailgating_penalty),
            'highway_under_cruise_penalty': float(highway_under_cruise_penalty),
            'highway_unnecessary_slow_penalty': float(highway_unnecessary_slow_penalty),
            'offroad_penalty': float(offroad_penalty_scale * (1.0 - on_road_reward)),
            'crash_penalty': float(crash_penalty),
            'completion_bonus': float(completion_bonus),
        }
        return float(shaped_reward)

    def episode_mean_speed(self):
        if self.episode_speed_count <= 0:
            return 0.0
        return float(self.episode_speed_sum / self.episode_speed_count)

    def episode_success(self):
        if self.collision_count > 0:
            return 0
        if self.road_scenario in ['merge', 'roundabout']:
            return int(self.scenario_completed and self.episode_mean_speed() >= self._minimum_success_speed())
        if self.road_scenario == 'highway':
            return int(
                self.highway_overtake_completion_count > 0 and
                self.step_num >= self.target_step_num and
                self.episode_mean_speed() >= self._minimum_success_speed()
            )
        return int(self.step_num >= self.target_step_num and self.episode_mean_speed() >= self._minimum_success_speed())

    def make_action(self, action):
        action_index = int(torch.argmax(action).item())
        show_live_progress = (self.mode == 'test')
        pre_step_highway_meta = self._capture_highway_step_context() if self.road_scenario == 'highway' else {}

        if show_live_progress and self.step_num % 10 == 0:
            print(f"\r🚗 正在马路上飞驰... 当前回合已开 {self.step_num} 步", end='', flush=True)


        next_state, reward_value, terminated, truncated, info = self.env.step(action_index)
        replenish_interval = SCENARIO_REPLENISH_INTERVAL.get(self.road_scenario)
        traffic_replenished = False
        if replenish_interval is not None and (self.step_num + 1) % replenish_interval == 0:
            self._top_up_scenario_traffic()
            traffic_replenished = True
        done = terminated or truncated

        if traffic_replenished:
            self.state_original = self._refresh_observation_from_env(next_state)
        else:
            self.state_original = np.array(next_state, copy=True)
        self.state_processed = self.state_to_tensor(self.state_original)
        self.step_num += 1
        self.episode_raw_return += float(reward_value)
        self.episode_speed_sum += self._current_speed(info)
        self.episode_speed_count += 1

        info_action = info.get('action', action_index)
        if info_action in self.lane_change_action_ids:
            self.lane_change_count += 1
        if bool(info.get('crashed', False)):
            self.collision_count += 1
        if self.road_scenario == 'merge':
            ego_vehicle = self._get_ego_vehicle()
            lane_index = getattr(ego_vehicle, 'lane_index', None) if ego_vehicle is not None else None
            on_stable_mainline = (
                self._merge_on_mainline(lane_index) and
                not bool(info.get('crashed', False)) and
                self._current_speed(info) >= MERGE_MAINLINE_MIN_SPEED
            )
            if on_stable_mainline:
                self.merge_mainline_step_count += 1
            else:
                self.merge_mainline_step_count = 0
        scenario_completed = self._scenario_completed_now(info)
        if scenario_completed and not bool(info.get('crashed', False)):
            self.scenario_completed = True

        shaped_reward = self._shape_reward(
            reward_value,
            info,
            info_action,
            done,
            pre_step_highway_meta=pre_step_highway_meta,
        )
        reward_breakdown = info.get('reward_breakdown', {})
        reward_breakdown['highway_slow_abort_penalty'] = 0.0
        highway_slow_abort = self._should_abort_low_speed_highway_episode(info)
        if highway_slow_abort:
            slow_abort_penalty = 8.0
            shaped_reward -= slow_abort_penalty
            reward_breakdown['highway_slow_abort_penalty'] = float(slow_abort_penalty)
            info['highway_slow_abort'] = True
            info['shaped_reward'] = float(shaped_reward)
        if scenario_completed and not bool(info.get('crashed', False)):
            self.scenario_completion_awarded = True
        self.episode_return += shaped_reward
        self.last_step_info = info

        done = done or scenario_completed or highway_slow_abort
        if done or self.step_num >= self.max_step_num:
            self.done_signal = 1
            if show_live_progress:
                print(f"\r🚗 💥 回合结束！最终活了: {self.step_num:2d} 步 | 步奖励: {shaped_reward:.2f} | 累计回报: {self.episode_return:.2f} | 原始回报: {self.episode_raw_return:.2f}    ")
        else:
            self.done_signal = 0

        reward_tensor = torch.tensor([shaped_reward], dtype=torch.float32, device=self.dev)
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

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


if __name__ == '__main__':
    env = GymLane(dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print('\033[91mFINISH: env_lane\033[0m')
