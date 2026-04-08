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
            'reward_speed_range': [20, 30],
            'collision_reward': -1.0,
            'lane_change_reward': -0.05,
        },
        'supports_vehicles_count': True,
    },
    'merge': {
        'env_id': 'merge-v0',
        'max_step_num':100,
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
MERGE_SUCCESS_X_THRESHOLD = 1500.0
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
        self.previous_action_index = None
        self.steps_since_lane_change = 1000
        self.target_step_bonus_awarded = False
        self.scenario_completed = False
        self.scenario_completion_awarded = False
        self.merge_mainline_step_count = 0
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
        if self.scenario_supports_vehicles_count:
            config['vehicles_count'] = self._resolve_vehicle_count(vehicles_count)
        self.env.unwrapped.configure(config)

    def _refresh_action_metadata(self):
        action_type = getattr(self.env.unwrapped, 'action_type', None)
        if action_type is None or not hasattr(action_type, 'actions_indexes'):
            self.lane_change_action_ids = {0, 2}
            self.left_lane_action_id = 0
            self.right_lane_action_id = 2
            return
        action_indexes = action_type.actions_indexes
        lane_change_ids = []
        self.left_lane_action_id = action_indexes.get('LANE_LEFT', 0)
        self.right_lane_action_id = action_indexes.get('LANE_RIGHT', 2)
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
        state_handle = np.copy(state).flatten()
        state_handle = np.clip((state_handle + 1.0) / 2.0, 0.0, 1.0)
        state_tensor = torch.FloatTensor(np.expand_dims(state_handle, axis=0)).to(self.dev)
        return state_tensor

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

    def _top_up_scenario_traffic(self):
        if self.env is None or self.road_scenario not in ['merge', 'roundabout']:
            return
        if self._other_vehicle_count() >= self.current_vehicle_target:
            return
        blueprint = MERGE_TRAFFIC_BLUEPRINT if self.road_scenario == 'merge' else ROUNDABOUT_TRAFFIC_BLUEPRINT
        min_gap = 18.0 if self.road_scenario == 'merge' else 16.0
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

    def _minimum_success_speed(self):
        if self.road_scenario == 'merge':
            return 12.0
        if self.road_scenario == 'roundabout':
            return 7.0
        return 18.0

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
        if episode_seed is None:
            self.state_original = self.env.reset()[0]
        else:
            self.state_original = self.env.reset(seed=int(episode_seed))[0]
        self._reposition_merge_ego_vehicle()
        self._top_up_scenario_traffic()
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

        # 【核心修改3】：重置并线场景的奖励权重，治好懒癌并加重碰撞威慑
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
            low_speed_penalty_scale = 2.20
            stop_penalty = 2.00 if (self.step_num > 2 and ego_speed_value < 1.5) else 0.0
            crash_penalty = crashed * 12.0
            completion_bonus_value = 10.0
        else:
            lane_change_penalty_scale = 0.75
            repeated_lane_change_scale = 0.90
            zigzag_penalty_scale = 0.25
            steady_action_bonus_scale = 0.20
            survival_bonus_scale = 0.25
            speed_bonus_scale = 0.60
            lane_bonus_scale = 0.10
            raw_reward_scale = 0.20
            merging_speed_penalty_scale = 0.0
            offroad_penalty_scale = 0.35
            low_speed_threshold = 18.0
            low_speed_penalty_scale = 2.20
            stop_penalty = 1.80 if (self.step_num > 5 and ego_speed_value < 3.0) else 0.0
            crash_penalty = crashed * 8.0
            completion_bonus_value = 0.0

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
            
            # 【核心修改4】：重构匝道博弈逻辑，建立“烫脚底板”的激励系统
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
        return int(self.step_num >= self.target_step_num and self.episode_mean_speed() >= self._minimum_success_speed())

    def make_action(self, action):
        action_index = int(torch.argmax(action).item())

        if self.step_num % 10 == 0:
            print(f"\r🚗 正在马路上飞驰... 当前回合已开 {self.step_num} 步", end='', flush=True)

        next_state, reward_value, terminated, truncated, info = self.env.step(action_index)
        replenish_interval = SCENARIO_REPLENISH_INTERVAL.get(self.road_scenario)
        if replenish_interval is not None and (self.step_num + 1) % replenish_interval == 0:
            self._top_up_scenario_traffic()
        done = terminated or truncated

        self.state_original = self._refresh_observation_from_env(next_state)
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

        shaped_reward = self._shape_reward(reward_value, info, info_action, done)
        if scenario_completed and not bool(info.get('crashed', False)):
            self.scenario_completion_awarded = True
        self.episode_return += shaped_reward
        self.last_step_info = info

        done = done or scenario_completed
        if done or self.step_num >= self.max_step_num:
            self.done_signal = 1
            # 【修改这里】：强制换行，并打印真实的 self.step_num
            print(f"\r🚗 💥 回合结束！最终活了: {self.step_num:2d} 步 | 步奖励: {shaped_reward:.2f} | 累计回报: {self.episode_return:.2f} | 原始回报: {self.episode_raw_return:.2f}    ")
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

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


if __name__ == '__main__':
    env = GymLane(dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print('\033[91mFINISH: env_lane\033[0m')