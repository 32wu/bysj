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
        'max_step_num': 120,
        'target_step_num': 120,
        'config': {
            'duration': 120,
            'lanes_count': 4,
            'vehicles_count': 60,
            'controlled_vehicles': 1,
            'initial_lane_id': 2,
            'ego_spacing': 1.55,
            'vehicles_density': 1.20,
            'high_speed_reward': 0.4,
            'reward_speed_range': [20, 30],
            'collision_reward': -1.0,
            'lane_change_reward': 0.0,
            'right_lane_reward': 0.1,
            'normalize_reward': True,
            'offroad_terminal': False,
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
    'light': 40,
    'standard': 48,
    'dense': 72,
}

HIGHWAY_TRAFFIC_ENV_PROFILE = {
    'light': {
        'vehicles_density': 0.90,
        'ego_spacing': 2.4,
    },
    'standard': {
        'vehicles_density': 0.95,
        'ego_spacing': 2.30,
    },
    'dense': {
        'vehicles_density': 1.35,
        'ego_spacing': 1.50,
    },
}

HIGHWAY_LOCAL_CLEAR_ZONE_RADIUS = {
    'light': 18.0,
    'standard': 14.0,
    'dense': 15.0,
}

HIGHWAY_REWARD_SPEED_RANGE = {
    'light': [20.0, 30.0],
    'standard': [20.0, 30.0],
    'dense': [19.0, 29.0],
}

HIGHWAY_DESIRED_CRUISE_SPEED = {
    'light': 29.0,
    'standard': 29.4,
    'dense': 28.2,
}

HIGHWAY_SUCCESS_MIN_SPEED = {
    'light': 23.5,
    'standard': 24.0,
    'dense': 22.5,
}

HIGHWAY_LANE_CONTEXT_THRESHOLDS = {
    'light': {
        'blocked_front_gap': 38.0,
        'blocked_speed_margin': 0.8,
        'clear_road_gap': 54.0,
        'left_front_clear': 20.0,
        'left_rear_clear': 10.0,
        'right_front_clear': 18.0,
        'right_rear_clear': 9.0,
    },
    'standard': {
        'blocked_front_gap': 36.0,
        'blocked_speed_margin': 0.4,
        'clear_road_gap': 46.0,
        'left_front_clear': 16.0,
        'left_rear_clear': 9.0,
        'right_front_clear': 14.0,
        'right_rear_clear': 8.5,
    },
    'dense': {
        'blocked_front_gap': 34.0,
        'blocked_speed_margin': 0.3,
        'clear_road_gap': 40.0,
        'left_front_clear': 13.0,
        'left_rear_clear': 8.0,
        'right_front_clear': 11.5,
        'right_rear_clear': 7.5,
    },
}

HIGHWAY_OVERTAKE_WINDOW_PROFILE = {
    'light': {
        'ego_target_speed_bonus': 2.0,
        'current_clear_rear_gap': 24.0,
        'current_clear_front_gap': 88.0,
        'left_clear_rear_gap': 40.0,
        'left_clear_front_gap': 132.0,
        'right_clear_rear_gap': 18.0,
        'right_clear_front_gap': 52.0,
        'current_lane_spawns': [
            {'offset': 20.0, 'speed_delta': 8.2, 'target_speed_delta': 7.2, 'min_gap': 18.0},
            {'offset': 48.0, 'speed_delta': 5.0, 'target_speed_delta': 4.2, 'min_gap': 18.0},
            {'offset': 82.0, 'speed_delta': 2.4, 'target_speed_delta': 1.8, 'min_gap': 20.0},
        ],
        'left_lane_spawns': [
            {'offset': -42.0, 'speed_delta': 0.5, 'target_speed_delta': -0.2, 'min_gap': 18.0},
            {'offset': 96.0, 'speed_delta': -0.4, 'target_speed_delta': -1.0, 'min_gap': 20.0},
        ],
        'right_lane_spawns': [
            {'offset': 26.0, 'speed_delta': 5.6, 'target_speed_delta': 4.8, 'min_gap': 18.0},
        ],
    },
    'standard': {
        'ego_target_speed_bonus': 2.0,
        'current_clear_rear_gap': 16.0,
        'current_clear_front_gap': 58.0,
        'left_clear_rear_gap': 30.0,
        'left_clear_front_gap': 88.0,
        'right_clear_rear_gap': 16.0,
        'right_clear_front_gap': 42.0,
        'current_lane_spawns': [
            {'offset': 18.0, 'speed_delta': 8.0, 'target_speed_delta': 7.0, 'min_gap': 16.0},
            {'offset': 38.0, 'speed_delta': 5.4, 'target_speed_delta': 4.4, 'min_gap': 16.0},
            {'offset': 72.0, 'speed_delta': 2.6, 'target_speed_delta': 2.0, 'min_gap': 18.0},
        ],
        'left_lane_spawns': [
            {'offset': -34.0, 'speed_delta': 0.5, 'target_speed_delta': -0.2, 'min_gap': 18.0},
            {'offset': 66.0, 'speed_delta': -0.2, 'target_speed_delta': -0.8, 'min_gap': 18.0},
            {'offset': 108.0, 'speed_delta': 0.8, 'target_speed_delta': 0.2, 'min_gap': 20.0},
        ],
        'right_lane_spawns': [
            {'offset': 24.0, 'speed_delta': 4.8, 'target_speed_delta': 4.0, 'min_gap': 16.0},
            {'offset': 58.0, 'speed_delta': 3.2, 'target_speed_delta': 2.6, 'min_gap': 16.0},
        ],
    },
    'dense': {
        'ego_target_speed_bonus': 2.0,
        'current_clear_rear_gap': 12.0,
        'current_clear_front_gap': 48.0,
        'left_clear_rear_gap': 26.0,
        'left_clear_front_gap': 68.0,
        'right_clear_rear_gap': 14.0,
        'right_clear_front_gap': 40.0,
        'current_lane_spawns': [
            {'offset': 16.0, 'speed_delta': 7.0, 'target_speed_delta': 6.2, 'min_gap': 14.0},
            {'offset': 32.0, 'speed_delta': 5.2, 'target_speed_delta': 4.5, 'min_gap': 14.0},
            {'offset': 52.0, 'speed_delta': 3.6, 'target_speed_delta': 3.0, 'min_gap': 14.0},
            {'offset': 78.0, 'speed_delta': 2.4, 'target_speed_delta': 2.0, 'min_gap': 15.0},
        ],
        'left_lane_spawns': [
            {'offset': -30.0, 'speed_delta': 0.8, 'target_speed_delta': 0.2, 'min_gap': 16.0},
            {'offset': 50.0, 'speed_delta': 0.6, 'target_speed_delta': 0.0, 'min_gap': 16.0},
            {'offset': 80.0, 'speed_delta': 1.4, 'target_speed_delta': 0.8, 'min_gap': 16.0},
        ],
        'right_lane_spawns': [
            {'offset': 20.0, 'speed_delta': 5.0, 'target_speed_delta': 4.2, 'min_gap': 14.0},
            {'offset': 38.0, 'speed_delta': 4.2, 'target_speed_delta': 3.5, 'min_gap': 14.0},
            {'offset': 64.0, 'speed_delta': 2.6, 'target_speed_delta': 2.0, 'min_gap': 15.0},
        ],
    },
}

HIGHWAY_OVERTAKE_PASS_MARGIN = 4.5
HIGHWAY_POST_OVERTAKE_SETTLE_STEPS = 6
HIGHWAY_LANE_CHANGE_EVAL_WINDOW = 8
HIGHWAY_MIN_EFFECTIVE_FRONT_GAP_GAIN = 3.0
HIGHWAY_MIN_EFFECTIVE_SPEED_GAIN = 0.25
HIGHWAY_MIN_EFFECTIVE_FRONT_SPEED_GAIN = 0.5
HIGHWAY_ASSIST_PROFILE = {
    'light': {
        'min_score_gain': 0.18,
        'keep_right_score_tolerance': 0.04,
        'accelerate_speed_margin': 1.2,
        'critical_front_gap': 13.0,
        'lane_change_cooldown_steps': 1,
        'purposeful_front_gap_ratio': 1.00,
        'blocked_gain_relaxation': 0.00,
        'faster_redirect_gap_ratio': 0.88,
        'anticipatory_block_gap_ratio': 0.72,
        'overtake_target_front_buffer': 3.0,
        'overtake_target_speed_gain': 0.4,
        'post_overtake_settle_steps': 3,
    },
    'standard': {
        'min_score_gain': 0.18,
        'keep_right_score_tolerance': 0.04,
        'accelerate_speed_margin': 1.2,
        'critical_front_gap': 13.0,
        'lane_change_cooldown_steps': 2,
        'purposeful_front_gap_ratio': 0.95,
        'blocked_gain_relaxation': 0.02,
        'faster_redirect_gap_ratio': 0.90,
        'anticipatory_block_gap_ratio': 0.76,
        'overtake_target_front_buffer': 3.5,
        'overtake_target_speed_gain': 0.5,
        'post_overtake_settle_steps': 4,
    },
    'dense': {
        'min_score_gain': 0.08,
        'keep_right_score_tolerance': 0.01,
        'accelerate_speed_margin': 0.9,
        'critical_front_gap': 22.0,
        'lane_change_cooldown_steps': 3,
        'purposeful_front_gap_ratio': 0.90,
        'blocked_gain_relaxation': 0.06,
        'faster_redirect_gap_ratio': 0.92,
        'anticipatory_block_gap_ratio': 0.95,
        'overtake_target_front_buffer': 5.0,
        'overtake_target_speed_gain': 0.5,
        'post_overtake_settle_steps': 8,
        'yield_gap_ratio': 0.80,
    },
}

HIGHWAY_TRAINING_SIMULATION_FREQUENCY = {
    'light': 15,
    'standard': 15,
    'dense': 18,
}

HIGHWAY_REWARD_PROFILE = {
    'light': {
        'lane_change_penalty_scale': 0.004,
        'repeated_lane_change_scale': 0.025,
        'zigzag_penalty_scale': 0.06,
        'survival_bonus_scale': 0.03,
        'speed_bonus_scale': 0.48,
        'lane_bonus_scale': 0.04,
        'raw_reward_scale': 1.28,
        'offroad_penalty_scale': 2.00,
        'low_speed_penalty_scale': 0.45,
        'stop_penalty_scale': 0.35,
        'crash_penalty_scale': 10.0,
        'cruise_bonus_scale': 0.22,
        'clear_road_bonus_scale': 0.18,
        'overtake_bonus_scale': 1.50,
        'blocked_penalty_scale': 0.32,
        'beneficial_lane_change_bonus_scale': 0.45,
        'missed_lane_change_penalty_scale': 0.22,
        'ineffective_lane_change_penalty_scale': 0.14,
        'keep_right_return_bonus_scale': 0.10,
        'momentum_bonus_scale': 0.12,
        'under_cruise_penalty_scale': 0.12,
        'blocked_accelerate_penalty_scale': 0.12,
        'progress_lane_change_bonus_scale': 0.35,
    },
    'standard': {
        'lane_change_penalty_scale': 0.003,
        'repeated_lane_change_scale': 0.020,
        'zigzag_penalty_scale': 0.06,
        'survival_bonus_scale': 0.03,
        'speed_bonus_scale': 0.56,
        'lane_bonus_scale': 0.03,
        'raw_reward_scale': 1.28,
        'offroad_penalty_scale': 2.20,
        'low_speed_penalty_scale': 0.52,
        'stop_penalty_scale': 0.40,
        'crash_penalty_scale': 8.5,
        'cruise_bonus_scale': 0.24,
        'clear_road_bonus_scale': 0.22,
        'overtake_bonus_scale': 1.85,
        'blocked_penalty_scale': 0.45,
        'beneficial_lane_change_bonus_scale': 0.65,
        'missed_lane_change_penalty_scale': 0.34,
        'ineffective_lane_change_penalty_scale': 0.18,
        'keep_right_return_bonus_scale': 0.14,
        'momentum_bonus_scale': 0.16,
        'under_cruise_penalty_scale': 0.16,
        'blocked_accelerate_penalty_scale': 0.18,
        'progress_lane_change_bonus_scale': 0.35,
    },
    'dense': {
        'lane_change_penalty_scale': 0.0022,
        'repeated_lane_change_scale': 0.030,
        'zigzag_penalty_scale': 0.060,
        'survival_bonus_scale': 0.10,
        'speed_bonus_scale': 0.56,
        'lane_bonus_scale': 0.02,
        'raw_reward_scale': 1.24,
        'offroad_penalty_scale': 2.20,
        'low_speed_penalty_scale': 0.52,
        'stop_penalty_scale': 0.38,
        'crash_penalty_scale': 14.0,
        'cruise_bonus_scale': 0.24,
        'clear_road_bonus_scale': 0.18,
        'overtake_bonus_scale': 1.75,
        'blocked_penalty_scale': 0.48,
        'beneficial_lane_change_bonus_scale': 0.74,
        'missed_lane_change_penalty_scale': 0.22,
        'ineffective_lane_change_penalty_scale': 0.16,
        'keep_right_return_bonus_scale': 0.05,
        'momentum_bonus_scale': 0.18,
        'under_cruise_penalty_scale': 0.16,
        'blocked_accelerate_penalty_scale': 0.40,
        'progress_lane_change_bonus_scale': 0.28,
        'risky_lane_change_penalty_scale': 0.30,
        'beneficial_post_gap_ratio': 0.44,
        'safe_progress_post_gap_ratio': 0.34,
        'post_overtake_stability_scale': 0.40,
    },
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
        self.assist_override_count = 0
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
        self.highway_overtaken_vehicle_ids = set()
        self.highway_relative_progress = {}
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

    def _get_highway_lane_context_thresholds(self):
        return HIGHWAY_LANE_CONTEXT_THRESHOLDS.get(
            self.traffic_level,
            HIGHWAY_LANE_CONTEXT_THRESHOLDS['standard'],
        )

    def _get_highway_overtake_setup_profile(self):
        return HIGHWAY_OVERTAKE_WINDOW_PROFILE.get(
            self.traffic_level,
            HIGHWAY_OVERTAKE_WINDOW_PROFILE['standard'],
        )

    def _get_highway_reward_profile(self):
        return HIGHWAY_REWARD_PROFILE.get(
            self.traffic_level,
            HIGHWAY_REWARD_PROFILE['standard'],
        )

    def _get_highway_assist_profile(self):
        return HIGHWAY_ASSIST_PROFILE.get(
            self.traffic_level,
            HIGHWAY_ASSIST_PROFILE['standard'],
        )

    def _highway_lane_change_gap_ok(
        self,
        front_gap,
        rear_gap,
        rear_speed,
        front_reference,
        rear_reference,
        ego_speed,
    ):
        dynamic_front_reference = float(front_reference)
        dynamic_rear_reference = float(rear_reference)
        closing_speed = 0.0
        if rear_speed is not None:
            closing_speed = max(0.0, float(rear_speed) - max(0.0, float(ego_speed)))

        if self.traffic_level == 'dense':
            dynamic_front_reference = max(8.5, dynamic_front_reference * 0.82)
            dynamic_rear_reference = max(5.0, dynamic_rear_reference * 0.72 + 0.45 * closing_speed)
        else:
            dynamic_rear_reference = max(dynamic_rear_reference, dynamic_rear_reference * 0.85 + 0.55 * closing_speed)

        front_safe = front_gap is None or float(front_gap) >= dynamic_front_reference
        rear_safe = rear_gap is None or float(rear_gap) >= dynamic_rear_reference
        return bool(front_safe and rear_safe)

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
            config['reward_speed_range'] = list(
                HIGHWAY_REWARD_SPEED_RANGE.get(
                    self.traffic_level,
                    HIGHWAY_REWARD_SPEED_RANGE['standard'],
                )
            )
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

    def _snapshot_highway_relative_progress(self):
        if self.env is None or self.road_scenario != 'highway':
            return {}
        ego_vehicle = self._get_ego_vehicle()
        lane_index = getattr(ego_vehicle, 'lane_index', None) if ego_vehicle is not None else None
        ego_longitudinal = self._lane_longitudinal(ego_vehicle, lane_index=lane_index)
        if ego_vehicle is None or ego_longitudinal is None:
            return {}
        snapshot = {}
        for vehicle in getattr(self.env.unwrapped.road, 'vehicles', []):
            if vehicle is ego_vehicle:
                continue
            vehicle_lane_index = getattr(vehicle, 'lane_index', None)
            vehicle_longitudinal = self._lane_longitudinal(vehicle, lane_index=vehicle_lane_index)
            if vehicle_longitudinal is None:
                continue
            snapshot[id(vehicle)] = float(ego_longitudinal - vehicle_longitudinal)
        return snapshot

    def _update_highway_overtake_count(self):
        if self.road_scenario != 'highway':
            return 0
        current_progress = self._snapshot_highway_relative_progress()
        new_overtakes = 0
        for vehicle_id, relative_progress in current_progress.items():
            previous_progress = self.highway_relative_progress.get(vehicle_id)
            if previous_progress is None or vehicle_id in self.highway_overtaken_vehicle_ids:
                continue
            if (
                previous_progress <= -HIGHWAY_OVERTAKE_PASS_MARGIN
                and relative_progress >= HIGHWAY_OVERTAKE_PASS_MARGIN
            ):
                self.highway_overtaken_vehicle_ids.add(vehicle_id)
                new_overtakes += 1
        self.highway_relative_progress = current_progress
        if new_overtakes > 0:
            self.highway_overtake_completion_count += int(new_overtakes)
        return int(new_overtakes)

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
            'left_front_speed': None,
            'left_rear_gap': None,
            'left_rear_speed': None,
            'left_lane_clear': False,
            'right_front_gap': None,
            'right_front_speed': None,
            'right_rear_gap': None,
            'right_rear_speed': None,
            'right_lane_clear': False,
        }
        if self.env is None or self.road_scenario != 'highway':
            return empty_context
        ego_vehicle = vehicle if vehicle is not None else self._get_ego_vehicle()
        lane_index = getattr(ego_vehicle, 'lane_index', None)
        if ego_vehicle is None or lane_index is None or len(lane_index) < 3:
            return empty_context

        thresholds = self._get_highway_lane_context_thresholds()
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
        ego_speed = float(getattr(ego_vehicle, 'speed', 0.0))
        left_lane_clear = (
            left_lane_index is not None and
            self._highway_lane_change_gap_ok(
                left_profile.get('front_gap'),
                left_profile.get('rear_gap'),
                left_profile.get('rear_speed'),
                thresholds['left_front_clear'],
                thresholds['left_rear_clear'],
                ego_speed,
            )
        )
        right_lane_clear = (
            right_lane_index is not None and
            self._highway_lane_change_gap_ok(
                right_profile.get('front_gap'),
                right_profile.get('rear_gap'),
                right_profile.get('rear_speed'),
                thresholds['right_front_clear'],
                thresholds['right_rear_clear'],
                ego_speed,
            )
        )
        return {
            'current_front_gap': current_profile.get('front_gap'),
            'current_front_speed': current_profile.get('front_speed'),
            'left_front_gap': left_profile.get('front_gap'),
            'left_front_speed': left_profile.get('front_speed'),
            'left_rear_gap': left_profile.get('rear_gap'),
            'left_rear_speed': left_profile.get('rear_speed'),
            'left_lane_clear': bool(left_lane_clear),
            'right_front_gap': right_profile.get('front_gap'),
            'right_front_speed': right_profile.get('front_speed'),
            'right_rear_gap': right_profile.get('rear_gap'),
            'right_rear_speed': right_profile.get('rear_speed'),
            'right_lane_clear': bool(right_lane_clear),
        }

    def _capture_highway_step_context(self):
        if self.env is None or self.road_scenario != 'highway':
            return {}
        ego_vehicle = self._get_ego_vehicle()
        lane_index = getattr(ego_vehicle, 'lane_index', None) if ego_vehicle is not None else None
        lane_id = int(lane_index[2]) if lane_index is not None and len(lane_index) >= 3 else None
        cruise_speed = self._target_highway_cruise_speed()
        thresholds = self._get_highway_lane_context_thresholds()
        context = self._highway_lane_context(ego_vehicle)
        neighbor_snapshot = self._lane_neighbor_snapshot(ego_vehicle, lane_index=lane_index)
        front_gap = context.get('current_front_gap')
        front_speed = context.get('current_front_speed')
        blocked_by_slower_front = (
            front_gap is not None and
            front_gap < thresholds['blocked_front_gap'] and
            front_speed is not None and
            front_speed < (cruise_speed - thresholds['blocked_speed_margin'])
        )
        clear_road_ahead = front_gap is None or front_gap >= thresholds['clear_road_gap']
        return {
            'lane_id': lane_id,
            'lane_index': lane_index,
            'speed': self._current_speed(),
            'context': context,
            'front_vehicle': neighbor_snapshot.get('front_vehicle'),
            'clear_road_ahead': bool(clear_road_ahead),
            'blocked_by_slower_front': bool(blocked_by_slower_front),
        }


    def _available_actions(self):
        if self.env is None:
            return {0, 1, 2, 3, 4}
        action_type = getattr(self.env.unwrapped, 'action_type', None)
        if action_type is None or not hasattr(action_type, 'get_available_actions'):
            return {0, 1, 2, 3, 4}
        try:
            return set(int(action_id) for action_id in action_type.get_available_actions())
        except Exception:
            return {0, 1, 2, 3, 4}

    def _highway_lane_option_score(self, front_gap, front_speed, rear_gap, cruise_speed, front_reference, rear_reference):
        gap_score = 1.20 if front_gap is None else float(np.clip(front_gap / max(1.0, front_reference), 0.0, 1.30))
        if front_speed is None:
            speed_score = 1.0
        else:
            speed_score = float(np.clip((front_speed - (cruise_speed - 7.0)) / 7.0, 0.0, 1.20))
        rear_score = 1.0 if rear_gap is None else float(np.clip(rear_gap / max(1.0, rear_reference), 0.0, 1.0))
        return 0.58 * gap_score + 0.28 * speed_score + 0.14 * rear_score

    def _highway_lane_change_window_ready(
        self,
        direction,
        context,
        thresholds,
        assist_profile,
        current_front_gap,
        current_front_speed,
        ego_speed,
    ):
        if direction == 'left':
            lane_clear = bool(context.get('left_lane_clear', False))
            if not lane_clear:
                return False
            if self.traffic_level != 'dense':
                return True
            left_front_gap = context.get('left_front_gap')
            left_front_speed = context.get('left_front_speed')
            current_front_gap_value = float(current_front_gap) if current_front_gap is not None else thresholds['blocked_front_gap']
            current_front_speed_value = float(current_front_speed) if current_front_speed is not None else float(ego_speed)
            target_front_gap = max(
                thresholds['left_front_clear'],
                min(
                    thresholds['clear_road_gap'] * 0.55,
                    current_front_gap_value + float(assist_profile.get('overtake_target_front_buffer', 4.0)),
                ),
            )
            target_speed_gain = float(assist_profile.get('overtake_target_speed_gain', 0.5))
            return bool(
                left_front_gap is None or
                float(left_front_gap) >= target_front_gap or
                (
                    left_front_gap is not None and
                    float(left_front_gap) >= thresholds['left_front_clear'] and
                    (
                        left_front_speed is None or
                        float(left_front_speed) >= (current_front_speed_value + target_speed_gain)
                    )
                )
            )
        if direction == 'right':
            return bool(context.get('right_lane_clear', False))
        return False

    def _derive_highway_tactical_plan(self, pre_step_highway_meta):
        default_plan = {
            'blocked': False,
            'current_score': 0.0,
            'left_score': -1.0,
            'right_score': -1.0,
            'best_lane_gain': 0.0,
            'desired_action': None,
            'reason': 'policy',
            'accelerate': False,
            'return_right': False,
        }
        if self.road_scenario != 'highway' or not pre_step_highway_meta:
            return default_plan

        thresholds = self._get_highway_lane_context_thresholds()
        assist_profile = self._get_highway_assist_profile()
        context = pre_step_highway_meta.get('context', {})
        cruise_speed = self._target_highway_cruise_speed()
        speed = float(pre_step_highway_meta.get('speed', 0.0))
        available_actions = self._available_actions()

        current_score = self._highway_lane_option_score(
            context.get('current_front_gap'),
            context.get('current_front_speed'),
            None,
            cruise_speed,
            thresholds['clear_road_gap'],
            thresholds['left_rear_clear'],
        )
        left_score = -1.0
        right_score = -1.0
        if context.get('left_lane_clear', False):
            left_score = self._highway_lane_option_score(
                context.get('left_front_gap'),
                context.get('left_front_speed'),
                context.get('left_rear_gap'),
                cruise_speed,
                thresholds['clear_road_gap'],
                thresholds['left_rear_clear'],
            )
        if context.get('right_lane_clear', False):
            right_score = self._highway_lane_option_score(
                context.get('right_front_gap'),
                context.get('right_front_speed'),
                context.get('right_rear_gap'),
                cruise_speed,
                thresholds['clear_road_gap'],
                thresholds['right_rear_clear'],
            )

        current_front_gap = context.get('current_front_gap')
        current_front_speed = context.get('current_front_speed')
        blocked = bool(pre_step_highway_meta.get('blocked_by_slower_front'))
        if self.traffic_level == 'dense' and not blocked:
            anticipatory_block_gap = thresholds['clear_road_gap'] * float(
                assist_profile.get('anticipatory_block_gap_ratio', 0.80)
            )
            blocked = (
                current_front_gap is not None and
                current_front_gap < anticipatory_block_gap and
                current_front_speed is not None and
                current_front_speed < max(speed - 0.4, cruise_speed - 1.8)
            )
        if not blocked:
            blocked = (
                current_front_gap is not None and
                current_front_gap < thresholds['blocked_front_gap'] and
                speed < cruise_speed + 1.0
            )

        left_window_ready = self._highway_lane_change_window_ready(
            'left',
            context,
            thresholds,
            assist_profile,
            current_front_gap,
            current_front_speed,
            speed,
        )
        right_window_ready = self._highway_lane_change_window_ready(
            'right',
            context,
            thresholds,
            assist_profile,
            current_front_gap,
            current_front_speed,
            speed,
        )

        best_lane_gain = 0.0
        desired_action = None
        reason = 'policy'
        if left_window_ready:
            left_gain = float(left_score - current_score)
            if left_gain > best_lane_gain:
                best_lane_gain = left_gain
                desired_action = self.left_lane_action_id
                reason = 'blocked_left_overtake'
        if right_window_ready:
            right_gain = float(right_score - current_score)
            if right_gain > best_lane_gain:
                best_lane_gain = right_gain
                desired_action = self.right_lane_action_id
                reason = 'blocked_right_escape'

        effective_min_score_gain = float(assist_profile['min_score_gain'])
        if self.traffic_level == 'dense' and blocked and current_front_gap is not None:
            blocked_gap_pressure = float(
                np.clip(
                    (thresholds['blocked_front_gap'] - float(current_front_gap)) / max(1.0, thresholds['blocked_front_gap']),
                    0.0,
                    1.0,
                )
            )
            effective_min_score_gain = max(
                0.03,
                effective_min_score_gain - float(assist_profile.get('blocked_gain_relaxation', 0.0)) * blocked_gap_pressure,
            )
        if best_lane_gain < effective_min_score_gain:
            desired_action = None
        if not blocked:
            desired_action = None
        if desired_action is not None and desired_action not in available_actions:
            desired_action = None

        critical_blocked = (
            blocked and
            current_front_gap is not None and
            current_front_gap < float(assist_profile['critical_front_gap'])
        )
        if desired_action is None and critical_blocked:
            if left_window_ready and left_score >= right_score:
                desired_action = self.left_lane_action_id
                reason = 'critical_left_escape'
            elif right_window_ready:
                desired_action = self.right_lane_action_id
                reason = 'critical_right_escape'
            elif left_window_ready:
                desired_action = self.left_lane_action_id
                reason = 'critical_left_escape'

        clear_road_ahead = bool(pre_step_highway_meta.get('clear_road_ahead'))
        accelerate = (
            clear_road_ahead and
            speed < (cruise_speed - float(assist_profile['accelerate_speed_margin'])) and
            self.faster_action_id in available_actions
        )
        return_right = (
            context.get('right_lane_clear', False) and
            not blocked and
            right_score >= (current_score - float(assist_profile['keep_right_score_tolerance']))
        )
        if self.traffic_level == 'dense':
            return_right = (
                return_right and
                clear_road_ahead and
                speed >= max(self._minimum_success_speed(), cruise_speed - 1.0)
            )

        if desired_action is None and blocked:
            front_gap = context.get('current_front_gap')
            front_speed = context.get('current_front_speed')
            yield_gap_ratio = float(assist_profile.get('yield_gap_ratio', 0.45))
            should_yield = (
                front_gap is not None and
                front_gap < max(12.0, thresholds['blocked_front_gap'] * yield_gap_ratio) and
                front_speed is not None and
                speed > (front_speed + 1.0) and
                self.slower_action_id in available_actions
            )
            if should_yield:
                desired_action = self.slower_action_id
                reason = 'yield_to_blocker'

        if desired_action is None:
            if return_right and self.right_lane_action_id in available_actions:
                desired_action = self.right_lane_action_id
                reason = 'keep_right'
            elif accelerate:
                desired_action = self.faster_action_id
                reason = 'clear_road_accelerate'

        return {
            'blocked': bool(blocked),
            'current_score': float(current_score),
            'left_score': float(left_score),
            'right_score': float(right_score),
            'best_lane_gain': float(best_lane_gain),
            'desired_action': desired_action,
            'reason': reason,
            'accelerate': bool(accelerate),
            'return_right': bool(return_right),
        }

    def _select_highway_assisted_action(self, requested_action, pre_step_highway_meta):
        if self.road_scenario != 'highway':
            return int(requested_action), {'reason': 'non_highway'}

        available_actions = self._available_actions()
        tactical_plan = self._derive_highway_tactical_plan(pre_step_highway_meta)
        desired_action = tactical_plan.get('desired_action')
        executed_action = int(requested_action)
        assist_reason = 'policy'
        context = (pre_step_highway_meta or {}).get('context', {})
        assist_profile = self._get_highway_assist_profile()
        thresholds = self._get_highway_lane_context_thresholds()
        dense_mode = self.traffic_level == 'dense'
        current_front_gap = context.get('current_front_gap')
        current_front_speed = context.get('current_front_speed')
        ego_speed = float((pre_step_highway_meta or {}).get('speed', 0.0))
        requested_lane_change_side_safe = (
            (requested_action == self.left_lane_action_id and context.get('left_lane_clear', False)) or
            (requested_action == self.right_lane_action_id and context.get('right_lane_clear', False))
        )
        requested_lane_change_window_ready = (
            (
                requested_action == self.left_lane_action_id and
                self._highway_lane_change_window_ready(
                    'left',
                    context,
                    thresholds,
                    assist_profile,
                    current_front_gap,
                    current_front_speed,
                    ego_speed,
                )
            ) or
            (
                requested_action == self.right_lane_action_id and
                self._highway_lane_change_window_ready(
                    'right',
                    context,
                    thresholds,
                    assist_profile,
                    current_front_gap,
                    current_front_speed,
                    ego_speed,
                )
            )
        )
        requested_lane_change_is_safe = requested_lane_change_side_safe and requested_lane_change_window_ready
        dense_lane_change_cooldown = (
            dense_mode and
            self.steps_since_lane_change < int(assist_profile.get('lane_change_cooldown_steps', 0))
        )
        purposeful_dense_lane_change = (
            tactical_plan.get('blocked', False) or
            desired_action == requested_action or
            (
                current_front_gap is not None and
                current_front_gap < thresholds['clear_road_gap'] * float(assist_profile.get('purposeful_front_gap_ratio', 1.0))
            )
        )
        dense_critical_lane_change = tactical_plan.get('reason') in ['critical_left_escape', 'critical_right_escape']
        dense_post_overtake_hold_steps = max(
            int(self.highway_post_overtake_settle_steps),
            int(self.highway_recent_overtake_completion_steps),
        )
        dense_post_overtake_settle_active = (
            dense_mode and
            dense_post_overtake_hold_steps > 0 and
            not dense_critical_lane_change
        )
        forced_escape = False
        if (
            dense_mode and
            current_front_gap is not None and
            float(current_front_gap) < float(assist_profile['critical_front_gap']) * 0.85 and
            desired_action in self.lane_change_action_ids and
            desired_action in available_actions
        ):
            if desired_action == self.left_lane_action_id:
                target_front_gap = context.get('left_front_gap')
                target_rear_gap = context.get('left_rear_gap')
                target_rear_speed = context.get('left_rear_speed')
                front_ref = thresholds['left_front_clear']
                rear_ref = thresholds['left_rear_clear']
                lane_clear = context.get('left_lane_clear', False)
                window_ready = self._highway_lane_change_window_ready(
                    'left',
                    context,
                    thresholds,
                    assist_profile,
                    current_front_gap,
                    current_front_speed,
                    ego_speed,
                )
            else:
                target_front_gap = context.get('right_front_gap')
                target_rear_gap = context.get('right_rear_gap')
                target_rear_speed = context.get('right_rear_speed')
                front_ref = thresholds['right_front_clear']
                rear_ref = thresholds['right_rear_clear']
                lane_clear = context.get('right_lane_clear', False)
                window_ready = self._highway_lane_change_window_ready(
                    'right',
                    context,
                    thresholds,
                    assist_profile,
                    current_front_gap,
                    current_front_speed,
                    ego_speed,
                )
            safe_gap = self._highway_lane_change_gap_ok(
                target_front_gap,
                target_rear_gap,
                target_rear_speed,
                front_ref,
                rear_ref,
                ego_speed,
            )
            forced_escape = bool(lane_clear and safe_gap and window_ready)
        if forced_escape:
            executed_action = int(desired_action)
            assist_reason = 'dense_forced_escape'
        dense_critical_lane_change = dense_critical_lane_change or forced_escape

        if executed_action == self.left_lane_action_id and not context.get('left_lane_clear', False):
            if desired_action is not None and desired_action in available_actions and desired_action != self.left_lane_action_id:
                executed_action = int(desired_action)
                assist_reason = 'unsafe_left_redirect'
            else:
                executed_action = self.faster_action_id if tactical_plan.get('accelerate') and self.faster_action_id in available_actions else (1 if 1 in available_actions else min(available_actions))
                assist_reason = 'unsafe_left_suppressed'
        elif executed_action == self.right_lane_action_id and not context.get('right_lane_clear', False):
            if desired_action is not None and desired_action in available_actions and desired_action != self.right_lane_action_id:
                executed_action = int(desired_action)
                assist_reason = 'unsafe_right_redirect'
            else:
                executed_action = self.faster_action_id if tactical_plan.get('accelerate') and self.faster_action_id in available_actions else (1 if 1 in available_actions else min(available_actions))
                assist_reason = 'unsafe_right_suppressed'

        if executed_action in self.lane_change_action_ids:
            if (
                dense_post_overtake_settle_active and
                (
                    not tactical_plan.get('blocked', False) or
                    current_front_gap is None or
                    float(current_front_gap) >= thresholds['blocked_front_gap'] * 0.82
                )
            ):
                if tactical_plan.get('accelerate') and self.faster_action_id in available_actions:
                    executed_action = self.faster_action_id
                    assist_reason = 'dense_post_overtake_settle_to_accelerate'
                elif tactical_plan.get('reason') == 'yield_to_blocker' and self.slower_action_id in available_actions:
                    executed_action = self.slower_action_id
                    assist_reason = 'dense_post_overtake_settle_to_yield'
                else:
                    executed_action = 1 if 1 in available_actions else min(available_actions)
                    assist_reason = 'dense_post_overtake_settle_hold'
            elif dense_lane_change_cooldown and not dense_critical_lane_change:
                if tactical_plan.get('accelerate') and self.faster_action_id in available_actions:
                    executed_action = self.faster_action_id
                    assist_reason = 'dense_lane_change_cooldown_to_accelerate'
                elif tactical_plan.get('reason') == 'yield_to_blocker' and self.slower_action_id in available_actions:
                    executed_action = self.slower_action_id
                    assist_reason = 'dense_lane_change_cooldown_to_yield'
                else:
                    executed_action = 1 if 1 in available_actions else min(available_actions)
                    assist_reason = 'dense_lane_change_cooldown'
            elif desired_action is None or desired_action != executed_action:
                if dense_mode and requested_lane_change_is_safe and purposeful_dense_lane_change:
                    assist_reason = 'safe_policy_lane_change'
                elif tactical_plan.get('accelerate') and self.faster_action_id in available_actions:
                    executed_action = self.faster_action_id
                    assist_reason = 'nontactical_lane_change_to_accelerate'
                elif tactical_plan.get('reason') == 'yield_to_blocker' and self.slower_action_id in available_actions:
                    executed_action = self.slower_action_id
                    assist_reason = 'nontactical_lane_change_to_yield'
                else:
                    executed_action = 1 if 1 in available_actions else min(available_actions)
                    assist_reason = 'nontactical_lane_change_suppressed'

        if executed_action not in available_actions:
            if desired_action is not None and desired_action in available_actions:
                executed_action = int(desired_action)
                assist_reason = 'mask_to_tactical_action'
            elif 1 in available_actions:
                executed_action = 1
                assist_reason = 'mask_to_idle'
            else:
                executed_action = min(available_actions)
                assist_reason = 'mask_to_available_action'
        elif desired_action is not None and desired_action in available_actions:
            if tactical_plan.get('reason') in ['blocked_left_overtake', 'blocked_right_escape', 'critical_left_escape', 'critical_right_escape']:
                if dense_mode:
                    faster_redirect_blocked = (
                        requested_action == self.faster_action_id and
                        current_front_gap is not None and
                        current_front_gap < thresholds['blocked_front_gap'] * float(assist_profile.get('faster_redirect_gap_ratio', 0.90))
                    )
                    should_override = (
                        not dense_lane_change_cooldown or dense_critical_lane_change
                    ) and (
                        requested_action in [1, self.slower_action_id] or
                        faster_redirect_blocked
                    )
                else:
                    should_override = requested_action in [1, self.faster_action_id, self.slower_action_id, self.left_lane_action_id, self.right_lane_action_id]
                if should_override:
                    executed_action = int(desired_action)
                    assist_reason = tactical_plan['reason']
            elif tactical_plan.get('reason') == 'keep_right':
                if not dense_mode and requested_action in [1, self.faster_action_id]:
                    executed_action = int(desired_action)
                    assist_reason = tactical_plan['reason']
            elif tactical_plan.get('reason') == 'clear_road_accelerate':
                if requested_action in [1, self.slower_action_id]:
                    executed_action = int(desired_action)
                    assist_reason = tactical_plan['reason']
            elif tactical_plan.get('reason') == 'yield_to_blocker':
                if dense_mode:
                    should_override = requested_action in [1, self.faster_action_id]
                else:
                    should_override = requested_action in [1, self.faster_action_id, self.left_lane_action_id, self.right_lane_action_id]
                if should_override:
                    executed_action = int(desired_action)
                    assist_reason = tactical_plan['reason']

        if executed_action not in available_actions:
            executed_action = 1 if 1 in available_actions else min(available_actions)
            assist_reason = 'fallback_available_action'

        tactical_plan = dict(tactical_plan)
        tactical_plan['assist_reason'] = assist_reason
        tactical_plan['requested_action'] = int(requested_action)
        tactical_plan['executed_action'] = int(executed_action)
        return int(executed_action), tactical_plan

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
        """Keep a small safety buffer around ego while preserving local traffic differences."""
        if self.env is None:
            return
        base_env = self.env.unwrapped
        road = getattr(base_env, 'road', None)
        ego_vehicle = self._get_ego_vehicle()
        if road is None or ego_vehicle is None:
            return

        clear_radius = 20.0
        if self.road_scenario == 'highway':
            clear_radius = float(
                HIGHWAY_LOCAL_CLEAR_ZONE_RADIUS.get(
                    self.traffic_level,
                    HIGHWAY_LOCAL_CLEAR_ZONE_RADIUS['standard'],
                )
            )
        elif self.road_scenario == 'merge':
            clear_radius = 20.0
        elif self.road_scenario == 'roundabout':
            clear_radius = 16.0

        for v in list(getattr(road, 'vehicles', [])):
            if v is not ego_vehicle:
                if np.linalg.norm(np.array(v.position) - np.array(ego_vehicle.position)) < clear_radius:
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
        target_lane_id = max(1, lane_count - 2) if lane_count > 2 else lane_count - 1
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
        if self.traffic_level == 'dense':
            same_lane_close = (6.0, 7.4)
            same_lane_mid = (4.0, 5.4)
            same_lane_far = (2.2, 3.4)
            left_lane_near = (-0.2, 0.8)
            left_lane_far = (0.0, 1.0)
            left_lane_rear = (-0.8, 0.4)
            right_lane_penalty = (4.0, 5.8)
        elif self.traffic_level == 'standard':
            same_lane_close = (6.2, 7.8)
            same_lane_mid = (4.4, 5.8)
            same_lane_far = (2.2, 3.4)
            left_lane_near = (0.3, 1.5)
            left_lane_far = (0.4, 1.8)
            left_lane_rear = (-0.4, 0.8)
            right_lane_penalty = (3.8, 5.8)
        else:
            same_lane_close = (7.0, 8.5)
            same_lane_mid = (4.8, 6.2)
            same_lane_far = (2.5, 4.0)
            left_lane_near = (0.6, 1.8)
            left_lane_far = (0.8, 2.0)
            left_lane_rear = (-0.2, 1.0)
            right_lane_penalty = (4.2, 6.0)
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
                if relative_longitudinal <= 30.0:
                    target_speed = cruise_speed - np.random.uniform(*same_lane_close)
                elif relative_longitudinal <= 60.0:
                    target_speed = cruise_speed - np.random.uniform(*same_lane_mid)
                else:
                    target_speed = cruise_speed - np.random.uniform(*same_lane_far)
            elif lane_delta < 0:
                if relative_longitudinal < -10.0:
                    target_speed = cruise_speed + np.random.uniform(*left_lane_rear)
                elif relative_longitudinal <= 55.0:
                    target_speed = cruise_speed + np.random.uniform(*left_lane_near)
                else:
                    target_speed = cruise_speed + np.random.uniform(*left_lane_far)
            elif lane_delta > 0:
                target_speed = cruise_speed - np.random.uniform(*right_lane_penalty)
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
        left_lane_index, right_lane_index = self._nearest_side_lanes(ego_lane_index)
        cruise_speed = self._target_highway_cruise_speed()
        profile = self._get_highway_overtake_setup_profile()
        ego_target_speed = min(30.0, cruise_speed + float(profile.get('ego_target_speed_bonus', 2.0)))
        if hasattr(ego_vehicle, 'target_speed'):
            ego_vehicle.target_speed = ego_target_speed
        if hasattr(ego_vehicle, 'speed'):
            ego_vehicle.speed = max(float(getattr(ego_vehicle, 'speed', ego_target_speed)), ego_target_speed)

        self._clear_highway_lane_window(
            ego_lane_index,
            ego_longitudinal,
            rear_gap=float(profile.get('current_clear_rear_gap', 16.0)),
            front_gap=float(profile.get('current_clear_front_gap', 72.0)),
        )
        if left_lane_index is not None:
            self._clear_highway_lane_window(
                left_lane_index,
                ego_longitudinal,
                rear_gap=float(profile.get('left_clear_rear_gap', 28.0)),
                front_gap=float(profile.get('left_clear_front_gap', 96.0)),
            )
        if right_lane_index is not None and profile.get('right_clear_front_gap', None) is not None:
            self._clear_highway_lane_window(
                right_lane_index,
                ego_longitudinal,
                rear_gap=float(profile.get('right_clear_rear_gap', 16.0)),
                front_gap=float(profile.get('right_clear_front_gap', 56.0)),
            )

        def _spawn_profiled_vehicles(lane_index, spawn_specs):
            if lane_index is None:
                return
            for spawn_spec in spawn_specs:
                longitudinal = ego_longitudinal + float(spawn_spec.get('offset', 0.0))
                speed = max(17.0, cruise_speed - float(spawn_spec.get('speed_delta', 0.0)))
                target_speed = max(17.0, cruise_speed - float(spawn_spec.get('target_speed_delta', spawn_spec.get('speed_delta', 0.0))))
                self._spawn_vehicle_on_lane(
                    lane_index=lane_index,
                    longitudinal=longitudinal,
                    speed=speed,
                    target_speed=target_speed,
                    min_gap=float(spawn_spec.get('min_gap', 16.0)),
                )

        _spawn_profiled_vehicles(ego_lane_index, profile.get('current_lane_spawns', []))
        _spawn_profiled_vehicles(left_lane_index, profile.get('left_lane_spawns', []))
        _spawn_profiled_vehicles(right_lane_index, profile.get('right_lane_spawns', []))
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
        del info
        # Keep HighwayEnv termination close to the official behavior: crash/off-road/time only.
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
        self.highway_overtaken_vehicle_ids = set()
        self.highway_relative_progress = {}
        self._reset_highway_lane_change_eval()
        if episode_seed is None:
            self.state_original = self.env.reset()[0]
        else:
            self.state_original = self.env.reset(seed=int(episode_seed))[0]
            
        self._reposition_merge_ego_vehicle()
        if self.road_scenario in ['merge', 'roundabout']:
            self._clear_ego_surroundings()
        self._top_up_scenario_traffic()
        if self.road_scenario == 'highway':
            ego_vehicle = self._get_ego_vehicle()
            cruise_speed = self._target_highway_cruise_speed()
            if ego_vehicle is not None and hasattr(ego_vehicle, 'target_speed'):
                ego_vehicle.target_speed = float(np.clip(cruise_speed, 20.0, 30.0))
            self.highway_relative_progress = self._snapshot_highway_relative_progress()

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

        highway_shaped_reward = None

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
            # Highway: PDF-based 5-term reward formula (论文2.2章)
            # Rtotal = cv*Rv + cc*Rc + cl*Rl + cd*Rd + ce*Re
            # Rv: 速度奖励, Rc: 碰撞奖励, Rl: 车道保持奖励, Rd: 变道奖励, Re: 动作稳定性奖励

            # 权重系数 (参考PDF表2.3，根据traffic_level调整)
            if self.traffic_level == 'dense':
                cv, cc, cl, cd, ce = 0.5, 5.0, 0.1, 0.5, 2.0
            elif self.traffic_level == 'light':
                cv, cc, cl, cd, ce = 0.6, 5.0, 0.08, 0.4, 1.5
            else:  # standard
                cv, cc, cl, cd, ce = 0.5, 5.0, 0.1, 0.5, 2.0

            # 速度参数设置 (PDF 2.2.3节)
            reward_speed_range = self.env.unwrapped.config.get(
                'reward_speed_range',
                HIGHWAY_REWARD_SPEED_RANGE.get(self.traffic_level, HIGHWAY_REWARD_SPEED_RANGE['standard']),
            )
            vmin = float(reward_speed_range[0])
            vmax = float(reward_speed_range[1])
            vtarget = float(HIGHWAY_DESIRED_CRUISE_SPEED.get(self.traffic_level, 29.0))
            v = ego_speed_value

            # 获取车道和变道信息
            highway_context = self._highway_lane_context()
            ego_vehicle = self._get_ego_vehicle()
            pre_context = (pre_step_highway_meta or {}).get('context', {})
            pre_lane_id = (pre_step_highway_meta or {}).get('lane_id')
            post_lane_index = getattr(ego_vehicle, 'lane_index', None)
            post_lane_id = int(post_lane_index[2]) if post_lane_index is not None and len(post_lane_index) >= 3 else pre_lane_id
            lane_changed = (
                action_index in self.lane_change_action_ids and
                pre_lane_id is not None and
                post_lane_id is not None and
                int(post_lane_id) != int(pre_lane_id)
            )

            # ========== 1. 速度奖励 Rv (PDF公式2.13) ==========
            vtolerance = max(vtarget - vmin, vmax - vtarget)
            if v <= vmin or v >= vmax:
                Rv = -2.0
            else:
                Rv = 1.0 - abs(v - vtarget) / vtolerance

            # ========== 2. 碰撞奖励 Rc (PDF公式2.14) ==========
            Rc = -5.0 if crashed else 0.0

            # ========== 3. 车道保持奖励 Rl (PDF公式2.15) ==========
            Rl = 0.0
            if ego_vehicle is not None and hasattr(ego_vehicle, 'lane') and ego_vehicle.lane is not None:
                try:
                    lane_center_y = ego_vehicle.lane.position(ego_vehicle.position[0], 0)[1]
                    lateral_deviation = abs(ego_vehicle.position[1] - lane_center_y)
                    k, d = 0.5, 3.5  # k*d为偏差阈值
                    Rl = 1.0 if lateral_deviation <= k * d else 0.0
                except:
                    Rl = 0.0

            # ========== 4. 变道奖励 Rd (PDF公式2.16) ==========
            # RC: 合理变道 +1.0, UC: 不合理变道 -0.5, NC: 无变道 0.0
            Rd = 0.0
            if lane_changed:
                # 判断合理变道的两个条件 (PDF 2.2.3节)
                # 1) 当前速度低于目标速度
                # 2) 相邻车道车辆平均速度至少比自车高10个单位
                condition1 = v < vtarget

                # 计算相邻车道平均速度
                current_front_speed = highway_context.get('current_front_speed')
                pre_front_speed = pre_context.get('current_front_speed')
                adjacent_lane_avg_speed = None
                if current_front_speed is not None:
                    adjacent_lane_avg_speed = float(current_front_speed)
                elif pre_front_speed is not None:
                    adjacent_lane_avg_speed = float(pre_front_speed)

                condition2 = False
                if adjacent_lane_avg_speed is not None:
                    condition2 = adjacent_lane_avg_speed >= v + 10.0

                # 判断是否为合理变道
                if condition1 and condition2:
                    Rd = 1.0  # 合理变道
                else:
                    Rd = -0.5  # 不合理变道
            else:
                Rd = 0.0  # 无变道

            # ========== 5. 动作稳定性奖励 Re (PDF公式2.17) ==========
            # 当动作变化时给予负奖励，随时间指数衰减
            lambda_decay = 0.05  # PDF建议的衰减系数
            if self.previous_action_index is not None and action_index != self.previous_action_index:
                Re = -np.exp(-lambda_decay * self.step_num)
            else:
                Re = 0.0

            # ========== 总奖励计算 (PDF公式2.12) ==========
            highway_shaped_reward = cv * Rv + cc * Rc + cl * Rl + cd * Rd + ce * Re

            # 记录奖励分解信息
            info['reward_breakdown'] = {
                'speed_reward_Rv': float(cv * Rv),
                'collision_reward_Rc': float(cc * Rc),
                'lane_keeping_reward_Rl': float(cl * Rl),
                'lane_change_reward_Rd': float(cd * Rd),
                'action_stability_reward_Re': float(ce * Re),
                'total': float(highway_shaped_reward),
            }

        # --- Non-highway (merge/roundabout) shared computations ---
        lane_change_penalty = 0.0
        repeated_lane_change_penalty = 0.0
        zigzag_penalty = 0.0
        steady_action_bonus = 0.0
        survival_bonus = 0.0
        merge_mainline_bonus = 0.0
        merge_progress_bonus = 0.0
        merge_window_bonus = 0.0
        merge_commit_bonus = 0.0
        merge_wait_penalty = 0.0
        merge_deadline_penalty = 0.0
        merge_gap_penalty = 0.0
        low_speed_penalty = 0.0
        merging_speed_penalty = 0.0
        completion_bonus = 0.0
        step_survival_bonus = 0.0

        if highway_shaped_reward is None:
            lane_change_penalty = lane_change_penalty_scale if lane_change_action else 0.0
            if lane_change_action and self.steps_since_lane_change < 6:
                repeated_lane_change_penalty = repeated_lane_change_scale * (6 - self.steps_since_lane_change) / 6.0
            zigzag_penalty = zigzag_penalty_scale if (
                self.previous_action_index in self.lane_change_action_ids and
                lane_change_action and
                self.previous_action_index != action_index
            ) else 0.0
            steady_action_bonus = steady_action_bonus_scale if not lane_change_action else 0.0
            survival_bonus = survival_bonus_scale if not crashed else 0.0

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

            if self.step_num > 2 and ego_speed_value < low_speed_threshold:
                low_speed_penalty = low_speed_penalty_scale * (low_speed_threshold - ego_speed_value) / max(1.0, low_speed_threshold)
            merging_speed_penalty = merging_speed_penalty_scale * max(0.0, merging_speed_reward)
            completion_bonus = completion_bonus_value if (scenario_completed and not crashed and not self.scenario_completion_awarded) else 0.0

            if not crashed:
                step_progress = float(self.step_num) / max(1.0, float(self.max_step_num))
                step_survival_bonus = 0.004 * (step_progress ** 1.5)

        if highway_shaped_reward is not None:
            shaped_reward = highway_shaped_reward
        else:
            shaped_reward = (
                survival_bonus
                + step_survival_bonus
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
        if highway_shaped_reward is None:
            info['reward_breakdown'] = {
                'survival_bonus': float(survival_bonus),
                'step_survival_bonus': float(step_survival_bonus),
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
        executed_action_index = int(action_index)
        highway_tactical_plan = None
        if self.road_scenario == 'highway':
            executed_action_index, highway_tactical_plan = self._select_highway_assisted_action(action_index, pre_step_highway_meta)
            pre_step_highway_meta['tactical_plan'] = highway_tactical_plan

        if show_live_progress and self.step_num % 10 == 0:
            print(f"\r🚗 正在马路上飞驰... 当前回合已开 {self.step_num} 步", end='', flush=True)


        next_state, reward_value, terminated, truncated, info = self.env.step(executed_action_index)
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

        info['policy_action'] = int(action_index)
        info['executed_action'] = int(executed_action_index)
        info['assist_overrode_policy'] = int(int(executed_action_index) != int(action_index))
        self.assist_override_count += int(info['assist_overrode_policy'])
        if highway_tactical_plan is not None:
            info['highway_assist_reason'] = highway_tactical_plan.get('assist_reason', 'policy')
            info['highway_tactical_reason'] = highway_tactical_plan.get('reason', 'policy')
            info['highway_best_lane_gain'] = float(highway_tactical_plan.get('best_lane_gain', 0.0))
        info_action = int(info.get('executed_action', info.get('action', executed_action_index)))
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
        if self.road_scenario == 'highway':
            info['highway_overtake_events'] = int(self._update_highway_overtake_count())
        else:
            info['highway_overtake_events'] = 0

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
        if self.highway_post_overtake_settle_steps > 0:
            self.highway_post_overtake_settle_steps -= 1
        if self.highway_recent_overtake_completion_steps > 0:
            self.highway_recent_overtake_completion_steps -= 1
        self.previous_action_index = info_action
        return self.state_processed, reward_tensor, done, info, step_record

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


if __name__ == '__main__':
    env = GymLane(dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print('\033[91mFINISH: env_lane\033[0m')
