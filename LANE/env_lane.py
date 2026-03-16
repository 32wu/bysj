# -*- coding: utf-8 -*-

import numpy as np
np.bool8 = np.bool_  # 魔法补丁必须保留
import torch
import sys
import os

# 全面启用现代的 gymnasium 接口
import gymnasium as gym
import highway_env  
from gymnasium.wrappers import RecordVideo

def print_info(input_string):
    print('\033[96mLANE_ENV_INFO|\033[0m', input_string)

class GymLane:
    def __init__(self, dev=torch.device('cpu')):
        # Settings
        # 🔓 第一把锁砸碎：外层最大步数放宽到 150 步！
        self.max_step_num = 150  
        self.state_dimension = 25  
        self.action_num = 5  
        self.dev = dev
        self.env = None
        # Variables
        self.mode, self.step_num, self.done_signal = None, None, None
        self.state_original, self.state_processed = None, None
        print_info('ACTION NUMBER: %6d' % self.action_num)
        print_info('STATE DIMENSION: %6d' % self.state_dimension)

    def state_to_tensor(self, state):
        # 把 5x5 的矩阵展平成 1D 的 25 维向量
        state_handle = np.copy(state).flatten()
        
        # ⚠️ 【重要修复】highway-env 的值可能包含负数，需要映射到 [0, 1] 概率区间
        # 假设原始范围域在 [-1, 1]，则 (x + 1) / 2
        state_handle = np.clip((state_handle + 1.0) / 2.0, 0.0, 1.0)
        
        state_cuda = torch.FloatTensor(np.expand_dims(state_handle, axis=0)).to(self.dev)
        return state_cuda
    def init_train(self):
        if self.env is not None:
            self.env.close()
        self.env = gym.make('highway-v0', render_mode=None)
        
        # 强制篡改底层配置：让训练和测试环境完全一致！
        self.env.unwrapped.configure({
            "duration": 150,               # 赛道拉长到 150 步
            "vehicles_count": 40,          # 加入40辆环境车，上点强度
            "high_speed_reward": 0.8,      # 逼迫小车踩油门，不准当龟速大爷
            "reward_speed_range": [20, 30],# 速度域定在 20-30
            "collision_reward": -1.0,      # 撞车直接大扣分
            "lane_change_reward": -0.05    # 惩罚瞎变道
        })
        self.state_original = self.env.reset()[0]
        self.step_num = 0

    def init_val(self):
        if self.env is not None:
            self.env.close()
        self.env = gym.make('highway-v0', render_mode=None)
        
        # 验证环境也要保持一样的配置
        self.env.unwrapped.configure({
            "duration": 150,               
            "vehicles_count": 40,          
            "high_speed_reward": 0.8,      
            "reward_speed_range": [20, 30],
            "collision_reward": -1.0,      
            "lane_change_reward": -0.05    
        })
        self.state_original = self.env.reset()[0]
        self.step_num = 0

    def init_test(self, variation_type='none', variation_param=0, record_video=False):
        if self.env is not None:
            self.env.close()
        
        render_mode = 'rgb_array' if record_video else None
        
        # 裁判刚把新车造出来
        self.env = gym.make('highway-v0', render_mode=render_mode)
        
        # ==========================================
        # 🔓 第二把锁砸碎：在没 reset 之前，强制篡改底层配置！
        # ==========================================
        try:
            self.env.unwrapped.configure({
                "duration": 150,          # 强制拉长跑道至 150 步
                "vehicles_count": 40      # 强制增加车流密度至 40 辆
            })
            print_info("成功黑入底层：赛道已延长至 150 步，车流已增加！")
        except Exception as e:
            print_info(f"黑入底层失败: {e}")
        # ==========================================

        if record_video:
            self.env = RecordVideo(self.env, video_folder='./video_logs_lane', name_prefix='highway_show')

        # 带着篡改过的配置启动环境！
        self.state_original = self.env.reset()[0]
        self.step_num = 0

    def get_observation(self):
        self.state_processed = self.state_to_tensor(self.state_original)
        return self.state_processed

    def get_train_observation(self, **kwargs):
        return self.get_observation()

    def get_val_observation(self, **kwargs):
        return self.get_observation()

    def get_test_observation(self, noise_type='none', noise_param=0, **kwargs):
        s_tensor = self.get_observation()
        if noise_type == 'none':
            return s_tensor
        if noise_type == 'gaussian':
            s_tensor.add_(torch.randn(s_tensor.size()).to(self.dev) * noise_param)
        else:
            s_tensor.add_((torch.rand(s_tensor.size()).to(self.dev) - 0.5) * 2 * noise_param)
        return s_tensor

    def make_action(self, action):
        action_index = torch.argmax(action).item()
        
        # 【心跳监视器】：让小车每开 10 步就报个平安，并在同一行刷新！
        if self.step_num % 10 == 0:
            print(f"\r🚗 正在马路上飞驰... 当前回合已开 {self.step_num} 步", end="", flush=True)

        s2, r, terminated, truncated, info = self.env.step(action_index)
        done = terminated or truncated
        
        reward = torch.FloatTensor(np.array([r])).to(self.dev)
        
        if done or self.step_num >= self.max_step_num - 1:
            done_flag = 1
            # 回合结束时换行
            print(f" -> 💥 回合结束！得分: {r:.2f}") 
        else:
            done_flag = 0
            
        self.done_signal = done_flag
        
        self.state_original = np.copy(s2)
        self.step_num += 1
        
        return reward, self.state_to_tensor(s2), [self.step_num, float(r)]

if __name__ == "__main__":
    env = GymLane(dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print('\033[91mFINISH: env_lane\033[0m')