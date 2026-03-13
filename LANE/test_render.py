# -*- coding: utf-8 -*-
import numpy as np
np.bool8 = np.bool_  # 之前管用的魔法补丁，必须带上

import gym
from gym.wrappers import RecordVideo

print("🎥 开始在后台搭建倒立摆影棚...")

# 1. 创建环境，必须设置 render_mode='rgb_array' 才能在无头服务器上画图
env = gym.make('InvertedPendulum-v4', render_mode='rgb_array')

# 2. 给环境套上“录像机”外壳，视频保存在当前目录的 video_logs 文件夹下
env = RecordVideo(env, video_folder='./video_logs', name_prefix='pendulum_show', episode_trigger=lambda x: True)

# 录制 3 个回合的视频
for episode in range(3):
    obs, info = env.reset()
    done = False
    print(f"开始录制第 {episode + 1} 回合...")
    
    while not done:
        # 因为提取训练好的 SNN 大脑太繁琐，这里我们给它输入“随机动作”
        # 你看到的将是一个“笨笨的”、还没学会平衡的初始倒立摆状态
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close() # 必须 close，否则视频文件会损坏
print("✅ 录制大功告成！快去左侧找 video_logs 文件夹吧！")