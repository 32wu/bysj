# -*- coding: utf-8 -*-
import os

import torch
import numpy as np

import checkpoint_utils
import env_lane
import model_rwta

print("🎬 自动驾驶导演组：【毕设压轴】执行器故障/方向盘失控测试启动...")

device = torch.device('cpu')
road_scenario = 'roundabout'
traffic_level = 'standard'

# 1. 实例化环境并开启录像机
env = env_lane.GymLane(dev=device, road_scenario=road_scenario, traffic_level=traffic_level)
env.init_test(record_video=True)
print(f"📹 本次录像将保存到: {env.video_folder}")

# 2. 组装类脑脉冲神经网络 (SNN)
model = model_rwta.RWTAspike(
    input_size=25, output_size=5, hid_num=8, hid_size=8,
    spk_response_window='uni', spk_full_time=42, spk_resp_time=40,
    remove_connection_pattern='none', optimizer_name='rmsprop',
    optimizer_learning_rate=0.001, entropy_ratio=5.0, device=device
)

# 3. 自动加载当前场景目录下最新的优化算法最佳权重
checkpoint_prefix = checkpoint_utils.find_latest_checkpoint_prefix(
    kind='ours',
    road_scenario=road_scenario,
    traffic_level=traffic_level,
    best_only=True,
)
checkpoint_path = checkpoint_utils.resolve_checkpoint_file(checkpoint_prefix + '_w_1')
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f'没有找到固定权重: {checkpoint_path}')
model.load_model(checkpoint_prefix)
print(f"✅ 大脑组装成功！当前使用模型: {checkpoint_prefix}")

FAILURE_RATE = 0.2
print(f"⚠️ 当前正在测试的方向盘失控率为: {FAILURE_RATE * 100}%")

done = False
step = 0

try:
    while not done:
        original_state = env.get_observation()

        with torch.no_grad():
            raw_out = model(original_state)
            if isinstance(raw_out, (tuple, list)):
                action_data = raw_out[0]
            else:
                action_data = raw_out

            if not torch.is_tensor(action_data):
                action_data = torch.tensor(action_data, dtype=torch.float32)

            if action_data.dim() > 1:
                action_scores = torch.sum(action_data, dim=0)
            else:
                action_scores = action_data

        if np.random.rand() < FAILURE_RATE:
            random_action = np.random.randint(0, 5)
            action_scores = torch.zeros_like(action_scores)
            if action_scores.dim() > 0:
                action_scores[random_action] = 100.0
            else:
                action_scores = torch.tensor([100.0 if i == random_action else 0.0 for i in range(5)])

        try:
            next_state, reward, done_flag, info, step_record = env.make_action(action_scores)
            step = step_record[0]
            if env.done_signal == 1 or done_flag:
                done = True
        except Exception as e:
            print(f"❌ 环境交互报错: {e}")
            break
finally:
    env.close()

print(f"✅ 杀青！在 {FAILURE_RATE*100}% 的方向盘失控率下，小车存活了 {step} 步。")
print(f"📁 录像已安全保存至: {env.video_folder}")
