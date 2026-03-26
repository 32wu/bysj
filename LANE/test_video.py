# -*- coding: utf-8 -*-
import os

import torch
import numpy as np
import env_lane
import model_rwta

print("🎬 自动驾驶导演组：【毕设压轴】执行器故障/方向盘失控测试启动...")

device = torch.device('cpu')

# 1. 实例化环境并开启录像机 (150步的超长赛道已经在 env_lane.py 里生效了)
env = env_lane.GymLane(dev=device)
env.init_test(record_video=True)
print(f"📹 本次录像将保存到: {env.video_folder}")

# 2. 组装类脑脉冲神经网络 (SNN)
model = model_rwta.RWTAspike(
    input_size=25, output_size=5, hid_num=8, hid_size=8,
    spk_response_window='uni', spk_full_time=42, spk_resp_time=40,
    remove_connection_pattern='none', optimizer_name='rmsprop',
    optimizer_learning_rate=0.001, entropy_ratio=5.0, device=device
)

# 3. 固定加载训练出的最佳 seed06 权重
checkpoint_prefix = 'ppo_gymip_rwtaspk_h8-8-40_none_rmsprop_0.000500_2.00_0.99500_4_0.1500_ro512_mb128_lam0.97_rs1.00_gc0.50_adaptive_seed06_best'
checkpoint_path = f'./log_model/{checkpoint_prefix}_w_1.pt'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f'没有找到固定权重: {checkpoint_path}')
model.load_model(checkpoint_prefix)
print(f"✅ 大脑组装成功！当前使用模型: {checkpoint_prefix}")

# ==========================================
# 🎛️ 实验参数控制台：方向盘失控率 (执行器故障)
# ==========================================
# 调节这个值来画你的毕设折线图：
# 0.0 = 车况完美；0.2 = 偶尔打滑；0.5 = 严重失控；0.8 = 彻底疯了
FAILURE_RATE = 0.2  
print(f"⚠️ 当前正在测试的方向盘失控率为: {FAILURE_RATE * 100}%")
# ==========================================

done = False
step = 0

try:
    while not done:
        # 这次我们给大脑完美的雷达数据，不干扰传感器
        original_state = env.get_observation()
        
        # --- 🧠 步骤 A：大脑进行完美决策 ---
        with torch.no_grad():
            raw_out = model(original_state)
            
            # SNN 脉冲信号解码与剥壳
            if isinstance(raw_out, (tuple, list)):
                action_data = raw_out[0]
            else:
                action_data = raw_out

            if not torch.is_tensor(action_data):
                action_data = torch.tensor(action_data, dtype=torch.float32)

            # 频率解码：将时间步维度累加，得到 5 个动作的原始得分
            if action_data.dim() > 1:
                action_scores = torch.sum(action_data, dim=0)
            else:
                action_scores = action_data

        # --- 💥 步骤 B：核心干扰（物理级执行器故障） ---
        # 即使大脑做出了正确决策，车辆的手脚却不听使唤！
        if np.random.rand() < FAILURE_RATE:
            # 强制抢走方向盘，在 0~4 这五个动作里纯随机选一个执行
            random_action = np.random.randint(0, 5)
            
            # 伪造一个极端的得分矩阵，强行覆盖大脑的决策，骗过底层的 argmax
            action_scores = torch.zeros_like(action_scores)
            if action_scores.dim() > 0:
                action_scores[random_action] = 100.0
            else:
                action_scores = torch.tensor([100.0 if i == random_action else 0.0 for i in range(5)])
        # ----------------------------------------------

        # --- 步骤 C：将最终动作传给环境物理引擎执行 ---
        try:
            next_state, reward, done_flag, info, step_record = env.make_action(action_scores)
            step = step_record[0]
            # 当小车撞毁或跑满最大步数时，跳出循环
            if env.done_signal == 1 or done_flag:
                done = True
        except Exception as e:
            print(f"❌ 环境交互报错: {e}")
            break
finally:
    # 安全关闭环境，确保录像写入硬盘
    env.close()

print(f"✅ 杀青！在 {FAILURE_RATE*100}% 的方向盘失控率下，小车存活了 {step} 步。")
print(f"📁 录像已安全保存至: {env.video_folder}")