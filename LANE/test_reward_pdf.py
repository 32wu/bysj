#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试PDF奖励函数的基本功能"""

import torch
import env_lane

def test_reward_function():
    print("=" * 60)
    print("测试PDF奖励函数 (方案E)")
    print("=" * 60)

    # 创建环境
    device = torch.device('cpu')
    env = env_lane.GymLane(dev=device, road_scenario='highway', traffic_level='standard')

    # 初始化训练环境
    env.init_train(vehicles_count=48)
    observation = env.get_train_observation()

    print(f"\n初始状态:")
    print(f"  观测维度: {observation.shape}")
    print(f"  动作空间: {env.action_num}")

    # 运行几步测试
    total_reward = 0.0
    for step in range(10):
        # 随机选择动作
        action_index = step % env.action_num
        action_onehot = torch.nn.functional.one_hot(
            torch.tensor([action_index], dtype=torch.long),
            num_classes=env.action_num
        ).float()

        # 执行动作
        next_state, reward, done, info, step_record = env.make_action(action_onehot)
        total_reward += float(reward.item())

        # 打印奖励分解
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            print(f"\nStep {step + 1} (动作={action_index}):")
            print(f"  速度奖励 Rv: {breakdown.get('speed_reward_Rv', 0.0):.4f}")
            print(f"  碰撞奖励 Rc: {breakdown.get('collision_reward_Rc', 0.0):.4f}")
            print(f"  车道保持 Rl: {breakdown.get('lane_keeping_reward_Rl', 0.0):.4f}")
            print(f"  变道奖励 Rd: {breakdown.get('lane_change_reward_Rd', 0.0):.4f}")
            print(f"  动作稳定 Re: {breakdown.get('action_stability_reward_Re', 0.0):.4f}")
            print(f"  总奖励: {breakdown.get('total', 0.0):.4f}")

        if done:
            print(f"\n回合结束于第 {step + 1} 步")
            break

        observation = next_state

    print(f"\n累计奖励: {total_reward:.4f}")
    print(f"回合步数: {env.step_num}")
    print(f"碰撞次数: {env.collision_count}")
    print(f"变道次数: {env.lane_change_count}")

    env.close()
    print("\n测试完成！")

if __name__ == '__main__':
    test_reward_function()
