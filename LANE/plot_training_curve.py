import matplotlib.pyplot as plt
import os
import re

# 设置学术画图风格和中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

log_path = 'log_text/YOUR_LOG_FILE_NAME.txt' # 替换为你的真实日志文件路径

episodes = []
returns = []

with open(log_path, 'r') as f:
    for line in f:
        # 解析带有 'val ' 的行（验证集得分通常更稳定）
        if line.startswith('val '):
            parts = [item for item in re.sub(',', ' ', line).split()]
            episodes.append(int(parts[1]))
            returns.append(float(parts[2])) # 假设第三列是Return

plt.figure(figsize=(8, 5), dpi=300)
plt.plot(episodes, returns, label='RWTA 模型 (Ours)', color='#1f77b4', linewidth=2)
plt.xlabel('训练回合数 (Episodes)', fontsize=12)
plt.ylabel('平均验证奖励 (Mean Validation Return)', fontsize=12)
plt.title('模型训练收敛曲线', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('training_curve.png', dpi=300)
print("收敛曲线图已保存为 training_curve.png")