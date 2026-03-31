import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.style.use('seaborn-v0_8-whitegrid')

# 读取生成的 CSV 文件 (请替换为真实路径)
csv_path = 'comparison_reports/compare_202X_XX/action_failure_eval.csv' 
df = pd.read_csv(csv_path)

# 提取不同模型的数据
baseline_data = df[df['model'] == 'baseline']
ours_data = df[df['model'] == 'optimized'] # 代码里把你们的模型叫做 optimized

plt.figure(figsize=(8, 5), dpi=300)
plt.plot(baseline_data['failure_rate'], baseline_data['success_mean'], marker='o', label='Baseline (MLP)', linestyle='--')
plt.plot(ours_data['failure_rate'], ours_data['success_mean'], marker='s', label='Ours (RWTA)', linewidth=2)

plt.xlabel('执行器故障率 (Action Failure Rate)', fontsize=12)
plt.ylabel('任务成功率 (Success Rate)', fontsize=12)
plt.title('不同故障率下的模型鲁棒性对比', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('robustness_comparison.png', dpi=300)
print("鲁棒性对比图已保存为 robustness_comparison.png")