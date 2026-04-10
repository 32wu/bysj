conda activate bysj  #激活环境

nvidia-smi  #检查GPU状态

python train.py | tee last_run.log  #实时保存运行日志


git add .
git commit -m "备份：完成于202X-XX-XX，实验结果xxx"
git push  #关机过后传输到github实时保存三部曲

conda env export > /root/autodl-tmp/SVPG2023/bysj_env.yml  #备份环境配置

watch -n 1 nvidia-smi  #实时监控显卡运行状态

htop  #监控CPU和内存

du -sh .  #查看autodl系统盘内存

df -h  #查看整个磁盘内存空间  

find . -type f -size +100M  #快速查找大文件

ps -ef | grep python  #查找python进程

kill -9 <进程号PID>   #杀死进程

git status   #查看备份状态

git log --oneline  #查看最近提交记录

git checkout -- <文件名> #撤销还没有push的更改

conda env create -f bysj_env.yml #备份的环境

python run_RL_ours.py --model rwtaspk --entropy 5.0 #训练小车大脑（默认 auto 档会自动收敛到更稳的参数）

python compare_experiments.py --device cuda:0  # 自动对比优化前/优化后，输出论文可用表格到 comparison_reports/


训练小车运行基础算法

python run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level standard  # 基线算法：直道标准车流
python run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level dense  # 优化算法：高密度直道
python run_RL_base.py --model rwtaspk --road_scenario merge  # 优化算法：匝道汇入场景
python run_RL_base.py --model rwtaspk --road_scenario roundabout  # 优化算法：环岛/弯道场景


训练小车运行改进算法

python run_RL_ours.py --model rwtaspk --road_scenario highway --traffic_level standard  # 基线算法：直道标准车流
python run_RL_ours.py --model rwtaspk --road_scenario highway --traffic_level dense  # 优化算法：高密度直道
python run_RL_ours.py --model rwtaspk --road_scenario merge  # 优化算法：匝道汇入场景
python run_RL_ours.py --model rwtaspk --road_scenario roundabout  # 优化算法：环岛/弯道场景



运行小车

python3 test_video.py --road_scenario highway --traffic_level standard

python3 test_video.py --road_scenario highway --traffic_level dense

python test_video.py --road_scenario merge

python test_video.py --road_scenario roundabout 

python3 test_video.py --road_scenario merge --checkpoint_kind base --device cuda

4.09finish
python3 test_video.py --road_scenario merge --device cuda --checkpoint_kind ours
python3 test_video.py --road_scenario roundabout --device cuda --checkpoint_kind ours
python3 test_video.py --road_scenario merge --device cuda --checkpoint_kind baseline

python3 test_video.py --road_scenario highway --traffic_level dense --device cuda --checkpoint_kind ours
python3 test_video.py --road_scenario highway --traffic_level standard --device cuda --checkpoint_kind ours
python3 test_video.py --road_scenario roundabout --traffic_level standard --device cuda --checkpoint_kind baseline

监控模式训练
python3 /root/autodl-tmp/SVPG2023/LANE/monitor_training_eta.py \
  --log /root/autodl-tmp/SVPG2023/LANE/training_runs/baseline/highway_standard/logs/log_ppo_gymip_rwtaspk_h8-8-40_none_rmsprop_0.001000_0.10_0.97000_5_0.2000_roadhighway_tfstandard_rep11.txt \
  --poll-seconds 30 --clear


  python3 /root/autodl-tmp/SVPG2023/LANE/monitor_training_eta.py \
  --log /root/autodl-tmp/SVPG2023/LANE/training_runs/baseline/highway_standard/logs/log_ppo_gymip_rwtaspk_h8-8-40_none_rmsprop_0.001000_0.10_0.97000_5_0.2000_roadhighway_tfstandard_rep11.txt \
  --once


