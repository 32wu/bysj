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