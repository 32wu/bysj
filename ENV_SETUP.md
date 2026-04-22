# SVPG2023 环境安装与汇总

本文档根据当前服务器实际状态整理，快照时间为 `2026-04-22 UTC`，目标是让你把仓库 `git clone` 到新的云服务器后，尽量用一条命令恢复可运行环境。

## 1. 当前机器已安装的 conda 环境

| 环境名 | 路径 | Python | 用途判断 |
|---|---|---|---|
| `base` | `/root/miniconda3` | `3.8.10` | Miniconda 基础环境，不是本项目推荐运行环境 |
| `bysj` | `/root/miniconda3/envs/bysj` | `3.10.20` | 本项目主环境，训练/测试/评估都建议使用它 |

当前 shell 默认落在 `base`，但项目真正应使用的是 `bysj`。

## 2. 当前服务器硬件快照

- GPU: `NVIDIA GeForce RTX 4080`
- Driver: `580.105.08`
- `nvidia-smi` 报告 CUDA Version: `13.0`

说明：
- 项目环境里实际安装的是 `torch 2.10.0 + cu12` 这一套 Python 侧 CUDA 依赖。
- 只要服务器 NVIDIA 驱动足够新，一般不需要单独再装完整 CUDA Toolkit。

## 3. 推荐的一键安装方式

仓库里已经放好了两个文件：

- `bysj_env.yml`
- `setup_bysj_env.sh`

从新服务器执行：

```bash
git clone <你的仓库地址>
cd SVPG2023
bash setup_bysj_env.sh
conda activate bysj
```

如果环境已经存在，脚本会自动走 `update --prune`；如果环境不存在，会自动创建。

## 4. 为什么我没有直接照抄当前 bysj 环境

我在当前机器上做了实际校验，发现现有 `bysj` 环境已经发生了依赖漂移：

- 当前实装 `numpy==2.2.6`
- 项目代码依赖 `gym==0.26.2` 和 `opencv-python==4.7.0.72`
- 这组版本在当前服务器上会触发 `cv2 / gym` 导入失败

也就是说，原样导出“此刻机器里的所有包”并不等于“可复现可运行”。

因此仓库里的 `bysj_env.yml` 我改成了“项目复现优先”的版本，核心修正是：

- 固定 `python==3.10.20`
- 固定 `numpy==1.23.5`
- 保留项目实际使用的 `torch / gym / gymnasium / highway-env / mujoco / snntorch / spikingjelly`
- 保留当前你环境里已经装过、且与项目流程相关的常用视频/图像/评估依赖

## 5. 关键依赖版本

| 组件 | 版本 |
|---|---|
| Python | `3.10.20` |
| torch | `2.10.0` |
| torchvision | `0.25.0` |
| gym | `0.26.2` |
| gymnasium | `1.2.3` |
| highway-env | `1.10.2` |
| mujoco | `2.2.0` |
| opencv-python | `4.7.0.72` |
| snntorch | `0.9.4` |
| spikingjelly | `0.0.0.0.8` |
| numpy | `1.23.5` |
| pandas | `2.3.3` |
| matplotlib | `3.10.8` |

## 6. 新服务器建议先装的系统依赖

如果是全新 Ubuntu 云服务器，建议先准备这些基础包：

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg libgl1 libglib2.0-0 libosmesa6 libglfw3
```

如果你已经能正常跑 `nvidia-smi`，通常说明显卡驱动层已经就绪。

## 7. 安装完成后的自检

```bash
conda activate bysj
python -c "import numpy, gym, cv2, torch, highway_env; print('numpy', numpy.__version__); print('gym', gym.__version__); print('opencv', cv2.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
cd LANE
python run_RL_base.py --help
python run_RL_ours.py --help
```

## 8. 常用启动命令

```bash
conda activate bysj
cd /path/to/SVPG2023/LANE

python run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level standard
python run_RL_ours.py --model rwtaspk --road_scenario merge
python test_video.py --road_scenario highway --traffic_level standard --checkpoint_kind ours
```

## 9. 后续维护建议

- 新装或升级包后，优先先跑一遍第 7 节自检命令。
- 如果训练能跑但视频录制报错，优先检查 `opencv-python`、`ffmpeg`、`libgl1`。
- 如果你后面又在服务器里手动 `pip install` 过新包，建议同步更新 `bysj_env.yml`，避免仓库文件和服务器实际环境再次漂移。
