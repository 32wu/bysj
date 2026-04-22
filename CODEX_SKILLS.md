# CODEX_SKILLS.md

## 目标
这份文档用于规范我与 Codex 的后续协作，目标不是继续无限制试错，而是围绕本科毕设任务书，把项目收口成一条完整、可答辩、可写进论文的证据链：

**SVPG / 类脑强化学习方法 → Gym 虚拟环境车道保持控制 → 系统性域随机化训练 → 鲁棒性测试 → 论文级实验结果、图表、表格与结论。**

---

## 一、总原则（每次都默认生效）

1. 现在处于**毕设收尾阶段**，优先做实验矩阵收口和论文结果整理，不做无上限开放式研究。
2. 坚持**最小必要修改**：优先复用现有脚本，不大规模重构训练逻辑。
3. 不破坏现有可用 checkpoint，尤其是已有 best checkpoint。
4. 所有论文级结果统一使用**当前 120 步口径**，不再混用旧的 200 步口径。
5. 论文级结论优先使用：
   - 多 seed
   - mean ± std
   - 固定 episode 数评估
6. 不要把 dense 场景当作主收敛图唯一目标。
7. **highway-standard** 负责主收敛图和“方法有效性”主证据。
8. **highway-dense** 负责复杂交通 clean / robustness 主验证。
9. **merge、roundabout** 作为补充泛化验证。
10. 必须单独做“**系统性域随机化训练**”的消融证据。

---

## 二、实验结构（论文收口版）

### A. 主收敛实验
用途：论文主图，证明方法能在主场景稳定学会车道保持控制。

- 主场景：`highway-standard`
- 对比组：`baseline` vs `ours`
- 横轴：`total environment timesteps`
- 每隔固定 timesteps 做一次 validation
- 每次 validation：20～50 个 episode
- 至少 3 个 seed，最好 5 个 seed
- 主图指标：
  - average reward
  - success rate
  - collision rate
  - episode length
  - （可选）average speed
- 绘图要求：
  - 原始曲线 + 平滑曲线（滑动平均或 EMA）
  - 最终图必须能体现“前期提升、后期平台”的收敛特征

### B. Clean Performance 对比
用途：论文表格，证明在干净环境下 baseline 和 ours 的性能差异。

优先场景：
- `highway-standard`
- `highway-dense`

补充场景：
- `merge`
- `roundabout`

统计指标：
- average reward
- average episode length
- success rate
- collision rate
- average progress
- average speed
- lane change count（若现有脚本易于统计）

### C. 系统性域随机化 / 课程训练消融
用途：证明性能提升不是偶然，而是“类脑方法 + 域随机化训练”共同作用。

最少组别：
1. `baseline`
2. `ours`
3. `ours 去掉域随机化 / 去掉噪声课程`

时间够时再加：
4. `baseline + 域随机化`

目标：
- 解释性能提升来源
- 回答“改进来自 SVPG/SNN，还是来自训练策略”
- 对接任务书中的“系统性域随机化训练”要求

### D. 鲁棒性实验
用途：对应任务书中的“鲁棒性测试”。

优先主场景：
- `highway-standard`
- `highway-dense`

若时间允许，再补：
- `merge`
- `roundabout`

扰动类型（沿用现有脚本）：
1. `failure_rate`
2. `input_noise`
3. `weight_noise`

建议档位：
- `failure_rate`: `0 0.1 0.2 0.3 0.4 0.5`
- `input_noise`: `0 0.05 0.10 0.15 0.20`
- `weight_noise`: `0 0.02 0.05 0.10`

每个点建议：
- 20～30 个 episode
- 3～5 个 seed

### E. 泛化验证
用途：说明方法不只适用于单一场景。

场景：
- `merge`
- `roundabout`

要求：
- baseline vs ours
- 至少 3 个 seed
- clean evaluation 为主
- robustness 如果时间不够，可以降级为补充实验

---

## 三、论文最终图表清单

### 图
1. 图1：`highway-standard` 主收敛曲线
2. 图2：动作失效鲁棒性曲线
3. 图3：输入噪声鲁棒性曲线
4. 图4：权重噪声鲁棒性曲线

### 表
1. 表1：clean performance 对比（四场景或主场景+补充场景）
2. 表2：域随机化/课程训练消融
3. 表3：多 seed `mean ± std` 汇总表

### 文字分析必须回答的问题
1. SVPG / 类脑强化学习是否能在主场景完成车道保持控制并收敛？
2. 系统性域随机化训练是否有效？
3. 在哪类扰动下 ours 更鲁棒？
4. dense / merge / roundabout 下有哪些短板？
5. 结果是否与任务书要求对齐？

---

## 四、与 Codex 沟通时的固定规则

### 1. 每次都先说清本轮目标
例如：
- “本轮只做实验盘点，不要改训练逻辑。”
- “本轮只做 highway-standard 主收敛图，不要跑 dense。”
- “本轮只做 best checkpoint 评估，不要重新训练。”

### 2. 强制 Codex 按固定格式回复
建议每轮都附上这段：

```text
请严格按以下格式回复：
1. 本轮目标
2. 你检查了哪些文件
3. 你修改了哪些文件
4. 关键 diff / 关键代码片段
5. 你生成了哪些结果文件
6. 可直接运行的命令
7. 本轮结果是否已经可以进论文
8. 还有哪些必须重跑
9. 下一轮最优先任务
```

### 3. 不要给模糊命令
不要说：
- “你看着办”
- “帮我优化一下”
- “继续搞”

要说成：
- “只做 highay-standard 主收敛图，baseline vs ours，3 个 seed，输出均值和平滑曲线。”
- “只做 dense clean evaluation 和碰撞失败模式分析，不重新训练。”

### 4. 每轮只解决一个层级的问题
推荐顺序：
1. 实验盘点
2. 主收敛图
3. clean 对比
4. dense 复杂场景分析
5. 域随机化消融
6. 鲁棒性曲线
7. 论文表格和文字总结

---

## 五、最常用的 Codex Prompt 模板

### 模板 1：总控入口
```text
你现在进入“毕设论文收口模式”。
项目目标不是继续无限调参，而是围绕以下证据链完成论文级实验：
SVPG/类脑强化学习
-> Gym 虚拟环境车道保持控制
-> 系统性域随机化训练
-> 鲁棒性测试
-> 实验结果总结与论文图表

请按下面原则工作：
1. 最小必要修改
2. 不破坏现有 best 模型
3. 优先复用 compare_experiments.py / test_robustness.py / plot_robustness.py
4. 所有结果统一使用当前 120 步口径
5. 论文级结果必须尽量使用多 seed、mean ± std，而不是单 seed 最优值
6. 不要把 dense 当作主收敛图唯一目标
7. 主收敛图优先做 highway-standard
8. highway-dense 作为复杂交通 clean+robustness 主验证场景
9. merge、roundabout 作为补充泛化验证
10. 必须单独做“系统性域随机化训练”的消融证据

回复时严格输出：
1. 当前项目状态判断
2. 哪些结果能直接用于论文
3. 哪些结果必须重跑
4. 建议的实验执行顺序
5. 可直接运行的命令
6. 预计生成哪些图表和表格
```

### 模板 2：实验盘点
```text
请先不要新增实验，先盘点当前项目结果并收口。

任务：
1. 扫描现有训练日志、comparison_reports、robustness 报告
2. 按场景整理：
- highway-standard
- highway-dense
- merge
- roundabout
3. 按模型整理：
- baseline
- ours
- ours 去掉域随机化/课程（如果已有）
4. 判断哪些结果：
- 可以作为论文阶段性结果
- 可以直接进最终论文表格
- 必须重跑
5. 单独指出：
- 哪些结果仍是旧 200 步口径
- 哪些结果只有单 seed
- 哪些结果缺少 mean ± std
- 哪些 quick report 不能直接当最终结论

输出：
- 一个实验状态总表
- 一个必须重跑清单
- 一个可直接进论文清单
```

### 模板 3：主收敛图
```text
请为 highway-standard 生成论文主收敛图流程，不要把 dense 作为主收敛图目标。

要求：
1. 使用 baseline vs ours
2. 至少 3 个 seed
3. 统一横轴为 total timesteps
4. 每次 validation 20~50 个 episode
5. 输出 reward / success / collision / episode length 曲线
6. 同时输出原始曲线和平滑曲线
7. 生成可直接用于论文的 png/pdf 图
8. 给出绘图脚本命令和输入数据路径
```

### 模板 4：dense 复杂场景验证
```text
请不要把 dense 当成主收敛图，而是作为复杂交通鲁棒性验证主场景。

要求：
1. 使用当前 best checkpoint 或多 seed 最优模型
2. clean evaluation：baseline vs ours
3. 输出 success / collision / timeout / reward / progress / speed
4. 分析 collision 失败模式：
- startup
- interaction
- car_following
- lane_change
- overtake
5. 输出 dense 结果表和失败案例统计表
6. 结论聚焦：
- dense 下鲁棒性不足体现在哪
- 域随机化或课程训练是否缓解
```

### 模板 5：域随机化消融
```text
请单独完成“系统性域随机化训练”的论文证据组。

组别：
1. baseline
2. ours
3. ours 去掉域随机化/噪声课程
4. 如实现简单，再加 baseline + 域随机化

目标：
1. 明确说明性能提升来自哪里
2. 输出 clean performance 对比
3. 输出 robustness 对比
4. 结论回答：
- 域随机化是否提升鲁棒性
- 提升主要体现在什么扰动类型下
- 是否存在 clean performance 与 robustness 的 trade-off
```

### 模板 6：鲁棒性曲线
```text
请用现有 test_robustness.py 和 plot_robustness.py，正式生成论文版鲁棒性结果。

要求：
1. 场景优先：
- highway-standard
- highway-dense
2. 扰动类型：
- failure_rate
- input_noise
- weight_noise
3. 输出：
- success 曲线
- collision 曲线
- reward 或 length 曲线
4. 至少 3 个 seed 汇总
5. 最终生成均值曲线和标准差
6. 图名、坐标轴、图例、保存路径统一规范
```

### 模板 7：论文结果汇总
```text
请根据最终实验结果，自动整理论文结果材料。

输出：
1. 表1：clean performance 对比
2. 表2：域随机化消融
3. 表3：多 seed mean ± std
4. 图1：主收敛曲线
5. 图2~4：三类鲁棒性曲线
6. 一段可直接写进论文“实验结果分析”章节的文字草稿
7. 一段可直接写进“结论与不足”的文字草稿

要求：
- 文字表述客观
- 不夸大 dense 表现
- 说明 standard 上已收敛、dense 上仍有短板
- 强调系统性域随机化训练和鲁棒性测试与任务书要求一致
```

---

## 六、推荐沟通节奏（你怎么用这份 skill）

### 用法 1：先发“总控入口”
每次开一个新阶段，先发“模板 1：总控入口”。
目的：让 Codex知道现在是“毕设收口模式”，不是自由发挥模式。

### 用法 2：再发“本轮具体模板”
例如：
- 你要盘点现有结果，就接“模板 2：实验盘点”
- 你要画主图，就接“模板 3：主收敛图”
- 你要做鲁棒性，就接“模板 6：鲁棒性曲线”

### 用法 3：附加你的限制条件
例如：
- “本轮不要重新训练，只做 best checkpoint 评估。”
- “本轮不要碰 merge、roundabout。”
- “本轮只输出命令和修改清单，不要直接跑实验。”

### 用法 4：要求它输出“下一轮最优先任务”
这样你下一轮就能顺着它给的结果继续，而不是每次重新想。

---

## 七、一个实际对话示例

### 例子：你要让 Codex 先盘点现有结果
你可以这样发：

```text
先进入毕设论文收口模式。
[粘贴模板1：总控入口]

本轮只做实验盘点，不改训练逻辑，不重新训练。
[粘贴模板2：实验盘点]
```

### 例子：你要让 Codex 生成 standard 主收敛图
```text
先进入毕设论文收口模式。
[粘贴模板1：总控入口]

本轮只做 highway-standard 主收敛图，不跑 dense，不做鲁棒性。
[粘贴模板3：主收敛图]
```

### 例子：你要让 Codex 做 dense 失败分析
```text
先进入毕设论文收口模式。
[粘贴模板1：总控入口]

本轮不重新训练，只分析当前 best checkpoint 在 dense 下的 clean evaluation 和碰撞失败模式。
[粘贴模板4：dense 复杂场景验证]
```

---

## 八、最后提醒（防止走偏）

1. 主图服务于“**standard 场景下方法有效收敛**”。
2. dense 服务于“**复杂交通交互下鲁棒性验证和短板分析**”。
3. merge、roundabout 是补充，不是当前第一优先。
4. quick report 只能做阶段性参考，不能直接当论文终稿主结论。
5. 最后答辩看重的是“证据链完整、自洽、能解释”，不只是某一张图特别好看。

