# 原MLP的OGG单变量消融

本轮已完成64回合，错误0。结果、运行开销和翻转案例见[结案报告](../../results/20261121/ogg_controlled/RESULTS.md)。

首轮固定16个历史条件与评测种子组合，普通Object10和Plus各8组，每组跑以下四个配置，共64回合。使用原MLP seed42权重、原冻结SmolVLA、原LVM状态机和20Hz控制，动作块50步、去噪10步。Plus用于历史条件诊断。

| 配置 | MSE系数 | 余弦距离系数 | eta | guidance_every |
|---|---:|---:|---:|---:|
| cosine_base | 0 | 1 | 1 | 3 |
| cosine_eta2 | 0 | 1 | 2 | 3 |
| cosine_every1 | 0 | 1 | 1 | 1 |
| mixed_mse003 | 0.03 | 1 | 1 | 3 |

同域先跑2号机的余弦基线。其余配置严格匹配同机基线的初始状态、首输入、首噪声、引导前控制及首候选。首个跨机检查中，旧回合与2号机的初始仿真状态及噪声一致，首个策略输入和实际动作存在差异。旧结果保留为跨机诊断，结局比较使用本次同机基线；首次检查的记录位于 `ogg_controlled_20261121_001`。

本地[冻结历史参考](REFERENCE.json)绑定四份旧余弦归档中的条件与种子、首输入签名和动作摘要。远端文件名为 `reference.json`。运行入口为 [corrector_ogg_controlled_ablation.py](../../../../scripts/corrector_ogg_controlled_ablation.py)，源码和参考SHA见[结案摘要](../../results/20261121/ogg_controlled/summary.json)。

每次引导保存损失分量、梯度和候选动作变动，并检查实际更新和去噪索引。首个触发事件保留真正提交的动作，回合结束后使用同一个纠偏目标比较首次候选动作与最终执行动作的视觉误差。这项诊断计时单列。统计保留块推理、监测、引导、墙钟、峰值显存及结局翻转。

2号机独占GPU，按域复用一次VLA和原MLP加载，沿用原视频、时间轴及执行命令采集。完整运动状态采集留给有需要的代表性复验。视频和原始记录默认留在2号机。

在2号机执行下列命令。状态包含普通环境和Plus的已保存回合数。暂停在回合边界生效，恢复时校验并跳过已经完成的回合。

```bash
R=/root/autodl-tmp/corrector/ogg_controlled_20261121_002
"$R/manage" status
"$R/manage" pause
"$R/manage" resume
```

各域的 `trials/` 和 `attempts/` 保留逐回合结果、跟踪和录像，诊断摘要在每回合的 `ogg_ablation.json`。两域完成后生成实验根目录的 `summary.json`。首轮用于筛查机制，独立条件上的复验单独立项。
