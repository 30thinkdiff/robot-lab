# Object10 完整数据单层视觉 Corrector 实验：严格控制变量开发任务

## 一、项目背景与研究目标

现有项目：

- Corrector 仓库：https://github.com/30thinkdiff/corrector
- 原始 VLA-Corrector：https://github.com/ZJU-OmniAI/vla-corrector
- RobotLab / Replay：https://github.com/30thinkdiff/robot-lab

历史参考分支：

- `corrector`：`feature/j36m-lvm-ogg-20261112`
- `robot-lab`：`feature/robotlab-replay`

请首先检查当前实际分支、最新代码、提交 SHA 和实验资产，不要假定上述分支一定是最新版本。

本轮研究目标：

**在尽可能严格保持原始 MLP Corrector 的架构、训练目标、LVM、OGG 和在线控制协议不变的前提下，仅替换视觉特征来源，研究 SmolVLA SigLIP 不同层级的视觉特征是否能改善视觉动态预测与在线执行异常监测。**

本轮只新增两种模型：

- `S2`：SigLIP Block2 + 单动作 Residual MLP
- `S8`：SigLIP Block8 + 单动作 Residual MLP

使用与原 MLP 相同规模的完整 Object10 数据，完成训练、离线评估、LVM/OGG 闭环实验与本地 Replay 导入。

**核心原则：这是严格的控制变量实验，不是重新设计 Corrector，也不是让 AI 自主优化现有算法。**

## 二、最高优先级约束：禁止未经授权的算法修改

除明确列出的必要适配外，不允许修改原始算法行为。

### 绝对禁止擅自改变

1. LVM 自适应阈值算法及状态机。
2. Threshold 的滑动窗口、Bootstrap、EWMA、MAD、触发计数、滞回与跳变规则。
3. Cooldown 参数及触发逻辑。
4. 动作块中断、队列消费、未到期预测的保留和取消规则。
5. OGG 的目标构造、Loss 公式、梯度计算路径、η 和 guidance frequency。
6. SmolVLA 基础策略、权重、Processor、动作空间及推理机制。
7. 预测时间跨度、数据和控制时间基准。
8. 原 Residual MLP 的隐藏层结构、宽度、残差结构和激活函数。
9. 原 MLP 的训练 Loss、优化器、学习率调度、checkpoint 选择规则。
10. 旧实验数据、checkpoint、配置、Replay 导入目录和结果报告。

禁止添加所谓的“稳定性改进”“检测增强”“智能阈值”“防止频繁重规划”“融合优化”“自动恢复策略”等未经批准的设计。

如果发现原算法存在缺陷，应当：

- 先提供可复现证据；
- 说明影响范围；
- 给出最小修复方案；
- 暂停涉及该算法改动的操作，等待明确授权。

不得以“工程最佳实践”为理由悄悄修改研究算法。

允许增加不影响数值和控制行为的日志、校验、异常检查和独立测试，但必须验证这些改动没有改变预测、触发或执行结果。

## 三、第一阶段：只读源码与实验协议审计

在修改代码前，必须审查以下文件及其最新对应实现：

- `scripts/corrector_online_runtime.py`
- `scripts/corrector_train.py`
- `scripts/corrector_models.py`
- `scripts/j36m/model.py`
- `scripts/j36m/reference_mlp/MLP.py`
- `scripts/j36m/reference_mlp/config.py`
- `scripts/j36m/training.py`
- `scripts/j36m/loss.py`
- `scripts/j36m/online.py`
- `scripts/j36m/calibration.py`

同时核对原始 VLA-Corrector 中的：

- `RobustThresholdState`
- `robust_threshold_step`
- `_combined_loss`
- `SiglipResidualMLP`

尤其要从旧 MLP 已完成实验的 `protocol.json`、checkpoint 元数据、实际运行入口及源码 SHA 中恢复**真实使用的参数**，而不只是读取上游默认值。

已知历史原 MLP 训练记录可从：

`studies/policy_generalization/results/20261111/training_runs.csv`

开始检查。

请输出一份 `BASELINE_AUDIT.md`，包含以下对照：

| 项目 | 原 MLP 实际运行值 | 旧 J36M 实际值 | 新 S2/S8 计划值 |
|---|---|---|---|
| 网络结构 | 必须查证 | 必须查证 | 沿用原 MLP |
| 视觉层与形状 | 必须查证 | 必须查证 | Block2/8 |
| 训练 Loss | 必须查证 | 必须查证 | 沿用原 MLP |
| 预测跨度 | 必须查证 | 必须查证 | 沿用原 MLP |
| Score 公式 | 必须查证 | 必须查证 | 沿用原 MLP |
| LVM 窗口及历史更新 | 必须查证 | 必须查证 | 沿用原 MLP |
| Bootstrap | 必须查证 | 必须查证 | 沿用原 MLP |
| Cooldown | 必须查证 | 必须查证 | 沿用原 MLP |
| OGG 参数和 Loss | 必须查证 | 必须查证 | 沿用原 MLP |
| 预测队列处理 | 必须查证 | 必须查证 | 沿用原 MLP |

已知需要特别审计的历史差异：

- 上游默认 LVM：窗口 15，Cooldown 0，在线自适应历史。
- J36M：冻结长历史、Cooldown 20、多层分数融合和低范数过滤。
- 原 MLP Loss 与 J36M Loss 的数值归一化、低范数处理可能不同。

**本轮 S2/S8 不得自动继承 J36M 的监测或训练改动。**

如果原 MLP 已完成实验中的实际参数不同于上游默认值，以可验证的原 MLP 实验协议为主要基线，并记录原因。

## 四、第二阶段：完整 Object10 数据准备

数据集：

`HuggingFaceVLA/libero`

固定历史 revision：

`86958911c0f959db2bbbdb107eb3e17c5f9c798e`

沿用原始 Object10 的 454 条 episode、364/90 训练验证划分、10Hz 数据频率和 k=10 的预测跨度。

保持原冻结 SmolVLA checkpoint、相机顺序、预处理器及动作归一化协议。

提取：

- Block2 `[N,2,1024,768]`
- Block8 `[N,2,1024,768]`

使用 FP16 原始中间层特征，不进行未经授权的空间池化、Token 丢弃或特征投影。

按 episode 分文件保存，采用 memory mapping 加载。

预计单层原始特征约 196.2GiB，两层约 392.5GiB。

新服务器为 RTX 4090，数据盘可扩容至约 600GB。必须先实际检查磁盘余量、系统内存和 GPU 显存；不得将数据盘容量当作 GPU 显存。

新增独立 `manifest454`，提供：

- SHA 校验和；
- episode/frame/timebase 对齐；
- 训练/验证划分一致性检查；
- 已验证旧缓存的安全复用；
- 可恢复的逐 episode 提取；
- 磁盘不足时安全暂停。

不得修改、覆盖或删除原 `raw70` 及已有历史数据。

## 五、第三阶段：S2/S8 单层 MLP

尽量复用现有 `ReferenceSingle` 与 `SiglipResidualMLP`。

输入：

- 视觉 `[B,2,1024,768]`
- 动作 `[B,7]`

内部仅进行必要 reshape：

`[B,2,1024,768] -> [B,2048,768]`

配置：

`token_dim=768`

输出恢复：

`[B,2048,768] -> [B,2,1024,768]`

预测目标：

`F_layer(t+10)-F_layer(t)`

不额外添加 `Linear(768,960)`、新的 Adapter、FiLM、Transformer、Attention、多层融合或新的预测头。

使用原 MLP 的单层预测 Loss，不擅自切换到 J36M 的归一化 MSE 或额外过滤设计。

保持原训练设置，包括 optimizer、scheduler、训练种子、有效 batch、训练轮数上限、early stopping 和验证指标。

由于中间层 Token 数增加，允许在显存不足时调整 microbatch 和梯度累积以尽可能保持有效 batch，但必须记录所有差异，并验证梯度累积实现正确。不得自动降低模型容量或者改变监督目标。

首先进行 CPU 单元测试、真实小 batch GPU 测试、Loss backward 和 checkpoint reload 测试。

正式训练使用新实验目录，所有输出保留完整来源信息。

## 六、第四阶段：保持原 LVM 不变

这是本轮最重要的算法约束。

必须直接复用原 MLP 使用的 `RobustThresholdState`、`robust_threshold_step` 和单动作 `OnlineCorrector` 行为。

对于 S2/S8，仅允许改变：

- 在线视觉特征提取位置；
- 输入输出张量维度；
- 对应层的预测变化与实际变化。

Score 保持原公式：

`1 - cosine_similarity(flatten(predicted_delta), flatten(observed_delta))`

不要引入 J36M 的：

- 多层加权融合；
- 冻结 1555 点正常历史；
- 预先替代原 Bootstrap 的校准机制；
- `cooldown_steps=20`；
- 基于 J36M 的逐层 norm_floor 跳过规则；
- 其他改变监测覆盖或触发时机的逻辑。

首轮保持原 MLP 实际使用的 LVM 参数。

如果正常分数分布不同，先报告分布、阈值和误报情况，不得自行调参以提高成功率。

后续如需校准 `base_noise_floor` 或其他阈值相关超参数，必须另建具有独立实验标识的消融实验，不能覆盖首轮结果。

尤其要保留原始单动作预测的队列语义，不允许重现早期 J36M 中断后清空未到期预测造成的监测空窗。

增加只读日志，记录每次评分的：

- 原始 score；
- EWMA；
- threshold_on / threshold_off；
- Bootstrap 状态；
- 连续触发计数；
- cooldown_remaining；
- prediction origin/due；
- score 是否有效；
- trigger 与实际 interrupted；
- 队列及重规划事件。

日志增强不得改变原状态机运行结果。

## 七、第五阶段：OGG

先验证 S2-LVM / S8-LVM，再运行 OGG。

OGG 的计算公式、目标构造、候选动作、梯度路径、η 和 guidance_every 必须保持原 MLP 协议。

仅允许视觉空间与维度适配。

要求验证：

- 单次 OGG 前向与反向正确；
- 预测器参数冻结；
- 梯度有限且非全零；
- 更新前后候选动作差异可测；
- 不引入额外控制步；
- 未触发时保持普通 VLA 动作行为。

不要擅自增大 η，也不要引入新的 OGG Loss 变体。

## 八、第六阶段：评测与历史对照

保留以下既有方法及其原始结果：

- `mlp_lvm`
- `mlp_ogg`
- `j_lvm`
- `j_ogg`

新增：

- `s2_lvm`
- `s2_ogg`
- `s8_lvm`
- `s8_ogg`

保留 H20 基线。

优先使用历史 Ordinary Object10 和 LIBERO-Plus 的相同 condition、initial state 与 eval seed 进行配对对照。

但必须明确：旧 MLP、旧 J 使用的训练数据规模和 LVM 协议可能不同，历史成功率对照不能直接证明纯模型架构优劣。

另外开展同轨迹被动监测实验，比较不同模型的 score、触发条件和检测时间，不让检测器介入改变该轨迹。

至少记录：

- Success rate；
- 重规划次数；
- 首次触发时间；
- score 与 threshold 曲线；
- 评分覆盖率；
- 预测离线 MSE、cosine 和零变化基线；
- OGG 梯度及动作更新；
- 运行时间和显存；
- 失败案例与成功恢复案例。

不得只根据触发次数多少判断检测性能。

如要进一步分析旧 J 的 LVM 修改，应复用原 J checkpoint 建立独立的阈值机制消融，不能修改或覆盖原 J 正式结果。

## 九、第七阶段：Replay 兼容和数据安全

读取 `robot-lab` 的最新：

- `tools/robotlab_replay/DATA_CONTRACT.md`
- `tools/robotlab_replay/MOTION_CAPTURE.md`
- 现有 Replay 导入、索引和运动数据适配代码。

新实验应兼容 `Replay-v1` 和已实现的 `robotlab_motion_v1`。

必须保留真实视频、frame_map、monitor trace、动作块、队列、真实控制器命令及可验证的实际机器人运动状态。

注意区分：

1. VLA 生成动作；
2. 经过处理器的控制命令；
3. OSC 实际目标；
4. 机器人实测位姿、速度和力矩。

不要把名义轨迹当作实际运动。

历史资料缺少某项数据时必须明确标记“未记录”，不得推断或补造。

旧四方法的所有本地 Replay 数据必须保持只读，不覆盖、不删除、不重新标记实验身份。

新增实验放入独立数据目录，只读导入本地 Replay，验收 S2/S8 能够与旧四方法配对查看视频、score、threshold、动作、控制器及运动数据。

## 十、阶段验收与执行控制

按以下顺序开展：

**阶段 A：只读审计**

提交：

- `BASELINE_AUDIT.md`
- 现有模型/数据/控制协议差异表
- 当前服务器资源审计
- 预计磁盘/显存占用
- 需要修改的代码文件清单
- 允许与禁止的变更清单

**阶段 B：最小代码适配**

完成 S2/S8 输入输出、数据工程、在线视觉 Hook 和 Replay 兼容。运行自动化测试。

**阶段 C：真实 GPU 验证**

小样本提取对齐、训练烟雾测试、LVM 原版状态机数值一致性、OGG 梯度及采集开关一致性。

**阶段 D：完整数据与训练**

只有 A–C 的门槛通过后，才能在明确训练配置和资源预算下启动 454 条数据的完整提取与 S2/S8 训练。

**阶段 E：正式评测与导出**

保存新模型 checkpoint、完整评测结果、Replay 数据以及可复核的分析报告。

所有阶段禁止覆盖已有实验；出现磁盘不足、OOM、数值异常或数据不一致时，应保存现场、停止相关任务并报告，不得擅自修改研究方案以强行继续。

## 十一、最终验收目标

我希望这次实验最终回答：

**在只替换视觉特征层级、尽量保持原始 Corrector 其他机制不变的情况下，Block2 或 Block8 是否比 Connector 更有利于未来视觉变化预测、LVM 执行异常监测和机器人任务完成？**

如果 S2/S8 最终表现更好，需要有证据解释为什么；如果没有改善，也必须完整报告，不得筛选有利案例或者继续擅自添加新模块。

特别强调：

**不要为使结果好看而自主进行算法设计。首先确保实验变量干净、源码可追溯、结果可复核。任何必要但超出本任务范围的算法修改，都必须先明确说明并获得授权。**