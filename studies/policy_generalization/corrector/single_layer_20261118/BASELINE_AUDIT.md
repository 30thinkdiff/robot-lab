# Object10完整数据单层MLP：基线审计

状态：**本地基础实现和CPU验证；GPU/完整提取/训练/闭环未启动。** 依据用户提供的[完整任务书](TASK.md)。本次不扩容、不重启实例、不修改数据迁移队列或冻结实验。

## 实际仓库与证据

| 来源 | 实际分支/版本 |
|---|---|
| 本地robot-lab / GitHub对应分支 | `feature/robotlab-replay`，HEAD `109d39d527774b2100f6bb8978c229c047d04967`，存在先前未提交修改，未覆盖 |
| 1号机Corrector实际HEAD | `feature/multilayer-feature-screening-20261112`，`39d6f61e33be5b1d928071ab8953f5ab8030c98d` |
| 本地历史Corrector参考克隆 | `feature/j36m-lvm-ogg-20261112`，`345c032849a9a13eb56b50973ff1703e528ce687`，不宣称它是远端当前版本 |
| GitHub冻结部署归档分支 | `archive/autodl-deployments-20261114`，`6c006207279987c1febe902a5bace010e90ca5b3` |
| 原始VLA-Corrector干净源码 | `927a06da8f00d7df483459e9039219e076c846de` |

已通过GitHub连接器核查两个仓库的当前分支SHA，并只读查询1号机HEAD；没有切换、提交或推送任何仓库。新适配器先放robot-lab独立目录`tools/robotlab_single_layer`，不修改Corrector历史部署或更换origin。

审查了原训练/模型/控制入口及J的model/reference_mlp/training/loss/online/calibration，也核对了上游Residual MLP、`_combined_loss`、阈值状态机和Replay数据/运动约定。复用旧S2/S8形状适配思想，但不使用其J归一化训练器或校准控制器。

基线选择**正式六方法实际使用的`night/mlp_s42`**，Checkpoint SHA为`05d55ba9452142be66840b3bef6fad10226480b6115be4bbbcf0842116732fad`。不混入最早`queue_4090_001`学生实现的不同样本锚点/权重。历史训练CSV、完整协议快照及此前的实际控制Trace核查相互印证；本次未为审计重新加载远端大Checkpoint或运行GPU。

冻结加载器逐文件校验：

- train：`8baaf7be9afaccf4bafc9ff05a287da29e80bd578ac6ae28659c1c2c8dcfd95b`
- model factory：`811386798aa491d8c0b43c08cd400bf00f397ad380484693689a1912976c4201`
- online runtime：`32a6daa67254f6e587e43df0f5c42a0cc83e29fa17854dbab874715f8ae70a5a`
- OGG/core：`5c076c017b086254ec63c3aa8089063d8fbae5840c01c3240f7a57afce99198f`

历史协议的必要字段、454个Episode及原动作/时间/元信息SHA整理在`tools/robotlab_single_layer/historical_mlp.json`；来源绑定先前私有只读快照SHA，未填入新实验数字。

## 核心对照表

| 项目 | 原MLP实际值 | 旧J正式值 | 新S2/S8计划值 |
|---|---|---|---|
| 网络 | 4个2048宽Residual MLP，SiLU/LayerNorm，动作嵌入256；38,052,800参数 | 共享Transformer主干、三层头；36,793,536参数 | **同原MLP隐藏结构**，仅token_dim768；37,266,176参数 |
| 当前视觉输入 | Connector `[B,128,960]` | Block2/8各`[B,2,1024,768]`及Connector | 单独一层`[B,2,1024,768]`，reshape到`[B,2048,768]` |
| 动作条件 | 当前单个归一化7维动作 | 当前单动作 | 同原MLP；不输入未来10动作 |
| 训练/验证 | 454条，364/90 | 70条，50/20 | 原454条及完全相同划分 |
| 训练loss | 原始MSE＋`1−cos`，beta1；不按RMS归一化，不过滤零向量 | 每层归一化MSE＋有范数筛选的cos，0.25/0.25/0.5 | 直接调用原`_combined_loss` |
| 时间跨度 | 数据10Hz/k10＝1秒；在线20Hz/due20 | 同左 | 同左 |
| 样本锚点 | h_window1，但共同anchor3；t≥2 | 正式在线恢复首due24 | 保持t≥2，首due24 |
| 评分 | flatten后`1−cos` | 归一化有效层加权cos，保留正常校准权重/范数筛选 | 原单层flatten `1−cos`；不加norm_floor |
| LVM历史/窗口 | 自适应历史，窗口15 | 最终expanded已恢复动态窗口15 | 直接继承原单动作控制器 |
| Bootstrap | 每回合前15个有效分数；初始阈值10 | 最终expanded同左 | 同左 |
| EWMA/MAD等 | alpha0.35，floor0.25，k_on3/k_off2，margin0.02，jump0.06，连续5 | 最终阈值状态机同左；输入score不同 | 同左，不为新层调参 |
| Cooldown | 0 | 最终expanded为0 | 0 |
| OGG目标/候选 | expected_delta−observed_delta；`x_tau−t*v`首动作7维 | 同目标，分层cos | 直接调用原函数，仅视觉空间变更 |
| OGG loss/参数 | 单层cos，无MSE；eta1、每3次去噪引导，10步去噪中4次 | 有效三层加权cos | 原单层cos及原参数 |
| 单动作预测队列 | 中断只清未执行动作，保留待到期预测 | 最终expanded保留 | 继承原逻辑，不新增取消 |
| VLA与在线编码 | 固定策略FP32，50动作块、10次去噪 | VLA FP32；额外BF16多层编码器 | 原FP32策略直接挂只读Hook，无独立J编码器 |

**历史J版本不能混称**：早期V2为正常长历史冻结/cooldown20/中断取消预测；早期V3只恢复保留，仍冻结历史且冷却20。最终Object10六方法expanded恢复动态历史/cooldown0/保留预测。父类仍有旧逻辑，必须看实际子类覆盖。[11月18日完整Trace重放审计](../../results/20261118/storage_protocol_audit/RESULTS.md)已验证240监测回合、30,389次评分阈值/触发全部一致。此次新适配不继承任何J在线类。

## 完整训练协议

原MLP实际使用AdamW、lr0.0003、weight_decay0.0001，CosineAnnealingLR T_max150/eta_min0.000003，梯度裁剪1，有效batch512，micro32/accumulate16，AMP BF16，loader2/pin_memory，seed42/43/44，最多150轮、patience5，按验证**total loss**选择best。原latest保留优化器、调度器、RNG及loader RNG，完整epoch边界恢复。

新适配直接复用原`run`及其训练/评估/累积/保存函数，唯一作用域替换是数据加载入口。输出协议记录所选层、原始特征SHA、原动作SHA和适配器SHA；换层或换数据不能冒充续训。默认保留micro32；显存不足时只允许有真实依据地改micro并保持有效batch512，不自动降模型或改训练目标。

原始分组与索引重建后为49,447训练样本、12,089验证样本，两个索引SHA均与历史协议一致。不要改成t≥0或将anchor3误解成模型输入3帧。

## 严格比较仍须声明的限制

1. 原Connector保存为INT8＋FP16 scale，而任务规定Block2/8使用FP16。此缓存量化差异会随层来源一起变化，不能声称完全隔离了量化影响；不擅自新增Connector-FP16模型或将新层量化。
2. 保持历史提取路径的BF16冻结编码及FP16原始层保存；在线保持原策略FP32。此处沿用原离线/在线精度约定，不自动套用J的独立BF16在线编码器。真实图像差异需在阶段C测量。
3. 原MLP与正式J训练规模、损失和融合筛选不同，旧成绩是系统参考，不能直接解释为纯架构因果效应。
4. Block2/8采用既有**零基索引2/8**，不是临时改成按人类序数理解的index1/7。

## 资源与变更边界

**17:50更新**：用户扩容后，2号机已实测为550GiB数据盘、剩余约541GiB；98个原始Parquet已全部SHA验证。独立环境资产准备正在后台执行，未开始特征提取/训练。详见[2号机准备状态](SERVER2_SETUP.md)。下段50GiB是本轮开发开始时的旧快照。

2号机最近只读资源核查：4090 24GiB、RAM cgroup120GiB、数据盘50GiB尚未扩容、系统盘30GiB。原始数据迁移在独立目录后台进行；本次不调整传输或启动处理。重启会终止活动进程，需要先确认迁移状态，重启后按同一清单续传/校验。

两层完整FP16载荷392.484GiB（421,426,888,704字节），Connector量化载荷7.682GiB，另计原始Parquet8.76GiB、环境、权重、训练状态及余量；以扩容后真实`df`为准。新模型参数＋梯度＋AdamW状态约0.56GiB量级，**不是显存峰值**；完整Token激活显著增加，真实峰值和microbatch只能在GPU门槛测试后报告。

允许：独立适配目录、shape reshape、来源校验、bounded memmap、原训练器的数据入口适配、只读Hook与可关闭诊断、CPU测试、新计划JSON。禁止：改原隐藏结构/损失/阈值/队列/OGG/编码器权重、擅自调参、池化/投影、覆盖旧缓存/Checkpoint、自动大规模执行。

新增代码清单：`baseline.py/model.py/manifest.py/data.py/training.py/online.py/extraction_plan.py/replay_contract.py`及只读开发CLI、测试和历史协议证据。没有改原训练/评测文件、第三方包、冻结源码或旧Replay数据库。

## 当前验收与下一门槛

CPU测试覆盖实际完整宽度MLP的loss backward、OGG动作梯度及checkpoint重载；完整2048 Token用meta检查形状。数据fixture采用明确的合成值和完整形状，不冒充真实图像。原始控制器与新控制器在monitor/LVM/OGG模式、日志开/关下，动作、阈值、触发、目标、预测队列及调用数一致；只读Hook不改前向并验证相机次序。

本轮19项CPU测试全部通过，具体源码SHA与验证范围见[CPU_TEST_RECEIPT.json](CPU_TEST_RECEIPT.json)。

**未完成**：真实GPU/AMP全形状一致性、真实采集开关、完整提取调度器/续接集成、正式训练/评测队列、同轨迹被动监测采集、真实Replay导入。只有CPU和真实GPU门槛均通过，才允许后续完整训练；当前CLI没有长期执行命令。详细运行方式见[基础代码README](../../../../tools/robotlab_single_layer/README.md)。
