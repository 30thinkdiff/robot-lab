# 实验结果总入口

**11月19日第三轮执行优化已验收并部署，从完整epoch122恢复`training_execution_001`。** [新入口与完整证据](../corrector/single_layer_20261118/EXECUTION_OPTIMIZATION.md)。延迟标量回传、训练阶段CUDA预取；全部科学设置不变。46项CPU测试、全量S2 49,447／12,089样本和97次更新状态严格一致；工程外推时间341.40→317.50秒，约1.0753倍；S8短测约1.0946倍，尚非其正式完整轮成绩。原断点和三代训练历史保留，旧manager保持paused；以`train_execution_manage.py status`查看实时状态和ETA。**正式epoch123已完成：343.14→331.28秒（5.52分钟），实测约1.036倍，低于工程对照1.075倍；best123/stale0，已进入124。** [正式首轮与权重等价收据](../corrector/single_layer_20261118/FIRST_EXECUTION_EPOCH_001.json)。

**11月19日：第二轮训练优化验收通过，当前从epoch68切换到`training_transport_002`。** [当前入口/原理/瓶颈与多GPU建议](../corrector/single_layer_20261118/TRANSPORT_OPTIMIZATION.md)。CPU保持原FP16特征值传输，GPU先恢复FP32再作同样的视觉差分；模型/精度/batch/loss/早停不变。40项CPU及真实GPU梯度/更新/RNG一致性通过，小规模训练132.98→181.28样本/秒；**完整epoch69已完成：8.38→5.96分钟，实测1.407倍；累计较最初17.77分钟约2.983倍。已进入epoch70。** [整轮收据](../corrector/single_layer_20261118/FIRST_TRANSPORT_EPOCH_002.json)。原队列和断点保留，尚未租卡、迁移或启动跨机训练。

**11月18日晚：S2/S8训练管线优化验收通过，原第7轮完整断点已迁入独立续训队列。** [真实等价验收与当前入口](../corrector/single_layer_20261118/PIPELINE_OPTIMIZATION.md)：4-worker及去除未消费历史拷贝，小规模训练143.97对64.02样本/秒；输入/梯度/参数更新/RNG、验证指标一致，两端37项CPU测试通过。原断点保留，六组科学设置不变。**首个完整加速轮已完成：17.77→9.18分钟，实测1.935倍；已进入epoch9。** [整轮收据](../corrector/single_layer_20261118/FIRST_FAST_EPOCH_001.json)。六组150轮情景剩余约136.5小时；50/60轮情景约44.7/53.9小时，不是早停预测。

**11月18日：S2/S8完整454条特征完成，正式六组训练队列已启动。** [训练参数/实时进度/ETA/暂停续接](../corrector/single_layer_20261118/TRAINING.md)、[特征制作记录](../corrector/single_layer_20261118/OPERATIONS.md)、[基线审计](../corrector/single_layer_20261118/BASELINE_AUDIT.md)。两层约392.5GiB完整SHA、原动作/时间及样本划分复核454/454通过；S2/S8分别seed42/43/44，首组S2_s42。保持四个2048隐藏宽度、原MSE＋cosine、batch32×16、AMP BF16、最多150轮/patience5，原训练器/优化器/调度器不改。ETA区分跑到轮数上限与后续无改善早停情景，不预先承诺实际早停时刻。**尚无新正式模型最终成绩或性能评测结果**；旧工程GPU/12短回合收据不混为本轮成绩。

**11月18日只读复核**：[多层缓存容量与Object10正式协议](20261118/storage_protocol_audit/RESULTS.md)、[CPU重放收据](20261118/storage_protocol_audit/receipt.json)。70条三层缓存约61.32GiB；扩至原454条/66,984帧约400.17GiB特征载荷。正式六方法采用动态LVM历史、cooldown0、保留待到期单动作预测，240监测回合30,389次阈值/触发重放全部一致。早期V3并非相同部署；J仍有分层权重/有效性校准及训练数据、精度差异。当前无卡2GiB RAM实例，未启动新训练或评测、未清理数据。

**11月17日跨任务泛化已完成：880/880、错误0、220组四方法完整配对，15:32提前结束。** [正式报告](20261117/crosssuite/RESULTS.md)、[成功率图](20261117/crosssuite/success_rates.svg)、[机器摘要](20261117/crosssuite/summary.json)、[配对获失](20261117/crosssuite/comparisons.csv)。固定原MLP与余弦OGG：累计Spatial/Goal/Long每方法60回合，H20分别47/60、49/60、42/60，LVM为38/60、42/60、35/60，OGG为38/60、43/60、31/60；90固定子集为0/40、1/40、0/40、2/40（H50/H20/LVM/OGG）。主要init0/1与补充init2分开留表。评分106501/106501、OGG3848次非零梯度，不支持原MLP收益稳定迁移的结论。[完成审计](20261117/crosssuite/completion_receipt.json)注明历史init字段为reset后游标；原数据未改写。已按需导入90的task29/init1、两个seed下的四方法共8条完整回放，其余原视频留远端；[Replay入口与LIBERO组成](20261117/crosssuite/LIBERO_SUITES.md)。

11月16日后续均已完成、错误0：[12回合运动与96回合Plus Horizon](20261116/gpu_followup/RESULTS.md)、[原24回合OGG Pilot](20261116/ogg_loss/RESULTS.md)、[新增72＋复用24的累计96回合OGG损失对照](20261116/ogg_loss_extension/RESULTS.md)。两组“96”是不同实验，不混算。H10/H5/H1为19/32、15/32、6/32，旧H20为20/32。累计loss对照J为余弦7/16、MSE9/16、组合8/16，MLP各11/16；存在个别结局翻转，尚不宣称稳定/显著提升。Replay已导入12条motion及96条loss回放，并提供六配置成败样本库与真实运动诊断。

以下旧扩展阶段最终快照截止：**2026-11-14 00:14:34 北京时间**。

- [本次完整指标报告](20261114/expanded_snapshot_20261113T161434Z/RESULTS.md)：新70条数据训练对照、普通LIBERO、Plus首轮17方法、复测快照和逐类结果。
- [机器可读摘要](20261114/expanded_snapshot_20261113T161434Z/summary.json) · [新训练汇总](20261114/expanded_snapshot_20261113T161434Z/training_runs.csv) · [逐层验证指标](20261114/expanded_snapshot_20261113T161434Z/training_layers.csv)。

| 实验 | 本次状态/规模 | 阅读入口 |
|---|---|---|
| Object训练MLP跨任务泛化 | 880回合、220完整配对；已完成，错误0 | [跨任务报告](20261117/crosssuite/RESULTS.md) |
| 历史五模型训练与闭环 | 15组训练、1200正式回合；已完成 | [历史总报告](20261111/RESULTS.md) |
| SmolVLA H50/H20/H10/H5/H2/H1 | 24回合小样本；已完成 | [Horizon Pilot](20261111/horizon_pilot/RESULTS.md) |
| 单动作/A10混合检测与引导 | 完整Pilot；已完成 | [Hybrid Pilot](20261111/hybrid_pilot/RESULTS.md) |
| 已执行动作被动监测 | 4回合；已完成，不干预 | [被动诊断](20261111/executed_consistency_pilot/RESULTS.md) |
| 已执行监测闭环对照 | 48回合；已完成 | [闭环Pilot](20261111/consistency_closed_loop/RESULTS.md) |
| 夜间扩展与预测精度诊断 | 核心600已完成，整体2486/3400按时限暂停 | [夜间报告](20261112/overnight_review/RESULTS.md) |
| Replay成败组合样本库 | 119/660按研究价值暂停；诊断样本非总体成功率 | [覆盖审计](20261112/casebank_research_v2/RESULTS.md) |
| J预测保留策略Pilot | 小样本、不同早期协议；已完成 | [保留预测Pilot](20261113/j36m_retention_preserve/RESULTS.md) |
| Object10正式六方法 | 旧240＋新J120；已完成，桥接另4次 | [六方法报告](20261113/j36m_expanded_object/RESULTS.md) |
| 70条数据新训练 | J/C0/J_equal/S2已完成，S8在24轮暂停 | [训练指标](20261114/expanded_snapshot_20261113T161434Z/training_layers.csv) |
| Plus环境/资产冒烟 | completed；17/17，错误0 | [分方法CSV](20261114/expanded_snapshot_20261113T161434Z/plus_plus_smoke/methods.csv) |
| Plus首轮（正式选定子集） | completed；544/544，错误0 | [分方法CSV](20261114/expanded_snapshot_20261113T161434Z/plus_plus/methods.csv) |
| Plus有限诊断复测 | completed；272/272，错误0 | [分方法CSV](20261114/expanded_snapshot_20261113T161434Z/plus_plus_replicates/methods.csv) |

## Plus首轮关键结果

| 方法 | 成功 | 成功率 | 覆盖率 |
|---|---:|---:|---:|
| H50 | 7/32 | 21.88% | 不适用 |
| H20 | 20/32 | 62.50% | 不适用 |
| MLP-LVM | 11/32 | 34.38% | 100.00% |
| MLP-OGG | 11/32 | 34.38% | 100.00% |
| J-LVM | 9/32 | 28.12% | 100.00% |
| J-OGG | 9/32 | 28.12% | 100.00% |

当前选定Plus子集中H20领先，J尚未显示终点收益；J覆盖率完整不等于任务成功率提高。两基础任务、多类扰动不代表官方全量Plus基准。复测按首轮差异选条件，单独报告，不并入首轮。

- [全部17方法关键指标](20261114/expanded_snapshot_20261113T161434Z/plus_plus/methods.csv)：步数、VLA调用、重规划、OGG调用、VLA/监测/回合计时、显存、覆盖率、空窗、梯度和更新量。
- [扰动类别](20261114/expanded_snapshot_20261113T161434Z/plus_plus/categories.csv) · [逐回合](20261114/expanded_snapshot_20261113T161434Z/plus_plus/cases.csv) · [严格配对差异](20261114/expanded_snapshot_20261113T161434Z/plus_plus/comparisons.csv)。
- [J逐层预测误差](20261114/expanded_snapshot_20261113T161434Z/plus_plus/prediction_layers.csv)：来自已记录RMS/cosine的推导，仅有效评分窗口，不是全量离线预测。

[Replay新增代表案例：J/MLP四方法与54条Plus完整回放](20261114/replay_representative_cases/RESULTS.md)；普通3组完整、1组旧MLP明确缺视频，Plus首轮与复测分开。

[最终复盘：配对得失、三种子复现与Replay候选](20261114/expanded_final_review_20261114_002/RESULTS.md)。

[动作块与实际运动数据审计：六回合、36块、1263步](20261114/motion_evidence/RESULTS.md)。旧数据可核验命令传递，缺逐步状态与OSC目标，不能计算运动gap；新采集器仅CPU验证，未启动GPU。

## 当前队列已完成：历史指标与状态核查

在**远程终端**运行只读健康检查，确认实际进程，而非只读旧running：

```bash
/root/autodl-tmp/corrector/bootstrap/bin/python /root/autodl-tmp/corrector/dev/replay-health-60bd07ea8d57.py
```

以下为已完成阶段的最终指标，未自动启动后续任务：

```bash
cat /root/corrector_plus_expanded_20261113_001/results/plus_replicates/execution.json
cat /root/corrector_plus_expanded_20261113_001/results/plus_replicates/analysis/overall.csv
tail -n 25 /root/corrector_plus_expanded_20261113_001/worker.log
```

首轮完整表：同一路径将`plus_replicates`换成`plus`。`analysis/trials.csv`逐回合、`analysis/summary.csv`逐条件；实际原始trial与monitor日志在`trials/`和`attempts/`。

在**本地Ubuntu终端**刷新本报告及CSV（需要可用SSH连接，密码只在终端输入；只读远程、不启动实验）：

```bash
python3 scripts/corrector_progress_report.py sync
```

本次不下载视频/权重/大张量，因此不会自动给Replay补齐所有Plus视频。按需另选少量病例导入。你之前打开的`multilayer_mvp.../remote_results_001`是早期5条数据/5轮MVP，不能当作本次70条正式训练结果。
