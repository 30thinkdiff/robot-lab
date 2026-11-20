**最终状态：seed42两组训练与240条Object10评测已完成。** 下文保留原六组计划及科学设置历史。[正式结果](../../results/20261120/single_layer_object60/RESULTS.md) · [交接记录](SEED42_OBJECT60.md)。

# 完整Object10 S2/S8正式训练 · 2026-11-18

**当前使用第三轮执行优化队列`training_execution_001`，从完整epoch122继承。** 第一版`training_001`保留epoch7，第一次加速`training_fast_001`保留epoch68，第二轮`training_transport_002`保留epoch122，均安全暂停。当前完整继承模型、优化器、调度器、RNG和历史；继续原FP16无损搬运与FP32目标计算，新增严格等价的延迟标量统计和训练阶段CUDA预取，科学参数不变。[第三轮实测与等价证据](EXECUTION_OPTIMIZATION.md)，[第二轮记录](TRANSPORT_OPTIMIZATION.md)。当前入口是`train_execution_manage.py`，旧入口显示paused是预期，请仅使用当前入口控制队列。

454条Episode、66,984帧的两层完整FP16特征制作完成后，重新读取两层全部文件并检查SHA、原动作/时间、Episode身份及划分，454/454通过。复核耗时599.36秒；结果位于新实验目录`training_validation_001/result.json`。该验证不使用GPU。

## 固定实验设置

| 项目 | 实际配置 |
|---|---|
| 训练顺序 | S2_s42、S8_s42、S2_s43、S8_s43、S2_s44、S8_s44 |
| 数据 | 原Object10全部454条，训练364条／验证90条 |
| 样本 | 训练49,447／验证12,089，按Episode划分，原索引不变 |
| 输入 | 当前时刻两相机原始单层特征＋当前单个7维动作；不输入未来帧/动作 |
| 模型形状 | `[B,2,1024,768] → [B,2048,768]`，无池化/额外投影/融合 |
| 模型 | 原Residual MLP四个2048隐藏宽度，每个37,266,176参数 |
| 预测目标 | 同Episode的`F[t+10]−F[t]`，10Hz数据、1秒跨度 |
| 样本锚点 | h_window=1、共同anchor_history=3，沿用原t≥2样本集合 |
| Loss | 原始MSE＋余弦损失，beta=1；不使用J归一化/过滤 |
| 优化器 | 原AdamW，lr=0.0003，weight_decay=0.0001 |
| 调度器 | 原CosineAnnealingLR，T_max=150，eta_min=0.000003 |
| Batch | microbatch32 × accumulate16，有效512；尾组按真实样本数加权 |
| 精度 | FP16特征储存；训练使用原AMP BF16，参数/优化器保留原实现 |
| 梯度裁剪 | 原L2范数上限1 |
| 上限与早停 | 最多150轮，验证**总loss严格降低**才改善；连续5轮未改善即早停 |
| 最佳模型 | 验证总loss最低的checkpoint；训练后新实例重载并复验指标 |
| CPU/loader | CPU线程4，DataLoader workers4（原2→4；搬运优化后复测4/8，最终保留4）；原FP16无损搬运，恢复FP32再作差；pin_memory开启；共享内存60GiB |
| 保存 | 每完整epoch原子更新latest.pt，保留best_model.pt；不额外生成每轮快照 |

不重新使用discarded工程探针权重。每组从相应seed初始化，原优化器、训练循环、调度器、RNG与早停实现直接复用冻结`night-code-1e80242789/corrector_train.py`。新层特征/动作/时间文件只读，新模型保存到独立训练目录。

## 查看进度和ETA

在**2号机终端**执行：

```bash
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/train_execution_manage.py status
```

每15秒刷新，Ctrl+C仅退出查看：

```bash
watch -n 15 /root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/train_execution_manage.py status
```

训练入口是`train_execution_manage.py`；特征制作的`manage_v2.py`继续显示已完成454/454，不用它查看模型训练。

| 字段 | 含义 |
|---|---|
| `runs_completed / runs_planned` | 已训练并通过最佳checkpoint重载检查的组数／6 |
| `current_job` / `current_output` | 当前模型/seed及实际attempt目录 |
| `phase` | 等待全量验收、逐层输入检查、初始验证、训练、验证或等待边界暂停 |
| `current_training.input_check` | 原训练器初始化时本层454条SHA检查的数量与剩余时间；原适配器每次启动/恢复都会重验 |
| `current_training.epochs_completed / epoch_cap` | 完整提交轮数／150，不把正在进行的半轮算完成 |
| `current_training.live` | 当前轮、反向完成的训练样本数、优化器更新计数、最近验证阶段已处理样本数（在下次验证开始时清零） |
| `best_epoch` / `best_validation` | 验证总loss最低轮及其MSE、cosine等实际指标 |
| `stale_epochs / patience` | 当前连续未改善轮数／5 |
| `eta_to_epoch_cap_hours` | 当前模型继续到150轮的剩余时间估计 |
| `eta_if_no_further_improvement_hours` | **假如此后不再改善**，当前模型触发早停前剩余时间；不是停止时刻预测 |
| `queue_eta_to_epoch_caps_hours` | 所有未完成组按最多150轮计算的队列剩余时间 |
| `estimated_finish_at_epoch_caps_beijing` | 上限轮数情景下的北京时间估算终点 |

续训的首个新完整epoch之前，使用已通过的小规模GPU吞吐验收做明确标记的临时外推；之后使用最近最多3个新epoch的训练＋验证时间，剔除继承的旧慢速轮耗时。尚未实测的S8暂借同形状S2的耗时，并在输出明确列出。输入检查、checkpoint写入、断连和资源竞争会带来额外耗时。

**早停可能大幅缩短总时间；无法事先知道何时最后一次改善，因此不提供伪精确的早停完成承诺。** 上限轮数ETA也只是按当前速度外推，不是严格的墙钟上界。

## 日志和模型文件

当前训练日志路径直接见status中的`current_log`。队列目录：

`/root/autodl-tmp/corrector/single_layer_20261118_001/training_execution_001/`

- `queue_status.json`：队列状态及当前子进程。
- `runs/<S2_s42等>/index.json`：实际attempt路径。
- `runs/<名称>/attempt_001/train.log`：原训练器日志，每轮实际loss/指标。
- 同目录`protocol.json`、`config.json`、`split_indices.npz`：原协议、形状与样本索引。
- `history.json`、`metrics.csv`：每轮训练/验证指标。
- `latest.pt`：包含优化器、调度器、最佳权重、RNG和DataLoader RNG的续训状态。
- `best_model.pt`、`result.json`：最佳模型与严格重载复核结果。
- `QUEUE_COMPLETION.json`：队列保存的完成文件SHA。

## 安全暂停与断点恢复

请求当前完整epoch结束后暂停整个队列，不启动下一组：

```bash
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/train_execution_manage.py pause
```

待paused且GPU子进程退出后恢复：

```bash
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/train_execution_manage.py resume
```

恢复使用同一完整latest，未提交的半轮重做；已完成模型校验后跳过。若在首次latest生成前启动阶段被中断，显式resume会新建编号attempt、保留旧半成品，并记录原因；不会把它冒充精确续训。源码/清单/门槛收据变化、外部GPU占用、数据错误和未知训练错误会停止并保留日志，不无限重试或自动改变batch/loss。

两盘5GiB边界和checkpoint写入余量受监控，空间不足会请求边界暂停。当前不需清理原始数据或历史Checkpoint。高cgroup内存主要为文件页缓存，启动审计未见OOM；不将宿主机总RAM当作实例可用内存。

## 工程验证和版本

原队列本地及2号机33项CPU测试通过，包括新进程模块路径传递、固定六组顺序、未提交启动保留、条件ETA，以及只读进度观察器对输出、loss、梯度范数、参数和RNG的精确一致性回归。观察器不新增模型前向或CUDA标量读取，原训练与验证张量运算保持原实现。

本次部署：`dev/slc-training-777f8060ba19`；独立ZIP SHA256：

`777f8060ba1951e8c9c775b837dc3a0700e37deec863799bc679838b32db928d`

本地ZIP/CPU日志和部署回执位于忽略目录`outputs/single_layer_training_20261118/`。完成本队列后不会自动启动新的机器人闭环评测；训练/离线验证结果与机器人成功率分开报告。

原速度审计阶段：首组前6轮约18分钟/轮，存在数据供给等待，见[SPEED_REVIEW.md](SPEED_REVIEW.md)。用户授权后已完成真实GPU管线等价验收并部署独立续训版本，本地与2号机各37项CPU测试通过。4-worker小规模训练吞吐为143.97样本/秒，对比原方案暖缓存64.02样本/秒；完整新epoch的实测时间另行记录，不把2.25倍的小规模吞吐收益直接当作全量耗时收益。

加速续训首轮已实测完成：epoch8耗时550.965秒（9.18分钟），原5—7轮均值1,066.166秒（17.77分钟），约1.935倍。已保存完整断点并进入epoch9，验证总loss0.3599367，best8/stale0。小规模吞吐ETA已由完整轮实测替换，见[FIRST_FAST_EPOCH_001.json](FIRST_FAST_EPOCH_001.json)。

11月19日第二轮验收：本地与生产快照各40项CPU回归通过，真实S2/S8输入/目标/梯度/参数更新/RNG精确一致，2,048个验证样本指标完全相同。小规模训练132.98→181.28样本/秒；当前从epoch68续接，整轮耗时以[TRANSPORT_OPTIMIZATION.md](TRANSPORT_OPTIMIZATION.md)及新status为准，不将小规模吞吐直接当作最终整轮成绩。

最终V2整轮验收：epoch69全部训练/验证完成，357.461秒（5.96分钟），原66—68轮均值502.898秒（8.38分钟），实测1.407倍；当前已保存新断点并进入epoch70，best69/stale0。新入口已切换整轮ETA；详见[FIRST_TRANSPORT_EPOCH_002.json](FIRST_TRANSPORT_EPOCH_002.json)。

第三轮执行验收：46项CPU测试通过，真实全量S2的97次更新及S8短测状态严格一致。候选先跑、原管线后跑，工程时间341.40→317.50秒（吞吐约+7.53%，不冒充正式history.seconds）；S8短测约+9.46%。原epoch122迁入`training_execution_001`，完整payload及科学协议相同。当前ZIP、显存与状态见[EXECUTION_OPTIMIZATION.md](EXECUTION_OPTIMIZATION.md)，本轮不启用多训练进程、MPS或改变batch。

第三轮正式首轮已提交：epoch123／11931更新，331.285秒（5.52分钟），旧120—122轮均值343.144秒，本轮约1.036倍，低于工程对照1.075倍。验证loss0.2598828100、best123/stale0，最新模型权重及全部验证指标与同源全量工程对照一致，已进入epoch124。当前status已采用新完整轮实测，详见[FIRST_EXECUTION_EPOCH_001.json](FIRST_EXECUTION_EPOCH_001.json)。
