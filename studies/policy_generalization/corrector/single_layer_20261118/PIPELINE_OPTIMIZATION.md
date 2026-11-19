# S2/S8训练数据管线优化与精确续接

2026年11月18日晚，用户授权在不改变训练语义的前提下提高单任务资源利用率。原`training_001`通过STOP在完整epoch7、679次更新后安全暂停，并完成最佳模型重载检查；没有强杀训练进程。原latest、best、协议和历史保留。第7轮验证总loss为0.3642168540，best_epoch7、stale0。

## 变更与不变项

- DataLoader从2个加载进程改为4个，原CPU线程4、batch32、累积16、有效batch512及pin_memory不变。
- MLP专用collate不再堆叠未被模型消费的`z_hist`，保留`z_t/a_t/delta_z`逐元素及dtype一致。该入口拒绝多帧历史，不套用于Transformer。
- 不改原数据适配器、冻结训练器或第三方包；模型37,266,176参数、全部2048个视觉Token、两相机、FP16存储、BF16训练、原MSE＋cosine、lr/优化器/调度器/梯度裁剪/尾组样本加权全部保持。
- 六组顺序、种子、样本划分、150轮上限、patience5和验证总loss选best不变。优化探针更新不计入正式训练。

batch32的三个浮点特征字段原约576MiB；去掉MLP未消费的历史字段后为384MiB，减少这部分CPU堆叠、共享内存与pin搬运。更多worker不必然更快，因此实际比较2/4/8后选择4。

## 真实GPU验收

用原epoch7完整断点的独立内存副本，以及完整分辨率的真实S2/S8数据测试。每个训练候选先运行512样本的一次更新并做字节哈希检查，再测2048样本、4次更新的端到端吞吐。

| 管线 | 加载进程 | 训练样本/秒 | GPU峰值GiB |
|---|---:|---:|---:|
| 原管线，首测 | 2 | 56.94 | 8.019 |
| MLP-only collate | 4 | 143.97 | 8.020 |
| MLP-only collate | 8 | 137.80 | 8.020 |
| 原管线，最后暖缓存复测 | 2 | 64.02 | 8.020 |

以原管线两次中更快的64.02为基准，小规模训练吞吐提高2.249倍。验证另外实测1024个真实样本，吞吐59.84→142.65样本/秒，total loss、MSE、cosine等各项指标**完全一致**。

S2/S8各自的真实输入/目标、loss、裁剪前梯度、梯度范数、更新后全部模型参数、AdamW状态、CPU/CUDA RNG与DataLoader RNG字节核对均通过；各候选前2560样本顺序一致。CPU另遍历完整49,447个训练索引，核对2/4/8进程的整轮顺序及generator/RNG一致。本地和2号机最终各37项CPU测试通过，包含无未来视觉/动作输入、无Episode泄漏、checkpoint迁移与ETA不混入旧慢速轮等回归。

这里的512/2048/1024均为工程验收规模，不是正式新训练集或验证集缩减。正式训练继续使用49,447/12,089样本。单层196GiB大于120GiB实例RAM，短测试的缓存命中率不能代表整轮，**不承诺全量每轮也有2.249倍加速**。首个新完整epoch之前，status仅将吞吐外推约428秒/轮标为临时估计；随后以新完整epoch替换。

续训已进入真实反向更新。另一个约30秒只读训练窗口的GPU平均利用率为53.8%，此前独立25秒训练窗口约34.8%；两个短窗口不代表全量epoch平均。采样未见OOM/OOM-kill；接近120GiB的cgroup记账主要含可回收文件页缓存，不能当作模型匿名RAM占用。

完整验收目录为`single_layer_20261118_001/pipeline_benchmark_002`。公开轻量证据见[PIPELINE_ACCEPTANCE_001.json](PIPELINE_ACCEPTANCE_001.json)；原样完整收据及失败尝试日志保留在独立目录，不覆盖。

## 首个完整加速epoch实测

第8轮完整处理49,447个训练样本、12,089个验证样本和97次更新，耗时**550.965秒，即9.18分钟**。原第5—7轮平均1,066.166秒，即17.77分钟，首轮加速为**1.935倍**；因此不沿用小规模2.249倍作为全量收益。当前已保存epoch8的完整latest，进入epoch9。验证总loss从epoch7的0.3642168540降到epoch8的0.3599367267，best_epoch8、stale0；这只是续训进展，不是最终模型质量结论。[首轮实测收据](FIRST_FAST_EPOCH_001.json)。

status已改用新轮实测：六组都跑满150轮的条件剩余约136.5小时；假设每组分别在50/60轮结束，约44.7/53.9小时。**这些不是早停预测**，只用于预算对照，不含后续每组SHA复核、checkpoint写入及资源变化。后续使用最近最多3个新epoch滚动更新，S8未实测前暂借同形状S2耗时。六组顺序及科学设置没有擅自缩减。

## 独立续接与操作

当前入口：

```bash
# 在2号机终端查看；Ctrl+C只退出watch
watch -n 15 /root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/train_fast_manage.py status

# 请求当前完整epoch结束后暂停
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/train_fast_manage.py pause

# paused且子进程退出后恢复
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/train_fast_manage.py resume
```

新结果写入`single_layer_20261118_001/training_fast_001/`；原`training_001`保持暂停，不要并行resume旧入口。当前日志路径见status的`current_log`，模型目录见`current_output`。`inherited_epochs=7`不算新管线已完成的轮数，`optimized_epochs_completed`单列。

迁移只更改协议中的`loader.num_workers`这一明确记录的工程字段。除protocol外，latest的所有payload（含模型、optimizer、scheduler、RNG、loader_rng、best权重、历史与stale）重新读取后哈希完全一致；原始latest SHA仍为`93cae193fdd5f89277b5ee524f65dca4a6a6bce52140741522286b2a5e22ba62`。原split/history/metrics/initial_validation按字节复制，最佳权重仅同步工程协议来源。证明保存在新attempt的`ENGINEERING_CONTINUATION.json`，轻量副本为[PIPELINE_CONTINUATION_001.json](PIPELINE_CONTINUATION_001.json)。

续训已重新通过本层454/454条SHA检查，并实际进入epoch8；现已完成epoch8的全部训练/验证并进入epoch9。几分钟启动等待不计为GPU训练。源码、清单、门槛、验收收据及工程字段均绑定SHA；不绕过原协议检查、不自动改batch、精度或loss。两盘保留5GiB边界及checkpoint写入余量。

生产代码快照为`dev/slc-fast-95cdbf87a58f`，ZIP SHA256：

`95cdbf87a58fd325cfe8a12b202fc02cc91ee4994b9deea347f08b77ed09b686`

本地包、测试日志、部署回执和完整探针证据保存在忽略目录`outputs/single_layer_pipeline_optimization_20261119/`（目录为版本标识；实际操作时间为11月18日晚）。当前没有修改Replay数据库、重训其他模型或启动闭环评测。
