# J36M-454 训练与评测

2号机工作目录为 `/root/autodl-tmp/corrector/j36m454_20261120_001`。训练结果保存在 `training/`，正式闭环结果保存在 `eval/ordinary/` 和 `eval/plus/`。独立后台队列依次执行数据核验、训练集RMS、首个完整epoch验收、正式训练、共同验证集、在线标定与真实GPU检查、普通Object10和Plus。训练和评测各有状态与日志。首次复核时间为北京时间2026年11月21日00:50。

在2号机终端运行进度总览。

```bash
watch -n 30 /root/autodl-tmp/corrector/j36m454_20261120_001/progress
```

单次查看可去掉 `watch -n 30`，机器可读输出可在 `progress` 后加 `--json`。训练剩余时间采用最近最多3个完整epoch，评测采用最近最多12个完成回合，资产传输按最近传输速率估计。尚无实测的阶段显示「待实测」，总体完成时间取决于本轮训练、验证和评测吞吐。

`manage status` 和 `manage_eval_plus status` 返回各自详细状态。相同路径下，`manage pause` 请求在Episode或epoch边界暂停，`manage_eval_plus pause` 请求在评测回合边界暂停，各用 `resume` 恢复。`manage_eval_plus` 为原评测入口设置本次Plus专用的系统库路径。Plus压缩包传输先写2号机 `evaluation_assets/assets.zip.partial`，完整SHA核验后发布。传输收据在本地 `outputs/j36m454_20261120/eval_assets_transfer.json`。

## 协议与资源

- Corrector正式训练源取冻结部署 `j36m-35ecdd7488bf`。模型、监督Loss和训练器与J70长期训练部署的对应文件逐字节相同。远端分支核对记录为 `feature/j36m-lvm-ogg-20261112/345c032`、`feature/multilayer-feature-screening-20261112/39d6f61`，本轮robot-lab基线为 `feature/robotlab-replay/091272f`。
- J36M有36,793,536个参数。训练固定seed42、FP32、60轮、microbatch8、累积4、AdamW 3e-4、weight decay 1e-4、历史余弦学习率调度与无早停。按验证Connector normalized MSE选择最优，这一选择优先反映Connector头。三层保持分辨率及0.25/0.25/0.50监督权重，保留原归一化MSE及余弦项。
- 454条Episode、364/90划分。样本锚点由旧J的 `t>=0` 对齐至原MLP和S2/S8的 `t>=2`，49,447个训练及12,089个验证样本。RMS沿用旧J对训练集全部帧与全部合法k10变化的统计口径。验证集未用于拟合RMS。共同离线对照固定在两模型共同留出的20条Episode上。
- 两层完整FP16缓存原地只读。Connector使用历史INT8及FP16 scale字节，2条来自迁移试验，其余缺失张量用原始图像、冻结编码器和历史量化函数在2号机恢复。使用原量化逐帧SHA与完整文件SHA验证后发布。454条 / 66,984帧，已完成，用时约38分37秒。2号机4090约24 GiB显存，cgroup约120 GiB内存，CPU配额16核，00:50数据盘剩余112.2 GiB。
- 128样本真实GPU测试的输入、loss、梯度诊断、权重和优化器状态一致，原搬运23.09样本/秒，无损packed、4个worker为72.21样本/秒。短测来自一个Episode，缓存和启动顺序可能影响差值。正式首轮训练加验证670.08秒、1,546次更新，完整断点通过严格重载后续跑。累计60轮约9小时27分钟。
- 迁移样本36.9 MiB，端到端2.44 MiB/s，两段理想并行上限估约3.30 MiB/s。单Episode本地恢复试验46.7帧/秒，正式批量平均28.9帧/秒。采用本地恢复，另行传输J70权重、Plus源码和资产。 [迁移收据](TRANSFER_PROBE.json)

## 在线与Replay

[预冻结评测清单](EVAL_PLAN.json)含120条普通Object10及64条历史Plus诊断回合。按最终J70在线部署复用原动态窗口15、bootstrap15、cooldown0、待到期单动作预测保留及OGG eta1/every3。层融合权重及可靠性下限按历史方法，在冻结的独立成功默认视角H50轨迹上重新标定；校准不使用正式评测条件。诊断强制触发和采集开关检查使用独立条件，正式回合使用自然触发。Plus仍是已知历史诊断条件。

只有最佳权重重载和在线GPU门槛通过后才运行闭环。队列保存逐回合控制、视频、监测、引导和运动数据，生成普通与Plus各自配对汇总。Replay导出在2号机生成带SHA的压缩包并留在远端。因本地磁盘空间有限，后续默认只读取汇总指标和必要日志；视频、动作轨迹、逐步采集文件按明确请求选取回合后下载。进度中的「本地Replay验收」在未请求导入时可以保持等待。

北京时间11月21日15:00核验训练与评测队列均完成，Object10 120/120、Plus 64/64，最终错误均为0。Plus首次初始化缺少MagickWand，随后复用1号机同版本Ubuntu包，按APT索引SHA核验后解包到独立目录，PNG测试及32个环境预检均通过。两套配对报告和Replay包已保存于2号机。[结案报告与轻量索引](../../results/20261121/j36m454/RESULTS.md)
