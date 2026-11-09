# SmolVLA相机扰动、执行窗口与Corrector

当前进展与行动安排统一见[工作日志](work_log.md)，按任务和日期维护；结论与证据见[研究报告](research_report.md)。

2026-11-09：单任务44条示范/6867帧的38,052,800参数Corrector已在T4训练并开发集验证；4090同步LVM/OGG闭环已接通，旧权重长批次2394回合、程序错误0，现已人工暂停。当前先扩大预测器训练：Object十任务454条/66984帧清单已冻结，4090特征已完成3条/472帧，原始分片三线程预下载进行中；三个完整新分片测速约2.41MiB/s。新的十任务训练尚未开始，不把旧单任务结果当成多任务或论文完整复现成绩。

- 运行入口：[数据与训练Notebook](../../scripts/corrector_data_prepare.ipynb)、[CPU/GPU命令行](../../scripts/corrector_train.py)。
- 技术参考：[上游源码导读](corrector_source_walkthrough.md)、[算法核心练习与审查](corrector/practice.md)。
- AutoDL入口：[并发预下载](../../scripts/corrector_prefetch.py)、[分片与特征流水线](../../scripts/corrector_stream_workbench.py)、[闭环评测](../../scripts/corrector_closed_loop_eval.py)。公开固定清单见[Object十任务manifest](corrector/manifests/libero_object10_8695891.json)，可用于重新打包；不包含图像、权重或账号信息。
- 既有结果：[执行窗口实验](replanning_results.md)、[T4 LoRA实测](training_results_20261105.md)、[离线评分](offline_eval_results_20261105.md)。
- 公开证据：[2026-10-31结果](results/20261031)、[2026-11-07数据与基线摘要](results/20261107)。

旧行动指南已合并，原路径仅跳转。历史文件中的“尚未提取/尚未运行”须结合日期阅读，以工作日志顶部状态为准。
