# Robot-Lab

**2026-11-05当前进展：[持续研究报告](studies/policy_generalization/research_report.md)。** 已完成30回合相机/窗口对照、真实T4 LoRA五步训练、保存重载与离线评分；Corrector进入数据协议与核心重实现阶段，尚无监控/恢复收益结论。

下一步：[Corrector机制与实践](studies/policy_generalization/corrector_mechanism_practice.md)。训练练习已收束，旧checkpoint闭环是可选工程验证，不作为Corrector前置条件。
动手参考：[LoRA机制与CPU/GPU实践](studies/policy_generalization/lora_training_practice.md)、[单单元格远程准备](scripts/smolvla_training_prepare.ipynb)、[训练结果](studies/policy_generalization/training_results_20261105.md)。

以 LeRobot / SmolVLA 为基础，研究相机视角变化下的操作失败、动作执行窗口及计算成本。
当前完成同一任务/物理初始化下的 30 个执行窗口评测回合，包含未参与窗口选择的两个保留视角。
结果是有限改善，尚未解决所有视角下的抓取与恢复问题；当前阶段使用已有数据进行离线复核。

- [最新中文进展与复现入口](studies/policy_generalization/progress_20261101.md)
- [实验结果、计算成本与结论边界](studies/policy_generalization/replanning_results.md)
- [队列、推理耗时、仿真时间与视频播放的区别](studies/policy_generalization/timing_and_queue.md)
- [六个真实案例的视频/数据复核](studies/policy_generalization/video_review_20261101.md)
- [公开的逐回合 CSV、配置、图表与来源哈希](studies/policy_generalization/results/20261031)

公开 Notebook 保留代码和说明，清除运行输出、身份元数据与私人实验链接。
原始 Tensor、日志、视频及完整运行笔记本在本地/私有归档，不随 Git 发布；公开 CSV 和图表来自实测记录，不是 CPU 替身测试结果。

## 调用第三方包

`policies/act` 和 `policies/smolvla` 来自 LeRobot，保留原始版权声明；不将其模型实现当作本项目新算法。
