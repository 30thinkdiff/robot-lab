# Robot-Lab

**2026-11-08：统一入口为[按任务与日期维护的工作日志](studies/policy_generalization/work_log.md)和[研究报告](studies/policy_generalization/research_report.md)。**

已完成472帧双相机视觉缓存、原始LIBERO/H50两回合冒烟（2/2成功）、上游算法接入及CPU保存恢复检查。CPU pilot完成三轮；上游约38M预测器完成一个完整epoch、独立episode验证、权重重载与预测可视化。尚无跨视角泛化或在线恢复收益证据。
用户已完成四个student_*训练步骤，审查后真实CPU运行通过；环境、归档、日志和对照路径由助手辅助。学习重实现不冒充正式算法贡献。
运行入口：[数据与训练Notebook](scripts/corrector_data_prepare.ipynb)、[命令行训练脚本](scripts/corrector_train.py)；本地.py变更后使用[sync工具](scripts/sync_corrector_notebook.py)刷新远程快照。

以 LeRobot / SmolVLA 为基础，研究相机视角变化下的操作失败、动作执行窗口及计算成本。
当前完成同一任务/物理初始化下的 30 个执行窗口评测回合，包含未参与窗口选择的两个保留视角。
结果是有限改善，尚未解决所有视角下的抓取与恢复问题；当前阶段使用已有数据进行离线复核。

- [历史中文进展（2026-11-01）](studies/policy_generalization/progress_20261101.md)
- [实验结果、计算成本与结论边界](studies/policy_generalization/replanning_results.md)
- [队列、推理耗时、仿真时间与视频播放的区别](studies/policy_generalization/timing_and_queue.md)
- [六个真实案例的视频/数据复核](studies/policy_generalization/video_review_20261101.md)
- [公开的逐回合 CSV、配置、图表与来源哈希](studies/policy_generalization/results/20261031)

公开 Notebook 保留代码和说明，清除运行输出、身份元数据与私人实验链接。
原始 Tensor、日志、视频及完整运行笔记本在本地/私有归档，不随 Git 发布；公开 CSV 和图表来自实测记录，不是 CPU 替身测试结果。

## 调用第三方包

`policies/act` 和 `policies/smolvla` 来自 LeRobot，保留原始版权声明；不将其模型实现当作本项目新算法。
