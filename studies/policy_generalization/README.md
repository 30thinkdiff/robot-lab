# SmolVLA相机扰动、执行窗口与Corrector

当前进展与行动安排统一见[工作日志](work_log.md)，按任务和日期维护；结论与证据见[研究报告](research_report.md)。

2026-11-08：472帧真实双相机特征已归档；新原始LIBERO/H50基线2/2成功。用户四个student_*训练步骤已审查并真实运行。CPU pilot完成三轮；38,052,800参数的上游宽度模型完成1epoch、独立episode验证和预测可视化，尚无跨视角泛化或在线恢复收益。用户已完成LVM/OGG三核心并通过CPU审查及真实预测器输入梯度探测；中间快照与离线评估入口已准备，流式提取与单/批闭环入口已准备，仍待T4实际冒烟。

- 运行入口：[数据与训练Notebook](../../scripts/corrector_data_prepare.ipynb)、[CPU/GPU命令行](../../scripts/corrector_train.py)。
- 技术参考：[上游源码导读](corrector_source_walkthrough.md)、[算法核心练习与审查](corrector/practice.md)。
- 既有结果：[执行窗口实验](replanning_results.md)、[T4 LoRA实测](training_results_20261105.md)、[离线评分](offline_eval_results_20261105.md)。
- 公开证据：[2026-10-31结果](results/20261031)、[2026-11-07数据与基线摘要](results/20261107)。

旧行动指南已合并，原路径仅跳转。历史文件中的“尚未提取/尚未运行”须结合日期阅读，以工作日志顶部状态为准。
