# SmolVLA 相机扰动与动作执行窗口

总入口：[持续研究报告](research_report.md)记录问题、方法、个人工作、实验设计、已验证结果、不足与待实现项；[公开CV与岗位调研](resume_research_sources.md)说明报告组织依据。

**2026-11-05当前主线：训练练习收束，进入[Corrector机制实践](corrector_mechanism_practice.md)。** 已完成真实五步、保存重载与离线评分；LIBERO安装后导入问题已修复并通过CPU真实渲染，远程VLA闭环尚未验证。下方日期记录保留为历史。

**2026-11-05 10:09：真实T4单步、官方5步、checkpoint归档与重载推理均通过。**
[训练核对结果与下一项评测任务](training_results_20261105.md)。尚无任务性能结论，下一步为独立验证与仿真视频，然后回到Corrector主线。

**2026-11-04 17:44：T4安装、导入检查和资产下载已通过；下一步为真实batch、用户单步函数、官方5步微调及保存重载。**
[最新核对与行动安排](lora_training_practice.md#2026-11-04进展核对与下一步)区分已实现、待实测和待实现项；全参数对照尚未完成。

**2026-11-03：进入训练闭环准备阶段。** [训练与Corrector准备单](training_and_corrector_20261103.md)记录已核对接口、真实示范索引、20步命令草案和验收门槛；尚未进行真实PEFT训练。下文为上一阶段结果。

本次动手入口：[LoRA机制、实测CPU证据与单单元格GPU准备](lora_training_practice.md)。CPU单模块实验已完成，完整SmolVLA训练尚未执行。

最新状态：2026-11-01，30 个执行窗口评测回合已完成，当前不需要远程 GPU。
20 步执行窗口在默认/view_15 选择条件下均为 3/3；保留的 view_5 为 0/3→1/3、view_10 为 1/3→2/3。
每组只有三个种子、同一任务和初始化，不能报告通用泛化或实时部署已成功。

建议按 [进展说明](progress_20261101.md) → [结果报告](replanning_results.md) → [案例复核](video_review_20261101.md) 阅读。
[时间与队列解释](timing_and_queue.md)说明同步等待为何在导出视频中不可见。

- GPU 实验入口：`scripts/libero_plus_camera_check.ipynb`，固定版本、语言和初始条件，逐回合归档。
- CPU 复核入口：`review_grasp_events.py`，筛选接触/抬升/相对位移线索并生成关键帧。
- 公开实测摘要：[results/20261031](results/20261031)，包括 30 行去重回合、冻结协议、图表与哈希。
- 历史学习记录：[trace/practice_notes.md](trace/practice_notes.md)；其中早期安排是历史，不是当前待重做清单。

核心用户实现与实践：Tensor 摘要的 bool 统计、位姿差/首次事件函数、真实 GPU 实验执行与人工案例复核。
助手辅助环境接入、重复采集/归档、对照审计和制图；模型主体来自 LeRobot，不宣称提出新算法。
