# SmolVLA 相机扰动与动作执行窗口

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
