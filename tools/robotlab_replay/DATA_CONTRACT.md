# 实验保留数据约定 · Replay v1

目标：让“哪一个回合、哪一步、哪个观测、哪些动作、什么预测与触发”可以复核。适用于后续新实验；不修改旧日志或正在运行的采集器。以下为建议约定，当前适配器已支持的旧格式见README；新增字段须通过适配器接入，不能仅因为写进文件就假定界面已支持。

## 最低可用资料

| 文件 | 必须保留的内容 | 缺少时的限制 |
| --- | --- | --- |
| `plan.json` | 实验ID、研究问题、冻结方法表、cases、任务、init、评测seed、模型训练seed、配置及代码/权重SHA | 只能看散落文件，无法确认公平配对 |
| `execution.json` | running/completed/paused/error、原因、计划/完成/错误数、起止UTC、截止时间 | 不能仅凭终端退出或日志不动判断结束 |
| `trials/<id>.json` | 原case、attempt、status、success、env_steps、墙钟、VLA调用数、重规划/引导次数、GPU峰值、视频路径及SHA | 无法确认指标来自哪个回合；程序错误不当作物理失败 |
| `actual_protocol.json` | 实际加载的模型/检测器/引导器、checkpoint SHA、处理器、视觉空间、频率、horizon、阈值算法、精度/设备 | 文件名相同不代表协议相同 |
| `monitor_steps.jsonl` | step、score、threshold、有效性、触发、跳过/取消、耗时；同一步允许多条去噪日志 | 不将缺失值补成0，不把skip+cancel当作覆盖率 |
| 视频及录制映射 | 至少保留选定配对案例的原始视频，见下一节 | 无视频仍可看曲线；无映射只能独立播放 |

纯H50/H20/H10无Corrector时不必伪造monitor文件；覆盖率标“不适用”。应明确记录实际生成块长和实际执行长度，不能都只写成一个H。

## 时间基准：精确同步的必要证据

1. 分别记录数据采样Hz、仿真控制Hz、视频编码时间基准、模型推理墙钟计时范围。记录控制时间戳/环境step，不以GPU耗时当作仿真时间。
2. 推荐保存`frame_map.jsonl`：`frame_index, pts_seconds, env_step, observation_id, phase`。phase明确为`before_action`或`after_action`，并说明reset/warmup/terminal、丢帧或重复帧规则。
3. 没有显式frame_map时，至少保存录制器源码版本/SHA、原trial SHA、视频SHA、每次render相对`env.step`的顺序及截取代码。仅“20FPS、280帧、280步”不足以排除一帧偏移。
4. 现有核验视频：frame n对应提交u_n之前的Z_n；最后一次动作后的Z_N不在N帧视频内。该结论只对有独立绑定证据的视频生效，不是所有新录像的默认定义。
5. 文件搬运必须携带原trial及其哈希。相同视频SHA可出现在两个原trial中；两个来源都需单独核验，不能只凭内容相同将录制证据扩散给所有记录。

## 建议增补的逐步诊断字段

- **预测生命周期**：`prediction_id, origin_step, due_step, plan_id, created_step, action_offsets, score_valid, skip_reason, cancel_reason`。到期评分记录关联prediction_id；窗口失效不等于异常事件。
- **动作与状态**：区分VLA规范化候选动作、经过postprocessor的动作、真正传给仿真控制器的命令、机器人测量状态。分别记录处理器版本/统计量SHA。动作采集应挂在真实提交点，不能把未执行计划填充成执行历史。
- **块与队列**：真实前向调用ID/开始结束、生成块长、计划ID、队列清理原因、实际出队位置。旧版Replay推导块号会明确标记，不替代这些实测记录。
- **调用与耗时口径**：区分一次完整动作块生成、内部去噪网络前向及Corrector前向，不能把`chunk_calls`写成所有子网络前向总次数。记录CUDA同步/异步、warmup是否计入，以及共享视觉采集、监测、OGG、仿真、诊断各自计时范围。
- **LVM**：原始score、EWMA、动态threshold、阈值状态、首个可评分步、stride、连续触发计数和cooldown。覆盖率的分母由协议定义；预热和监测空窗分开。
- **OGG**：触发step/event_id、检测器生成目标的origin/due、target shape/范数/内容SHA、引导器checkpoint SHA；每次去噪索引、Loss、梯度L2、候选动作更新L2、逐动作更新L2、作用位置、eta、调用耗时。Loss下降不等于任务成功。
- **预测准度**：预测/真实变化的MSE、cosine、目标能量、视觉空间维度与相机顺序、动作条件类型（planned/executed/demo）、共同窗口ID。整回合误差可能是失败的后果，不能直接称因果关系。
- **异常标签**：任务最终失败不是每个step的异常标签。若研究检测能力，应另存接触/掉落/扰动等独立标签来源、标注区间与置信度；没有标签不输出“异常检测AUROC”。

当前工具显示已记录的JSON摘要；没有记录的梯度/候选Tensor显示缺失。若以后保留Tensor，优先限定小量安全NPZ/JSON和显式shape/dtype，不默认加载pickle/checkpoint。写trace需可选、低侵入，另测采集开销，保证不改变控制命令。

## 节省空间的保留策略

- **所有回合长期保留**：plan/actual protocol、trial、逐步轻量trace、阈值、事件、状态、SHA和汇总CSV。不要只留overall.csv，它不能重建曲线或公平配对。
- **事先指定少量配对条件，所有方法都保留视频**：包含成功、失败、纠错恢复与监测空窗。只保留失败视频会丢掉“为什么纠错后成功”的关键证据。录制真实分辨率即可，不用无意义放大。
- **仅诊断子集保留**：真实控制命令、目标/动作Tensor、逐帧视觉特征或模型输出。设单回合/总量预算，先汇总指标，避免全量视频和高维Tensor淹没50GiB数据盘。
- 保留可复核的指标计算代码、软件版本、文件清单、导出选择规则、缺失/未留存原因。视频删除策略应体现在清单中；不要事后静默删除研究者以为保留的材料。
- 数据与工具分离：原始资料只读；转换、缓存、截图和导出写独立派生目录。私人地址、命令参数中的凭据、权重和大型视频不提交公共Git。

## 导入与复查

2026-11-13扩展评测：已接入J-36M方法和Plus条件元数据；纯基线的覆盖率仍为“不适用”。原始动作/队列实测值与派生控制返回标志严格区分，后者在界面警告及actual_protocol.derivation中说明。`import_expanded.py`用于只读转换已完成J-v1归档；新采集器直接满足Replay-v1。`scripts/corrector_replay_export.py`按完成回合快照导出，保留source_execution、原计划SHA和未导出文件理由，不将运行中快照冒充完整评测。

有限复测以已观察结局签名和关键方法翻转选例，同一条件更换seed是随机复现，不增加物理状态计数；光照/语言变化也不自动视为新增物理布局。无需枚举全部理论成败组合。保存完整首轮统计，不能将筛选后的子集成功率作为总体性能。已发布视频SHA参与断点恢复校验，活动队列不能事后删视频；空间下限处先停止，再处理可回收缓存。

旧格式目录/ZIP可用`server.py --source`读取。当前服务在启动时索引；取回新视频后仅重启本地Replay服务，不重启远程实验。JSON、视频、录制证据并不要求全部复制：先取轻量结果，再按具体回合取视频/trace。当前不在线挂载远程、不提供浏览器SSH；SSHFS等额外挂载属于未来方案，不是当前已验收能力。

人工复盘注释保存在派生导出：人工锚点包含主选/对照原始step和事件说明，诊断摘要区分日志事实与缺失证据。它们不是物理阶段真值、机器人状态或失败原因标签，不能反向覆盖原trial。动作热图是实际控制数组，OGG热图是逐动作更新范数，两者与完整候选动作/梯度/视觉Tensor分别声明。

## 可选运动诊断扩展：robotlab_motion_v1

新回合可在attempt下附加`motion/protocol.json`、`motion/chunks/*.json`、`motion/steps.jsonl`及`motion/status.json`。每个真实控制周期记录提交前/返回后仿真时间、末端实际位姿、7关节角/速度、夹爪状态、真实OSC目标、限幅前后力矩与块/slot。终末观测单独保留，不能因原视频缺终帧而丢掉最后控制返回状态。

区分动作块名义轨迹、控制器实际目标和机器人实际运动；记录坐标系、单位和每步相位。归一化动作、反归一化控制输入、控制器缩放后的增量均不可称为关节目标角。OGG最终动作块不能标成无引导反事实；缺少实际状态时轨迹gap必须null。完整候选轨迹属于名义解释，不当作真实未来测量。

采集器保留异常现场，使用单回合字节预算；不额外调用模型或物理步，GPU到CPU复制和IO开销需要实测。现在仅通过CPU接口与历史命令链测试，未完成真实LIBERO/GPU验收；专用三轨迹/3D视图未实现。格式与未来最小采集开关对照命令见[MOTION_CAPTURE.md](MOTION_CAPTURE.md)。

## motion-v1回放接口（2026-11-16）

原`Replay-v1`为基础，`result.motion_capture.files_sha256`绑定`motion/protocol.json`、`status.json`、`steps.jsonl`和所引用的`chunks/NNNN.json`。可选扩展缺失时只显示“未记录”，不生成替代状态。适配层按回合惰性读取、SHA校验、case id/condition/seed/method绑定、连续步号及`after.sim_time-before.sim_time=1/control_fps`核验后，向UI提供`motion.rows[n]`。任一关键校验失败时关闭运动联动，保留该回合其他可读资料。

`before`对应视频帧n；`after`对应执行u_n后的状态，误差曲线标注为本步执行后。`goals[0]`为真实OSC目标；`nominal`来自本块记录的规则解释，不能混称未来真实状态。关节位置/速度与力矩命令/动力学作用力分开；不存在逐步关节目标角时保持缺失。可选子步默认不记录，不能声称保留完整高频力矩曲线。

新增OGG损失试验使用`protocol.ogg_loss`和`case.config.ogg_loss_mode`标识余弦、原始MSE及MSE+余弦。`monitor_steps.jsonl`保留真实引导步/去噪序号/损失/梯度/动作更新；额外`guidance_loss_details.jsonl`按实际引导调用顺序对应，分析器须先核对次数与损失一致才能关联，不按墙钟猜测。

## 跨loss组合视图与状态去重

`reviewed_replay_comparisons_v1`的`comparison_kind=ogg_loss_modes`引用既有trial及其SHA。模型/原method/loss模式需与case、protocol、result三处一致；六配置使用同一冻结控制协议、相同condition/seed、初始状态指纹及经数值核对的首候选证据。未发生实际引导时只接受全段命令一致的明确证据，不补造候选/梯度。原始trial不复制，衍生清单与证据放在忽略的`.local/comparison_sets`。

样本库分别计数测试条件、condition/seed组和已记录的初始仿真状态指纹。历史`physical_state`字段仅兼容旧condition标识，不等于已测物理状态；真实记录可通过`initial_state_sha256`另行去重。指纹也不编码相机/语言或全部静态模型资产。缺少指纹时不猜测“独立物理场景数”。
