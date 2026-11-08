# 研究与工程工作日志：按任务、日期更新

更新：2026-11-08。以后行动安排、运行命令和进展统一在本文件更新，不再新增按天命名的指南。
研究问题、方法、正式实验结论与个人贡献另见[研究报告](research_report.md)；源码原理文档保留为技术参考，不作为另一份待办清单。

## 当前状态

| 主线 | 已完成 | 当前阻塞或下一步 |
| --- | --- | --- |
| 数据 | 3条示范/472帧原始归档与T4视觉提取，Drive SHA复查完整 | 把真实features取回本机，或下次远程复用；尚未扩大到44条 |
| 算法理解 | 用户完成配对、模型、损失和单步更新练习，审查通过 | 练习用于理解；正式训练调用上游数学实现 |
| 训练工程 | 上游算法接入、日志、独立验证、保存重载、epoch边界续训的对照路径已检验 | 4个student_*训练步骤留给用户；真实472帧训练未运行 |
| 闭环评测 | 原始LIBERO/H50新入口完成2回合，2/2成功 | 50回合计划、在线Corrector尚未执行 |

## 任务1：数据下载、特征提取与训练样本

### 2026-11-08：核对昨天结果与调用关系

这里是3条episode（示范轨迹），不是3个epoch；epoch是完整遍历一轮训练集。

| episode | 帧数 | 划分 | k=1样本对 | k=10样本对 |
| --- | ---: | --- | ---: | ---: |
| 813 | 155 | train | 154 | 145 |
| 818 | 164 | train | 163 | 154 |
| 825 | 153 | validation | 152 | 143 |
| 合计 | 472 | 训练319帧/验证153帧 | 469 | 442 |

每帧两相机256×256 RGB，共944张图像观测；原图字节、8维state、7维action、episode/frame/timestamp等保存在Parquet。原始三轨迹归档约61MiB；视觉缓存数组载荷约55.46MiB。它们不是整个LIBERO的数据。
数据Hub为HuggingFaceVLA/libero@8695891，三份物理分片file-213/214/215包含筛选所需轨迹；元信息下载不等于原始帧全量下载。

Notebook具体调用链：

```text
corrector-data-1：写出脚本快照、建立独立解释器与环境
corrector-data-3：subprocess启动 corrector_data_workbench.py prepare --download
  → snapshot_download：固定revision、allow_patterns限定meta和三份Parquet
  → episode_rows：按实际episode列筛选、按frame排序，核对长度/任务/动作
  → frames.parquet、第一帧PNG、provenance.json
  → publish：复制到.partial、核对SHA、最后发布COMPLETE
corrector-data-5：另起进程运行extract
  → 固定上游SmolVLAExtractor：双相机视觉编码
  → 上游_quantize_per_token：int8 z_q和float16 scale
  → checkpoint保存的preprocessor处理action
  → actions/raw_actions、索引、时间、metadata；逐轨迹归档
corrector_train.py
  → 校验features → 上游SiglipDynamicsDataset反量化并构造(t,t+k)
```

视觉特征实测每帧[128,960]。raw和features均位于Drive的`robot-lab/experiments/corrector_data_pilot_20261107`下；昨天保存的Notebook显示各三条COMPLETE。GPU训练新单元格尚无执行输出，不能据数据完成推断模型已训练。

### 2026-11-08：本地CPU的可行范围

- 准备原始轨迹、构造样本、训练pilot、验证和保存恢复：可以在本地.py进程中完成。
- 本地已有三轨迹raw，但没有云端真实features，也没有完整SmolVLA权重缓存。可用内存约4.5GiB、工作盘剩余约4.8GB是本次检查的瞬时值，不作为固定硬件容量。
- 优先把云端features整个目录下载/解压到`outputs/corrector_local/features`，保留每条metadata/COMPLETE和全部数组；约55MiB，不需要为体验命令行重复视觉提取。
- CPU视觉提取在理论和上游接口上可行，项目现增加`--device cpu --allow-cpu-extraction`显式分支。默认只找已有模型缓存，不自动下载GB级权重；缺失权重会停止。此分支尚未在本机完成真实模型验证，不保证速度、内存或与CUDA逐位一致。
- 如以后明确需要下载CPU提取权重，再加`--allow-model-download`；当前没有启动该下载或CPU全模型推理。CPU提取也要求完整SmolVLA依赖，不是已有轻量data-cpu-venv能直接完成。

当前10Hz数据k=10约1秒；20Hz仿真10步仅0.5秒。在线接入前核对时间跨度、相机顺序和动作尺度，不仅核对shape。

## 任务2：掌握训练脚本并进行真实Corrector训练

### 2026-11-08：用户负责四步，助手负责外围

在[scripts/corrector_train.py](../../scripts/corrector_train.py)文件开头填写：

| 顺序 | 用户方法 | 输出与核心操作 |
| --- | --- | --- |
| A | student_components | 创建上游模型并放到device；AdamW；CosineAnnealingLR；返回三者 |
| B | student_loaders | 用已分好的groups创建训练/验证DataLoader；训练shuffle，验证不shuffle |
| C | student_forward | model(z_t,a_t)；调用原上游loss_fn；返回pred及三个标量Tensor |
| D | student_update | backward；到累积边界才裁剪、step、zero_grad；返回grad_norm或None |

每个方法已列出输入、返回、可用接口和限制。先写A，不要求从零写整个平台，也不要求一次补齐4处。写完一处请审查，再进入下一处。
上游MLP/Dataset/损失仍是正式数学实现；这次学习的是把它们组装成实际训练流程，不是再次重写算法。
默认`--implementation student`，任何占位未完成都会明确报错；不会自动运行reference冒充用户实现。inspect不受占位影响。
`--implementation reference`保留之前经过检查的显式对照，用于定位工程问题；若选择此路，结果协议会标明reference，不能称学生训练逻辑已完成。
日志、校验、按轨迹划分、累积组的样本权重、验证汇总、异常处理和checkpoint保存恢复由外围提供。C在训练和no_grad验证中都会被真实调用；D接收已缩放的loss，避免把梯度累积误写成每个micro-batch都更新。

拿到真实features后，本地终端先运行（无需Notebook、无需GPU）：

```bash
"$HOME/ws_lerobot/.venv/bin/python" scripts/corrector_train.py inspect \
  --features outputs/corrector_local/features \
  --source-root outputs/source_audit/vla-corrector \
  --output outputs/corrector_local/inspect_001 \
  --device cpu --profile pilot --implementation student --k-step 10
```

完成A～D并审查后，将inspect换为fit，使用新output，增加`--epochs 3 --batch-size 2`开始小规模真实训练。现在直接fit会报告未完成，不是环境坏了。
已有协议或代码变化都要求新实验目录；同协议断连才用`--resume`。续训从完整epoch恢复，半个epoch重做。

需要自己读懂的实际结果：train/validation episode与样本数、参数量、优化器更新次数、验证cosine与MSE、零变化MSE基线、重载验证。cosine目标改善不保证幅度误差改善；训练loss下降也不代表机器人成功率提高。

### 2026-11-08：与论文规模的关系

pilot约82.5万参数；paper-width在当前D=960/A=7下为38,052,800参数，采用4个2048宽block。先验证小规模完整链路，再实测T4显存与耗时，不能把pilot说成论文规模。
现有3条→完整单任务44条/6867帧→预算内约200–400条多任务是工程阶段目标，后两阶段未下载完成，不是论文已验证数据量。
全量特征估算约31.38GiB，超过15GB Drive，需分组或扩容。详细原论文核对和既有命令保留在下方11-07历史。

## 任务3：闭环基线、失败诊断与后续在线Corrector

### 2026-11-08：已有证据与未完成项

默认原始LIBERO task0，真实初始状态0/1、seed1000、H50，两回合完成且成功，平均3次动作块生成。只是2/2冒烟，不是完整benchmark。
早期Plus窗口实验已完成30回合；view_5/view_10是开发条件，不应重复作为未见测试。尚未训练在线Corrector或得到Corrector任务收益。
先完成真实预测器训练和独立验证，再依次接入只监控、事件触发普通重规划、OGG；保留H50/H20对照及调用/耗时。视频由闭环评测产生，不是离线训练自动产生。

## 任务4：本地开发、远程执行、提交与知识积累

### 2026-11-08：本地.py到底怎样送到Colab

原Notebook首格包含源码字符串快照，以write_text写到`/content/smolvla-training/*.py`，后续subprocess用指定Python执行这些文件。它并不是跨机器直接import本地Ubuntu上的.py。
因此只保存本地.py不足以更新远程。现在每次修改脚本后执行：

```bash
python3 scripts/sync_corrector_notebook.py
```

此命令刷新Notebook内嵌脚本并保留本地输出。保存Notebook后，在远程重跑环境同步格，核对打印的训练脚本SHA256；再运行inspect/fit。
Notebook只是启动器，实际训练仍是独立.py进程；以后租用服务器可直接运行同一命令，省去Notebook这一层。VSCode文件在本机不代表远程内核拥有同一路径。

### 2026-11-08：文档与公开提交约定

后续任务/日期/命令/当前阻塞仅追加或更新本文件；研究报告提炼已完成证据，不复制多份待办。
旧新指南合并到下方历史，并保留原路径跳转；已有原理导读和实验结果报告继续保留。
提交包含代码、学习占位、测试、统一日志和脱敏实验摘要。完整数据/权重/视频/私人路径日志留在被忽略的outputs；公开Notebook清除运行输出，保留本地完整副本与本地可读输出，避免推送账号和会话信息。

## 下一次工作的最短路径

1. 用户先完成任务A，助手审查；不要求默写其他训练流程。
2. 将已保存的features取回本机，运行CPU inspect确认299/143样本对；不重提视觉特征。
3. 完成其余训练步骤并检查后，CPU pilot真实训练3epoch；后续T4仅做更大模型/数据与闭环评测。

以下为合并的11-07历史，保留当时已完成/待完成状态；与当前状态不同时以上方11-08为准。

<details>
<summary>2026-11-07｜任务1历史：数据与存储（历史原文）</summary>

# 数据在磁盘上是什么、在哪里运行、怎样保留评测

2026-11-07。当前只给出已验证的三轨迹数据链路，不把代码准备算作预测器已训练。

## 1．下载的LeRobot数据：先看具体revision，而不是只看仓库名

当前HuggingFaceVLA/libero固定8695891，meta/info.json中codebase_version=v3.0，图像feature的dtype=image。
这里的v3.0是数据格式版本，与LeRobot Python包0.6.1不是同一个版本号。

```text
dataset/
  meta/
    info.json                   字段、dtype、shape、FPS、路径模板
    stats.json                  状态/动作等统计量
    tasks.parquet               任务编号与语言
    episodes/chunk-000/file-000.parquet   episode边界、长度等元信息
  data/chunk-000/
    file-213.parquet            实际核对过包含episode813
    file-214.parquet            实际核对过包含episode818
    file-215.parquet            实际核对过包含episode825
```

Parquet是带类型的二进制表格，概念上像表，每行代表一个记录时刻；不是.py文件，也不需要用文本编辑器打开。
当前每行字段实际包括：

| 字段                              | 磁盘中的值/结构                                               |
| --------------------------------- | ------------------------------------------------------------- |
| observation.images.image / image2 | struct：bytes是PNG压缩字节，path是图片名字；解码后[256,256,3] |
| observation.state                 | 8个float的列表                                                |
| action                            | 7个float的列表，同一记录时刻的动作，不是已生成50步动作块      |
| episode_index / frame_index       | 轨迹编号、轨迹内部帧编号                                      |
| timestamp / index / task_index    | 时间戳、全局行编号、任务编号                                  |

同一Parquet可以包含多条episode，不应由文件名直接推断episode。这个revision还出现过元信息文件号与实际分片不符，已按行核对。
一般LeRobot v3数据也可能把视觉流放在videos/下的MP4，Parquet保存低维字段，靠元信息与时间索引关联。
但本份数据两路图像已在Parquet里；info.json存在video_path模板不代表这些image字段实际使用MP4。
一般格式说明见[官方文档](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)，具体处理以实际feature与文件内容为准。

## 2．提取后的数据不是另一个LeRobot原始数据集

原extract输出一套数组和metadata；本项目外围将其按episode拆开，另保留帧编号、时间、原动作和完成清单：

```text
Drive/…/corrector_data_pilot_20261107/
  raw/episode_813/
    frames.parquet              筛选出的原始155帧，仍含两相机图片和动作
    image_frame0.png / image2_frame0.png
    provenance.json / COMPLETE.json
  features/episode_813/         GPU完成后才会出现有效目录
    z_q.npy
    z_scale.npy
    actions.npy
    raw_actions.npy
    episode_index.npy / frame_index.npy / timestamp.npy
    metadata.json / COMPLETE.json
```

| 数组                        | shape   | dtype         | 意义                                             |
| --------------------------- | ------- | ------------- | ------------------------------------------------ |
| z_q                         | [N,L,D] | int8          | N帧、两相机合计L个token、每token D个特征的量化值 |
| z_scale                     | [N,L]   | float16       | 每帧每token的量化尺度                            |
| actions                     | [N,7]   | float16       | 当前checkpoint保存的processor所定义尺度的动作    |
| raw_actions                 | [N,7]   | float32       | 原示范动作，供核对                               |
| episode_index / frame_index | [N]     | int32 / int64 | 确保监督不跨轨迹且时序正确                       |
| timestamp                   | [N]     | float64       | 保留实际记录时间                                 |

N=155是这条轨迹帧数，不是训练batch_size；L/D以实际视觉输出为准。当前GPU提取尚未执行，不能把预期形状说成实测。
反量化为float32特征：z≈z_q.float32*z_scale.float32[...,None]。
delta_z不是原始文件里人工标好的类别，而是Dataset取两帧后做z_future-z_current得到。
不需要为每个k重复保存所有特征，改变k只需重建行号对。
训练使用Corrector的缓存Dataset，不用再把这些.npy目录交给LeRobotDataset或重复视觉推理。

## 3．脚本与Notebook：不是本地/远程二选一

corrector_data_workbench.py是处理程序；corrector_data_prepare.ipynb负责选择路径、启动程序、显示日志/图片。
它们在Notebook所连接的Python内核所在机器执行，文件在本地VSCode中打开不代表计算在本机。
现有Notebook已内嵌处理脚本，执行首格会把它写到远程/content，因此不强制clone整个仓库。
也可以clone仓库后直接运行.py，但克隆只恢复代码，不恢复已安装依赖、数据或模型缓存。

| 机器/会话  | 做什么                                                 | 留什么                                                        |
| ---------- | ------------------------------------------------------ | ------------------------------------------------------------- |
| 本地Ubuntu | 阅读、写核心方法、小张量检查、少量CPU结果分析          | 代码、报告、小摘要；不再下载完整大数据                        |
| Colab CPU  | 下载选定原始数据、筛选/校验、后续CPU结果统计与视频查看 | 归档写Drive；/content副本可丢弃                               |
| Colab T4   | 冻结视觉模型提特征，之后训练/运行机器人实验            | 完成的特征、checkpoint、回合结果/视频写Drive                  |
| Drive      | 跨会话持久化                                           | 数据/特征/权重/日志/视频和来源SHA；不是已经安装好的Python环境 |

本地CPU当然可以跑prepare/verify，但无需为了执行它继续占用紧张的本地磁盘。
CPU↔GPU切换后，按新会话处理：挂载同一Drive、重建对应轻/重环境，读取已完成归档。
环境文件夹不能简单跨Python/Torch版本复制就当作已恢复环境。训练时将本轮特征缓存复制到临时盘，避免每个batch从Drive随机读大量小块。

## 4．现在Notebook怎样用

入口：[corrector_data_prepare.ipynb](../../scripts/corrector_data_prepare.ipynb)。

1. Colab界面选CPU，首格RUNTIME_MODE='cpu'。只安装独立的pyarrow/Pillow/Hub/NumPy，不加载VLA或模拟器。
2. 运行原始示范准备格：下载缺少的固定分片，选三条轨迹，验图像、帧编号和动作，逐条保存Drive。
3. 切换T4会话，首格改RUNTIME_MODE='gpu'，重新挂载同一Drive并准备模型依赖。这个变量不负责切硬件，硬件在Colab界面选。
4. 重跑原始准备格会先验Drive已有轨迹，全部完成则不用本会话保留下载缓存、不再次下载。
5. 显式打开RUN_GPU_EXTRACTION才提取。当前入口尚未远程实测，先一小批核对，再扩大规模。

缓存完成标记最后写入，重启时验SHA再复用；中断的当前episode重做，已完成的跳过。
这只是数据续跑；真正训练的optimizer/scheduler/RNG/step续训仍需后续训练入口实现，不能混为一谈。

## 5．核心方法框架和真实调用入口

见[五处实践接口](corrector/practice.md)。初始化已调用你的make_transition_pairs；你接着写getitem、MLP、loss和单batch更新。
CPU检查入口scripts/check_corrector_practice.py逐个真实调用它们；目前保留NotImplementedError，退出码2表示尚未完成。

## 6．批量评测：先沿用真实基线，不伪造Corrector结果

已有基础是30回合配对、初始状态/首噪声/语言审计、动作块耗时、视频和物理事件采集。
新增batch_evaluation.py负责冻结条件×seed×方法、逐回合状态、重试、统计及视频索引；不是并行训练mini-batch。
优先顺序执行，n_envs=1；小规模冒烟通过后再扩大条件和seed，避免一次在T4启动大量模型/环境。

[旧相机Notebook](../../scripts/libero_plus_camera_check.ipynb)新增第20节：

- 20a定义调度器。H50/H20 callback调用已有真实run_controlled_camera_pilot，RUN_BATCH默认False。
- Corrector callback尚未接入，训练checkpoint和对应SHA未固定前不能正式运行该组；不回退成基线冒充Corrector。
- 20b可以在CPU重新读旧实验统计；20c选择失败回合播放视频，无需加载模型。

每组保存计划数、完成数、成功数、错误数与pending。成功率=successes/completed，断连/程序错误不是任务失败。
环境success=False且正常执行完才算失败。完成覆盖率另看completed/planned，不能只报已完成子集的百分数。
每回合有独立attempt目录与状态JSON；中断重试保留旧日志，已完成且视频哈希正确的跳过，配置改变需换实验目录。
保存summary.csv/json、逐回合trials.csv、failure_videos.json和可点击HTML索引；模型调用数、policy/monitor耗时分别记录，缺少的指标不填0。
新baseline wrapper只修改当前实例的录像FPS为控制频率20，不修改物理控制规律；完整新批量GPU执行尚未实测。
同seed只是名义配对条件，还要沿用之前对实际状态、语言和首噪声的Tensor审计；不能据此声称全程随机性完全一致。

本地CPU已实际读取旧12回合：view_5 H50/H20=0/3与1/3，view_10=1/3与2/3；8个完整失败回合进入视频索引。
已用ffprobe读取一段真实失败视频：280帧、80FPS。旧片播放约3.5秒不等于20Hz控制下的14秒；复核事件按控制帧编号理解。
当前只是旧结果重新整理，不是新增12回合，更不是Corrector改善成绩。

CPU复核命令：

```bash
python scripts/corrector_evaluation_report.py \
  --legacy-root outputs/smolvla_heldout_camera_20261031_001 \
  --output outputs/corrector_eval_audit_20261107/legacy_report
```

后续方法比较至少报告H50、H20、只监控、事件截断普通重规划；OGG另作一组，避免把所有组件收益混在一起。
原view_5/view_10已经看过，用于开发；最终评测需提前固定新的条件/初始化与预算，不能看完再改阈值后称为未见测试。

</details>

<details>
<summary>2026-11-07｜任务2历史：正式训练准备（历史原文）</summary>

# 从核心练习转入真实Corrector训练

2026-11-07。练习实现用于主动理解与对照；正式训练默认调用固定上游927a06d，不把练习代码当作必须维护的算法分支。

## 已经拿到什么

数据来源是HuggingFaceVLA/libero@8695891。下载了元信息和file-213/214/215三个数据分片，从实际episode列筛选并归档以下轨迹；不等于下载了整个数据集。

| episode | 帧数 | 当前划分             | k=1时的时序对 | k=10时的时序对 |
| ------- | ---: | -------------------- | ------------: | -------------: |
| 813     |  155 | train                |           154 |            145 |
| 818     |  164 | train                |           163 |            154 |
| 825     |  153 | validation           |           152 |            143 |
| 合计    |  472 | 319帧训练、153帧验证 |           469 |            442 |

一个时间点包含两张256×256 RGB图像，因此共944张图像观测，图像字节存储在Parquet中，不是944个独立PNG文件。另含状态、7维动作、episode/frame/timestamp等字段。
原始选定轨迹在raw/episode_*中，本地同一归档约61MiB；每条还保存两相机的第一帧PNG便于核对。
features/episode_*保存int8 z_q、float16 scale和归一化动作、原始动作、帧/轨迹索引和时间戳。视觉特征实测每帧[128,960]，当前三轨迹数组载荷约55.46MiB，文件头/归档标记另计。

云盘根目录：`MyDrive/robot-lab/experiments/corrector_data_pilot_20261107`。
raw/features都逐轨迹SHA校验后写COMPLETE；临时环境的完整Parquet下载缓存、模型权重和Python环境不等于已备份到Drive的数据。
已经完成两个原始LIBERO/H50基线回合，2/2成功；尚未完成正式50回合基线或真实Corrector训练。

## CPU和GPU分别负责什么

图像→冻结视觉特征已经用T4完成。现在MLP训练读取特征，不再加载整个SmolVLA；因此小缓存、小模型的真实训练可以在CPU完成，不是必须有T4。
本地目前没有下载云端的真实特征，助手只运行了明确标注的小构造缓存验证。要在本地训练，先把已完成features目录完整取回，包括metadata/COMPLETE与数组，不能只拿PNG。
已有Colab CPU数据环境只有下载/解码依赖，并没有Torch；Notebook的CPU训练入口使用宿主解释器先核对Torch/NumPy，GPU训练沿用已有隔离venv。
论文宽度模型、批量视觉提取和闭环VLA评测优先T4或租用GPU。不能仅凭MLP参数量承诺T4一定能放下论文batch512；需实测峰值和单epoch时间。

## 正式入口怎样接到上游

`scripts/corrector_train.py`：

```text
校验逐episode归档 → 复制到临时本地盘
  → 上游SiglipDynamicsDataset，各轨迹独立配对
  →
  → 上游SiglipResidualMLP
  → 上游train.py中的_combined_loss
  → AdamW、梯度裁剪、cosine scheduler
  → 独立轨迹验证、最佳权重、完整epoch续训
```

损失加载器只从固定原文件加载两个纯函数，不执行原train()，避免额外导入W&B和默认随机拆分；函数正文未重写，上游文件未修改。模型/数据集/损失来源路径与文件SHA进入实验协议。
外围训练循环由项目提供，加入归档、逐轨迹划分和续训；这属于工程适配，不声称逐行照搬上游训练器。用户练习方法不在调用链上。

模式inspect只校验和打印，不更新参数；fit显式启动；resume从最近完整epoch恢复model/optimizer/scheduler、随机状态和DataLoader生成器。中断epoch重做，尚未支持任意batch位置恢复。
同目录改变数据、预测间隔或优化配置会拒绝续训；以新实验目录表示新配置。

## 下一次连接时怎么做

继续使用[原数据Notebook](../../scripts/corrector_data_prepare.ipynb)，不另起一套环境。

1. 运行首个环境同步格，写出新训练脚本。raw/features已完整时不重做提取，基线也不用重跑。
2. 新增“冻结特征上的真实Corrector训练”一节，保持TRAIN_PROFILE='pilot'，先将RUN_TRAIN_INSPECT=True。
3. 读报告确认train为813/818、validation为825，k=10时299/143对；然后打开RUN_CORRECTOR_TRAINING，跑3个epoch。
4. 看history/曲线、独立轨迹损失、预测幅度和零变化基线MSE，以及checkpoint_reload_verified。训练损失下降本身不代表成功率提升。
5. 中断后同环境配置重建，使用同一TRAIN_RUN，RESUME_TRAINING=True；已保存epoch可以复用。

输出在`predictor/<profile>_k10_<run>/`：train.log、protocol.json、config.json、initial_validation.json、metrics.csv、history.json、latest.pt、best_model.pt、result.json和Notebook生成的learning_curves.png。
latest含优化器/调度器/RNG/最佳状态，best只作推理导出；最后实际重载best并重算验证指标。每个epoch先写临时文件再替换，云盘突然断连仍须在恢复时检查完整性，文件损坏会明确报错。
离线特征训练不产生机器人执行视频；视频和任务success来自之后的闭环策略评测。
cosine目标不直接约束预测幅度，所以cosine改善而MSE未下降不一定是代码错误，也不能解释为数值预测整体更准；若要比较both目标，应另建实验编号并保留对照。

## 论文规模与资源预算

核对[论文v2附录C.1–C.4](https://arxiv.org/html/2607.01804v2#A3)：四个2048宽残差块，约38–42M参数；AdamW lr=3e-4、weight_decay=1e-4，cosine调度最低为初始lr的1%，30epoch、早停5。部署h1-k10配置使用cosine目标、batch512；ratio-sweep另用128/256。原始代码默认配置不等于论文所有实验配置。
本项目当前D=960、A=7时，上游结构实际参数量为38,052,800；pilot为825,024，不把pilot当成论文规模。
paper-width使用同四块宽度，初始micro-batch8、累积64次，有效batch上限512；尾组按实际样本数归一化。小batch累积是资源适配，不能声称与A100物理batch的浮点结果逐位一致，T4峰值尚待实测。

论文数据缩放按episode先固定80/10/10的训练池/验证/测试，多seed研究要求每个集合至少20条。当前2/1轨迹没有测试集，只是管线试运行。
目前核对的说明没有能直接对应我们SmolVLA/LIBERO组合的完整训练轨迹绝对数量，不能把任意N条宣传成复现了论文数据规模。

扩展路线按预算设定，不是论文成绩：

1. 现有3条：真实训练/验证/保存恢复跑通。
2. 完整单任务44条、6867帧：约0.788GiB特征，先检验同任务独立轨迹泛化；余下41条物理分片映射与下载提取尚未实现。
3. 约200–400条多任务示范：在实际总帧数预算内选择任务，正式冻结80/10/10 episode清单；该规模才有条件满足每个集合至少20条。继续保留按任务的统计，不只报总体均值。
4. 全库1693条/273465帧若按当前特征布局约31.38GiB，超过15GB Drive总容量；不能承诺当前云盘保存全量。可按任务分组训练和报告，或以后扩容再统一训练；分组训练不等同于全库单模型训练。

扩展时原始分片先存Colab临时盘，特征提取校验后保留特征与元信息，不长期双份保存全库图像。给checkpoint/日志/视频留至少约2GiB预算，不能把15GB全部拿来放特征。
训练数据10Hz，k=10预测约1秒后的变化；原在线仿真20Hz，10步仅0.5秒。上线前必须显式对齐时间和动作归一化；现缓存可重新构造不同k的时序对，不必为改k重复视觉提取。

## 这次需要掌握的工程能力

不要求背训练平台。先能操作并解释三件事：inspect怎样确认输入协议；一处配置怎样改变实际训练而非只改打印；checkpoint恢复了哪些训练状态。
运行后自己给出一次判断：验证方向误差是否改善、MSE与零变化预测的差别、是否出现训练下降而验证恶化。助手负责检查证据和定位异常，随后再接入在线监控，不用新语法练习拖延真实实验。

本轮验证：62项CPU检查通过；构造缓存下运行了真实CLI的3epoch更新、验证与最佳权重重载。相同CPU环境下，中断在完整epoch后恢复的模型权重与连续训练逐位一致；梯度累积的尾组权重、配置变更拒绝、缓存损坏拒绝及导入路径不污染均有检查。构造数据只是工程验证，472帧真实缓存训练和T4训练尚未运行。

</details>

<details>
<summary>2026-11-07｜任务3历史：资源与基线核对（历史原文）</summary>

# Colab资源、公开数据与SmolVLA基线核对

核对日期：2026-11-07。入口：[数据与独立基线Notebook](../../scripts/corrector_data_prepare.ipynb)。

最新执行状态：独立Torch回退和三轨迹GPU特征提取已在远程通过；实测472帧、每帧[128,960]、7维动作，Drive复查完整。
随后Matplotlib后端错误已修复，远程repair-sim与两个原始LIBERO/H50基线回合通过，2/2成功。正式上游Corrector训练入口已准备，但真实472帧训练仍待执行；见[训练运行说明](corrector_training_runbook.md)。下方保留此前排错过程。

## 当前T4错误与修复

Notebook原始输出确认CPU成功处理813/818/825并在Drive写入COMPLETE，合计472帧。
切换T4后失败在prepare的Torch版本约束检查，尚未提视觉特征。旧日志没有打印实际Torch版本，不能猜成某个具体版本。
硬件输出为Tesla T4、系统内存12.67GiB、可用约11.25GiB、临时磁盘112.54GiB。

修复：首格自动识别实际分配硬件，记录Torch版本/来源/CUDA；保留LeRobot的torch>=2.7,<2.12要求。
宿主不兼容时用独立torch-runtime-2.11-cu128安装官方Torch2.11.0、torchvision0.26.0、torchaudio2.11.0，
在新的解释器验证CUDA张量运算与torchvision算子，再准备项目venv。不在已经import Torch的宿主进程里降级，不改Colab全局安装。
兼容宿主仍走原来的复用路径；新分支首次需下载CUDA wheels，后续同会话复用。
安装命令来源：[PyTorch官方历史版本](https://pytorch.org/get-started/previous-versions/)。
CPU回归验证覆盖选择分支和隔离目标；本地没有执行CUDA wheel大下载，修复后远程GPU安装/推理仍待重试。

重新执行更新后的首格即可；CPU原始数据格可重跑，已归档数据会先验SHA再跳过，无需重做三轨迹。
旧Notebook报错和图片输出保留；完整修改前副本位于outputs/corrector_env_fix_20261107/before_torch_fallback.ipynb。

## Colab CPU是不是免费、是否影响GPU

官方FAQ说明Colab可免费使用，但资源不保证且不无限；会话寿命、空闲超时与总体限额动态变化，不公布固定数值。
所以免费标准CPU可用，不应称为“无额度限制”。官方也没有公布CPU/GPU配额的完整联动算法，不能保证账号限额完全独立。
官方明确建议不使用GPU时将硬件加速器设为None，以减少触及GPU使用限制。
仅把RUNTIME_MODE或device改成cpu并不会释放界面上分配的T4；需要在Colab界面切换运行时。
无需因此默认全程占用GPU：长时间下载/整理/分析可用CPU，实际视觉推理和评测用T4。
依据：[Colab官方FAQ的资源限制及GPU未使用说明](https://research.google.com/colaboratory/faq.html)。

## 实际公开数据清单（仅下载元信息）

通过Hub API固定SHA清单与info.json核对，不凭网页显示的“行数”推断训练帧数：

| 数据集 | 固定revision | 轨迹/帧数 | FPS | 全部文件字节数 | 视觉存储 |
| --- | --- | --- | ---: | ---: | --- |
| HuggingFaceVLA/libero | 86958911c0f959db2bbbdb107eb3e17c5f9c798e | 1693 / 273465 | 10 | 34926157548，约32.528GiB | PNG字节在Parquet中 |
| lerobot/libero_plus | f3f49f426d75030177b18778374005bc12ccd588 | 14347 / 2238036 | 20 | 15832116939，约14.745GiB | front/wrist双路AV1 MP4 |

文件清单/元信息本地在outputs/corrector_env_fix_20261107/dataset_inventory，未下载全量训练分片/视频。
用户确认Drive总容量15GB、已用约660MB，剩余约14.3GB；不启动全库下载。已按实际episode元信息列出当前完整单任务44条/6867帧范围，
计划保存在single_task_scope_plan.json。除三个pilot外，物理分片映射仍需逐个核对，不能用曾出错的元信息文件号直接下载。
Notebook新增只读清单入口，可用scripts/inspect_robot_datasets.py重新获得固定版本清单。
完整原始数据体积不等于特征缓存体积；以首次pilot的实际L/D估算int8特征+scale+动作/索引等开销，再决定规模。
以用户当前总容量15GB、已用约660MB计算，剩余空间不足以保留Plus全库及所需特征/checkpoint；容量单位与实际配额应以Drive界面为准。
本地Ubuntu磁盘紧张，不在本地下载全库。Colab临时盘112GiB与Drive额度是两种资源。

LIBERO-plus官方已经公开RLDS/LeRobot训练集及OpenVLA-OFT+混合微调模型，并非只有修改评测环境。
依据：[官方仓库](https://github.com/sylvestf/LIBERO-plus)、[LeRobot官方数据说明](https://huggingface.co/docs/lerobot/en/libero_plus)、
[公开数据集](https://huggingface.co/datasets/lerobot/libero_plus)。

任务元信息实际有40条语言；alphabet soup放篮子为task_index=3，不是原LIBERO数据中的24。
没有view_5字符串；目前没有证明某条视频对应我们固定4976dc3环境的view_5/view_10参数。
Plus的20FPS、front/wrist视频和不同任务索引需要单独读取适配，不把它直接塞进当前PNG/10FPS/固定三轨迹处理器。
下一步先确保三条真实特征缓存正常，再扩展单任务完整示范或suite；全库下载须先满足Drive和缓存预算。

普通失败执行MP4可以辅助标注/评估，但单路渲染视频缺少对齐动作和另一相机，不能直接成为当前双相机动作条件预测器的监督数据。
若重新采集仿真rollout，保存两路实际策略输入、执行动作、episode/frame/time及配置，并明确成功示范训练与失败检测评估的分工。
把目标域相机示范用于训练属于目标域适配；不能同时把同样回合/相机称为完全未见的最终测试。

## 基线与论文复现分开

核对论文v2：Table1的SmolVLA在MetaWorld（Baseline平均61.90%）；LIBERO的Table3为π0.5，
论文主要权重/完整数据未随代码提供，不能用当前smolvla_libero在LIBERO跑出一个数就声称复现Table1或Table3。
我们先做“冻结公开SmolVLA、协议明确的LIBERO/Plus基线”，后续比较Corrector时保持同一权重和初始化集合。
论文来源：[VLA-Corrector v2](https://arxiv.org/abs/2607.01804v2)，第4节与附录C.1。

新独立入口scripts/smolvla_baseline_benchmark.py不依赖用户的Corrector练习实现，也不依赖五步LoRAcheckpoint。
Notebook末尾先准备独立原始LIBERO仿真环境，再显式启动：

1. SMOKE=True：Object task0，真实初始状态0/1，seed1000，共2回合。
2. 通过输入/录像/返回指标检查后，SMOKE=False：Object 10任务×5个真实初始状态，共50回合。
3. 预算允许再扩大任务、真实初始状态和独立重复种子，预先冻结，避免只增加噪声seed却仍固定物理init_state0。

实现显式设置并核对init_state_id范围，拒绝取模回绕；每回合新环境，保存初始完整sim state、语言/图像/状态输入、首噪声和MP4。
动作块CUDA同步计时，回合success与模型调用数分开；success/已完成与完成/计划数分开，程序错误不计作机器人失败。
模型一次加载、顺序评测、每回合归档，完整回合可校验后跳过。视频20FPS、帧数须与实际控制步数一致。
统计/失败索引沿用batch_evaluation.py；Plus旧入口继续作为相机扰动开发对照，不与原始LIBERO的组混算。

本地CPU实际检查：相同seed1000、物理init_state0/1产生不同的110维settled sim states，证明选择的是物理状态而非只改seed。
该检查没有完整VLA推理，不计为基线成功率。新的完整策略T4入口尚未实测，两个smoke回合通过后才执行50回合计划。

本轮52项CPU基础/回归检查通过，包含独立Torch安装调度、初始状态边界和总体成功率加权。
Notebook内嵌脚本一致性、语法、旧输出保留和diff检查通过；这些不替代远程CUDA wheel安装与真实策略执行。

</details>

<details>
<summary>2026-11-07｜任务3历史：任务选择与规模（历史原文）</summary>

# SmolVLA阶段：先确认真实链路，再选可分析的恢复任务

2026-11-07。先完成SmolVLA上的数据→预测器→监控→重规划/OGG→任务评测；π0.5作为后续架构迁移，不在当前同时铺两套环境。

## 本次实际进展

依据用户保存的远程Notebook输出，宿主Torch为2.14.1+cu130，已成功走独立环境回退到2.11.0+cu128；T4约14.56GiB显存。
三轨迹813/818/825均提取并归档，原图10FPS，两相机合计128 token，每token960维；N分别155/164/153，总472帧。
action_dim=7，训练/验证标记保持813/818与825。脚本逐轨迹SHA复查均返回COMPLETE。
这些是Notebook运行证据，尚未由助手另行下载全部数组核验数值，也尚未训练Corrector。

按实际L/D与已记录dtype，三条缓存数组约55.46MiB；完整当前单任务44条/6867帧约0.788GiB，仅为特征/动作/索引载荷估算，未含原图、备份和checkpoint。
全库31.38GiB/256.79GiB特征估算无法适配当前15GB Drive。当前路线：先单任务完整示范，再视实际空间与效果逐任务扩展。
41条新增轨迹的物理分片映射尚待核对；不根据已知出错的meta文件号直接下载。此文没有把数据范围计划算作已下载。

## 当前错误：Notebook绘图配置泄漏到CLI

独立基线尚未加载VLA。仿真pip安装、libero导入、pip check均通过，首次check-sim在Matplotlib导入时报错：

```text
MPLBACKEND=module://matplotlib_inline.backend_inline
ValueError: Key backend: ... is not a valid value for backend
```

Colab宿主选择的inline模块不一定安装在隔离sim-venv，LIBERO导入matplotlib.cm时触发。
修复在runtime_env统一为CLI设置MPLBACKEND=Agg，不改Notebook内核的显示配置，不为后台进程加装整套Jupyter。
准备格发现已安装sim环境/源码/配置后使用repair-sim，只复查版本、路径和真实场景，不重复pip/apt。
本地在父环境明确设为inline的情况下运行真实repair-sim已通过reset/step、初始状态配对和MP4编码；新的完整T4基线仍待用户重试。

后续执行更新：用户保存的Notebook execution 17/18/19已确认远程repair-sim和完整策略冒烟通过。task0、init_state0/1、seed1000、H50共2回合，2/2成功，无错误或待完成回合；每回合平均3次块生成，块生成总耗时的回合均值约2.291秒。仅代表两个默认场景回合，未运行50回合计划，也未检验Corrector。

同次提取中只读NumPy/旧torch_dtype参数警告已在项目外围改为可写RGB数组及兼容参数名；不修改上游文件、不改变dtype，不要求重做已完成缓存。

## 不把LIBERO任务硬贴成论文Easy/Hard

论文v2的Easy/Medium/Hard/Very Hard是MetaWorld难度分组；其SmolVLA主表也在MetaWorld。
本项目的原始LIBERO套件按Object/Spatial/Goal/Long等组织，不能直接借用那套难度标签。
Plus还有相机/布局等扰动维度与配置强度，它们也不等于某个冻结策略上的成功率等级。
研究时标注“该checkpoint与协议下的基线表现”，保留权重SHA、H、初始化集合和成功分子/分母。

## 怎样选择任务，而不是预设必须提高

训练预测器与选择评测任务是两件事。当前监督目标是从示范中的视觉特征和动作预测未来视觉变化，不需要先给每条示范贴Easy/Hard或失败标签。先用成功示范建立这种预测关系，再用独立轨迹检查误差；在线偏离检测阈值和恢复效果还要另行验证。仅在默认视角训练，并不能保证预测器在偏移视角也准确；若它把视角差异当成执行偏离，可能出现误触发，因此需要保留正常执行误报率和扰动条件的评测。

1. 先默认场景冒烟：task0真实初始状态0/1，确认模型、动作尺度、图像朝向、录像、环境success都正常。
2. 再运行预先固定的小范围baseline筛选。已有Object 10任务×5初始化共50回合的入口，按时间预算执行并逐回合归档。
3. 开发优先看基线有失败且存在可恢复性的任务：偶发抓空、掉落后仍执行旧动作、H20能挽回部分H50失败。
4. 既有Plus开发线索保留，不再次花GPU重复旧12回合：view_15为H50 1/3→H20 3/3；view_5为0/3→1/3，view_10为1/3→2/3。
5. 这些只说明值得检验重规划/引导机制，不保证Corrector提高。H50为0/3但H20能成功的条件仍可有价值；所有方法均失败可能涉及基础策略能力不足。
6. 保留默认高成功条件检查是否退化，也保留难失败案例说明方法边界，不能只挑Corrector获胜视频。

五次或三次试验都不能稳定估计“固有难度”；暂不人为设官方Easy/Hard标签。
基线筛选和阈值调参只用开发条件，筛选过程与全部候选结果要记录；在看Corrector测试结果前冻结正式任务/初始化/seed和指标。
正式报告写清“在基线筛选出的恢复困难条件上”，不外推为全面泛化；报告成功率、完成覆盖率、调用数、监控/策略耗时和失败类型。

原view_5/view_10已被看过，属于开发条件；Plus任务名可能固定initstate_0，不能只改seed就称改变物理初始化。
后续Plus正式初始条件需按实际任务/初始化文件核对，不能依靠取模回绕得到“新样本”。

## π0.5迁移时保留与重做什么

可复用：时序配对原则、缓存/归档协议、训练循环、监控状态机思路、批量评测和结果分析。
需重新核对/生成：π0.5视觉空间的维度和token布局、预处理、动作表示与归一化、相同控制时间含义的数据配对，以及新Corrector训练和阈值。
SmolVLA预测器权重不能因为“VLA被冻结”就直接用于π0.5的另一视觉空间；冻结参数也不消除视觉提取、策略推理及OGG输入梯度的算力成本。
SmolVLA上至少完成一次独立验证的预测器和基线/监控/重规划对照后，再启动该迁移。

近期只做三件事：修复仿真检查并跑基线冒烟；用户完成已搭好的核心方法；用现有缓存验证训练接口后扩完整单任务数据。
来源：[论文v2](https://arxiv.org/abs/2607.01804v2)、[已完成窗口结果](replanning_results.md)、[资源与基线核对](resources_and_baseline_20261107.md)。

</details>
