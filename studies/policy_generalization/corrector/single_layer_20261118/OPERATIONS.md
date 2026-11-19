# Object10 S2/S8：2号机特征制作操作

**后续状态：454条特征制作已完成，用户明确授权后已启动独立正式训练队列。训练进度、ETA和暂停续接请看[TRAINING.md](TRAINING.md)，使用`train_manage.py`。** 本页继续保留特征制作的操作记录，`manage_v2.py`显示其已完成状态。

特征制作使用原454条、364/90划分、66,984帧和两相机顺序，保存两层完整FP16 `[N,2,1024,768]`，约392.5GiB；制作阶段未自动启动训练或性能评测。

## 查看进度与预计剩余时间

在**2号机远程终端**执行：

```bash
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/manage_v2.py status
```

每15秒刷新，按Ctrl+C只退出查看，不停止远程任务：

```bash
watch -n 15 /root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/manage_v2.py status
```

主要字段：

| 字段 | 含义 |
|---|---|
| `status` / `process_alive` | 保存的任务状态与实际进程存活情况；旧running进程不存在会明确报需恢复 |
| `completed` / `planned` | 已完成两层完整SHA检查并发布的Episode数／454 |
| `frames_completed` / `frames_total` | 完整发布的帧数／66,984；每帧包含两相机、两层 |
| `current_frames_completed` | 当前Episode已处理到的帧数，约每50帧更新 |
| `frames_per_second` | 最近最多5个完成Episode的端到端平均帧率，包含IO、SHA和发布 |
| `eta_hours` | 估算剩余提取小时，不包括正式训练或后续评测 |
| `estimated_finish_beijing` | 根据当前速度估算的北京时间完成点，仅running时显示 |
| `eta_basis` | 估算使用的样本数和口径；首Episode前显示等待，不虚构ETA |
| `data_free_gib` / `system_free_gib` | 两块盘各自剩余空间 |
| `status_age_seconds` | 距离最近状态写入的秒数；模型冷启动或SHA检查期间可能增加 |

首个Episode包含模型冷启动；暂停后续接也需要重新加载模型，因此最初几个ETA可能偏大。完成约5～6个新Episode后通常更有参考性。ETA是滚动估算，不是固定完成承诺。

当前配置：`/root/autodl-tmp/corrector/single_layer_20261118_001/extraction_config_v2.json`。完整特征：同目录`raw454_v2/episode_<id>/`。

## 查看日志

最新worker日志路径由同目录`raw454_v2/launcher.json`的`log`字段给出。也可在远程终端动态读取：

```bash
/root/autodl-tmp/corrector/runtime/venv/bin/python -c "import json,pathlib; p=pathlib.Path('/root/autodl-tmp/corrector/single_layer_20261118_001/raw454_v2'); print(pathlib.Path(json.loads((p/'launcher.json').read_text())['log']).read_text()[-6000:])"
```

每完成一个Episode，日志会打印已完成数、帧数、实测速度和ETA。发生错误保留完整栈、状态及未完成attempt，不静默更换模型、精度、特征分辨率或数据定义。

## 安全暂停与续接

请求在**当前完整Episode校验发布后**暂停，不强杀：

```bash
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/manage_v2.py pause
```

确认`status=paused`或进程已退出，且无其他任务占GPU后，显式续接：

```bash
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/manage_v2.py resume
```

续接会归档STOP请求，重验已完成Episode并跳过，不重写已发布张量。断电留下的半成品保留在`.work/`，新尝试使用新编号，不冒充完整数据。新进程初始化期间可能暂时看到上一次状态，结合`launcher.json`的新PID/日志核对。不要并发重复执行resume。

两盘分别保留5GiB；每个新Episode开始前还检查其完整输出所需空间，空间不足以paused状态停下。不会自动删除原始数据、旧视频或Checkpoint。程序错误后先查日志，禁止靠修改阈值、loss或精度绕过；未知错误不无限重启。

## 版本与验收

- 活动数据/配置为`raw454_v2` / `manifest454_v2.json`；原`raw454`及`control_acceptance_001`为失败现场，保留但不混入当前进度。
- 实际提取代码：`dev/slc-extract-2374758902ae`，ZIP SHA `2374758902ae12b2b21da2c208d8be6897d30d8c402de06bc94619a80e0c75a3`。
- GPU完整形状反向、重载和非零首动作OGG梯度已通过；真实LIBERO短回合12/12通过日志/采集开关一致性。
- 12回合没有自然触发纠偏，不能声称验证了恢复成功率或GPU自然触发下的OGG收益；触发分支数值/队列由CPU回归覆盖，GPU梯度单独验收。
- 完整Episode807的143帧已做逐帧Connector精确匹配、两层完整SHA与动作/时间绑定、样本形状和边界暂停验收；随后通过管理入口续接。
- 自有管理入口补齐了子进程`PYTHONPATH`、`LEROBOT_SRC`及CPU线程环境；保留补丁前脚本和`MANAGER_ENVIRONMENT_FIX.json`。冻结源码、科学协议、已完成张量未改变。
- 缺失EGL/GLES系统运行库已通过APT补齐（连带Mesa依赖更新）；没有安装/升级Python、Torch或修改冻结第三方源码。

门槛证据、配置、完整Episode验收和所有失败日志都位于新实验目录。正式训练须在454条特征完成并通过数据验证后显式启动；本次不会把工程探针权重当作正式S2/S8模型。
