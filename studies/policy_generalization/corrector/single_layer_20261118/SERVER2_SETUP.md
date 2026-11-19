# 2号机扩容后准备 · 2026-11-18

北京时间17:50只读核查：数据盘实际550GiB，已用约8.76GiB，剩余约541.24GiB；系统盘30GiB、余约29.26GiB。RAM cgroup上限120GiB，使用约0.36GiB；4090共24,564MiB，显存占用0、利用率0、无GPU计算进程。以本次实测替代此前“50GiB尚未扩容”的资源状态，不修改历史实验。

## 已完成：原始数据迁移

- 98个Parquet，9,402,105,058字节（8.756GiB）。
- 17:32:31完成，98/98逐文件SHA通过；目标清单与本地源清单逐项一致。
- 原Object10 manifest SHA仍为`61c9f63f16553ae2f81d57478fdeeaccc8cb653223da9931705102c935d4acb9`。
- 目标：`/root/autodl-tmp/corrector_imports/object10_raw_20261118_001/source/`。
- 此次传输的临时公钥已撤销、目标私钥已移除；1号机原authorized_keys2的SHA精确恢复。新资产准备使用独立、限时、只读授权，不复用已撤销密钥。

私有完成/清理收据保存在本地忽略目录`outputs/object10_migration_20261118_001/`。

## 正在运行：已有环境与最小探针资产准备

2号机原先只有Python3.10，没有Torch。旧成功环境依赖同项目内Python3.12 bootstrap、外部共享Torch目录和LeRobot可编辑源码；不是单独复制venv即可使用。

已启动独立后台资产准备任务，迁移**整套对应路径**：bootstrap、runtime/venv、共享Torch 2.11 cu128、LeRobot及固定上游源码、已有HF模型缓存、冻结night/core源码与一个真实Episode813缓存。总计46,695个文件/链接，常规文件约8.61GiB；不包含raw70、其他训练数据或历史训练Checkpoint。不从公网下载新模型或数据。

采用4路无重叠文件分片、压缩rsync、断点文件保留、有限网络重试，先在独立payload目录逐文件检查SHA和链接依赖，完整通过后才发布至当前不存在的`/root/autodl-tmp/corrector`。若目标项目根目录出现，不覆盖；来源始终只读。资产清单及路径/外部链接/重复/字节数拒绝测试已通过，未修改LVM/OGG算法。

准备任务最后只执行：

1. Python/Torch/NumPy/Transformers版本及CUDA可用性查询。
2. 在新机重跑现有19项**CPU**测试。
3. 写READY并停止，不自动启动真实Corrector GPU冒烟、454条提取、训练或仿真。

18:05首次启动检查：`running/compressed_transfer`，4路均实际接收数据、无错误。这个状态是运行中快照，不能当作环境已经验证完成。源文件体积为未压缩字节，界面中的落盘速率也不是压缩后的实际网络吞吐。

在2号机终端查看：

```bash
cat /root/autodl-tmp/corrector_imports/runtime_assets_20261118_001/status.json
tail -n 30 /root/autodl-tmp/corrector_imports/runtime_assets_20261118_001/.transfer/launch.log
```

成功标志为`status=completed`、`phase=ready_for_real_gpu_probe`，表示环境与CPU检查完成，**不是Corrector GPU验证完成**。具体兼容性/CPU结果保存在该任务`.transfer/environment_probe.log`及`destination_cpu_tests.log`。发生错误会保留现场并停止，不自动安装不同版本以强行通过。

传输阶段若断连/重启，已接收文件保留；确认旧进程退出、目标尚未发布且限时授权仍有效后，可用同一worker恢复。若已发布后验证失败，需要检查具体日志再恢复验证，不重新覆盖目标目录。

## 仍待执行

环境成功后，先做少量真实图像的原始层对齐、完整分辨率GPU前向/反向、checkpoint重载、OGG动作梯度、原状态机及运动采集开关检查。然后再集成完整提取/恢复协调器及正式队列，保持任务书的阶段门槛。当前没有新模型训练成绩或闭环成功率。

## 后续实测更新：环境与真实GPU小检查已完成

上述18:05状态作为历史快照保留。环境迁移现已完成46,695项文件/链接校验。首次CPU检查暴露了传输清单遗漏Git空目录和被通用过滤器排除的原版`uv.lock`；已按源端目录与原Git blob精确补齐，两个源码仓库`git status --porcelain`均为空，未放宽源码SHA或安装不同依赖。原失败日志保留，修复证据另存`RECOVERY_RECEIPT.json`。资产`status.json`现在为`completed/runtime_ready_after_verified_recovery`。本任务临时传输公钥和私钥均已撤销/移除，1号机原授权文件SHA保持不变。

454条Episode的六个原始动作/时间/元信息文件全部SHA通过，保持364/90划分和66,984帧。约6.12MiB元数据ZIP另含全部帧的原Connector量化字节SHA，源端还完整校验了原z_q/z_scale文件SHA；没有伪造缺失张量或传输无关训练缓存。2号机数据位于`/root/autodl-tmp/corrector_imports/object10_metadata454_20261118_001`，校验回执为`VALIDATED_V2.json`。

V2代码在本地与2号机均通过21项CPU测试。随后真实GPU探针已完成：

| 项目             | 实测结果                                              |
| ---------------- | ----------------------------------------------------- |
| 原始输入         | Episode813真实示范前13帧、两相机                      |
| 特征             | Block2/8各`[13,2,1024,768]`，FP16、不池化           |
| Connector复量化  | 13/13帧与旧缓存字节一致                               |
| Block2/8原始特征 | 两层各13/13帧与旧FP16缓存逐帧SHA一致                  |
| 提取测量         | 已加载模型后的提取/检查阶段8.700秒，不包括冷启动加载  |
| MLP              | 每个37,266,176参数，四个2048隐藏宽度不变              |
| GPU训练形状      | `[32,2048,768]`，原MSE＋cosine、反向与一步AdamW通过 |
| Checkpoint       | 两层严格重载后同输入输出完全一致                      |
| OGG              | 有限非零动作梯度，仅更新首动作前7维；预测器冻结       |
| 探针峰值显存     | S2约7.606GiB、S8约7.615GiB                            |

**探针显存测试将一个真实t2→t12样本重复32次；不是32个独立示范、正式epoch或预测准确率。** OGG流候选是明确标记的接口测试张量，不是机器人执行命令。探针权重保留`discarded_probe`标记，不能用于正式对比。

在2号机查看此次完成结果：

```bash
cat /root/autodl-tmp/corrector/single_layer_20261118_001/gpu_probe_001/status.json
cat /root/autodl-tmp/corrector/single_layer_20261118_001/gpu_probe_001/result.json
```

本地完整收据位于忽略目录`outputs/single_layer_foundation_20261118_v2/gpu_probe_result.json`，代码ZIP SHA为`bab8f73e0b5bd88426b02c4e83e126923647823bb7134cea5d52ac4be78d9685`。

新增`extraction.py`完成逐Episode Connector哈希验证、全层SHA后原子发布、STOP/5GiB磁盘边界暂停、独占锁与中断attempt保留；新恢复测试加入后本地25项CPU通过。该提取器是后续候选实现，**尚未启动454条完整提取**。原成功导出器、旧环境和数据保持只读。当前尚缺真实GPU在线控制与motion采集开关一致性验收，故`all_stage_C_gates_passed=false`；正式训练未启动，不把工程探针当作完成任务书阶段C或训练成绩。

### 后续CPU数据审计与候选提取器部署

V3候选已在2号机独立目录部署，25项CPU测试全部通过（5.762秒）；ZIP SHA为`faa11bc9d5e4cfd9490986d9187e51e7713f92b9be6541d828c27ed190abec60`。这次部署不覆盖V2真实GPU探针代码，也没有重跑或改写旧结果。

另运行`raw_audit.py`，只读完整98个Parquet并重新核对原来源SHA，以及全部454条示范的真实动作、frame_index、timestamp和划分。66,984帧全部通过，Parquet动作转FP32后与原`raw_actions.npy`逐元素相等。耗时14.091秒，不用GPU、不提取图像。

10个原数据task_id为20～29，各Episode数依次45/45/45/46/44/45/47/45/42/50；这些是原始语料task_id，不是在线LIBERO Object suite的0～9编号。原364/90划分保持不变，未搜索或改选验证样本。

结果：`/root/autodl-tmp/corrector/single_layer_20261118_001/RAW_AUDIT_001.json`。本地忽略目录`outputs/single_layer_foundation_20261118_v3/`保存完整CPU日志、部署回执和审计JSON。最后核对数据/系统盘余量约532.03/28.59GiB，GPU显存0MiB；GPU探针已结束，无正式提取/训练后台任务。

## 后续：验收通过，完整454条特征已开始制作

上述“尚未启动”保留为当时状态；当前活动任务见[OPERATIONS.md](OPERATIONS.md)。

复用1号机普通LIBERO仿真环境，1,784,928,798字节、10,012项文件/链接校验后复制到2号机原先不存在的路径。2号机缺少EGL/GLES系统库，已用APT安装`libegl1`、`libgles2`及其依赖（含3个Mesa库的依赖升级）；下载约9.4MB、增加约8.8MB系统空间。未安装/升级Python、Torch或修改冻结第三方源码。

真实GPU控制验收使用两层工程探针权重，task1/init0/seed1000，80步上限；每层分别对LVM/OGG比较原控制器＋必要raw视觉观测、适配器日志关、日志＋motion采集开，共12个短回合。全部12/12完成，动作/仿真状态/首输入/噪声/动作块/分数/阈值状态/待到期预测/视频SHA逐项完全一致。**12回合没有自然触发纠偏**；它们仅证明观测与采集接入一致，不是恢复成功率或OGG收益测试。非零首动作OGG梯度已在完整GPU形状探针中单独通过，实际触发/队列分支由带触发的CPU回归覆盖。

修正了三个工程入口问题并保留现场：

1. 精度检查原先错误要求整个wrapper全FP32；实测冻结视觉197个参数与Connector1个参数为FP32，语言和动作专家部分参数原本为BF16。仅将检查限定于视觉路径，未转换任何权重或精度。
2. 新提取入口漏传`LEROBOT_SRC`，使上游初始化重排导入路径、误导入旧LeRobot。补齐变量并验证实际策略类来源，不安装缺包绕过；旧`raw454`未发布完成数据，原错误目录保留。
3. 续接子进程未继承父进程`sys.path`，补齐自有管理入口的`PYTHONPATH`和线程环境，并保留补丁前入口与`MANAGER_ENVIRONMENT_FIX.json`。本地再加入全新Python进程启动回归；实际续接已经连续完成多个Episode。

活动代码ZIP SHA：`2374758902ae12b2b21da2c208d8be6897d30d8c402de06bc94619a80e0c75a3`；该部署28项CPU通过，本地补充启动回归后29项通过。新的`manifest454_v2.json`、`gates_v2/`、`extraction_config_v2.json`明确版本化，旧计划与日志不覆盖。新manifest只改变独立输出根路径，仍对全部原始数据重做来源、动作与时间审计。

首个Episode807完整143帧通过原Connector逐帧字节检查、两层完整SHA、原动作/时间绑定和训练样本形状检查，已在完整边界暂停并使用正式管理入口续接。随后正式提取器生成的Episode813两层前13帧再次与旧原始FP16缓存逐帧SHA一致。

最后读取的真实快照：14/454 Episode、2,042/66,984帧完整发布；正在制作下一个Episode。最近5个完成Episode约15.48帧/秒，预计剩余1.165小时，估算北京时间20:37完成，仅为提取滚动ETA，不包括训练。数据/系统盘余量约518.05/28.42GiB；cgroup内存约27.41/120GiB（包含页缓存），GPU占用1,783MiB。实时状态优先于此静态快照。

**当前仅提取训练特征，尚未正式训练新S2/S8模型，不自动接训练。** 完成后仍需核验454条特征和完整训练输入，再显式启动既定的训练任务。暂停/续接与ETA命令见操作说明。

## 后续：特征全部完成并开始正式训练

活动`raw454_v2`已完成454/454、66,984帧。用户明确授权后，重新读取两层约392.5GiB全部原始特征复核SHA和原动作/时间/划分，599.36秒内454条全部通过。2号机GPU空闲、数据/系统盘余量约137.8/28.4GiB，共享内存60GiB，未清理历史资料。

新增独立`training_001`六组队列，S2/S8各seed42/43/44，保持原模型宽度、MSE＋cosine、batch32×16、AMP BF16、AdamW与Cosine调度、最多150轮/patience5。实际S2_s42的参数和原样本索引SHA完全匹配。初始验证已处理12,089条样本并保存epoch0完整续训状态，随后第1轮已完成反向与首个optimizer更新；GPU实测约9.6GiB/100%（瞬时状态）。该阶段不是已训练完成或性能提升结论。

训练版本`slc-training-777f8060ba19`在本地与2号机均通过33项CPU测试，队列/进度使用独立模块，不修改冻结训练器。新查看入口为`train_manage.py status`，特征制作旧入口保持显示已完成。参数表、条件ETA、日志和安全暂停/恢复见[TRAINING.md](TRAINING.md)。
