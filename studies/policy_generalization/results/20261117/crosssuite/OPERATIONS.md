# Object训练Corrector的跨任务泛化评测

**完成更新：880/880、错误0，2026-11-17 15:32:05结束，无需恢复或等待16:07预算到期。** [最终结果与完整性审计](RESULTS.md)。下文保留当时的计划及操作命令；不为用满时间追加实验。初版环境元数据`init`是reset后游标，具体修正口径见结果报告，原文件不改写。

这是用户授权的**最多4小时**、四方法闭环实验。没有新训练、下载、阈值调参或删除旧数据。此次是原MLP，不是J多层模型。

## 固定协议与研究范围

四方法为`H50`、`H20`、`mlp_lvm`、`mlp_ogg`。冻结SmolVLA仍用原50步生成块、10次去噪、FP32策略、20Hz控制；H20只执行前20步。MLP采用seed42权重，原动态LVM状态机、单动作预测保留策略和余弦OGG，eta=1、每3次去噪引导一次。训练10Hz/k=10，部署预测到期20步，首评分24步，均不改变。

MLP权重SHA256：`05d55ba9452142be66840b3bef6fad10226480b6115be4bbbcf0842116732fad`。已核对真实454条fit/validation记录：364训练、90验证，语言覆盖Object10十任务；新套件任务语言和名称与它们不重叠。模型卡未明确列出基础SmolVLA的任务级训练范围，因此这轮评价**Object训练Corrector的任务迁移**，不能称SmolVLA零样本能力。

| 优先级 | 套件/任务 | 初始状态 | 评测seed | 四方法回合数 | 含义 |
|---|---|---|---|---:|---|
| 1 | Spatial、Goal、Long各全部10任务 | 0、1 | 1000、1001 | 480 | 主要比较，各套件单独汇报 |
| 2 | LIBERO-90固定ID：0/9/19/29/39/49/59/69/79/89 | 0、1 | 1000、1001 | 160 | 预先选定子集，不能写成完整LIBERO-90 |
| 3 | Spatial、Goal、Long各全部10任务 | 2 | 1000、1001 | 240 | 扩充物理初始状态 |

880是预先冻结的**回合上限**，不是必须凑满的运行目标。四方法相邻执行；首120回合即覆盖三个主套件的全部30任务各一组配对。之后按固定顺序补init/seed，不能根据成功失败选下一任务。重复seed不是新的物理场景。工程Object复测4回合及40次reset+一步检查另计，不混入跨任务成绩。

各套件采用本机实际LeRobot声明的回合上限：Spatial 280、Goal 300、Long 520、90子集400。图像与控制协议保持不变，未用Object的280步截断Long。

## 远程路径、状态和安全暂停

远程根目录：`/root/corrector_crosssuite_20261117_001`，新源码在其`code/`，未覆盖原部署。

本次预算从北京时间**2026-11-17 12:07:10**起，截止**16:07:10**。预算包括环境检查和Object工程复测；最多允许完成截止时正在执行的一回合，不强杀。

在已登录的**远程终端**查看：

```bash
cat /root/corrector_crosssuite_20261117_001/status.json
cat /root/corrector_crosssuite_20261117_001/transfer/execution.json
cat /root/corrector_crosssuite_20261117_001/transfer/analysis/transfer_by_suite.csv
tail -n 30 /root/corrector_crosssuite_20261117_001/transfer.log
```

检查真实进程，避免只依赖旧running字段：

```bash
pgrep -af '[c]orrector_crosssuite_(queue|eval)[.]py'
nvidia-smi
```

安全请求暂停，仅写本队列STOP，在完整回合边界执行：

```bash
touch /root/corrector_crosssuite_20261117_001/STOP
```

在**原4小时窗口仍有余额**且确认旧进程退出后恢复；已完成视频及诊断文件SHA会复核，旧attempt保留。此命令不增加4小时预算：

```bash
nohup /root/autodl-tmp/corrector/bootstrap/bin/python -u \
  /root/corrector_crosssuite_20261117_001/code/corrector_crosssuite_queue.py run \
  --root /root/corrector_crosssuite_20261117_001 --resume \
  >> /root/corrector_crosssuite_20261117_001/launch.log 2>&1 < /dev/null &
```

未知程序错误停止并保留现场，不无限重启。系统盘与数据盘分别留5GiB；监督器在5.15GiB附近请求暂停，另有容器内存检查。数据存系统盘独立目录，未复制权重或大型缓存。极短时间内的外部空间占用或硬关机仍可能造成错误，需要核对后恢复。

## 配对、数据与结果解释

原冻结runner逐组核对初始模拟器状态、首策略输入和首噪声SHA；动作队列缓存为空才允许真实前向，每块必须是`[1,50,7]`。每回合额外记录真实suite、task语言、原生长度、OSC相对控制器和模拟器控制步长。两条路径分叉后，不声称全程噪声逐时刻一致。

`transfer/analysis/transfer_summary.json`和`transfer_by_suite.csv`给出：分suite/phase原始完成数、四方法完整配对数、配对成功率，以及环境步数、VLA调用、推理/监测/回合时间、干预/OGG次数、显存。只完成部分方法的组单列，不把待完成或程序错误算失败。原生`analysis/overall.csv`会混合套件，**不能用它替代分套件主要比较**。

保存全部完成回合视频、帧PTS/动作前相位、实际提交动作、块/队列记录、监测和OGG实际梯度/更新摘要；仅前两次引导保留候选数组，未记录完整未来状态轨迹。每组保留一份首策略输入，其他方法保留引用和SHA。所有资料默认只读。

CPU轻量快照可在本地取回，SSH别名需指向用户自己的已配置主机：

```bash
python3 scripts/corrector_crosssuite_remote.py status --host <SSH别名>
python3 scripts/corrector_crosssuite_remote.py fetch --host <SSH别名>
```

若没有别名，用`--host root@你的主机 --port 你的端口`。地址/凭据不写入公共仓库。fetch只取文本状态和分套件汇总，不下载权重和视频。需Replay回放时，使用已有`corrector_replay_export.py`按完成trial ID导出精选回合，再以`corrector_casebank_import.py`校验后追加；不自动导入880条。导入器已支持有明确原生协议的300/400/520步，界面不再将所有套件标成Object。

## 验证与部署

- 新增9项CPU测试：原生上限、Object隔离、固定优先顺序、四方法相邻、稳定case ID、partial配对不计失败及Replay长度检查。
- Replay原107项回归通过（8项因缺相应私有归档而跳过），JavaScript语法检查通过。本次仅改套件标题与导入上限，未新增浏览器交互功能；没有将语法检查称作浏览器验收。
- 真实远程40任务全部reset+一步通过，控制dt约0.05秒。
- Object四方法与旧同条件参考的实际命令及队列精确一致，工程桥接通过。
- 未修改第三方安装、冻结模型、旧实验或checkpoint；新环境配置只作用于当回合实例。
- 部署ZIP SHA256：`4c918b7907478f22b7ddf9e52b7cc471c23bd35bdc2b500afbf193157e3599d1`。源码及SHA保存在本地忽略目录`outputs/autodl_setup/crosssuite_20261117/`，远程保留`deployment.zip`与`launch_plan.json`。

冻结loader会额外加载未参与本轮控制的J/C0和视觉副本，以复用已验收加载路径。因此记录的显存是当前评测进程峰值，不能作为最小纯SmolVLA部署显存。
