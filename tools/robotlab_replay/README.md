# RobotLab Replay

本地机器人实验回放与诊断工作台。面向现有SmolVLA / VLA-Corrector的真实结果，不加载模型、不启动仿真、不控制机器人。

## 启动

在robot-lab根目录执行：

```bash
python3 -X utf8 tools/robotlab_replay/server.py --open
```

默认地址 **http://127.0.0.1:8870**。终端按 **Ctrl+C** 关闭，服务及其文件句柄随之退出。

Ubuntu也可以运行：

```bash
tools/robotlab_replay/start.sh
```

Windows安装Python 3.10+后运行`start.bat`，或：

```powershell
py -3 -X utf8 tools/robotlab_replay/server.py --open
```

运行只需要Python标准库和现代浏览器，不需要安装Python/Node包。视频帧核验和PNG导出使用已有的`ffprobe` / `ffmpeg`；若没有，视频可用原生控件独立播放，但禁用未经核验的逐步联动与关键帧导出。Ubuntu为本轮实际验收平台；Windows启动脚本已提供，未做Windows实机验收。端口冲突可加`--port 8871`。

## 已接入的真实资料

自动发现下列现有只读归档，均位于项目`outputs/`：

| 来源 | 内容 |
| --- | --- |
| `corrector_audit_20261111/results_bundle.zip` | 五模型正式1200回合及15次强制接口检查；trace、协议与已完成回合 |
| `hybrid_pilot_20261111/results.zip` | A/B/C自然配对12回合，另3次强制检查；较完整OGG梯度/Loss/动作更新与目标记录 |
| `consistency_closed_loop_20261111/results.zip` | H50/H10/规划LVM/已执行LVM，48个完整任务回合及真实动作 |
| `executed_consistency_pilot_20261111_v2/results.zip` | 4回合被动诊断，规划/已执行条件双评分 |
| `smolvla_horizon_pilot_20261111/results.zip` | H50/H20/H10/H5/H2/H1共24回合 |

当前共索引1306条回合记录，默认不展示18条强制接口检查。它是归档记录数，不是独立初始状态数；同基线在不同模型归档中的重复记录不合并统计成功率。

已只读取回混合Pilot的task1/init0/seed1000三段真实视频：A成功168步、B失败280步、C失败280步。视频SHA与各原回合记录一致，共计约471KiB的导入包（还包含录制源码证据）。原始视频仍在远端，没有移动或修改。**没有把相机实验或论文演示视频冒充Corrector视频。**

首次在另一台机器导入这些可访问的旧视频时：

```bash
python3 tools/robotlab_replay/import_remote.py --ssh-host <SSH配置中的别名>
```

这一步仅通过SSH读取3个**已完成**回合与固定录制源码，不写远程、不安装、不启动GPU，不接触正在运行的夜间实验。导入后重启本地工作台以重新发现视频。SSH地址、私人路径和视频清单只写入被忽略的`.local/`，示例不包含服务器凭据。重复导入拒绝覆盖。

不访问远程也可以使用已在本机的JSON/ZIP；没有视频的回合会明确提示，事件/曲线/配对仍可用。

## 一次可用的失败案例复盘

1. 打开后默认选中**混合Corrector B，task1/init0/seed1000，失败280步**。B是单动作检测器+A10引导器。
2. 输入环境步 **61**，或点击第61步的中断/OGG事件。当前视频解码PTS是 **3.05秒**，右侧显示4次真实OGG调用、Loss、梯度范数、动作更新量和目标shape。
3. 展开“前后5步上下文”查看分数/阈值变化；箭头或键盘左右键逐步检查抓取/提举过程。任务失败是记录的结果，具体抓取成败仍需要视频/独立物理状态证据，不能仅由score或画面重叠自动断言接触成功。
4. 点“配对对比”，默认找到同条件的**A：单动作检测+单动作OGG，成功168步**。两回合初始状态/输入/首噪声SHA一致；首次触发前的检测目标也可在原始事件记录中核对。两边共用环境步位置，但不假定相同步数处于相同操作阶段。
5. 在61步比较两种引导的已记录梯度和动作更新。切换右侧“主选/对照”可看各自原始日志；下方有成败、步数、调用、耗时、覆盖、首次事件的完整表格。
6. 跳到200步，成功的A已结束：明确显示冻结末帧167，右侧不复用末次评分。不能把末帧当作200步观测。
7. 选C（A10检测+A10引导），将事件筛选改为“监测空窗”。点击空窗后，阴影和“无到期评分”状态联动；最长空窗19步，不能将skip/cancel次数直接相加称为覆盖率。
8. 写入复盘备注并“导出复盘”：得到来源上下文、事件CSV、LVM SVG、可核验关键帧PNG与SHA清单。

小屏可点“专注回放”暂时隐藏概览，让视频和分数曲线同时留在视口。播放支持暂停、0.25/0.5/1/2/4倍速、单步前后、拖动环境步与指定步跳转。方法H50/H20等是纯VLA，模型筛选在这些行上代表归属的对照实验组，不表示Corrector参与了控制。

## 时间对齐规则

显示的控制秒、视频PTS和累计推理/监测墙钟是不同量，不互相替代。

精确联动的必要证据同时包括：

- 视频文件SHA匹配已归档回合；该SHA绑定到本次只读导入的录制证据。
- 固定LeRobot Git版本与干净的录制文件SHA匹配；AST核查reset后首先render、每次env.step后render以及视频截取顺序。
- `ffprobe`读取**每帧PTS**，不是仅根据FPS乘除；帧数等于本回合环境步数，PTS严格递增且起点为0。
- 对已验证三段视频，帧n对应动作u_n提交前的Z_n；采集缓存含初始观测，输出截取到N帧，因此最后一次动作后的Z_N不在视频中。

若仅匹配帧数/SHA而缺录制相位证据，标成“顺序对应”；若连帧数/PTS也不足，则标为未验证。**两种情况都提供独立视频播放，禁用逐步联动和标为环境步的PNG导出**。一份录制证据不能自动套用到其他视频。

分数为`1−cosine_similarity`，无量纲，非失败概率。高于图范围的初始Threshold（例如10）显式标注，没有把原阈值改小。缺分数断线、不插值；预热与预热后的空窗不同。覆盖率复用原`corrector_results_analysis.coverage_stats`，按明确首到期步/stride取分母；缺协议时不以第一条已观测分数反推“100%覆盖”。

原始trace未逐次记录预测生成时刻或候选Tensor时，不伪造这些事件。显式记录的到期、跳过、取消、纠偏目标与OGG调用会展示。动作块号从固定执行窗口和中断记录重建，**只有重建块数等于真实总VLA调用次数时显示**，标为“推导”；它不是逐次调用时间戳。视觉残差MSE只在有记录的维度证据和范数/cosine时推导，并标明来源。

## 数据格式与扩展

```text
evaluation/
  plan.json                       # cases + protocol
  trials/<id>.json                # case / status / attempt / result / video_artifacts
  attempts/<id>/attempt_001/
    actual_protocol[.export].json
    monitor_steps.jsonl           # 按step归并，允许同一步多个OGG调用
    executed_actions.json         # 可选
    controller_commands.json      # 可选
    targets.json                  # 可选，仅展示JSON摘要，不加载.pt
    videos/*.mp4                  # 可选；原文件不复制/改写
```

可以读取这样的目录或同结构ZIP，保留ZIP内多层评测目录：

```bash
python3 -X utf8 tools/robotlab_replay/server.py --source /path/to/evaluation --source /path/to/results.zip
```

默认只读索引元信息；选中回合才解压读取该trace/动作、探测该视频PTS。视频经HTTP Range按需读取，不整批读入内存。20回合LRU内存缓存；结构化成员24MiB上限。CSV只有汇总、缺少plan/trial协议时，第一版不将其推断成完整可同步回合。

代码分层：`data.py`只读适配与规范化，`server.py`本地API/Range/派生导出，`web/`视频/曲线与状态展示。未来接入新策略或JSON Tensor摘要时优先扩展适配层/显式证据字段，不需要改原来的训练、控制或推理接口。当前不自动读取`.pt/.npy`张量、训练模型或连接实时机器人。

## 导出、安全与文件边界

- 原实验默认只读。所有新导入、缓存、截图、验收记录、备注导出在`tools/robotlab_replay/.local/`，被此目录自己的`.gitignore`忽略。
- 每次导出新建唯一目录，含`SUMMARY.md`、事件CSV、SVG、可核验PNG、`provenance.json`。记录方法/模型、任务、init、seed、冻结配置与权重SHA、原trial成员/哈希、环境步/PTS、对齐限制和导出文件SHA；不携带私人绝对原始数据路径。
- 主选或对照已结束时，不导出伪称该步的关键帧；没有视频也不生成假的PNG。
- 服务只绑定127.0.0.1；限制Host、跨站写入、请求体和Range；文件API只接收已索引回合ID或独立导出白名单，没有任意路径文件浏览接口。拒绝路径穿越、源外符号链接、误指向checkpoint的“视频”。
- 没有修改现有Python环境、LeRobot、LIBERO、训练/评测脚本或远程运行任务。测试浏览器使用本机已有Playwright/Chromium；运行工作台无需它们。

## 验证

自动数据/导出/安全测试（真实数据存在时额外执行真实回合验收，否则明确skip）：

```bash
python3 -X utf8 -m unittest discover -s tools/robotlab_replay/tests -v
```

可选浏览器端到端测试需要现成Playwright及Chromium；它们不是应用运行依赖。启动服务后在Node18+环境执行，必要时通过`REPLAY_PLAYWRIGHT`/`REPLAY_BROWSER`指定已安装位置：

```bash
node tools/robotlab_replay/tests/browser_check.cjs
```

验证记录及真实浏览器截图写`.local/review/`。本轮实测内容见[ACCEPTANCE.md](ACCEPTANCE.md)，不是仅用手写JSON或静态截图验收。

## 当前边界

目前只有三个Corrector真实视频已取回；其他回合可看数据，缺视频清楚提示。未提供操作阶段自动对齐、全量原始Tensor热图、真实机器人测量状态补全、自动失败原因分类、训练曲线专页或实时控制。它们需要额外记录或独立标签，不能由现有CSV推造。原始视频256×256，放大不会产生新细节。

关闭后若导入了新数据，重启本地服务重新索引；当前不提供后台目录监控、共享多用户服务或远程公开部署。Python服务在前台运行，Ctrl+C退出；不启动常驻worker或GPU进程。
