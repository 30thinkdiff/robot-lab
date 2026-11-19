# LIBERO任务组成与本次90子集回放

核对日期：2026-11-17。依据[LIBERO官方README](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/README.md)和[LeRobot官方套件表](https://huggingface.co/docs/lerobot/libero)。这里只解释任务组织，不将通用文档等同于我们固定checkpoint的训练清单。

LIBERO原始基准有130个任务：Spatial 10、Object 10、Goal 10，以及LIBERO-100的100个。LIBERO-100进一步拆为90和10，因此运行时代码通常提供下面五个套件。

| 套件 | 代码名称 | 任务数 | 主要关注 |
|---|---|---:|---|
| Spatial | `libero_spatial` | 10 | 空间关系与位置条件 |
| Object | `libero_object` | 10 | 不同物体的操纵 |
| Goal | `libero_goal` | 10 | 目标变化与目标条件操作 |
| LIBERO-90 | `libero_90` | 90 | LIBERO-100中较短操作任务与综合知识迁移 |
| Long / LIBERO-10 | `libero_10` | 10 | LIBERO-100中长程、多步骤任务 |

90与Long共同组成100；Spatial、Object、Goal是另外三个集合。官方持续学习设计中，90用于预训练策略、10用于下游持续学习评测；这只是基准的设计用途，不证明本项目SmolVLA训练过90。

LeRobot常规四套件评测入口是Spatial/Object/Goal/Long，共40任务，90是额外的另一个任务集合。任务数字表示数量，不是难度等级；不能说90必然比Long更长、更难。我们上次从90中预先按ID分散选十任务，不是全部90，也不是按失败筛选的十任务。

本次是Object训练的冻结MLP Corrector迁移实验。基础SmolVLA的完整任务级训练覆盖尚未核实，不能把90低成功率直接断言成零样本结果、90任务本身更难，或仅是Corrector导致失败。

## 已导入的两个完整四方法配对

任务为`libero_90/task_29/init_1`：**把黑碗放进柜子的顶层抽屉**（`KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet`）。一个命名task/init条件、两个eval seed，初始状态向量字节完全一致；不当作两种不同物理场景。

| eval seed | H50 | H20 | MLP-LVM | MLP-OGG |
|---|---|---|---|---|
| 1000 | 失败，400步 | 成功，189步 | 失败，400步 | 成功，224步 |
| 1001 | 失败，400步 | 失败，400步 | 失败，400步 | 成功，229步 |

四方法顺序的成败组合为`0101`和`0001`。两条OGG各记录3次事件、12次真实引导更新。全部八回合有原视频、帧映射、控制命令、动作块和队列；MLP方法有逐步评分与引导记录。没有新增运动采集，实际关节状态/力矩等未记录字段仍显示缺失。

## 在Replay打开

重启本地Replay以重新索引新批次，浏览器刷新以加载新预设。选择：

1. **成败组合样本库 → 方法组合：原MLP：H50 / H20 / LVM / OGG**。
2. 来源：**跨任务泛化 · 原MLP四方法**。
3. 阶段：**LIBERO-90固定子集**；保留“完整资料”，可勾选“仅本地有视频”。
4. 选`0001`查看seed1001，再选MLP-LVM与MLP-OGG打开：直接比较失败与恢复。
5. 选`0101`查看seed1000，可对照H20与OGG两种成功路径。统一环境步不代表相同操作阶段。

真实Chromium已验收四方法表、两个seed/组合筛选、双视频step140联动、OGG Loss/梯度、400步与229步回合结束边界，以及189/224步双成功对照；页面错误0。8段视频SHA、原trial SHA、录制相位和PTS/步数绑定均通过校验；对照提示读取真实首状态/输入/首噪声SHA，显示日志签名一致，不用条件名称推造签名。只在独立目录生成导出包，没有启动GPU或改写原实验。

## 存储与复建

- 压缩包：`tools/robotlab_replay/.local/casebank_downloads/crosssuite90_947fd6be5dc01368.zip`，2,983,072字节（约2.85MiB）。
- 实际回放：`tools/robotlab_replay/.local/import_batches/casebank_947fd6be5dc01368/results/attempts/<trial>/attempt_001/`。
- 视频位于上述目录`videos/eval_episode_0.mp4`；8条新记录加入后，本机索引1732条，索引数不是独立实验场景数。
- ZIP SHA256：`947fd6be5dc013680eb8ffb3b9036b39abcdb88c7b65a2dec21a1edb40cd127c`。
- 私有导入/浏览器收据及截图：`.local/review/crosssuite90/`；公开[验收摘要](replay_import_validation.json)。原始MP4/NPZ与私有路径清单不提交Git。

已有资料无需重复下载；相同包重复导入会复用，不复制trial或覆盖旧批次。以后在已配置SSH别名的机器上可复建：

```bash
python3 tools/robotlab_replay/import_crosssuite.py --plan
python3 tools/robotlab_replay/import_crosssuite.py --ssh-host <SSH别名>
```

无别名时可指定`--ssh-host root@你的主机 --ssh-port 端口`，服务器地址与凭据不写入公共文件。导出只取这8条，省略pickle/PT文件，原始文件仍在远端；不会下载模型或全部880回合。
