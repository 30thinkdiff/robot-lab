# Replay四方法与Plus代表案例 · 2026-11-14

本次只读导入已完成结果，没有运行新仿真、训练或特征提取。原始结果、权重和远程文件保持不变。新增54条Plus回放，索引从1562增至1616；普通LIBERO仅新增引用视图，不复制回合。

## 模型身份

`j_lvm`与`j_ogg`使用正式J-36M联合多层预测器（36,793,536参数），共享主干、多层输出头，输入Connector、SigLIP Block 2/8视觉特征与单动作条件，预测未来视觉变化。当前冻结J权重SHA为`12b324888342ba17bf1035f94ed4c61fd1cd62f8b4f40d6b6c0e48c901e52ef1`。两方法共享检测协议，后者在事件重规划时启用OGG。它不是A10动作序列模型，也不是J_equal或0.767M早期MVP。

## 成败组合与资料完整性

下表四位顺序始终为 **J-LVM / J-OGG / MLP-LVM / MLP-OGG**；1成功、0失败。按结局挑选的案例用于诊断，不能据此估计总体成功率。

| 来源 | 条件 | 评测seed | 四位组合 | 四方法完整回放 |
|---|---|---:|---|---|
| Plus首轮 | plus/cheese/background_textures/1 | 1000 | `0000` | 是 |
| Plus首轮 | plus/cheese/camera_viewpoints/1 | 1000 | `0000` | 是 |
| Plus首轮 | plus/cheese/light_conditions/1 | 1000 | `0000` | 是 |
| Plus首轮 | plus/cheese/robot_initial_states/1 | 1000 | `1111` | 是 |
| Plus首轮 | plus/cheese/sensor_noise/1 | 1000 | `0011` | 是 |
| Plus首轮 | plus/default | 1000 | `1111` | 是 |
| Plus首轮 | plus/soup/language_instructions/1 | 1000 | `1101` | 是 |
| Plus首轮 | plus/soup/objects_layout/2 | 1000 | `0010` | 是 |
| Plus选例复测 | plus/cheese/sensor_noise/1 | 1002 | `0011` | 是 |
| 普通LIBERO（跨批次审查） | libero_object/task_0/init_0 | 1000 | `1111` | 是 |
| 普通LIBERO（跨批次审查） | libero_object/task_0/init_2 | 1001 | `0000` | 是 |
| 普通LIBERO（跨批次审查） | libero_object/task_9/init_2 | 1000 | `1100` | 是 |
| 普通LIBERO（跨批次审查） | libero_object/task_1/init_0 | 1001 | `0011` | 否：旧MLP缺视频及执行动作 |

普通LIBERO原60组匹配条件只观察到`1111/0000/1100/0011`。先选每类一个，优先已有完整回放：3组完整；`task_1/init_0/seed1001`的旧MLP成功回合未保存视频和逐步动作，保持缺失，不拿其他seed或重跑结果替代。原Trace/指标仍可读取。

普通对照引用原Case Bank、原正式MLP及J补测，保留逐回合原协议；4回合桥接已通过，核对同task/init/eval seed、冻结SmolVLA版本及各自checkpoint。界面明确标“桥接审查·跨批次同条件诊断”。这是既有实验的诊断对照，并非重新声明同批训练或完全相同训练数据量。

Plus首轮选8组×6方法＝48条，覆盖实际观察的5种四方法结局（`0000/1111/0011/1101/0010`）、七类扰动和历史相机条件。另导入已有`cheese/sensor_noise/1`的seed1002复测6条，保持独立来源，不混算首轮。每组还保留H50/H20基线。`1101`与`0010`可分析MLP-LVM和MLP-OGG的真实结局翻转；此次精选未发现J-LVM与J-OGG的结局翻转，不制造未观察到的模式。

## 如何打开

1. 重启本地Replay服务并刷新网页。点击“成败组合样本库”。
2. 方法组合选择 **J多层 / 原MLP：LVM与OGG四方法**。
3. 来源选择“普通LIBERO · J与MLP四方法（桥接审查）”或“Plus首轮 · 成败与扰动精选”。
4. 选成败模式，并从“主选/对照”中任选两种方法打开。要看缺视频的普通`0011`，取消“视频/trace/动作齐全”及左侧“仅本地有视频”；它不会出现在完整回放筛选里。
5. 推荐先看Plus `cheese/sensor_noise/1` seed1000：J两方法失败、MLP两方法成功；再看`soup/language_instructions/1`的`1101`与`soup/objects_layout/2`的`0010`。统一环境步对齐不代表同一操作阶段。

## 数据与验证

保留原trial字节、独立视频绑定SHA、录制源码证据、逐帧PTS映射、真实提交动作、控制返回、队列、评分、阈值、OGG Loss/梯度及更新摘要。完整候选Tensor/NPZ、噪声和仿真状态PT仍留远端；本次没有导入，不能在本地声称这些字段可用。新增解压资料约29MiB，另有小型压缩包。

普通配对清单逐项验证源trial SHA、方法/模型、条件、seed和桥接证据SHA；损坏/不匹配的整份清单不进入样本库。引用视图不生成重复实验记录。详细筛选及来源SHA见[selection.json](selection.json)。

新增5项测试（含54条真实视频逐项同步/动作/评分验收）通过；Chromium实际交互两组通过，页面错误0。完整CPU回归70项，67通过、3跳过；系统Python无Torch的3项Hook测试明确跳过。临时验收服务关闭。
