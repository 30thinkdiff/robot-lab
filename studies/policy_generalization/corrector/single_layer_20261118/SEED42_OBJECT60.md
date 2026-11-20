# seed42两模型完成后自动Object10评测

**最终状态：2026-11-20 04:34（北京时间）全部完成，训练2/2，评测240/240，错误0。** 历史交接验收保留在下文。[正式结果](../../results/20261120/single_layer_object60/RESULTS.md)。

用户2026年11月19日缩减计划：仅完成S2_s42和S8_s42，不自动开始seed43/44。训练质量协议不变，仍是最多150轮、patience5、验证总loss选best及重载验证。旧六组计划、源码、数据与checkpoint全部保留。

原`training_execution_001`已在S8 epoch77／7469更新的完整边界暂停。独立`training_seed42_001`引用只读的已完成S2结果，给S8复制模型/optimizer/scheduler/RNG/历史的精确断点。仅调整训练组数，不改变数据、模型、loss、batch或早停。

## 自动顺序

1. S2两个自然完整回合检查LVM/OGG、视频、动作、运动采集接口。这是正式S2相同冻结计划的前两条，验收通过后保留，不重复计入新样本。
2. 从完整断点继续S8_s42；S2/S8均完成且best重载通过、训练进程退出后才占用GPU评测。
3. S2两方法，再S8两方法：Object10 task0–9、init0/1/2、eval seed1000/1001，每方法60回合，共240条新回合。
4. 复用旧H50/H20/MLP-LVM/MLP-OGG共240条，生成`comparison.json`：八方法各60条。历史四方法不重跑。

旧参照来自`night/evaluation/mlp_s42/05d55ba9452142be/paired`，保存于工具的`historical_object60.json`，不是最早student试跑。每次核对历史评测入口SHA、策略/LeRobot/LIBERO版本、任务/初始状态/指令、20Hz、280步上限和LVM/OGG参数；不同机器上的首状态/输入/噪声位级一致性未独立与旧原始Tensor逐一核验，不能把仅配置匹配写成该证据已具备。

S2/S8沿用动态窗口15/bootstrap15、cooldown0、保留单动作待到期预测、原首动作余弦OGG eta1/every3。模型固定后不训练、不调阈值。新模型使用完整454条原始FP16层特征，与旧Connector量化特征的存储精度差异保留为限制。

## 记录及恢复

全部新回合保存视频、执行前frame_map、实际控制命令、动作队列、监测Trace、OGG轻量梯度/更新摘要、首状态/输入/噪声审计和MotionCapture的实际位姿/目标/力矩。未存的完整中间Tensor不补造；本轮不扩展Replay界面，原始记录留独立目录，之后可按需导入。

新队列只对进程丢失做最多两次训练恢复；明确程序错误停止保留现场，不把失败程序当机器人失败。评测按已提交完整回合跳过、未提交回合新建attempt；显式resume可接续。不强杀GPU进程，不删除历史文件。遇GPU占用或冻结协议不符会停止。

2号机总状态：

```bash
/root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/seed42_v3_manage.py status
```

训练进度/滚动ETA（只有两组）：

```bash
watch -n 30 /root/autodl-tmp/corrector/runtime/venv/bin/python /root/autodl-tmp/corrector/single_layer_20261118_001/train_seed42_manage.py status
```

总队列安全暂停/恢复：将`seed42_v3_manage.py status`改为`pause`或`resume`。暂停在epoch/回合边界生效，旧六组manager保持paused，勿用旧manager启动seed43。

结果根目录：`/root/autodl-tmp/corrector/single_layer_20261118_001/seed42_object60_002/`。查看`status.json`、`S2/execution.json`、`S8/execution.json`及两组`analysis/overall.csv`；日志为`S2_evaluation.log`、`S8_evaluation.log`及带时间的pipeline日志。全部结束后生成`comparison.json`。`status`不含完整训练ETA，需用上面的训练管理入口。

## 验收与状态

最终49项CPU测试通过，包括仅允许原六组或明确seed42两组、防止把暂停模型/缺重载证据模型用于评测。第一版部署索引父目录缺失及评测包缺MotionCapture的错误现场保留；前者未改原训练，后者进入受限两组训练恢复，不会继续seed43/44。V2补齐实际采集模块并保持冻结代码只读。实际GPU两个自然回合验收及最终上线状态见下方追加，不将CPU测试冒充GPU验证。

V2真实回合发现临时工厂替换使日志回调错误引用工厂函数。V3固定绑定原控制器基类，新增精确回归；没有改评分/阈值/动作逻辑。旧失败根目录001保留，新正式结果用002。交接bootstrap已独立后台化，网络断连不终止检查。V3独立ZIP SHA256：`ac44cd24fb217cb1894bd8caf2b4fc393e405d69f4dc96ebc25b7e115d41c40d`。

## 11月19日上线验收快照

V3已完成两个真实完整自然回合：S2-LVM成功、137步、1次重规划；S2-OGG成功、129步、1次重规划、4次实际引导。两条视频和全部控制命令/运动状态均完成保存，错误0。它们是固定S2正式120条计划的前两条，后续跳过，不重复算样本；少量成功不代表性能提升。

49项CPU回归通过。旧六组目录仍paused，新受限两组队列正在从S8 epoch77断点恢复，后台pipeline进程206267实际存活并进入`training_seed42_pair`。上线收据见[SEED42_HANDOFF_RECEIPT_003.json](SEED42_HANDOFF_RECEIPT_003.json)。该快照记录S8续训开始和首两条S2回合完成时的状态。

新结果源代码为`dev/slc-seed42-v3-ac44cd24fb21`；旧失败001目录和原六组计划保留，未删除缓存或权重。最终结果以本页开头链接的正式报告为准。
