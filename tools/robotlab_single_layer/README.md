# Object10单层MLP：受控适配基础代码

截至2026-11-19：完整形状GPU反向、重载和OGG首动作梯度通过，真实LIBERO的12个短回合日志/运动采集一致性通过。**454条完整特征全部通过SHA及时间/划分检查，六组正式训练已启动；当前采用独立加速续训版本，当前从完整epoch122继承。** 训练/ETA/暂停续接见[TRAINING.md](../../studies/policy_generalization/corrector/single_layer_20261118/TRAINING.md)，最新执行优化等价验收见[EXECUTION_OPTIMIZATION.md](../../studies/policy_generalization/corrector/single_layer_20261118/EXECUTION_OPTIMIZATION.md)。进度、滚动ETA与安全暂停/续接见[OPERATIONS.md](../../studies/policy_generalization/corrector/single_layer_20261118/OPERATIONS.md)。工程探针与短回合不是正式训练或机器人成功率成绩。

任务依据及逐项差异见[BASELINE_AUDIT.md](../../studies/policy_generalization/corrector/single_layer_20261118/BASELINE_AUDIT.md)。

## 已实现

- 从实际运行的`night-code-1e80242789`按SHA加载原MLP、损失、训练器及在线控制器；不导入J训练/校准/在线类，不自动下载或checkout源码。
- S2/S8仅将token_dim改为768，保持四个2048宽Residual MLP块；`LayerShapeAdapter`只做`[B,2,1024,768] ↔ [B,2048,768]`。实际37,266,176参数，减少的是输入/输出维度，不是隐藏容量。
- 原454条、364/90划分和共同anchor3；训练/验证样本49,447/12,089，与原MLP的样本顺序索引SHA一致。
- 只读FP16 memmap，逐Episode验证原始层SHA、原动作/元信息SHA、相机、时间和Episode身份。按需复制当前帧/动作及目标，不将未来输入模型；每进程最多16个映射。
- 在线直接继承历史单动作控制器，只覆盖视觉观测与可选日志；动态窗口15、bootstrap15、cooldown0、单动作预测保留、原余弦OGG保持。零向量仍遵循原公式，不引入J的过滤。
- 可选日志记录EWMA/on-off阈值/计数/预测origin-due/队列，以及实际首候选动作梯度、速度/候选更新、OGG loss。梯度Hook只观察，不返回替换值。
- Replay-v1方法身份、时间协议和motion-v1需求声明；保留现有读取器，不改UI或旧数据库。

Block2/8沿用已有**零基索引**`vision.encoder.layers[2]`和`[8]`输出，不悄悄换成`[1]`/`[7]`。

## 本地只读/CPU入口

本地使用已有环境，没有升级Python依赖。2号机为真实渲染补齐EGL/GLES系统运行库及APT依赖，Python/Torch和冻结源码不变。以下为本机已实际执行的命令。

```bash
# 完整分辨率的meta形状与参数检查，不分配GPU/真实激活
/home/hhd/ws_lerobot/.venv/bin/python tools/robotlab_single_layer/run.py inspect-model \
  --baseline-dir outputs/corrector_j36m_repository/night-code-1e80242789 \
  --source-root outputs/source_audit/vla-corrector

# CPU单元测试。实值梯度测试采用完整模型宽度和少量Token；
# 全2048 Token使用meta检验，不能将两者混称为真实全分辨率GPU测试。
PYTHONPATH=tools /home/hhd/ws_lerobot/.venv/bin/python \
  -m unittest robotlab_single_layer.test_core robotlab_single_layer.test_train_queue robotlab_single_layer.test_fast_pipeline robotlab_single_layer.test_packed_pipeline robotlab_single_layer.test_execution_pipeline -v
```

测试源目录可通过`ROBOTLAB_SLC_BASELINE_DIR`、`ROBOTLAB_SLC_SOURCE_ROOT`显式指定，实际成功的冻结导出脚本用`ROBOTLAB_SLC_EXPORTER`指定。无指定冻结源码时测试会报错，不能把缺依赖写成通过。

生成**计划清单**（只写一个新JSON，不制作特征）：

```bash
python3 tools/robotlab_single_layer/run.py plan \
  --old-manifest ORIGINAL_OBJECT10_MANIFEST.json \
  --baseline-protocol tools/robotlab_single_layer/historical_mlp.json \
  --raw-root /root/autodl-tmp/corrector_single_layer/raw454 \
  --connector-root /root/autodl-tmp/corrector_imports/object10_features_20261118_001 \
  --output NEW_MANIFEST454.plan.json
```

原manifest必须与历史SHA一致；重复Episode、改变划分/帧数、降分辨率或时间基准会被拒绝。已生成本地计划位于忽略目录`outputs/single_layer_foundation_20261118/manifest454.plan.json`，状态明确为`planned_not_extracted`。

扩容后且原缓存已准备好，`inspect-cache --manifest ... --layer block_02`可只读校验该层；它会计算真实特征SHA，可能读取约196GiB，不要把它当作瞬间完成的JSON检查。本轮未对远端执行。

## 后续门槛与执行状态

1. 已确认550GiB数据盘、120GiB RAM、4090以及98个原始Parquet的逐文件SHA，匹配环境和固定模型已迁入，不安装/升级旧依赖。
2. 454条原动作/时间/划分全部SHA通过；Connector仅传一个真实探针缓存，其余用原66,984帧量化字节SHA保留严格对齐验证，详见[V2_DATA_PREPARATION.md](V2_DATA_PREPARATION.md)。未伪造完整Connector缓存。
3. 真实13帧原始层字节对齐、完整分辨率GPU反向、重载及OGG梯度已通过；CPU原状态机和日志开/关一致性通过。真实GPU在线控制/motion采集12短回合已通过（未自然触发，不能作恢复收益结论）。视觉路径要求FP32；冻结语言/动作专家保留原加载的混合dtype，不强制转换整个策略。
4. 首先测试原microbatch32/accumulate16；只有真实显存约束需要时，才记录理由并改为整除512的microbatch和对应累积。禁止自动降容量/分辨率或改loss。
5. 真实门槛收据绑定manifest/数据适配器/训练适配器/原训练器SHA及microbatch，`training.fit`才允许调用原训练循环。`job.py`提供显式的制作特征start/resume/status/pause；不自动启动fit或正式评测。
6. `extraction.py`实现逐Episode原Connector哈希验证、全SHA后原子发布、STOP/两盘边界暂停、残留attempt保留、独占锁及滚动ETA。首143帧实际验收并通过恢复后继续制作；正式LVM→OGG评测、同轨迹被动评测及Replay导入待正式训练完成后集成。

训练复用原AdamW、CosineAnnealingLR、MSE＋cosine、beta1、梯度裁剪1、seed42/43/44、最多150轮/patience5及验证total-loss选best。epoch恢复/STOP由原训练器负责。提取与训练是独立阶段；制作完成不会自动将discarded工程权重用于正式实验。

## Replay与运动采集接法

`replay_contract.case_config`生成`s2_lvm/s2_ogg/s8_lvm/s8_ogg`所需配置与显示标签。生成case时method显式使用这些新身份，不能重标旧MLP/J。

后续真实评测需要复用现有`MotionCapture`，安装顺序为：先`controller.installed()`，再`MotionCapture(...)`，最后原`eval_policy(...)`。这样运动包装器看见的是纠偏后的真实返回动作及真实控制提交。视频、frame_map及motion SHA随新trial保存；本轮只完成接口要求声明，尚未用S2/S8验证真实采集或导入。

沿用原框架的单环境、单回合实例约定；每个新Episode创建新的Corrector和采集器，不跨回合复用预测队列/阈值历史。

原始资料只读，计划/日志/导出写独立目录。不把名义路径当实测状态、不补造缺失运动、不将CPU假环境测试当机器人成功率。
