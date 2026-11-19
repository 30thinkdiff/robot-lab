# V2：2号机数据准备与最小GPU门槛

本版不改MLP架构/损失、LVM或OGG。V1开发ZIP保留，新的模块和结果另建版本目录。

## 元数据包

S2/S8不需要把原Connector张量输入训练，只需要原动作、帧号、时间、划分及完整来源证明。`scripts/corrector_conditioning_export.py`只读原454条缓存，验证所有原文件SHA后复制这些小文件；同时完整流式读取原z_q/z_scale并验证其原始NPY SHA，生成66,984个逐帧`sha256(q[t].bytes+scale[t].bytes)`。

因此仅传约6.12MiB的ZIP，而非约8GiB张量；新提取器仍可逐帧验证原Connector量化字节完全一致，没有改成近似比较或降低校验覆盖率。

新目录使用`METADATA_COMPLETE.json`和原样`SOURCE_COMPLETE.json`，明确为metadata-only，**没有冒充完整Connector缓存的COMPLETE.json**。原动作/时间文件字节保持历史SHA；`metadata.py`和`data.py`支持两种明确格式，不补造缺失张量。旧缓存、模型和报告不改写。

## 实际GPU小检查

`python -m robotlab_single_layer.gpu_probe --execute-real-gpu ...`仅供显式的小检查：

1. 调用已成功的冻结导出器，读取真实Episode813的前13帧；复用原图像处理、BF16视觉编码及FP16保存。
2. 验证13帧Connector量化完全匹配；Block2/8的FP16帧字节还须匹配1号机旧原始缓存SHA。
3. 取真实t2→t12变化与原a2，测试原完整宽度MLP、microbatch32、MSE+cosine backward、checkpoint reload及原OGG首动作7维梯度。

为测试microbatch32显存，仅将这一个真实样本重复32次；它不是32个独立样本、训练epoch或验证指标。OGG候选流张量是明确的接口测试输入，不是VLA实际执行的动作。探针权重均标为discarded，不作为正式checkpoint。

13帧前缀写独立probe目录，不发布到raw454，也不能通过完整数据校验。OOM/数值/对齐错误直接保留现场并停止，不自动缩模型、降分辨率、改loss或换microbatch。

探针即使通过，结果仍明确`all_stage_C_gates_passed=false`，因为真实闭环/motion采集一致性等门槛尚未完成；不会由此自动启动454条完整提取或正式训练。
