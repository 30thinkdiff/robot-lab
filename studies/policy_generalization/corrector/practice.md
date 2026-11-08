# 核心实现练习：写哪里、怎样调用、怎样验收

2026-11-07。只实现下面五处；不要求你再写环境、下载器、日志或批量评测调度。
这些方法用于在本地CPU主动理解算法并与上游对照，不作为正式训练的默认依赖；正式入口优先使用固定上游实现。

最新状态：用户已填写五处核心。2026-11-07审查修复了MLP漏最后一个block、损失的方向/批次维/返回值问题；五阶段检查通过，并在两种小构造结构中与上游同权重前向及参数梯度对齐。尚未用472帧真实缓存训练预测器。

| 次序 | 文件与方法 | 你需要完成 | 上游参考 |
| --- | --- | --- | --- |
| A | model.py：ResidualMLPBlock.forward | 用已注册层完成分支，最后与输入相加 | MLP.py：_MLPBlock.forward |
| B | model.py：VisualDynamicsMLP.forward | 动作投影/展开/拼接→残差块→输出 | MLP.py：SiglipResidualMLP.forward |
| C | losses.py：corrector_loss | MSE与逐样本展平后的余弦项，返回三个Tensor | train.py：_combined_loss |
| D | dataset.py：CachedTransitionDataset.__getitem__ | 反量化前后帧，取动作，计算delta，返回Tensor字典 | dataset.py：__getitem__ |
| E | training_step.py：train_corrector_batch | 将已学的单batch训练迁移到双输入模型及新标签 | 原train_one_batch与上游train循环 |

初始框架提供层结构、初始化、文件读取、形状检查、分割组合；核心方法现已由用户填写并经审查修正。
不要把答案写进Markdown，直接填写对应.py方法。参考源码位于outputs/source_audit/vla-corrector/src/siglip_dynamics。

真实调用关系：

```text
check_corrector_practice.py
  → 用户block/model/loss/getitem/train_corrector_batch

练习中的训练调用关系：
build_episode_splits → 用户Dataset/配对 → DataLoader
  → 用户train_corrector_batch → 用户model/loss → backward/step

正式入口corrector_train.py：固定上游Dataset/MLP/loss
  → 项目提供的逐轨迹划分、训练/验证与checkpoint编排
```

初始化数据集已经真正调用你的配对函数；不是把它留在独立演示文件里。
训练/验证按每条缓存的metadata.split组合；相同episode跨集合会报错，不对配对样本随机拆分。
自写MLP保留上游的层/权重命名，但第一版只接收带batch输入，默认宽度较小；不声称配置与论文规模相同。

```bash
python scripts/check_corrector_practice.py --stage block
python scripts/check_corrector_practice.py --stage model
python scripts/check_corrector_practice.py --stage loss
python scripts/check_corrector_practice.py --stage dataset
python scripts/check_corrector_practice.py --stage step
```

也可使用--stage all。未写的方法显示“待用户实现”，退出码2；这不是环境安装失败。
形状/数学断言失败则保留实际traceback，用它定位错误；不修改检查来迁就错误实现。
这些检查用明确的小构造张量，不是视觉编码器实测或预测器效果评测。通过后再与上游同权重输出做进一步对齐。

第一轮可以对照源码写，不要求默写。每完成一个方法，说明输入、每行的形状变化、输出给谁，再运行检查。
优先理解并亲自写通这几十行；数据工作台和Notebook内嵌的长环境代码不用逐行背诵。

## 本轮审查：用户注释中的问题与关键修正

1. `drop`是`nn.Dropout`：训练时以概率p将元素置零，保留的元素除以1-p，以保持期望值；不是删除token或改变shape。当前默认p=0，因此不改变数值；eval模式也不随机置零。例如p=0.5时，输入[2,4,6,8]的一种可能输出是[4,0,12,0]，不是固定丢掉哪几项。
2. 对[B,C]输入，`unsqueeze(1)`与`[:,None,:]`都得到[B,1,C]；随后expand到[B,L,C]，把一个样本的动作特征交给它的每个视觉token。每个样本仍使用自己的动作。`expand`通常用共享存储的视图，不要原地修改展开结果。
3. block在同一宽度内计算并做残差相加；transition连接相邻block，当宽度不同则投影，宽度相同则Identity。例如widths=(256,128,128)，计算为block0(256)→transition0(256→128)→block1(128)→transition1(Identity)→block2(128)。3个block只有2个transition；zip按较短序列截断，会漏block2。现在遍历全部block，按索引判断其后是否还有transition。
4. `reshape(-1)`和默认`flatten()`都将所有轴展成一维，数值顺序一致；这里应保留batch轴。实际特征为[L,D]=[128,960]，若batch=16，则余弦比较输入要是[16,122880]，而非[1966080]。前者逐样本比较再平均，后者混合全部样本后只比较一次，损失和权重含义不同。
5. 相似度越大越好，损失要越小越好，因此用`1-cosine_similarity`。例如预测[1,2]、目标[2,4]，MSE=2.5、cos=1、方向损失=0：方向对了但幅度仍不对；两项结合才兼顾方向与数值。直接最小化cos会鼓励反方向。实现返回(total,mse,cosine_loss)三个标量Tensor，和训练调用中的三变量接收对应。
6. Dataset的`scale[:,None]`是[L]→[L,1]；例如量化特征[[10,-20],[3,4]]与scale=[0.1,0.5]相乘，恢复为[[1,-2],[1.5,2]]。每个token有自己的scale，其各维共用这个scale。原用户实现正确，只去掉return后不可达的占位异常。

残差块和单batch更新顺序保留用户实现。`.item()`在backward与step之后用于返回日志是合适的；不能先把loss变成Python数值再反向传播。单步入口目前是无AMP/梯度裁剪的最小实现；参考训练器的梯度裁剪、验证、保存恢复等由完整训练入口另行接入，不声称已经复现整个上游训练配置。

本次58项CPU检查通过，其中含五阶段计算检查、逐样本余弦回归，以及同权重的等宽(8,8)/变宽(8,6,4)上游输出与各参数梯度对齐。这里是小构造数据的实现验证，不计作真实缓存训练或Corrector成功率提升。

用户后续明确：本轮重实现的主要目的为理解与语法补足，不要求将其作为生产分支。已据此调整正式调用链，见[真实训练运行说明](../corrector_training_runbook.md)。
