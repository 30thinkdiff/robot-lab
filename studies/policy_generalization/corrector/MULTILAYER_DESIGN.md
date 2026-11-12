# 多层视觉变化预测：已验证接口与下一步最小方案

目标保持为**动作条件下未来多层视觉特征变化**。不训练任务失败分类器，不以最终success/failure作为预测器标签。现有Case Bank没有逐步多层视觉特征；不能从旧Connector缓存恢复中间层。

## 冻结模型实际结构

以固定SmolVLA revision `a24f1f380461b665158d52acd140152c79e8b36d`的完整权重为准。实际视觉模块类为 `SmolVLMVisionTransformer`（`smolvlm_vision`、SigLIP式视觉编码路径），不能直接按另一个标准`SiglipModel`类名挂Hook。

真实缓存配置和已安装Transformers源码核对、三帧双相机实测均确认：

```text
官方预处理：RGB → 保比例pad到512×512 → [0,1]变[-1,1]
vision embeddings：16×16 patch，32×32 = 1024 tokens，宽768
vision_model.encoder.layers：12个视觉Transformer block
vision_model.post_layernorm：末层后LayerNorm
connector：pixel shuffle factor4
           1024×768 → 64×12288 → 线性模态投影 → 64×960
两路相机按固定顺序拼接：128×960（现有Corrector表示）
```

视觉12层与日志中“VLM layers reduced to16”不是同一层数：后者指语言/VLM文本层。中间block输出位于末层LayerNorm之前，不能直接把其误差与末层Connector误差当作同分布数值，也不直接将训练在末层的Connector强套到中间层。

## 已实现的独立只读接口

[corrector_multilayer.py](../../../scripts/corrector_multilayer.py)提供：

- `ReadOnlyVisionTap`：仅在独立eval、参数冻结实例上临时注册forward hook，默认block索引2/5/11（从0计数），另观测post-layernorm与Connector。Hook返回None，捕获detach副本，不替换forward、不改原输出、不改模型类或安装包。逐相机上下文拒绝嵌套、重复前向、缺层和无界缓存；异常也清除Hook。
- `write_frame`：明确episode、frame_index、真实数据timestamp、相机顺序、各层位置与shape；安全数值NPZ（无pickle），不覆盖既有帧。中间层按明确的非重叠空间均值池化为8×8，Connector保持原64tokens；这是新缓存协议，不冒称原特征空间。
- `aligned_pairs`：同episode、相机顺序、层形状、池化与dtype严格匹配；k10只对应10Hz数据的1秒，不误用20个数据帧。重复帧或实际时间跨度异常即拒绝。

接口实现不依赖全量`output_hidden_states`，也不重算中间层的Connector。当前只缓存指定时刻，不接入在线LVM/OGG，不改变正在使用的SmolVLA代码。

## 三帧真实示范核验（已完成，无训练）

[corrector_multilayer_probe.py](../../../scripts/corrector_multilayer_probe.py)仅在GPU无进程、旧评测completed/paused时运行，独立目录、100MiB产物上限、系统盘至少5GiB；全程本地缓存/离线模式。复用原严格完整checkpoint加载、bf16视觉精度和官方预处理；显式`LEROBOT_SRC`与`sys.path`指向原官方冻结LeRobot，参考仓库只提供Extractor。

使用原示范parquet file213、episode813的frame0/10/20，两相机共6幅真实图像，构成0→10、10→20两个同轨迹1秒变化对。没有用视频帧或模拟图片代替示范，没有下载新数据。

| 指标 | 实测 |
| --- | --- |
| block2/5/11，每相机原shape | 各 `[1,1024,768]` |
| post-LayerNorm / Connector | `[1,1024,768]` / `[1,64,960]` |
| 池化后各中间层缓存 | `[2,64,768]`；两相机维不丢失 |
| Connector缓存 | `[2,64,960]` |
| 原前向 / Hook前向均值 | 4.321 / 4.629 ms，每图，仅embed_image；18次同步对照，不含加载、预处理或VLA去噪 |
| 原输出一致性 | 18/18逐元素完全相等，Connector捕获也一致 |
| Hook前向相对调用前常驻量的峰值增长 | 26.751 MiB，含前向临时分配；不是与无Hook峰值相减，不能当作纯Hook额外显存 |
| 完成时模型/张量已分配 | 0.876 GiB；进程退出后GPU无任务 |
| 三帧压缩文件 | 1,952,919字节；未压缩数值2,506,752字节 |
| 安全性 | 全有限、所有Hook移除、权重SHA未变、冻结LeRobot跟踪文件无修改 |

这里只是小规模同步微基准，不是全控制器延迟，也未测多层预测网络或OGG梯度成本；池化、CPU传输及压缩写盘未计入上述embed_image时间，尚不能据此宣称端到端提取吞吐。按实际shape及native bf16计算，单相机Hook快照副本约6.117MiB，两相机约12.234MiB；无Hook版本的峰值未单独保存，后续严格显存对照应补这个指标。真实CPU Torch的3项接口/异常/缓存测试也已通过；本机未安装Torch，未为测试改变其环境。

结构、每图原值、来源SHA与成本详见[summary.json](../results/20261112/multilayer_probe/summary.json)。少量原NPZ只在远程独立探针目录，未导入Case Bank，也不提交Git。V1/V2探针在参考Extractor重排LeRobot导入来源时停止，未产生特征；V3复用原提取流程的显式`LEROBOT_SRC`后通过，旧代码现场保留，没有安装imageio或修改第三方包。

## 缓存格式与存储边界

```text
MANIFEST.json：schema、策略/权重/来源SHA、层位置、池化、dtype、时间基准、相机顺序
cache/episode_813_frame_000000.npz
    block_02  float16 [camera=2, patch=64, dim=768]
    block_05  float16 [2,64,768]
    block_11  float16 [2,64,768]
    connector float16 [2,64,960]
records：episode/frame/timestamp、每数组shape、原始shape、字节数、文件SHA
```

视觉小缓存本身不是独立完整训练集。动作通过[只读动作引用](../results/20261112/multilayer_probe/action_reference.json)连接到原episode813的`actions.npy`，已CPU核验SHA、`[155,7]`、10Hz及`checkpoint_processor_normalized`表示。两对视觉变化分别使用原示范动作切片`[0:10]`、`[10:20]`，不拿规划动作冒充已执行示范，也不重复存一份动作数组。

计算沿用native bf16，缓存转换为float16，有限值校验后写入；没有任意缩放修补。数据10Hz/控制20Hz分别记录，格式转换timestamp仍不等于物理采集时钟已证实。

当前方案每帧未压缩0.796875MiB，全66,984帧约**52.127GiB**，远超当前数据盘剩余8GiB；不启动全量缓存。三帧压缩率不能保证代表全数据。下一阶段先限定少量完整episode与训练/验证拆分，再比较4×4/8×8空间池化、仅缓存两层、per-token int8+显式scale及复用原Connector缓存的误差/存储折中。每种压缩协议独立版本化，不能默默改变训练目标来适配磁盘。

## 后续模型设计与最小测试

令每层和相机的视觉表示为Fᶩ_t，预测器输入历史表示与动作条件，输出每层变化：

`fθ({Fᶩ_t}, A[t:t+k]) → {ΔF̂ᶩ_t}`，目标 `ΔFᶩ_t = Fᶩ_(t+k) − Fᶩ_t`。

建议先选少量完整示范，保留原episode划分；逐层投影或融合输入，但保持每层独立输出头及损失，避免较大幅值的一层支配训练。归一化/权重只在训练集确定，验证集评估预测误差，不用任务失败标签训练分类器。当前A10的“示范实际动作 vs 部署未来计划”分布差异仍在，不能靠多层输入自动解决。

LVM以后比较各层真实与预测变化，用独立正常示范/验证数据校准分数及阈值；不直接沿用现有单层阈值。OGG以后对多层纠偏目标形成明确加权损失，通过新预测器对候选动作求导；验证每层梯度尺度、作用slot及控制接口，再做monitor-only、最后少量配对闭环。

本轮最小真实shape/数值/时间/显存/存储核验已经完成。下一步先用CPU比较缓存协议的数值误差与容量，再决定少量episode规模。**尚未实现或启动多层预测器训练、多层LVM阈值校准、多层OGG或SmolVLA微调。**
