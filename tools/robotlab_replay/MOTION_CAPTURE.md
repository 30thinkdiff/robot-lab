# 动作命令—控制目标—实际运动采集

**11月16日进展：HF缓存链接检查已修复，真实H50采集开关验收及12个运动诊断回合已通过；新数据包已取回并导入Replay，运动/视频联动与导出已通过真实浏览器验收。**见[GPU后续报告](../../studies/policy_generalization/results/20261116/gpu_followup/RESULTS.md)。下文“无卡/未GPU验证”是11月14日准备阶段的历史状态，旧部署ZIP保留，不作为11月16日的新版本使用。

11月14日准备阶段完成历史数据只读核查、少量动作块取回、CPU数值审计和可选采集代码；当时无卡，尚未进行真实采集。11月16日已完成上方所述GPU采集与回放验收。不修改安装包、冻结评测源码、旧回合和权重。

## 历史资料能回答什么

六个真实Plus配对回合已取回36个完整动作块与小型处理器统计，压缩包约462KiB。1263步的“生成块 → 出队slot → 反归一化 → 真正提交命令”全部一致。它们没有逐步实际位姿、关节状态、OSC目标和力矩；不能计算实际运动gap，更不能把命令链误差0写成运动误差0。

数据在忽略的`.local/motion_archive/historical_*/`。报告及原始来源SHA见[历史审计](../../studies/policy_generalization/results/20261114/motion_evidence/RESULTS.md)。这份审计只描述检查过的六回合；结合冻结采集源码可解释字段缺失，不断言所有未知旧目录都经过逐文件扫描。

已取回的动作是最终返回块。OGG回合可能已经受引导；不包含额外生成的无引导反事实。未执行候选slot是计划，不是实际动作，也不是观测到的未来状态。

```bash
# 本地CPU复算到一个新报告路径，不写原数据目录
python3 tools/robotlab_replay/analyse_motion_archive.py \
  tools/robotlab_replay/.local/motion_archive/historical_5c462ce34da26963 \
  /tmp/motion-command-audit-new.json
```

## 可选采集器

`motion_capture.MotionCapture`安装在既有单回合`eval_policy`调用外围：

```python
from motion_capture import ActionDecoder, MotionCapture

decoder = ActionDecoder.from_snapshot(frozen_policy_snapshot)
with MotionCapture(
    env.envs[0].unwrapped, policy, attempt_dir / "motion", decoder,
    method=case["method"], provenance={"case": case}, substep_stride=0,
):
    info = eval_policy(...)  # 既有调用，参数与控制逻辑保持原样
```

已提供`motion_pilot.py`适配器，无需手改冻结源码。仅支持已审查的`expanded-2e6296eff15f`部署、单Panda、相对OSC_POSE、20Hz及完整`[1,50,7]`动作块；不支持的接口/频率/解码会报错停下，不自动修补协议。

采集不增加模型前向、采样或物理步，不调用额外的`controller.update`/`sim.forward`。它会增加CPU拷贝和写盘耗时，GPU→CPU传输也可能同步GPU，因此需要实测开销，不能把诊断采集当作原版速度基准。

## 输出格式

```text
attempt_001/motion/
  protocol.json       # 来源SHA、解码统计SHA、实际控制器参数、初始状态、时间相位
  chunks/0001.json    # 完整归一化块、解码命令、块起点状态、名义轨迹及角色
  steps.jsonl         # 每个真实控制提交对应一行，完成一行即flush
  status.json         # completed / interrupted_or_error、完成数、预算和边界
```

每行含：`step/chunk_id/slot`、出队动作、实际控制命令、`before/after`实际状态、真实`set_goal`目标、单步跟踪误差、块锚定名义轨迹误差、限幅前后力矩、控制更新次数及可选子步记录。

- `before`为提交u_n之前的Z_n，`after`为返回后的Z_(n+1)；仿真时间直接读取，不以墙钟或FPS估造。终末Z_N也记录，但原N帧视频可能不含它。
- 位置使用米、关节角和旋转误差使用弧度；旋转矩阵按控制器世界坐标约定。夹爪命令与实际位置分开。
- `tracking_error`：真实OSC目标与本控制周期结束状态的偏差；不把非零值自动认定为控制故障。
- `nominal_block_error`：假设每个增量完美执行，从块起点构造的轨迹与实际状态之差；不是世界模型预测或保证正确的未来状态。
- 原始控制器输出、限幅后力矩命令、MuJoCo最后一次动力学计算的actuator force分别记录；后者不是硬件传感器测量。
- 没有虚构逐步7关节目标角；OSC主要追踪末端位姿，零空间关节参考另存协议。
- 尚未实现3D幽灵机械臂/影子仿真或Replay中的三轨迹专用面板。当前阶段保证采集及可追溯格式；旧数据缺失时不画假曲线。

## 未来有卡后的最小验收

先将独立部署包解到一个新的`dev/motion-capture-<SHA>/`，不要覆盖历史目录。以下命令均在服务器执行，`MOTION_CODE_DIR`指向解压后的独立目录；现在只可做inspect，不要执行run。

```bash
# 无卡也可检查冻结文件SHA、案例集合；不加载模型、不创建仿真
/root/autodl-tmp/corrector/runtime/sim-venv/bin/python \
  "$MOTION_CODE_DIR/tools/robotlab_replay/motion_pilot.py" inspect
```

恢复空闲4090后，先做一对采集开关对照：

```bash
/root/autodl-tmp/corrector/runtime/sim-venv/bin/python \
  "$MOTION_CODE_DIR/tools/robotlab_replay/motion_pilot.py" run \
  --case libero_object/task_9/init_2:1000 --methods H50 --capture off \
  --output /root/corrector_motion_reference_001 --hours 0.5

/root/autodl-tmp/corrector/runtime/sim-venv/bin/python \
  "$MOTION_CODE_DIR/tools/robotlab_replay/motion_pilot.py" run \
  --case libero_object/task_9/init_2:1000 --methods H50 --capture on \
  --output /root/corrector_motion_smoke_001 --hours 0.5
```

人工核对两目录中的初始state/input/noise SHA、`controller_commands.json`、结果、视频和耗时。采集smoke的`motion/status.json`必须completed，行数与真实步数相同；在真实LIBERO通过这一关前，不宣称物理轨迹未改变，不启动扩大队列。

通过后才考虑以下12回合：两个已知J/MLP结局翻转条件，各含H50/H20/MLP-LVM/MLP-OGG/J-LVM/J-OGG：

```bash
/root/autodl-tmp/corrector/runtime/sim-venv/bin/python \
  "$MOTION_CODE_DIR/tools/robotlab_replay/motion_pilot.py" run \
  --output /root/corrector_motion_pilot_001 --hours 2
```

默认仅保存每控制步首末力矩及统计；`--substep-stride 1`可记录所有实际控制器更新，最多1000条/控制步、32MiB/回合，超预算停止并保留现场。没有无界记录或自动扩展任务。

## 查看、暂停和恢复

```bash
cat /root/corrector_motion_pilot_001/ordinary/execution.json
tail -n 2 /root/corrector_motion_pilot_001/ordinary/attempts/<case-id>/attempt_001/motion/steps.jsonl
touch /root/corrector_motion_pilot_001/STOP
```

STOP按原队列在完整回合边界暂停，不强杀。恢复前先核对没有旧进程，确认暂停原因后将本次STOP改名留档，再运行相同命令。原队列跳过完成回合，并额外核验motion文件SHA；中断回合创建新的attempt，不在旧运动轨迹后拼接。配置或源码SHA变化必须换新输出目录。

CPU测试：

```bash
python3 -m unittest discover -s tools/robotlab_replay/tests -p test_motion_capture.py -v
```

NumPy来自现有本地/仿真环境；不安装或升级关键依赖。伪控制循环用于接口、异常和不额外执行的测试；真实数值验收来自旧六回合动作记录。二者均不等于真实GPU/LIBERO采集验收。
