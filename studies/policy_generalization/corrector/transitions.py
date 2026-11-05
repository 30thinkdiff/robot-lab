"""用户亲写的Corrector时序配对；由scripts/corrector_pair_audit.py实际调用。"""


def make_transition_pairs(episode_index, frame_index, k_step=1):
    """返回[(当前行号, 未来行号), ...]，不是episode内部的frame编号。

    两个输入为等长整数列表；同一episode连续排列，内部frame按升序排列。
    一对必须属于同一episode，且未来frame恰好比当前frame大k_step。
    不足k_step的轨迹尾部不能配对；k_step必须为正整数。
    示例：[813,813,818,818]、[0,1,0,1]、k=1 → [(0,1),(2,3)]。
    这里只有索引，不提视觉特征、不训练网络。监督标签随后才由z[j]-z[i]构造。
    """
    # 1. 检查输入长度和k_step。
    # 2. 遍历候选行，检查episode相同、frame相差k_step。
    # 3. 收集并返回行号对。你可以先使用熟悉的for循环，不要求向量化。
    raise NotImplementedError("请亲写make_transition_pairs；运行corrector_pair_audit.py检查。")
