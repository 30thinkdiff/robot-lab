from lerobot.datasets import LeRobotDataset
import torch

REPO_ID = "lerobot/svla_so101_pickplace"
CHUNK_SIZE = 16

# 只使用 Episode 0，且不下载视频
dataset = LeRobotDataset(
    REPO_ID,
    episodes=[0],
    download_videos=False,
)

print("Episode 0 的帧数:", len(dataset))

# get_raw_item 读取原始 Parquet 行，不进行视频解码
first = dataset.get_raw_item(0)

print("原始数据字段:", list(first.keys()))
print("单帧状态 shape:", first["observation.state"].shape)
print("单帧动作 shape:", first["action"].shape)


def inspect_window(start):
    """展示指定起点的动作块，以及超出 Episode 末尾的 Padding。"""

    n = len(dataset)

    if not 0 <= start < n:
        raise ValueError("起点超出 Episode 范围")

    # 不允许索引越过当前 Episode
    indices = [
        min(start + i, n - 1)
        for i in range(CHUNK_SIZE)
    ]

    padding = torch.tensor([
        start + i >= n
        for i in range(CHUNK_SIZE)
    ])

    actions = torch.stack([
        dataset.get_raw_item(i)["action"]
        for i in indices
    ], dim=0)

    state = dataset.get_raw_item(start)["observation.state"]

    print(f"\n起点: {start}")
    print("输入状态维度:", tuple(state.shape))
    print("目标动作块维度:", tuple(actions.shape))
    print("动作索引:", indices)
    print("Padding:", padding.tolist())


inspect_window(0)
inspect_window(10)
inspect_window(len(dataset) - 3)