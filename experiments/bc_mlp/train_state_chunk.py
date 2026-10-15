
"""
实验：使用真实 LeRobot 数据训练一个小型动作块预测网络。

这不是 ACT，而是用于学习模仿学习训练流程的 MLP。

输入：
    当前机器人状态，形状 [6]

输出：
    未来 16 个时间步的动作，形状 [16, 6]

训练数据：
    Episode 0、1、2

验证数据：
    Episode 3（不参加训练）

只下载数值数据，不下载相机视频。
"""

from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from lerobot.datasets import LeRobotDataset


# ------------------ 1. 基本配置 ------------------

REPO_ID = "lerobot/svla_so101_pickplace"
CHUNK_SIZE = 16
ACTION_DIM = 6

TRAIN_EPISODES = [0, 1, 2]
VAL_EPISODES = [3]

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

torch.manual_seed(42)

# 轻薄本上限制 CPU 并行线程数量
torch.set_num_threads(2)


# ------------------ 2. 加载真实数据 ------------------

print("正在加载数据集...")

dataset = LeRobotDataset(
    REPO_ID,
    episodes=TRAIN_EPISODES + VAL_EPISODES,
    download_videos=False,
)

# grouped 的结构：
# {
#   0: [(state0, action0), (state1, action1), ...],
#   1: [...],
#   ...
# }
#
# 按 Episode 分组，是为了避免 Action Chunk 跨越
# 两次独立演示的边界。

grouped = defaultdict(list)

for i in range(len(dataset)):
    frame = dataset.get_raw_item(i)

    episode_id = int(frame["episode_index"])

    state = frame["observation.state"].float().clone()
    action = frame["action"].float().clone()

    grouped[episode_id].append((state, action))

for ep, frames in grouped.items():
    print(f"Episode {ep}: {len(frames)} 帧")


# ------------------ 3. 构建训练样本 ------------------

def make_samples(episode_ids):
    """
    假设某个 Episode 的动作序列是：
        a0, a1, a2, ..., a100

    当 CHUNK_SIZE=16 时：
        输入 state0 → 目标 [a0, ..., a15]
        输入 state1 → 目标 [a1, ..., a16]
        输入 state2 → 目标 [a2, ..., a17]

    这就是滑动窗口构造 Action Chunk。

    为方便学习，我们只使用完整的 16 步窗口，
    不在 Episode 结束处添加 padding。
    """
    xs = []
    ys = []

    for ep in episode_ids:
        frames = grouped[ep]

        for t in range(len(frames) - CHUNK_SIZE + 1):

            # 当前状态 [6]
            x = frames[t][0]

            # 未来 16 步动作 [16, 6]
            y = torch.stack([
                frames[j][1]
                for j in range(t, t + CHUNK_SIZE)
            ])

            xs.append(x)
            ys.append(y)

    return torch.stack(xs), torch.stack(ys)


x_train, y_train = make_samples(TRAIN_EPISODES)
x_val, y_val = make_samples(VAL_EPISODES)

print("\n训练输入:", x_train.shape)
print("训练目标:", y_train.shape)
print("验证输入:", x_val.shape)
print("验证目标:", y_val.shape)


# ------------------ 4. 数据归一化 ------------------

# 各关节的数值分布不同。
# 归一化能让优化器更稳定地处理不同维度。
#
# 均值和标准差只能从训练集计算，
# 不允许提前使用验证集信息。

x_mean = x_train.mean(dim=0)
x_std = x_train.std(dim=0).clamp_min(1e-3)

y_flat = y_train.reshape(-1, ACTION_DIM)

y_mean = y_flat.mean(dim=0)
y_std = y_flat.std(dim=0).clamp_min(1e-3)

x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std

x_val_norm = (x_val - x_mean) / x_std


# ------------------ 5. 创建小型神经网络 ------------------

class ChunkPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Linear：可学习的线性变换
        # ReLU：让网络学习非线性关系
        #
        # 输入 6 维状态
        # 最终输出 16 * 6 = 96 个数

        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, CHUNK_SIZE * ACTION_DIM),
        )

    def forward(self, state):
        output = self.net(state)

        # [batch, 96] → [batch, 16, 6]
        return output.reshape(
            -1, CHUNK_SIZE, ACTION_DIM
        )


model = ChunkPolicy()

print(
    "\n模型参数量:",
    sum(p.numel() for p in model.parameters())
)


# ------------------ 6. 训练准备 ------------------

train_loader = DataLoader(
    TensorDataset(x_train_norm, y_train_norm),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# MSE：均方误差
# 预测越接近真实演示动作，损失通常越小
loss_fn = nn.MSELoss()

# Adam：根据梯度更新网络参数
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ------------------ 7. 真正开始训练 ------------------

print("\n开始训练...")

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0.0

    for batch_states, batch_actions in train_loader:

        # 前向传播：预测动作
        predicted = model(batch_states)

        # 计算预测与演示动作的误差
        loss = loss_fn(predicted, batch_actions)

        # 清除上一次迭代留下的梯度
        optimizer.zero_grad()

        # 反向传播：计算各个参数的梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_loss += (
            loss.item() * batch_states.size(0)
        )

    avg_loss = total_loss / len(x_train)

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {avg_loss:.6f}"
        )


# ------------------ 8. 在独立 Episode 上验证 ------------------

model.eval()

with torch.no_grad():

    # 模型预测的是归一化后的动作
    predicted_norm = model(x_val_norm)

    # 转回原始数据数值尺度
    predicted = predicted_norm * y_std + y_mean


# 评价指标：
# 每一维的绝对误差先除以训练集标准差，
# 避免数值变化范围大的关节主导总误差。
def normalized_mae(pred, target):
    return (
        (pred - target).abs() / y_std
    ).mean().item()


model_error = normalized_mae(predicted, y_val)

# 基线 A：直接把当前状态当作未来动作目标
# 本数据集是关节位置目标，才有这种比较的意义。
hold_prediction = (
    x_val.unsqueeze(1).expand_as(y_val)
)
hold_error = normalized_mae(
    hold_prediction, y_val
)

# 基线 B：永远预测训练集的平均动作
mean_prediction = (
    y_mean.reshape(1, 1, ACTION_DIM)
    .expand_as(y_val)
)
mean_error = normalized_mae(
    mean_prediction, y_val
)

# ========== 公平比较：第 1 步动作预测 ==========
#
# Chunk=1 和 Chunk=16 都会预测当前时刻的第一个动作。
# 因此可以把两种模型的第一个预测动作单独拿出来比较。
#
# 同时保证两个实验评价的是 Episode 3 中相同的前 168 帧。
# 对于 Chunk=1，需要排除最后 15 帧。

common_n = len(x_val)

if CHUNK_SIZE == 1:
    common_n = len(x_val) - 15

# 取出每个预测动作块的第 1 个动作
pred_first = predicted[:common_n, 0, :]

# 第 1 个真实动作
target_first = y_val[:common_n, 0, :]

# 使用验证集相同时间点的当前状态作为基线
hold_first = x_val[:common_n]

# 两个 Chunk 设置使用相同的尺度进行评价：
# 从所有训练 Episode 的动作中计算标准差。
all_train_actions = torch.stack([
    action
    for ep in TRAIN_EPISODES
    for _, action in grouped[ep]
])

eval_std = all_train_actions.std(dim=0).clamp_min(1e-3)

first_error = (
    (pred_first - target_first).abs() / eval_std
).mean().item()

hold_first_error = (
    (hold_first - target_first).abs() / eval_std
).mean().item()

print("\n===== 第 1 步共同预测任务 =====")
print("共同验证时间点数:", common_n)
print(f"模型第 1 步 MAE: {first_error:.4f}")
print(f"保持当前姿态 MAE: {hold_first_error:.4f}")

# 16 步模型还可以检查远期预测。
# Chunk=1 没有预测第 16 步，因此不进行这种比较。
if CHUNK_SIZE >= 16:
    future_error = (
        (predicted[:, 15, :] - y_val[:, 15, :]).abs()
        / eval_std
    ).mean().item()

    print(f"模型第 16 步 MAE: {future_error:.4f}")

print("\n========== 验证结果 ==========")
print(f"MLP 模型误差:       {model_error:.4f}")
print(f"保持当前姿态误差:   {hold_error:.4f}")
print(f"预测平均动作误差:   {mean_error:.4f}")

print("\n一条验证样本：")
print("当前状态:", x_val[0])
print("真实第 1 步动作:", y_val[0, 0])
print("预测第 1 步动作:", predicted[0, 0])
print("真实第 16 步动作:", y_val[0, -1])
print("预测第 16 步动作:", predicted[0, -1])


# ------------------ 9. 保存模型 ------------------

output = Path("outputs")
output.mkdir(exist_ok=True)

torch.save(
    {
        "model": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "chunk_size": CHUNK_SIZE,
    },
    output / "state_chunk_mlp.pt",
)

print("\n模型已保存到 outputs/state_chunk_mlp.pt")
print("注意：离线动作误差不等于机器人实际任务成功率。")
