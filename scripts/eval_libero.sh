
#!/usr/bin/env bash

# 遇到命令失败、未定义变量或管道错误时停止。
# 这样可以避免前一步已经出错，后面却继续运行。
set -euo pipefail

# 这个程序只负责执行实验。
# SmolVLA 和 LIBERO 仍然由已经安装的 LeRobot 提供。

# 读取环境变量，允许外部覆盖。
# :- 表示变量不存在或为空时使用默认值。
RUN_NAME="${RUN_NAME:-smolvla_test}"
N_EPISODES="${N_EPISODES:-1}"

# Colab 上的实验输出目录。
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/outputs}"

# 默认使用之前验证过的模型。
POLICY="${POLICY:-lerobot/smolvla_libero}"

# MuJoCo 使用无界面 EGL 渲染。
export MUJOCO_GL=egl

# 先检查当前环境有没有安装 lerobot-eval。
# 在本地轻薄本上，可以只做脚本语法检查。
if ! command -v lerobot-eval >/dev/null 2>&1; then
    echo "错误：当前环境找不到 lerobot-eval"
    echo "请在已安装 LeRobot 与 LIBERO 的远程环境执行。"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

echo "========== LIBERO Experiment =========="
echo "Policy: ${POLICY}"
echo "Episodes: ${N_EPISODES}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================="

# 调用官方评估程序。
# 参数沿用你之前已经成功验证的配置。
lerobot-eval \
    --policy.path="${POLICY}" \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids="[0]" \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --env.control_mode=relative \
    --env.camera_name_mapping='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}' \
    --eval.batch_size=1 \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.use_async_envs=false \
    --seed=1000 \
    --output_dir="${OUTPUT_DIR}"

echo "评估命令已完成。请检查日志与输出目录。"
