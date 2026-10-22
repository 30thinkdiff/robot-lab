#!/usr/bin/env bash

# set -e：命令失败时停止
# set -u：使用未定义变量时报错
# pipefail：管道中的命令失败时返回错误
set -euo pipefail

# ===== 1. 实验参数 =====
# ${VAR:-default} 表示：
# 如果环境变量 VAR 没有设置，就使用 default。

RUN_NAME="${RUN_NAME:-baseline_001}"
N_EPISODES="${N_EPISODES:-1}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_DRIVER="${EVAL_DRIVER:-native}"

# 本地默认写入当前项目；远程可指定 /content/outputs
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/outputs}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_libero}"

# ===== 2. 检查参数是否合法 =====

# Episode 数量必须是正整数。
if ! [[ "$N_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：N_EPISODES 必须是正整数"
    exit 1
fi

# 限制实验名称中的字符，防止意外生成复杂路径。
if ! [[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "错误：RUN_NAME 含不合法字符"
    exit 1
fi

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "错误：DRY_RUN 只能是 0 或 1"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}"

# ===== 3. 构造命令 =====
# Bash 数组能正确保存每一个独立参数，
# 避免空格和引号造成参数拆分错误。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$EVAL_DRIVER" == "brightness" ]]; then
    EVAL_CMD=(
        python
        "${SCRIPT_DIR}/../studies/policy_generalization/eval_with_brightness.py"
    )
elif [[ "$EVAL_DRIVER" == "native" ]]; then
    EVAL_CMD=(lerobot-eval)
elif [[ "$EVAL_DRIVER" == "trace" ]]; then
    EVAL_CMD=(
        python
        "${SCRIPT_DIR}/../studies/policy_generalization/trace/eval_with_trace.py"
        "--trace-dir=${TRACE_DIR:-${OUTPUT_DIR}/trace}"
    )
else
    echo "Unknown EVAL_DRIVER: $EVAL_DRIVER"
    exit 1
fi
CMD=(
    "${EVAL_CMD[@]}"
    "--policy.path=${POLICY_PATH}"
    "--policy.device=cuda"
    "--env.type=libero"
    "--env.task=libero_object"
    '--env.task_ids=[0]'
    "--env.observation_height=256"
    "--env.observation_width=256"
    "--env.control_mode=relative"
    '--env.camera_name_mapping={"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}'
    "--eval.batch_size=1"
    "--eval.n_episodes=${N_EPISODES}"
    "--eval.use_async_envs=false"
    "--seed=1000"
    "--output_dir=${OUTPUT_DIR}"
)

echo "========== Experiment =========="
echo "Name: ${RUN_NAME}"
echo "Policy: ${POLICY_PATH}"
echo "Episodes: ${N_EPISODES}"
echo "Output: ${OUTPUT_DIR}"

# ===== 4. 显示将要运行的命令 =====

printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

# 如果开启 Dry Run，到这里就结束。
# 不需要 GPU，甚至不需要安装 LIBERO。
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry Run 成功：没有实际启动模型。"
    exit 0
fi

# ===== 5. 正式运行 =====

if ! command -v lerobot-eval >/dev/null 2>&1; then
    echo "错误：未找到 lerobot-eval"
    exit 1
fi

# MuJoCo 在无显示器的服务器上使用 EGL 渲染。
export MUJOCO_GL=egl

# 真正执行前面构造的命令数组。
exec "${CMD[@]}"
