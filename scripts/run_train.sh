#!/bin/bash

# 检查是否提供了配置文件参数
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_config_file.yaml>"
  exit 1
fi

CONFIG_FILE=$1

# 设置 PYTHONPATH 以确保 llm_factory 包可被找到
# 假设脚本从项目根目录运行 (./scripts/run_train.sh)
export PYTHONPATH=$(pwd):$PYTHONPATH

# 打印将要执行的命令
echo "Running training with config: ${CONFIG_FILE}"
echo "Command: python -m llm_factory.main train --config ${CONFIG_FILE}"

# 执行训练
# NCCL_P2P_DISABLE=1 # 如果遇到多GPU P2P 问题，可以尝试取消注释
# NCCL_IB_DISABLE=1  # 如果使用 InfiniBand 遇到问题

# 使用 accelerate launch 进行多GPU训练（如果配置支持）
# 这需要你的训练脚本 (如 sft_trainer.py) 和 TrainingArguments 配置为与 accelerate 兼容
# NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader)
# if [ "$NUM_GPUS" -gt 1 ]; then
#   echo "Detected $NUM_GPUS GPUs. Using accelerate launch."
#   accelerate launch --config_file ./accelerate_config.yaml llm_factory/main.py train --config "${CONFIG_FILE}"
# else
#   echo "Detected 1 GPU or CPU. Running directly."
  python -m llm_factory.main train --config "${CONFIG_FILE}"
# fi

echo "Training script finished."