#!/bin/bash

# 默认模型路径，可以被环境变量覆盖
DEFAULT_MODEL_PATH="./output/your_default_finetuned_model"
MODEL_PATH=${LLM_MODEL_PATH:-$DEFAULT_MODEL_PATH} # 使用 LLM_MODEL_PATH 环境变量，否则用默认

# 默认推理配置文件
DEFAULT_INFERENCE_CONFIG="./configs/inference_config.yaml" # 你需要创建一个这样的文件
CONFIG_FILE=${LLM_INFERENCE_CONFIG:-$DEFAULT_INFERENCE_CONFIG}

# API 服务端口
PORT=${API_PORT:-8000}

# Tensor Parallel Size for vLLM
TP_SIZE=${TP_SIZE:-1}

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# 打印将要执行的命令
echo "Starting inference API server..."
echo "Model Path: ${MODEL_PATH}"
echo "Config File: ${CONFIG_FILE}"
echo "Port: ${PORT}"
echo "Tensor Parallel Size: ${TP_SIZE}"

# 将模型路径作为环境变量传递给 vLLM 引擎
export MODEL_PATH # vLLM engine in api_server.py will pick this up
export TP_SIZE    # vLLM engine in api_server.py will pick this up

# 启动 API 服务
# 注意：如果 inference_config.yaml 中也定义了模型路径，环境变量会优先
python -m llm_factory.main inference_api --config "${CONFIG_FILE}" --port "${PORT}"
# main.py 的 inference_api 也需要接收 --port 参数

echo "Inference API server script finished (or running in background if detached)."