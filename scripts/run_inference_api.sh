#!/bin_bash

# Ensure script exits on error
set -e

# Configuration file for the API server
# First argument to script, or default
CONFIG_FILE=${1:-"./configs/inference_config.yaml"}

# Model path: Use LLM_MODEL_PATH env var if set, otherwise use a default
# This default path should ideally point to a generally available model for out-of-box testing
# or a placeholder indicating user needs to set it.
DEFAULT_MODEL_PATH_FOR_SCRIPT="./output/default_model_for_api" # User should change this or set LLM_MODEL_PATH
export MODEL_PATH=${LLM_MODEL_PATH:-$DEFAULT_MODEL_PATH_FOR_SCRIPT}

# Tensor Parallel size for vLLM: Use TP_SIZE env var if set, otherwise default to 1
export TP_SIZE=${TP_SIZE:-1}

# API server port: Use API_PORT env var if set, otherwise default to 8000
API_SERVER_PORT=${API_PORT:-8000}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Inference config file '${CONFIG_FILE}' not found."
    echo "Please provide a valid config file or ensure './configs/inference_config.yaml' exists."
    exit 1
fi

# Set PYTHONPATH
SCRIPT_DIR_REALPATH=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR_REALPATH")
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

echo "PYTHONPATH set to: ${PYTHONPATH}"
echo "Starting inference API server..."
echo "  Config File: ${CONFIG_FILE}"
echo "  Model Path (from env MODEL_PATH): ${MODEL_PATH}"
echo "  Tensor Parallel Size (from env TP_SIZE): ${TP_SIZE}"
echo "  API Port: ${API_SERVER_PORT}"

if [ "$MODEL_PATH" == "$DEFAULT_MODEL_PATH_FOR_SCRIPT" ]; then
    echo "Warning: Using default model path '$DEFAULT_MODEL_PATH_FOR_SCRIPT'."
    echo "Ensure MODEL_PATH environment variable is set to your desired model."
    # Optionally, exit if a default model isn't truly usable:
    # echo "Please set the LLM_MODEL_PATH environment variable to point to your model directory."
    # exit 1
fi


# Start the API server using the main entry point
python -m llm_factory.main inference_api --config "${CONFIG_FILE}" --port "${API_SERVER_PORT}"

echo "Inference API server script initiated."