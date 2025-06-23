#!/bin/bash

# Ensure script exits on error
set -e

# Check if a configuration file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_config_file.yaml>"
  echo "Example: $0 configs/lora_llama3_8b.yaml"
  exit 1
fi

CONFIG_FILE=$1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '${CONFIG_FILE}' not found."
    exit 1
fi

# Set PYTHONPATH to include the project root, assuming script is run from project root
# or this script is in ./scripts/ and called from project root
SCRIPT_DIR_REALPATH=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR_REALPATH")
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

echo "PYTHONPATH set to: ${PYTHONPATH}"
echo "Running training with config: ${CONFIG_FILE}"

# You might need to set Hugging Face cache directory if default is not writable or too small
# export HF_HOME="/path/to/writable/hf_cache"
# export TRANSFORMERS_CACHE="/path/to/writable/transformers_cache"

# Determine number of GPUs for accelerate
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader || echo 0)

# Default accelerate config path (user can create this with `accelerate config`)
ACCELERATE_CONFIG_FILE="${PROJECT_ROOT}/accelerate_default_config.yaml"

# Check if accelerate is installed and if multi-GPU is intended
# For simplicity, this example directly calls python -m.
# For multi-GPU or FSDP, DeepSpeed, etc., uncomment and configure accelerate:
# if [ "$NUM_GPUS" -gt 1 ] && command -v accelerate &> /dev/null; then
#   echo "Detected $NUM_GPUS GPUs. Using accelerate launch."
#   if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
#     echo "Warning: Accelerate config file $ACCELERATE_CONFIG_FILE not found. Trying default."
#     # Attempt to run with default accelerate settings if config file is missing
#     accelerate launch --num_processes "$NUM_GPUS" "${PROJECT_ROOT}/llm_factory/main.py" train --config "${CONFIG_FILE}"
#   else
#     accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" "${PROJECT_ROOT}/llm_factory/main.py" train --config "${CONFIG_FILE}"
#   fi
# else
  echo "Running with python -m (single GPU or CPU or accelerate not used for launch)."
  python -m llm_factory.main train --config "${CONFIG_FILE}"
# fi

echo "Training script finished."