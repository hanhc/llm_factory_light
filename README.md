```markdown
# LLM Factory

A comprehensive pipeline for Large Language Model (LLM) projects, including data processing, training (SFT, LoRA, RLHF), evaluation, and high-performance inference with vLLM.

## Features

*   **Modular Design**: Easily extendable and maintainable.
*   **Multiple Data Formats**: Supports JSON, JSONL, Parquet out-of-the-box.
*   **Versatile Training**: Implements SFT, LoRA, and a foundational RLHF (PPO) trainer.
*   **High-Performance Inference**: Uses vLLM for fast and efficient text generation via API.
*   **Configuration Driven**: Manage experiments via YAML files.
*   **Dockerized**: Ready for quick deployment.
*   **Logging**: Comprehensive logging for debugging and monitoring.

## Prerequisites

*   Python 3.8+
*   NVIDIA GPU with CUDA (for training and vLLM inference)
*   Docker (for containerized deployment)

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <your-repo-url>
    # cd llm-factory
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: vLLM installation can be tricky. Ensure your CUDA and PyTorch versions are compatible. Refer to the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html).*

## Configuration

All pipeline operations are driven by YAML configuration files located in the `configs/` directory. Examples:
*   `configs/sft_qwen_7b.yaml`
*   `configs/lora_llama3_8b.yaml`
*   `configs/rlhf_config.yaml`
*   `configs/inference_config.yaml`

Modify these or create new ones to define your model, data, training parameters, etc.

**Important:** You will need to provide your own datasets and ensure paths in the config files point to them. For models, specify Hugging Face model identifiers or local paths.

## Usage

### 1. Training

Use the `run_train.sh` script or directly call `main.py`.

**Using script:**
```bash
bash scripts/run_train.sh configs/your_training_config.yaml
```

**Directly:**
```bash
export PYTHONPATH=$(pwd) # Ensure llm_factory is in PYTHONPATH
python -m llm_factory.main train --config configs/your_training_config.yaml
```

### 2. Evaluation

After training, you can evaluate your model. Ensure your config file has an `evaluation` section pointing to the trained model and evaluation dataset.

```bash
export PYTHONPATH=$(pwd)
python -m llm_factory.main eval --config configs/your_evaluation_config.yaml
```
Results will be printed and optionally saved to a file specified in the config.

### 3. Inference API (with vLLM)

Start the vLLM-powered API server.

**Using script:**
```bash
# Set environment variables for the model path and tensor parallel size
export LLM_MODEL_PATH="/path/to/your/finetuned_model_or_hf_model" # e.g., ./output/lora_llama3_8b
export TP_SIZE=1 # Tensor Parallel size for vLLM

bash scripts/run_inference_api.sh configs/inference_config.yaml
```
The `MODEL_PATH` environment variable in `run_inference_api.sh` should point to the model you want to serve (either a base model or your fine-tuned one). For LoRA, you'll typically merge the adapter into the base model first or ensure vLLM can load adapters if supported for your specific version/setup.

**Directly:**
```bash
export PYTHONPATH=$(pwd)
export MODEL_PATH="/path/to/your/finetuned_model_or_hf_model"
export TP_SIZE=1
python -m llm_factory.main inference_api --config configs/inference_config.yaml --port 8000
```

**API Endpoint:**
Once running, the API will be available at `http://localhost:8000/generate`.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/generate" \
-H "Content-Type: application/json" \
-d '{
  "prompts": ["你好，请介绍一下你自己。", "What is the capital of France?"],
  "max_tokens": 128,
  "temperature": 0.7
}'
```

## Docker Deployment

1.  **Build the Docker image:**
    ```bash
    docker build -t llm-factory:latest .
    ```

2.  **Run the Docker container for inference:**
    ```bash
    docker run -d --gpus all \
        -v /path/to/your/models:/app/output \
        -e MODEL_PATH="/app/output/your_finetuned_model" \
        -e TP_SIZE=1 \
        -p 8000:8000 \
        --name llm-api \
        llm-factory:latest
    ```
    *   Replace `/path/to/your/models` with the actual path on your host machine where your models are stored.
    *   `MODEL_PATH` inside the container should point to the model directory accessible within the container (e.g., mounted via `-v`).

## Project Structure

(Include the tree structure provided earlier here)

## Extending the Factory

*   **New Data Loaders**: Add a new loader class in `llm_factory/data/` inheriting from `BaseDataLoader` and register it in `DataProcessor.LOADER_MAPPING`.
*   **New Training Techniques**: Add a new trainer class in `llm_factory/training/` inheriting from `BaseTrainer` and add it to the `TrainerFactory`.
*   **New Evaluation Metrics**: Extend `Evaluator` in `llm_factory/evaluation/` or integrate new tools.

## Logging

Log files are stored in the `logs/` directory by default (can be configured in YAML). Console output levels and formats are also configurable.

---
```