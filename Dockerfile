# Use a vLLM compatible base image or an official PyTorch image with correct CUDA
# Example: Using a PyTorch base image. You might need to adjust for vLLM compatibility.
# Check vLLM documentation for recommended base images.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# FROM nvcr.io/nvidia/pytorch:24.03-py3 # Alternative if it suits vLLM

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential build tools and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Consider using a specific vLLM wheel if direct pip install causes issues
RUN pip install --no-cache-dir -r requirements.txt
# Example for a specific vLLM version if needed:
# RUN pip install vllm==0.4.0 --no-build-isolation

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH so llm_factory module can be found
ENV PYTHONPATH=/app

# Create log directory and set permissions (if not running as root, adjust user/group)
# The application's logging_setup.py will also try to create this.
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Expose the API port
EXPOSE 8000

# Default command to run the inference API server
# MODEL_PATH and TP_SIZE should be passed as environment variables during `docker run`
# The default in CMD is a fallback if not provided.
CMD ["sh", "-c", "export MODEL_PATH=${MODEL_PATH:-/app/output/default_model} && \
                  export TP_SIZE=${TP_SIZE:-1} && \
                  python -m llm_factory.main inference_api --config /app/configs/inference_config.yaml --port 8000"]