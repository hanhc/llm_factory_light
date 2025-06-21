# 使用包含 CUDA 和 PyTorch 的官方镜像
# vLLM 对 CUDA 和 PyTorch 版本有严格要求，请根据 vLLM 官网选择合适的基础镜像
FROM nvcr.io/nvidia/pytorch:24.03-py3

# 设置工作目录
WORKDIR /app

# 安装项目依赖
# 拷贝 requirements.txt 并安装，以利用 Docker 缓存
COPY requirements.txt .

# vLLM 安装可能需要特殊处理，确保与 PyTorch 和 CUDA 版本兼容
# 建议使用 vLLM 官方提供的 wheel 或从源码编译
RUN pip install -r requirements.txt
# 例如: pip install vllm==0.4.0 --no-build-isolation

# 拷贝整个项目代码
COPY . .

# 设置 PYTHONPATH
ENV PYTHONPATH=/app

# 暴露 API 端口
EXPOSE 8000

# 默认启动命令 (可以被 docker run 命令覆盖)
# 启动推理服务，模型路径通过环境变量传入
CMD ["sh", "-c", "export MODEL_PATH=${MODEL_PATH:-/app/output/my-model} && python -m llm_factory.main inference_api --config /app/configs/inference_config.yaml"]