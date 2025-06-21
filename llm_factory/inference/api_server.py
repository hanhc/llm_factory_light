from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .vllm_engine import VLLMEngine
import uvicorn
import os

app = FastAPI()

# 从环境变量或配置文件中获取模型信息
MODEL_PATH = os.environ.get("MODEL_PATH", "path/to/your/finetuned/model")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))

# 初始化推理引擎
engine = VLLMEngine(model_path=MODEL_PATH, tensor_parallel_size=TP_SIZE)


class GenerationRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@app.post("/generate")
async def generate(request: GenerationRequest):
    return engine.generate(
        prompts=request.prompts,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )


def start_server(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)
