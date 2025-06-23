# llm_factory/inference/__init__.py
from .vllm_engine import VLLMEngine
from .api_server import app as fastapi_app # Expose the FastAPI app