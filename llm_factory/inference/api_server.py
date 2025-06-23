from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import uvicorn
import os

from .vllm_engine import VLLMEngine  # Singleton vLLM engine

logger = logging.getLogger(__name__)
app = FastAPI(title="LLM Factory Inference API", version="0.1.0")

# Global variable for the engine, initialized when server starts
vllm_engine_instance: Optional[VLLMEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initializes the VLLMEngine when the FastAPI server starts."""
    global vllm_engine_instance
    model_path = os.environ.get("MODEL_PATH")
    tp_size_str = os.environ.get("TP_SIZE", "1")

    if not model_path:
        logger.error("MODEL_PATH environment variable not set. Cannot start API server.")
        # This will prevent server from starting if raised during startup.
        # Alternatively, handle this more gracefully, but vLLM needs a model.
        raise RuntimeError("MODEL_PATH environment variable is required to start the vLLM engine.")

    try:
        tp_size = int(tp_size_str)
    except ValueError:
        logger.warning(f"Invalid TP_SIZE '{tp_size_str}', defaulting to 1.")
        tp_size = 1

    # Additional vLLM kwargs can be passed from env or config file if needed
    # For example, from a dedicated section in inference_config.yaml
    vllm_extra_args = {}  # Populate this from config if desired
    # Example: vllm_extra_args = config.get('vllm_engine', {}) in main.py, then pass here

    logger.info("FastAPI server starting up. Initializing VLLMEngine...")
    try:
        vllm_engine_instance = VLLMEngine(
            model_path=model_path,
            tensor_parallel_size=tp_size,
            **vllm_extra_args
        )
        logger.info("VLLMEngine initialized successfully for API server.")
    except Exception as e:
        logger.critical(f"Failed to initialize VLLMEngine during server startup: {e}", exc_info=True)
        # Depending on severity, might want to exit or prevent server from fully starting.
        # FastAPI might handle this by not starting if startup event fails hard.
        raise  # Re-raise to signal critical failure


class GenerationRequest(BaseModel):
    prompts: List[str] = Field(..., description="A list of prompts to generate completions for.")
    max_tokens: int = Field(default=256, gt=0, description="Maximum number of new tokens to generate per prompt.")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0,
                                         description="Sampling temperature. 0 means greedy.")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling 'top_p' parameter.")
    top_k: Optional[int] = Field(default=None, ge=-1, description="Top-k sampling. -1 means disabled.")
    stop: Optional[List[str]] = Field(default=None, description="List of strings to stop generation at.")
    # Add other vLLM sampling parameters as needed
    # e.g., presence_penalty, frequency_penalty


class GenerationResponseItem(BaseModel):
    prompt: str
    generated_text: str


class GenerationResponse(BaseModel):
    results: List[GenerationResponseItem]


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    Generates text completions for the given prompts using the vLLM engine.
    """
    global vllm_engine_instance
    if vllm_engine_instance is None or not hasattr(vllm_engine_instance, 'llm'):
        logger.error("Attempted to generate text but VLLMEngine is not available.")
        raise HTTPException(status_code=503, detail="VLLM engine is not initialized or not ready.")

    logger.info(f"Received generation request for {len(request.prompts)} prompts.")
    logger.debug(f"Generation request details: {request.dict(exclude_none=True)}")

    try:
        # Pass through relevant sampling parameters
        additional_sampling_params = {}
        if request.top_k is not None:
            additional_sampling_params['top_k'] = request.top_k
        if request.stop is not None:
            additional_sampling_params['stop'] = request.stop

        outputs = vllm_engine_instance.generate(
            prompts=request.prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature is not None else 0.0,
            # vLLM requires temperature >= 0
            top_p=request.top_p if request.top_p is not None else 1.0,  # vLLM requires top_p <= 1.0
            **additional_sampling_params
        )
        return GenerationResponse(results=outputs)
    except Exception as e:
        logger.error(f"Error during text generation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during generation: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    global vllm_engine_instance
    if vllm_engine_instance and hasattr(vllm_engine_instance, 'llm'):
        return {"status": "ok", "message": "VLLM engine appears to be initialized."}
    return {"status": "degraded", "message": "VLLM engine not initialized or unavailable."}


def start_server(host: str = "0.0.0.0", port: int = 8000, uvicorn_log_level: str = "info"):
    """
    Starts the FastAPI server with Uvicorn.
    """
    logger.info(f"Uvicorn server starting on http://{host}:{port}")
    uvicorn.run(
        app,  # Can also be "llm_factory.inference.api_server:app" as string
        host=host,
        port=port,
        log_level=uvicorn_log_level.lower()  # Uvicorn's own log level
    )

# Example for direct execution (though typically main.py handles this)
# if __name__ == "__main__":
#     # This part is more for testing the API server directly.
#     # In the project, `main.py` calls `start_server`.
#     os.environ["MODEL_PATH"] = "facebook/opt-125m" # Example small model for testing
#     os.environ["TP_SIZE"] = "1"
#     # Setup basic logging if run directly for testing
#     logging.basicConfig(level=logging.INFO)
#     start_server()