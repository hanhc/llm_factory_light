from vllm import LLM, SamplingParams
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class VLLMEngine:
    """
    Encapsulates vLLM's high-performance inference capabilities.
    Designed as a singleton to ensure the model is loaded only once.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VLLMEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None, tensor_parallel_size: int = 1, **vllm_kwargs):
        """
        Initializes the vLLM engine.

        Args:
            model_path (str, optional): Path or HuggingFace ID of the model.
                                        If None, reads from MODEL_PATH env var.
            tensor_parallel_size (int, optional): Degree of tensor parallelism.
                                                 If None, reads from TP_SIZE env var.
            vllm_kwargs: Additional keyword arguments to pass to vLLM `LLM(...)`.
                         e.g., dtype="auto", max_model_len=2048, gpu_memory_utilization=0.9
        """
        if self._initialized:
            return

        resolved_model_path = model_path or os.environ.get("MODEL_PATH")
        resolved_tp_size_str = os.environ.get("TP_SIZE")

        if resolved_tp_size_str is not None:
            try:
                resolved_tp_size = int(resolved_tp_size_str)
            except ValueError:
                logger.warning(f"Invalid TP_SIZE environment variable '{resolved_tp_size_str}'. Defaulting to 1.")
                resolved_tp_size = 1
        else:
            resolved_tp_size = tensor_parallel_size

        if not resolved_model_path:
            logger.error("VLLMEngine: Model path not provided and MODEL_PATH environment variable is not set.")
            raise ValueError("Model path is required for VLLM engine.")

        logger.info(f"Initializing vLLM engine for model: {resolved_model_path}")
        logger.info(f"Tensor Parallel Size: {resolved_tp_size}")
        if vllm_kwargs:
            logger.info(f"Additional vLLM LLM kwargs: {vllm_kwargs}")

        try:
            self.llm = LLM(
                model=resolved_model_path,
                tensor_parallel_size=resolved_tp_size,
                trust_remote_code=True,  # Often needed for custom models
                **vllm_kwargs
            )
            # self.tokenizer = self.llm.get_tokenizer() # Not strictly needed if using llm.generate
            logger.info("vLLM engine initialized successfully.")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}", exc_info=True)
            # This is a critical failure, re-raise or handle appropriately
            raise RuntimeError(f"vLLM engine initialization failed: {e}")

    def generate(self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, **kwargs) -> List[
        Dict[str, Any]]:
        """
        Generates text completions for a list of prompts.

        Args:
            prompts (List[str]): A list of prompt strings.
            max_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling p value.
            **kwargs: Additional sampling parameters for vLLM (e.g., top_k, stop).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing
                                  'prompt' and 'generated_text'.
        """
        if not self._initialized or not hasattr(self, 'llm'):
            logger.error("vLLM engine is not initialized. Cannot generate.")
            # Depending on desired behavior, could try to re-initialize or raise error
            raise RuntimeError("vLLM engine not initialized.")

        sampling_params_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs  # Allow overriding or adding other vLLM sampling params
        }

        # Filter out None values, vLLM might not like them for some params
        sampling_params_dict = {k: v for k, v in sampling_params_dict.items() if v is not None}

        sampling_params = SamplingParams(**sampling_params_dict)
        logger.debug(f"Generating with prompts: {prompts}, SamplingParams: {sampling_params_dict}")

        try:
            # vLLM's generate method is batch-oriented
            outputs = self.llm.generate(prompts, sampling_params)
        except Exception as e:
            logger.error(f"Error during vLLM generation: {e}", exc_info=True)
            # Return empty or error structure based on API contract
            return [{"prompt": p, "generated_text": f"Error: {e}"} for p in prompts]

        results = []
        for output in outputs:
            prompt_text = output.prompt
            # Assuming single output per prompt for simplicity
            generated_text = output.outputs[0].text
            results.append({"prompt": prompt_text, "generated_text": generated_text})
            logger.debug(f"Prompt: '{prompt_text[:100]}...' Generated: '{generated_text[:100]}...'")

        return results