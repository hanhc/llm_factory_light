from vllm import LLM, SamplingParams
from typing import List


class VLLMEngine:
    """
    封装 vLLM 的推理功能。
    这是一个单例模式的理想候选，确保模型只被加载一次。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VLLMEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        # 防止重复初始化
        if hasattr(self, 'llm'):
            return
        print(f"Initializing vLLM engine for model: {model_path}")
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = self.llm.get_tokenizer()
        print("vLLM engine initialized.")

    def generate(self, prompts: List[str], max_tokens: int, temperature: float, top_p: float):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        # vLLM 的 generate 方法是批处理的
        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            results.append({"prompt": prompt, "generated_text": generated_text})

        return results
