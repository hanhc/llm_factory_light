from transformers import AutoTokenizer
from datasets import Dataset
from .json_loader import JsonLoader
# from .parquet_loader import ParquetLoader # 未来可扩展

class DataProcessor:
    """
    协调数据加载和分词。
    """
    LOADER_MAPPING = {
        "json": JsonLoader,
        "jsonl": JsonLoader,
        # "parquet": ParquetLoader,
    }

    def __init__(self, data_config: dict):
        self.config = data_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['tokenizer_name_or_path'],
            trust_remote_code=True,
            use_fast=False
        )

    def _get_loader(self):
        file_format = self.config['file_format']
        file_path = self.config['file_path']
        loader_class = self.LOADER_MAPPING.get(file_format)
        if not loader_class:
            raise ValueError(f"Unsupported file format: {file_format}")
        return loader_class(file_path)

    def process(self) -> Dataset:
        """加载、分词并返回 Hugging Face Dataset 对象。"""
        loader = self._get_loader()
        raw_data = loader.load() # [{ "prompt": "...", "response": "..."}, ...]

        # 模板化和分词
        def formatting_function(example):
            # 这里的模板可以从config中读取，实现灵活配置
            text = f"### Human: {example['prompt']}\n### Assistant: {example['response']}"
            return {"text": text}

        processed_data = [formatting_function(item) for item in raw_data]
        return Dataset.from_list(processed_data)