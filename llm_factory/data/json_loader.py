import json
from typing import List, Dict, Any
from .base_loader import BaseDataLoader


class JsonLoader(BaseDataLoader):
    """加载 JSON或JSONL 文件。"""
    def load(self) -> List[Dict[str, Any]]:
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {self.file_path}")
        return data
