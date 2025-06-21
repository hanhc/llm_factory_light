from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDataLoader(ABC):
    """
    数据加载器的抽象基类。
    每个具体的数据加载器都必须实现 load 方法。
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """从文件路径加载数据并返回一个字典列表。"""
        pass
    