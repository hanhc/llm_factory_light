from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Each specific data loader must implement the `load` method.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Loads data from the specified file path and returns a list of dictionaries."""
        pass