import logging
from typing import List, Dict, Any
import pandas as pd
from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)

class ParquetLoader(BaseDataLoader):
    """
    Loads data from a Parquet file.
    Assumes each row in the Parquet file represents a data sample
    and can be converted to a dictionary.
    """
    def __init__(self, file_path: str, columns: List[str] = None):
        super().__init__(file_path)
        self.columns = columns # Specific columns to load, if any

    def load(self) -> List[Dict[str, Any]]:
        data = []
        try:
            logger.info(f"Attempting to load Parquet file: {self.file_path}")
            df = pd.read_parquet(self.file_path, columns=self.columns)
            data = df.to_dict(orient='records')
            logger.info(f"Successfully loaded {len(data)} records from {self.file_path}.")
        except FileNotFoundError:
            logger.error(f"Parquet file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading Parquet file {self.file_path}: {e}", exc_info=True)
            raise
        return data