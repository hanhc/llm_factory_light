import pandas as pd
# Alternatively, use pyarrow directly if you prefer:
# import pyarrow.parquet as pq
from typing import List, Dict, Any
import logging
from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class ParquetLoader(BaseDataLoader):
    """
    Loads data from a Parquet file.
    Assumes each row in the Parquet file represents a data sample
    and can be converted to a dictionary.
    """

    def __init__(self, file_path: str, columns: List[str] = None):
        """
        Initializes the ParquetLoader.

        Args:
            file_path (str): Path to the Parquet file.
            columns (List[str], optional): Specific columns to load.
                                           If None, all columns are loaded.
        """
        super().__init__(file_path)
        self.columns = columns

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads data from the Parquet file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a row from the Parquet file.
        """
        data = []
        try:
            logger.info(f"Attempting to load Parquet file: {self.file_path}")
            # Using pandas for simplicity, it handles various Parquet aspects well.
            df = pd.read_parquet(self.file_path, columns=self.columns)

            # Convert DataFrame to a list of dictionaries
            data = df.to_dict(orient='records')
            logger.info(f"Successfully loaded {len(data)} records from {self.file_path}.")

        except FileNotFoundError:
            logger.error(f"Parquet file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading Parquet file {self.file_path}: {e}", exc_info=True)
            raise
        return data