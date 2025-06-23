import json
import logging
from typing import List, Dict, Any
from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)

class JsonLoader(BaseDataLoader):
    """Loads data from JSON or JSONL files."""
    def load(self) -> List[Dict[str, Any]]:
        data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Try to load as a single JSON array first
                try:
                    content = f.read()
                    data = json.loads(content)
                    if isinstance(data, list):
                        logger.info(f"Loaded {len(data)} records from JSON file: {self.file_path}")
                        return data
                    else: # If it's a single JSON object, treat as JSONL attempt
                        logger.debug("Loaded a single JSON object, will attempt to parse as JSONL.")
                        data = [] # Reset data and fall through to JSONL parsing
                        f.seek(0) # Reset file pointer
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse {self.file_path} as a single JSON array. Attempting JSONL.")
                    f.seek(0) # Reset file pointer for line-by-line reading

                # If not a single JSON array or failed, try JSONL
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {self.file_path}: {line[:100]}...")
            logger.info(f"Loaded {len(data)} records from JSON/JSONL file: {self.file_path}")
        except FileNotFoundError:
            logger.error(f"JSON/JSONL file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON/JSONL file {self.file_path}: {e}", exc_info=True)
            raise
        return data