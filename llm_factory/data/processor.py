import logging  # Add this line
from transformers import AutoTokenizer
from datasets import Dataset
from .json_loader import JsonLoader
from .parquet_loader import ParquetLoader


logger = logging.getLogger(__name__)  # Get a logger specific to this module


class DataProcessor:
    LOADER_MAPPING = {
        "json": JsonLoader,
        "jsonl": JsonLoader,
        "parquet": ParquetLoader,
    }

    def __init__(self, data_config: dict):
        self.config = data_config
        logger.info(f"Initializing DataProcessor with config: {data_config}")
        logger.debug(f"Attempting to load tokenizer: {self.config['tokenizer_name_or_path']}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['tokenizer_name_or_path'],
                trust_remote_code=True,
                use_fast=self.config.get('use_fast_tokenizer', False)  # Example of using config for options
            )
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{self.config['tokenizer_name_or_path']}'. Error: {e}",
                         exc_info=True)
            raise

    def _get_loader(self):
        file_format = self.config.get('file_format', '').lower()
        file_path = self.config['file_path']
        logger.debug(f"Attempting to find loader for format '{file_format}' and path '{file_path}'.")

        loader_class = self.LOADER_MAPPING.get(file_format)
        if not loader_class:
            logger.error(
                f"Unsupported file format: {file_format}. Available formats: {list(self.LOADER_MAPPING.keys())}")
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Using loader '{loader_class.__name__}' for file '{file_path}'.")
        return loader_class(file_path)

    def process(self) -> Dataset:
        logger.info("Starting data processing pipeline...")
        loader = self._get_loader()

        try:
            raw_data = loader.load()
            logger.info(f"Loaded {len(raw_data)} raw data entries.")
        except Exception as e:
            logger.error(f"Error loading data from '{self.config['file_path']}': {e}", exc_info=True)
            raise

        # Example: Log a sample of raw data at debug level
        if raw_data:
            logger.debug(f"Sample raw data entry: {raw_data[0]}")

        # ... (rest of the processing logic)
        # Add more logging as needed
        # processed_data = ...
        # logger.info(f"Formatted {len(processed_data)} entries.")
        # ...

        # Example of final dataset creation
        # dataset = Dataset.from_list(processed_data)
        # logger.info(f"Data processing complete. Final dataset size: {len(dataset)} entries.")
        # return dataset
        # Placeholder for actual data processing
        if not raw_data:  # If raw_data is empty or loading failed
            logger.warning("Raw data is empty, returning an empty dataset.")
            return Dataset.from_list([])

        def formatting_function(example):
            text = f"### Human: {example.get('prompt', '')}\n### Assistant: {example.get('response', '')}"
            return {"text": text}

        processed_data = [formatting_function(item) for item in raw_data]
        logger.debug(f"Formatted {len(processed_data)} entries.")
        dataset = Dataset.from_list(processed_data)
        logger.info(f"Data processing complete. Created dataset with {len(dataset)} entries.")
        return dataset

    # In main.py, when mode is 'train' and method is 'rlhf':
    # train_dataset = data_processor.process_for_rl() # Instead of data_processor.process()
    def process_for_rl(self) -> Dataset:
        """Loads and tokenizes prompts for RL training."""
        loader = self._get_loader()
        raw_prompts_data = loader.load()  # Expects list of dicts, e.g., [{"prompt": "text"}, ...]

        if not raw_prompts_data:
            logger.warning("No raw prompts data loaded for RL. Returning empty dataset.")
            return Dataset.from_list([])

        def tokenize_prompts(example):
            if "prompt" not in example:
                logger.warning(f"RL prompt data missing 'prompt' field: {example}")
                return {"input_ids": [], "attention_mask": [], "raw_prompt": ""}

            prompt_text = example["prompt"]
            tokenized = self.tokenizer(prompt_text, truncation=True,
                                       max_length=self.config.get('max_prompt_length', 512))
            return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"],
                    "raw_prompt": prompt_text}

        # Check if the data is already in the Hugging Face Dataset format, or list of dicts.
        # Assuming loader.load() returns List[Dict]
        processed_prompts = [tokenize_prompts(item) for item in raw_prompts_data]
        return Dataset.from_list(processed_prompts)
    # In main.py, when mode is 'train' and method is 'rlhf':
    # train_dataset = data_processor.process_for_rl() # Instead of data_processor.process()