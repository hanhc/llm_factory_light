import logging
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from typing import Callable, Optional

from .json_loader import JsonLoader
from .parquet_loader import ParquetLoader

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Coordinates data loading, preprocessing, and tokenization.
    """
    LOADER_MAPPING = {
        "json": JsonLoader,
        "jsonl": JsonLoader,
        "parquet": ParquetLoader,
    }

    def __init__(self, data_config: dict):
        self.config = data_config
        self.tokenizer_name_or_path = self.config['tokenizer_name_or_path']
        self.use_fast_tokenizer = self.config.get('use_fast_tokenizer', True)

        logger.info(f"Initializing DataProcessor with tokenizer: {self.tokenizer_name_or_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                trust_remote_code=True,
                use_fast=self.use_fast_tokenizer
            )
            # Set pad token if not set, common practice for Causal LM
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info(f"Set tokenizer.pad_token_id to tokenizer.eos_token_id: {self.tokenizer.eos_token_id}")
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{self.tokenizer_name_or_path}'. Error: {e}", exc_info=True)
            raise

    def _get_loader(self):
        file_format = self.config.get('file_format', '').lower()
        file_path = self.config.get('file_path')  # For train/eval main file

        if not file_path:  # Could be for RL prompt dataset which has a different key
            file_path = self.config.get('prompt_dataset_path')

        if not file_path:
            logger.error("No file_path or prompt_dataset_path specified in data config.")
            raise ValueError("Data file path is missing in configuration.")

        loader_class = self.LOADER_MAPPING.get(file_format)
        if not loader_class:
            logger.error(f"Unsupported file format: {file_format}. Available: {list(self.LOADER_MAPPING.keys())}")
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Using loader '{loader_class.__name__}' for file '{file_path}'.")
        return loader_class(file_path)

    def _default_formatting_function(self, example: dict) -> dict:
        """
        Default formatting for SFT/LoRA assuming 'prompt' and 'response' fields.
        Produces a 'text' field.
        """
        prompt = example.get('prompt', '')
        response = example.get('response', '')

        # This template can be made configurable
        # E.g., read from self.config.get('chat_template')
        # For Llama3-Instruct, a more specific template might be better if not using HF's chat template system.
        # text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        text = f"### Human: {prompt}\n### Assistant: {response}"  # Generic template
        return {"text": text}

    def process(self, formatting_func: Optional[Callable[[dict], dict]] = None) -> Dataset:
        """
        Loads, formats (if needed), and tokenizes data for supervised fine-tuning.
        The SFTTrainer typically expects a 'text' field in the dataset.
        """
        logger.info("Starting data processing for supervised fine-tuning...")
        loader = self._get_loader()
        try:
            raw_data = loader.load()  # Expects List[Dict]
            logger.info(f"Loaded {len(raw_data)} raw data entries.")
        except Exception as e:
            logger.error(f"Error loading data from '{loader.file_path}': {e}", exc_info=True)
            raise

        if not raw_data:
            logger.warning("Raw data is empty. Returning an empty dataset.")
            return Dataset.from_list([])

        # Determine formatting function
        # Users can provide their own formatting_func_name in config to load dynamically,
        # or SFTTrainer can take formatting_func. Here, we use a default or passed one.
        _formatting_func = formatting_func if formatting_func else self._default_formatting_function

        dataset_text_field = self.config.get('dataset_text_field', 'text')

        # Apply formatting if the expected text field is not present or custom formatting is given
        formatted_data = []
        if dataset_text_field not in raw_data[0] or formatting_func:
            logger.info(f"Applying formatting function '{_formatting_func.__name__}' to create '{dataset_text_field}'.")
            for item in raw_data:
                formatted_item = _formatting_func(item)
                # Ensure the output field matches dataset_text_field if it's not 'text'
                if dataset_text_field != 'text' and 'text' in formatted_item:
                    formatted_item[dataset_text_field] = formatted_item.pop('text')
                formatted_data.append(formatted_item)
        else:
            logger.info(f"Using existing field '{dataset_text_field}' from raw data.")
            formatted_data = raw_data

        if not formatted_data or dataset_text_field not in formatted_data[0]:
            logger.error(f"After formatting, field '{dataset_text_field}' is still missing or data is empty.")
            raise ValueError(f"Data processing failed: '{dataset_text_field}' not found after formatting.")

        logger.debug(f"Sample formatted entry (first one): {formatted_data[0]}")

        # Convert to Hugging Face Dataset
        # Note: Tokenization is often handled by the Trainer (e.g. SFTTrainer, Trainer)
        # to take advantage of its packing, max_seq_length, etc.
        # However, returning a Dataset object is standard.
        # SFTTrainer will tokenize if dataset_text_field is provided.
        hf_dataset = Dataset.from_list(formatted_data)

        logger.info(f"Data processing complete. Created HF Dataset with {len(hf_dataset)} entries.")
        return hf_dataset

    def process_for_rl(self) -> Dataset:
        """Loads and tokenizes prompts for RL training."""
        logger.info("Starting data processing for RL prompts...")
        loader = self._get_loader()  # _get_loader will use prompt_dataset_path if file_path is not set
        try:
            raw_prompts_data = loader.load()  # Expects list of dicts, e.g., [{"prompt": "text"}, ...]
            logger.info(f"Loaded {len(raw_prompts_data)} raw prompts for RL.")
        except Exception as e:
            logger.error(f"Error loading RL prompts from '{loader.file_path}': {e}", exc_info=True)
            raise

        if not raw_prompts_data:
            logger.warning("No raw prompts data loaded for RL. Returning empty dataset.")
            return Dataset.from_list([])

        max_prompt_length = self.config.get('max_prompt_length', 128)  # Max length for tokenized prompts

        def tokenize_prompts(example):
            if "prompt" not in example:
                logger.warning(f"RL prompt data missing 'prompt' field: {example}")
                return {"input_ids": [], "attention_mask": [], "raw_prompt": ""}

            prompt_text = str(example["prompt"])  # Ensure it's a string
            tokenized = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_prompt_length,
                # PPOTrainer typically expects input_ids, not necessarily attention_mask in the batch for query_tensor
                # padding=True # Or handle padding in collator if PPOTrainer requires it.
                # For PPOTrainer, queries are usually lists of tensors, not padded batches.
            )
            return {"input_ids": tokenized["input_ids"],
                    # "attention_mask": tokenized["attention_mask"], # PPOTrainer might not need this for queries
                    "query": prompt_text  # TRL PPOTrainer often logs the original query text
                    }

        processed_prompts = [tokenize_prompts(item) for item in raw_prompts_data]

        # Filter out any entries that failed tokenization or were invalid
        processed_prompts = [p for p in processed_prompts if p["input_ids"]]
        if not processed_prompts:
            logger.warning("All RL prompts were filtered out or failed tokenization.")
            return Dataset.from_list([])

        logger.debug(f"Sample tokenized RL prompt (first one): {processed_prompts[0]}")

        hf_dataset = Dataset.from_list(processed_prompts)
        logger.info(f"RL prompt processing complete. Created HF Dataset with {len(hf_dataset)} entries.")
        return hf_dataset