import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from evaluate import load as load_metric  # Hugging Face evaluate library
import os

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles model evaluation on specified metrics and datasets."""

    def __init__(self, config: dict):
        self.config = config
        self.eval_config = config.get('evaluation')
        if not self.eval_config:
            logger.error("Missing 'evaluation' section in the configuration for Evaluator.")
            raise ValueError("Evaluator requires an 'evaluation' section in the config.")

        # Model and tokenizer paths can be from eval_config or main model/data config
        self.model_path = self.eval_config.get('model_to_evaluate_path', config.get('model', {}).get('name_or_path'))
        self.tokenizer_path = self.eval_config.get('tokenizer_path',
                                                   config.get('data', {}).get('tokenizer_name_or_path',
                                                                              self.model_path))

        if not self.model_path:
            logger.error("No model path specified for evaluation.")
            raise ValueError("Model path for evaluation is required.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Evaluator initialized. Using device: {self.device}. Model: {self.model_path}")

        self._load_model_and_tokenizer()
        self._load_evaluation_dataset()  # Loads self.eval_dataset and self.eval_dataset_tokenized
        self._load_metrics_to_compute()  # Loads self.metrics_to_compute

    def _load_model_and_tokenizer(self):
        logger.info(f"Loading model for evaluation from: {self.model_path}")
        # Add quantization config if specified in eval_config or main model config
        quantization_cfg_eval = self.eval_config.get('quantization_config',
                                                     self.config.get('model', {}).get('quantization_config'))
        model_load_kwargs = {"trust_remote_code": True}

        if quantization_cfg_eval:
            from transformers import BitsAndBytesConfig
            if 'bnb_4bit_compute_dtype' in quantization_cfg_eval and isinstance(
                    quantization_cfg_eval['bnb_4bit_compute_dtype'], str):
                try:
                    quantization_cfg_eval['bnb_4bit_compute_dtype'] = getattr(torch, quantization_cfg_eval[
                        'bnb_4bit_compute_dtype'].split('.')[-1])
                except AttributeError:
                    logger.error(
                        f"Invalid torch dtype string for eval quantization: {quantization_cfg_eval['bnb_4bit_compute_dtype']}")
                    raise
            bnb_config = BitsAndBytesConfig(**quantization_cfg_eval)
            model_load_kwargs["quantization_config"] = bnb_config
            logger.info(f"Applying quantization config for evaluation model: {quantization_cfg_eval}")

        # If model_path is a PEFT adapter, ensure base model is also implicitly handled or specified
        # For simplicity, assume model_path is a fully merged model or HF can handle adapters.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_load_kwargs
        ).to(self.device)
        self.model.eval()

        logger.info(f"Loading tokenizer for evaluation from: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
            use_fast=self.config.get('data', {}).get('use_fast_tokenizer', True)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Evaluator: Set tokenizer.pad_token_id to eos_token_id: {self.tokenizer.eos_token_id}")

    def _load_evaluation_dataset(self):
        dataset_identifier = self.eval_config.get('dataset_path')  # Can be HF ID or local path
        dataset_name_hf = self.eval_config.get('dataset_name')  # For HF datasets with sub-configs
        dataset_split = self.eval_config.get('dataset_split', 'test')
        self.text_column = self.eval_config.get('text_column', 'text')  # Column for perplexity

        if not dataset_identifier:
            logger.warning("No evaluation 'dataset_path' specified. Some evaluations might not be possible.")
            self.eval_dataset = None
            self.eval_dataset_tokenized = None
            return

        logger.info(
            f"Loading evaluation dataset: {dataset_identifier} (name: {dataset_name_hf}, split: {dataset_split})")
        try:
            # Check if it's a local file path
            if os.path.exists(dataset_identifier):
                file_ext = os.path.splitext(dataset_identifier)[1].lower()
                if file_ext == '.jsonl':
                    self.eval_dataset = load_dataset("json", data_files={dataset_split: dataset_identifier},
                                                     split=dataset_split)
                elif file_ext == '.json':
                    self.eval_dataset = load_dataset("json", data_files={dataset_split: dataset_identifier},
                                                     split=dataset_split)
                elif file_ext == '.txt':
                    self.eval_dataset = load_dataset("text", data_files={dataset_split: dataset_identifier},
                                                     split=dataset_split)
                else:
                    raise ValueError(f"Unsupported local file type for evaluation: {file_ext}")
            else:  # Assume it's a Hugging Face dataset identifier
                self.eval_dataset = load_dataset(dataset_identifier, name=dataset_name_hf, split=dataset_split)

            logger.info(f"Loaded evaluation dataset with {len(self.eval_dataset)} samples.")

            # Pre-tokenize for perplexity if perplexity is a target metric
            if 'perplexity' in self.eval_config.get('metrics', []):
                self._tokenize_for_perplexity()

        except Exception as e:
            logger.error(f"Failed to load or process evaluation dataset '{dataset_identifier}': {e}", exc_info=True)
            self.eval_dataset = None
            self.eval_dataset_tokenized = None
            # raise # Optionally re-raise

    def _tokenize_for_perplexity(self):
        if not self.eval_dataset or self.text_column not in self.eval_dataset.column_names:
            logger.warning(
                f"Cannot tokenize for perplexity: dataset not loaded or text_column '{self.text_column}' missing.")
            self.eval_dataset_tokenized = None
            return

        logger.info(f"Tokenizing evaluation dataset for perplexity using text column '{self.text_column}'...")

        def tokenize_function(examples):
            # No truncation for perplexity, model should process full sequences.
            return self.tokenizer(examples[self.text_column], truncation=False)

            # Remove other columns to keep only tokenized fields

        cols_to_remove = [col for col in self.eval_dataset.column_names if col != self.text_column]

        self.eval_dataset_tokenized = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=cols_to_remove  # Keep only input_ids, attention_mask
        )
        logger.info("Evaluation dataset tokenized for perplexity.")

    def _load_metrics_to_compute(self):
        self.metrics_to_compute = {}
        metric_names = self.eval_config.get('metrics', [])
        for name in metric_names:
            if name.lower() == "perplexity":  # Perplexity is handled customly
                continue
            try:
                self.metrics_to_compute[name] = load_metric(name)
                logger.info(f"Loaded metric '{name}' for evaluation.")
            except Exception as e:
                logger.warning(f"Could not load metric '{name}': {e}. It will be skipped.")

    def compute_perplexity(self) -> Optional[float]:
        if not hasattr(self, 'eval_dataset_tokenized') or self.eval_dataset_tokenized is None:
            logger.warning("Tokenized evaluation dataset not available. Skipping perplexity calculation.")
            return None

        logger.info("Calculating perplexity...")
        # max_length = self.model.config.max_position_embeddings # Or tokenizer.model_max_length
        # Using tokenizer's model_max_length if available, else model's config.
        # However, vLLM or other inference engines might have their own limits.
        # For perplexity, we often use a sliding window approach with the model's actual context limit.
        max_length = self.tokenizer.model_max_length
        if max_length > 2048 and self.device == torch.device("cpu"):  # Cap for CPU to avoid OOM for very large contexts
            logger.warning(f"Reducing max_length for perplexity on CPU from {max_length} to 2048 to avoid OOM.")
            max_length = 2048

        stride = self.eval_config.get('perplexity_stride', max_length // 2)  # Default stride to half max_length
        eval_batch_size = self.eval_config.get('batch_size', 1)

        nlls = []  # Negative log-likelihoods
        num_tokens_processed = 0

        for i in tqdm(range(0, len(self.eval_dataset_tokenized), eval_batch_size), desc="Perplexity Batches"):
            batch_data = self.eval_dataset_tokenized[i: i + eval_batch_size]
            # batch_data is dict {'input_ids': [[...], [...]], 'attention_mask': [[...], [...]]}

            for single_sequence_input_ids in batch_data['input_ids']:
                if not single_sequence_input_ids: continue  # Skip empty sequences

                for begin_loc in range(0, len(single_sequence_input_ids), stride):
                    end_loc = min(begin_loc + max_length, len(single_sequence_input_ids))

                    # input_ids for the current window
                    current_input_ids = torch.tensor([single_sequence_input_ids[begin_loc:end_loc]], device=self.device)

                    if current_input_ids.size(1) < 2:  # Need at least 2 tokens for label
                        continue

                    # Target IDs are same as input_ids, loss calculated only on non-context part
                    target_ids = current_input_ids.clone()

                    # For causal LMs, labels are shifted input_ids.
                    # Transformers' CausalLM models handle shifting internally if labels are provided.
                    # If begin_loc > 0, the first part is context.
                    # We want to calculate loss on tokens that are "new" in this window.
                    # A common approach: calculate loss on the entire window.

                    with torch.no_grad():
                        outputs = self.model(current_input_ids, labels=target_ids)
                        neg_log_likelihood = outputs.loss * current_input_ids.size(
                            1)  # Loss is mean, scale by num tokens in window

                    nlls.append(neg_log_likelihood.to('cpu'))  # Move to CPU to save GPU memory
                    num_tokens_processed += current_input_ids.size(1)

                    if end_loc == len(single_sequence_input_ids):
                        break

        if not nlls or num_tokens_processed == 0:
            logger.warning("No NLLs calculated or no tokens processed, perplexity cannot be computed.")
            return None

        total_nll = torch.stack(nlls).sum().item()
        perplexity = torch.exp(torch.tensor(total_nll / num_tokens_processed)).item()
        logger.info(
            f"Calculated Perplexity: {perplexity:.4f} (Total NLL: {total_nll:.2f}, Tokens: {num_tokens_processed})")
        return perplexity

    def run(self) -> dict:
        logger.info("Starting model evaluation run...")
        results = {}

        if 'perplexity' in self.eval_config.get('metrics', []):
            ppl = self.compute_perplexity()
            if ppl is not None:
                results['perplexity'] = ppl

        # Example for other metrics (e.g., ROUGE, BLEU for generation tasks)
        # This would require generating text from prompts in the eval dataset.
        # if 'rouge' in self.metrics_to_compute:
        #     logger.info("Computing ROUGE score...")
        #     # 1. Get prompts from eval_dataset (e.g., a 'prompt' column)
        #     # 2. Generate predictions using self.model.generate()
        #     # 3. Get references (e.g., a 'reference_response' column)
        #     # predictions = [...]
        #     # references = [...]
        #     # rouge_scores = self.metrics_to_compute['rouge'].compute(predictions=predictions, references=references)
        #     # results['rouge'] = rouge_scores
        #     logger.warning("ROUGE (and other generation metrics) computation is not fully implemented yet.")

        logger.info(f"Evaluation finished. Results: {results}")
        return results