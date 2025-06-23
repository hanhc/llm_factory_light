import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset  # For loading evaluation data
from ..data.processor import DataProcessor  # If using same data processing
from evaluate import load as load_metric  # Hugging Face evaluate library

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Handles model evaluation.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary, expected to have 'evaluation' section.
                           And 'model' section for model_path, 'data' for tokenizer.
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})
        self.model_config = config['model']
        self.data_config = config['data']  # For tokenizer path primarily

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Evaluator initialized. Using device: {self.device}")

        self._load_model_and_tokenizer()
        self._load_evaluation_dataset()
        self._load_metrics()

    def _load_model_and_tokenizer(self):
        model_path = self.eval_config.get('model_to_evaluate_path', self.model_config.get('name_or_path'))
        if not model_path:
            logger.error(
                "No model path specified for evaluation in 'evaluation.model_to_evaluate_path' or 'model.name_or_path'.")
            raise ValueError("Model path for evaluation is required.")

        logger.info(f"Loading model for evaluation from: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            # Add quantization, etc. if needed and consistent with how model was saved/trained
            # torch_dtype=torch.bfloat16 # Example
        ).to(self.device)
        self.model.eval()  # Set model to evaluation mode

        tokenizer_path = self.data_config.get('tokenizer_name_or_path', model_path)
        logger.info(f"Loading tokenizer for evaluation from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=self.data_config.get('use_fast_tokenizer', False)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Tokenizer pad_token_id set to eos_token_id: {self.tokenizer.eos_token_id}")

    def _load_evaluation_dataset(self):
        dataset_path = self.eval_config.get('dataset_path')
        dataset_name = self.eval_config.get('dataset_name')  # For datasets library
        dataset_split = self.eval_config.get('dataset_split', 'test')
        text_column = self.eval_config.get('text_column', 'text')  # Column with text data for perplexity

        if dataset_path:
            logger.info(f"Loading evaluation dataset from path: {dataset_path} (assuming preprocessed)")
            # Assuming a simple .jsonl or similar file that DataProcessor can handle
            # Or, directly use datasets.load_dataset if it's in a known format
            try:
                # Minimal data processing for eval - just tokenization
                # This part needs careful design: are we re-using DataProcessor?
                # For simplicity, let's assume a simple text file or Hugging Face dataset
                if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
                    self.eval_dataset = load_dataset("json", data_files={dataset_split: dataset_path},
                                                     split=dataset_split)
                elif dataset_path.endswith(".txt"):
                    self.eval_dataset = load_dataset("text", data_files={dataset_split: dataset_path},
                                                     split=dataset_split)
                else:  # Try as a Hugging Face dataset identifier
                    self.eval_dataset = load_dataset(dataset_path, name=dataset_name, split=dataset_split)

            except Exception as e:
                logger.error(f"Failed to load evaluation dataset from {dataset_path}: {e}", exc_info=True)
                raise
        else:
            logger.warning(
                "No evaluation dataset path specified. Cannot perform perplexity or other dataset-based evals.")
            self.eval_dataset = None

        if self.eval_dataset:
            logger.info(f"Loaded evaluation dataset with {len(self.eval_dataset)} samples for split '{dataset_split}'.")
            # Pre-tokenize for perplexity
            if 'perplexity' in self.eval_config.get('metrics', []):
                logger.info(f"Tokenizing evaluation dataset for perplexity using text column '{text_column}'...")

                def tokenize_function(examples):
                    return self.tokenizer(examples[text_column], truncation=False)  # No truncation for PPL

                self.eval_dataset_tokenized = self.eval_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[col for col in self.eval_dataset.column_names if
                                    col not in ["input_ids", "attention_mask"]]
                )
                logger.info("Evaluation dataset tokenized.")

    def _load_metrics(self):
        self.metrics_to_compute = {}
        metric_names = self.eval_config.get('metrics', [])  # e.g., ["perplexity", "rouge"]
        for name in metric_names:
            try:
                if name != "perplexity":  # Perplexity is handled customly
                    self.metrics_to_compute[name] = load_metric(name)
                    logger.info(f"Loaded metric '{name}' for evaluation.")
            except Exception as e:
                logger.warning(f"Could not load metric '{name}': {e}. It will be skipped.")

    def compute_perplexity(self):
        if not hasattr(self, 'eval_dataset_tokenized') or self.eval_dataset_tokenized is None:
            logger.warning("Tokenized evaluation dataset not available. Skipping perplexity calculation.")
            return None

        logger.info("Calculating perplexity...")
        max_length = self.model.config.max_position_embeddings  # Or tokenizer.model_max_length
        stride = self.eval_config.get('perplexity_stride', 512)

        nlls = []
        for i in tqdm(range(0, len(self.eval_dataset_tokenized), self.eval_config.get('batch_size', 1)),
                      desc="Perplexity"):
            batch_encodings = self.eval_dataset_tokenized[i:i + self.eval_config.get('batch_size', 1)]

            for encodings in batch_encodings['input_ids']:  # Process each sample in the batch
                seq_len = len(encodings)
                prev_end_loc = 0
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end_loc  # Ensure positive target length
                    input_ids = torch.tensor(encodings[begin_loc:end_loc], device=self.device).unsqueeze(0)
                    target_ids = input_ids.clone()
                    target_ids[:, :-trg_len] = -100  # Ignore context tokens for loss calculation

                    if input_ids.size(1) < 2:  # Skip if sequence is too short
                        continue

                    with torch.no_grad():
                        outputs = self.model(input_ids, labels=target_ids)
                        neg_log_likelihood = outputs.loss * trg_len  # Loss is already mean, scale by target length

                    nlls.append(neg_log_likelihood)
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

        if not nlls:
            logger.warning("No negative log-likelihoods calculated, perplexity cannot be computed.")
            return None

        ppl = torch.exp(torch.stack(nlls).sum() / sum(len(enc) for enc in self.eval_dataset_tokenized['input_ids']))
        logger.info(f"Calculated Perplexity: {ppl.item()}")
        return ppl.item()

    # Add other metric computation methods here, e.g., for ROUGE, BLEU if doing generation tasks
    # def compute_generation_metrics(self):
    #     # ... generate responses, then use self.metrics_to_compute['rouge'].compute(...)
    #     pass

    def run(self) -> dict:
        logger.info("Starting model evaluation...")
        results = {}

        if 'perplexity' in self.eval_config.get('metrics', []):
            perplexity = self.compute_perplexity()
            if perplexity is not None:
                results['perplexity'] = perplexity

        # Add calls to other metric computations
        # if 'rouge' in self.metrics_to_compute:
        #     # ...
        #     pass

        logger.info(f"Evaluation finished. Results: {results}")
        return results