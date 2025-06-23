import logging
from transformers import TrainingArguments, Trainer as HuggingFaceTrainer  # Standard HF Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import os
import torch

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class LoRATrainer(BaseTrainer):
    """
    Trainer for Low-Rank Adaptation (LoRA) fine-tuning.
    This uses the standard Hugging Face Trainer with a PEFT-modified model.
    """

    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset=None):
        super().__init__(config, model, tokenizer, train_dataset, eval_dataset)
        self.lora_config_args = config['lora']  # LoRA specific parameters like r, alpha
        self.training_args_config = config['training_args']

    def _prepare_lora_model(self):
        """Applies LoRA configuration to the base model."""
        lora_peft_config = LoraConfig(
            r=self.lora_config_args['r'],
            lora_alpha=self.lora_config_args['lora_alpha'],
            lora_dropout=self.lora_config_args['lora_dropout'],
            target_modules=self.lora_config_args['target_modules'],
            bias=self.lora_config_args.get('bias', "none"),  # Default to none
            task_type=TaskType[self.lora_config_args.get('task_type', "CAUSAL_LM")]  # Default to CAUSAL_LM
        )
        logger.info(f"Applying LoRA PEFT config: {lora_peft_config}")

        # Prepare model for k-bit training if quantization was applied during base model load
        if hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit or \
                hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit:
            logger.info("Preparing model for k-bit training (e.g., QLoRA).")
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.training_args_config.get("gradient_checkpointing", True)
                # Default to True for memory saving
            )
            # For QLoRA, gradient_checkpointing is highly recommended.
            # Ensure TrainingArguments also has gradient_checkpointing=True.

        self.model = get_peft_model(self.model, lora_peft_config)
        logger.info("LoRA model prepared. Trainable parameters:")
        self.model.print_trainable_parameters()

    def train(self):
        logger.info("Initializing LoRA training process.")

        # Prepare the model with LoRA layers
        self._prepare_lora_model()

        # TrainingArguments
        training_args = TrainingArguments(**self.training_args_config)
        logger.debug(f"TrainingArguments: {training_args}")
        self.output_dir = training_args.output_dir  # Ensure output_dir is consistent
        os.makedirs(self.output_dir, exist_ok=True)

        # Standard Hugging Face Trainer
        # Data collator might be needed, especially if using packing or specific formatting.
        # For simple text datasets prepared by DataProcessor, default might work.
        # from transformers import DataCollatorForLanguageModeling
        # data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # For LoRA, the dataset should be tokenized text. DataProcessor's 'process' method
        # should provide a dataset with a 'text' field (or whatever is configured).
        # The standard Trainer will then tokenize this 'text' field.
        # If your DataProcessor already tokenizes, ensure Trainer doesn't re-tokenize,
        # or pass pre-tokenized 'input_ids', 'attention_mask', 'labels'.

        # Let's assume DataProcessor returns a dataset that Trainer can directly use
        # (e.g., after tokenization if DataProcessor does it, or text for Trainer to tokenize)
        # SFTTrainer often implies that the dataset has a 'text' field.
        # For standard Trainer, if your dataset from DataProcessor is already tokenized and has
        # 'input_ids', 'attention_mask', 'labels', it will use them.
        # If it has a 'text' field, Trainer will tokenize it.

        # Let's clarify: The provided `DataProcessor.process` creates a dataset with a 'text' field.
        # The standard HuggingFaceTrainer, when given such a dataset, will tokenize this 'text' field on the fly.
        # This is generally fine.

        hf_trainer = HuggingFaceTrainer(
            model=self.model,  # This is now the PeftModel
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,  # Useful for saving and potentially for on-the-fly tokenization by Trainer
            # data_collator=data_collator, # Add if needed
        )

        logger.info("Starting LoRA model training with HuggingFace Trainer...")
        train_result = hf_trainer.train()
        logger.info("LoRA model training finished.")
        logger.info(f"TrainOutput: {train_result}")

        # Save the LoRA adapter and tokenizer
        # The base model is not saved here, only the adapter.
        super().save_model()  # Calls BaseTrainer's save_model which saves PeftModel (adapter)