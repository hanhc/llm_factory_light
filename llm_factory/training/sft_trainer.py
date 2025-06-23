import logging
from transformers import TrainingArguments, AutoModelForCausalLM
from trl import SFTTrainer as HF_SFTTrainer  # Rename to avoid conflict with class name
from peft import LoraConfig, TaskType, get_peft_model
import os
import torch

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) using TRL's SFTTrainer.
    Can handle full fine-tuning or LoRA-based SFT if `sft_lora` config is provided.
    """

    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset=None):
        super().__init__(config, model, tokenizer, train_dataset, eval_dataset)
        self.sft_config_args = config.get('sft_args', {})  # SFTTrainer specific args like max_seq_length
        self.training_args_config = config['training_args']  # Transformers TrainingArguments
        self.sft_lora_config = config.get('sft_lora')  # Optional LoRA config for SFT

    def _prepare_peft_config_for_sft(self) -> Optional[LoraConfig]:
        """Prepares LoraConfig if 'sft_lora' is specified in the main config."""
        if self.sft_lora_config:
            logger.info("LoRA configuration found under 'sft_lora'. Applying PEFT for SFT.")
            # Ensure target_modules are appropriate for the model architecture
            # Common ones are often q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
            return LoraConfig(
                r=self.sft_lora_config.get('r', 16),
                lora_alpha=self.sft_lora_config.get('lora_alpha', 32),
                lora_dropout=self.sft_lora_config.get('lora_dropout', 0.05),
                target_modules=self.sft_lora_config.get('target_modules', ["q_proj", "v_proj"]),
                bias=self.sft_lora_config.get('bias', "none"),
                task_type=TaskType.CAUSAL_LM,
            )
        return None

    def train(self):
        logger.info("Initializing SFT training process.")

        # TrainingArguments
        training_args = TrainingArguments(**self.training_args_config)
        logger.debug(f"TrainingArguments: {training_args}")
        self.output_dir = training_args.output_dir  # Ensure output_dir is consistent
        os.makedirs(self.output_dir, exist_ok=True)

        # PEFT Config (LoRA) for SFTTrainer, if specified
        peft_config = self._prepare_peft_config_for_sft()

        # The model passed to BaseTrainer is the base model.
        # If PEFT is used, SFTTrainer handles wrapping it.
        # If quantization is used, it should have been applied when loading the base model.

        dataset_text_field = self.config['data'].get('dataset_text_field', 'text')
        if not dataset_text_field:
            logger.error("SFTTrainer requires 'dataset_text_field' to be specified in data_config.")
            raise ValueError("'dataset_text_field' is missing in data configuration for SFT.")

        logger.info(f"Instantiating TRL SFTTrainer. Dataset text field: '{dataset_text_field}'.")

        # SFTTrainer handles model quantization if the base model was loaded with BitsAndBytesConfig
        # and peft_config is supplied.
        hf_sft_trainer = HF_SFTTrainer(
            model=self.model,  # Pass the base model (potentially quantized)
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field=dataset_text_field,
            peft_config=peft_config,  # Pass LoRA config here if SFT with LoRA
            max_seq_length=self.sft_config_args.get('max_seq_length', 1024),
            packing=self.sft_config_args.get('packing', False),  # Efficiently packs sequences
            neftune_noise_alpha=self.sft_config_args.get('neftune_noise_alpha'),  # Optional regularization
            # formatting_func= # Can be used if dataset_text_field is not sufficient
            # data_collator= # Can supply a custom data collator
        )

        logger.info("Starting SFT model training...")
        if peft_config:
            logger.info("PEFT (LoRA) enabled for SFT. Trainable parameters:")
            hf_sft_trainer.model.print_trainable_parameters()

        train_result = hf_sft_trainer.train()
        logger.info("SFT model training finished.")
        logger.info(f"TrainOutput: {train_result}")

        # Save model using SFTTrainer's method, which handles adapters correctly
        self._save_sft_model(hf_sft_trainer)

    def _save_sft_model(self, trainer_instance: HF_SFTTrainer):
        """Saves the model using the TRL SFTTrainer's save_model method."""
        logger.info(f"Saving SFT model to {self.output_dir}...")
        trainer_instance.save_model(self.output_dir)  # Saves adapter if PEFT, full model otherwise

        # SFTTrainer's save_model usually also saves tokenizer if it's part of the model components.
        # However, explicitly saving tokenizer is good practice.
        if self.tokenizer:
            try:
                self.tokenizer.save_pretrained(self.output_dir)
                logger.info(f"Tokenizer explicitly saved to {self.output_dir}.")
            except Exception as e:
                logger.error(f"Could not save tokenizer explicitly: {e}")

        logger.info(f"SFT Model (and adapter if PEFT) and tokenizer saved to {self.output_dir}.")

    def save_model(self):  # Override BaseTrainer's save_model
        logger.warning("SFTTrainer uses its internal saving mechanism via _save_sft_model called during train(). "
                       "Direct call to SFTTrainer.save_model() is a no-op here.")
        pass