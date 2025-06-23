import logging
from transformers import TrainingArguments, AutoModelForCausalLM
from trl import SFTTrainer as HF_SFTTrainer  # Rename to avoid conflict
from peft import LoraConfig  # SFTTrainer can also handle LoRA internally if needed
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) using TRL's SFTTrainer.
    """

    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset=None):
        super().__init__(config, model, tokenizer, train_dataset, eval_dataset)
        self.sft_config = config.get('sft_args', {})
        self.training_args_config = config['training_args']

    def _prepare_sft_config(self) -> dict:
        """Prepares SFTConfig or LoraConfig if specified for SFTTrainer."""
        sft_specific_lora_config = self.config.get('sft_lora')  # Check for LoRA config under SFT
        if sft_specific_lora_config:
            logger.info("LoRA configuration found for SFT. Applying PEFT for SFT.")
            return LoraConfig(
                r=sft_specific_lora_config.get('r', 8),
                lora_alpha=sft_specific_lora_config.get('lora_alpha', 16),
                lora_dropout=sft_specific_lora_config.get('lora_dropout', 0.05),
                target_modules=sft_specific_lora_config.get('target_modules', ["q_proj", "v_proj"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
        # If no specific LoRA, SFTTrainer will do full fine-tuning or use model's quantization
        return None

    def train(self):
        logger.info("Initializing SFT training process.")

        # TrainingArguments
        training_args = TrainingArguments(**self.training_args_config)
        logger.debug(f"TrainingArguments: {training_args}")

        # PEFT Config (LoRA) for SFTTrainer, if specified
        peft_config = self._prepare_sft_config()
        if peft_config:
            logger.info("Using PEFT (LoRA) configuration for SFT.")

        # Quantization config (already part of model if loaded with it)
        # SFTTrainer handles models loaded with `quantization_config` from `transformers`

        # Determine dataset text field or formatting function
        dataset_text_field = self.config['data'].get('dataset_text_field')
        # formatting_func_path = self.sft_config.get('formatting_func')
        # formatting_func = None
        # if formatting_func_path:
        #     # Dynamically import formatting_func (requires careful path management)
        #     pass
        if not dataset_text_field:  # and not formatting_func:
            logger.warning("No 'dataset_text_field' or 'formatting_func' specified for SFTTrainer. "
                           "TRL will default to 'text' field or expect pre-formatted data.")
            # TRL SFTTrainer defaults to 'text' if dataset_text_field is None.

        logger.info(f"Instantiating TRL SFTTrainer. Dataset text field: '{dataset_text_field}'.")

        trainer = HF_SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field=dataset_text_field,  # Field in dataset containing text to train on
            # formatting_func=formatting_func, # Custom formatting function
            max_seq_length=self.sft_config.get('max_seq_length', 512),
            peft_config=peft_config,  # Pass LoRA config here if SFT with LoRA
            packing=self.sft_config.get('packing', False),
            neftune_noise_alpha=self.sft_config.get('neftune_noise_alpha'),
            # data_collator can be added if needed
        )

        logger.info("Starting SFT model training...")
        train_result = trainer.train()
        logger.info("SFT model training finished.")
        logger.info(f"TrainOutput: {train_result}")

        # Save model and tokenizer
        # If PEFT was used, SFTTrainer saves the adapter.
        # If full fine-tuning, it saves the full model.
        self.save_model(trainer)  # Pass trainer to save_model

    def save_model(self, trainer_instance):  # Modified to accept trainer
        """Saves the model using the trainer's save_model method."""
        output_dir = self.training_args_config['output_dir']
        logger.info(f"Saving model to {output_dir}...")
        trainer_instance.save_model(output_dir)  # SFTTrainer's save_model handles adapters correctly

        # Tokenizer might not need saving again if it wasn't modified,
        # but it's good practice, especially if new tokens were added (though unlikely for SFT).
        if self.tokenizer:
            try:
                self.tokenizer.save_pretrained(output_dir)
                logger.info(f"Tokenizer saved to {output_dir}.")
            except Exception as e:
                logger.error(f"Could not save tokenizer: {e}")

        logger.info(f"Model (and adapter if PEFT) and tokenizer saved to {output_dir}.")