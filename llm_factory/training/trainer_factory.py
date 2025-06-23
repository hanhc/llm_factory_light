import logging
from .sft_trainer import SFTTrainer
from .lora_trainer import LoRATrainer
from .rl_trainer import RLTrainer # Import RLTrainer

logger = logging.getLogger(__name__)

def get_trainer(config, model, tokenizer, train_dataset, eval_dataset=None):
    """
    Factory function to get the appropriate trainer based on the configuration.
    """
    training_method = config.get('training_method', '').lower()
    logger.info(f"Attempting to create trainer for method: '{training_method}'")

    if training_method == 'sft':
        logger.info("Selected SFTTrainer.")
        return SFTTrainer(config, model, tokenizer, train_dataset, eval_dataset)
    elif training_method == 'lora':
        logger.info("Selected LoRATrainer.")
        return LoRATrainer(config, model, tokenizer, train_dataset, eval_dataset)
    elif training_method == 'rlhf':
        logger.info("Selected RLTrainer (for PPO).")
        # For RLHF, 'model' is SFT model, 'train_dataset' is prompt dataset
        return RLTrainer(config, model, tokenizer, train_dataset, eval_dataset)
    else:
        logger.error(f"Unknown or unspecified training method: '{training_method}'")
        raise ValueError(f"Unknown or unspecified training method: {training_method}. "
                         "Supported methods: 'sft', 'lora', 'rlhf'.")