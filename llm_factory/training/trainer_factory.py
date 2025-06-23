import logging
from .sft_trainer import SFTTrainer
from .lora_trainer import LoRATrainer
from .rl_trainer import RLTrainer

logger = logging.getLogger(__name__)

def get_trainer(config: dict, model, tokenizer, train_dataset, eval_dataset=None):
    """
    Factory function to get the appropriate trainer based on the configuration.
    Args:
        config (dict): The full configuration dictionary.
        model: The base model instance.
        tokenizer: The tokenizer instance.
        train_dataset: The training dataset.
        eval_dataset (optional): The evaluation dataset.
    Returns:
        An instance of a trainer class (SFTTrainer, LoRATrainer, RLTrainer).
    """
    training_method = config.get('training_method', '').lower()
    logger.info(f"Attempting to create trainer for method: '{training_method}'")

    if not training_method:
        logger.error("No 'training_method' specified in the configuration.")
        raise ValueError("Missing 'training_method' in configuration.")

    if training_method == 'sft':
        logger.info("Selected SFTTrainer.")
        return SFTTrainer(config, model, tokenizer, train_dataset, eval_dataset)
    elif training_method == 'lora':
        logger.info("Selected LoRATrainer.")
        return LoRATrainer(config, model, tokenizer, train_dataset, eval_dataset)
    elif training_method == 'rlhf':
        logger.info("Selected RLTrainer (for PPO).")
        # For RLHF, 'model' is the SFT/base actor model.
        # 'train_dataset' is the prompt dataset from DataProcessor.process_for_rl().
        return RLTrainer(config, model, tokenizer, train_dataset, eval_dataset)
    else:
        logger.error(f"Unknown training method: '{training_method}'")
        raise ValueError(f"Unsupported training method: {training_method}. "
                         "Supported methods: 'sft', 'lora', 'rlhf'.")