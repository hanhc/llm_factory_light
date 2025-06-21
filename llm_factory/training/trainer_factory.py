from .sft_trainer import SFTTrainer
from .lora_trainer import LoRATrainer
from .rl_trainer import RLTrainer


def get_trainer(config, model, tokenizer, train_dataset, eval_dataset=None):
    """根据配置返回相应的训练器实例。"""
    training_method = config['training_method']

    if training_method == 'sft':
        # return SFTTrainer(config, ...) # 假设已实现
        pass
    elif training_method == 'lora':
        return LoRATrainer(config, model, tokenizer, train_dataset, eval_dataset)
    elif training_method == 'rlhf':
        # return RLTrainer(config, ...) # 假设已实现
        pass
    else:
        raise ValueError(f"Unknown training method: {training_method}")