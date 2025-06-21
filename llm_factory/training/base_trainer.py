from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    训练器的抽象基类。
    """
    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    @abstractmethod
    def train(self):
        """执行训练循环。"""
        pass

    def save_model(self):
        """保存训练好的模型。"""
        output_dir = self.config['training_args']['output_dir']
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
        