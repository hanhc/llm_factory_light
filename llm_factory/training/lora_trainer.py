from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from .base_trainer import BaseTrainer


class LoRATrainer(BaseTrainer):
    """使用 PEFT LoRA 进行训练。"""
    def _setup_model(self):
        lora_config = self.config['lora']
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
            bias="none"
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train(self):
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name_or_path'],
            trust_remote_code=True
        )
        self._setup_model()

        training_args = TrainingArguments(**self.config['training_args'])

        # 使用 transformers.Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            # data_collator 可以按需添加
        )
        trainer.train()
        self.save_model()
