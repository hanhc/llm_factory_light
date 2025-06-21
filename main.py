import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data.processor import DataProcessor
from .training.trainer_factory import get_trainer
from .utils.config_loader import load_config
from .inference.api_server import start_server


def main():
    parser = argparse.ArgumentParser(description="LLM Factory Pipeline")
    parser.add_argument("mode", choices=['train', 'eval', 'inference_api'], help="Pipeline mode to run")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == 'train':
        print("--- Starting Training Mode ---")
        # 1. 初始化数据处理器
        data_processor = DataProcessor(config['data'])

        # 2. 加载和处理数据
        train_dataset = data_processor.process()
        tokenizer = data_processor.tokenizer

        # 3. 加载模型 (对于 LoRA，这里是基础模型)
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name_or_path'],
            trust_remote_code=True
        )

        # 4. 使用工厂获取并运行训练器
        trainer = get_trainer(config, model, tokenizer, train_dataset)
        trainer.train()
        print("--- Training Completed ---")

    elif args.mode == 'eval':
        print("--- Starting Evaluation Mode ---")
        # 实现评估逻辑
        # evaluator = Evaluator(config)
        # evaluator.run()
        pass

    elif args.mode == 'inference_api':
        print("--- Starting Inference API Server ---")
        # API 服务器的配置可以通过环境变量传递
        start_server()


if __name__ == "__main__":
    main()
