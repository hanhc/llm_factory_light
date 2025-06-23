import argparse
import logging
import os  # For environment variables
from llm_factory.utils.config_loader import load_config
from llm_factory.utils.logging_setup import setup_logging

# Module-level logger, will be configured by setup_logging
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM Factory Pipeline")
    parser.add_argument("mode", choices=['train', 'eval', 'inference_api'], help="Pipeline mode to run")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server (if mode is inference_api)")
    # Add other global arguments if needed

    args = parser.parse_args()

    # Load main configuration
    config = load_config(args.config)

    # Initialize logging as early as possible
    setup_logging(logging_config_override=config.get('logging'))

    logger.info(f"LLM-Factory started in '{args.mode}' mode with config: {args.config}")
    logger.debug(f"Full configuration: {config}")

    if args.mode == 'train':
        # Lazy import to avoid loading heavy libraries if not needed
        from llm_factory.data.processor import DataProcessor
        from llm_factory.training.trainer_factory import get_trainer
        from transformers import AutoModelForCausalLM  # Keep for base model loading

        logger.info("--- Training Mode Initialized ---")

        logger.info("Initializing DataProcessor...")
        data_processor = DataProcessor(config['data'])  # Pass data sub-config

        # Determine dataset processing based on training method for flexibility
        training_method = config.get('training_method', '').lower()
        if training_method == 'rlhf':
            logger.info("Processing data for RLHF (prompts)...")
            # Ensure DataProcessor has a method like `process_for_rl` or handles it in `process`
            train_dataset = data_processor.process_for_rl()
        else:  # SFT, LoRA, etc.
            logger.info("Processing data for supervised fine-tuning...")
            train_dataset = data_processor.process()

        tokenizer = data_processor.tokenizer
        logger.info(f"Data processing complete. Training dataset samples: {len(train_dataset)}")

        # Model loading (base model for LoRA/SFT/RLHF actor)
        model_name_or_path = config['model']['name_or_path']
        logger.info(f"Loading base model: {model_name_or_path}")
        # Quantization config can be part of model config if SFTTrainer/LoRATrainer need it directly
        quantization_cfg = config['model'].get('quantization_config')

        model_load_kwargs = {"trust_remote_code": True}
        if quantization_cfg:
            from transformers import BitsAndBytesConfig
            import torch  # for dtypes
            # Example: Convert string dtype to actual torch dtype
            if 'bnb_4bit_compute_dtype' in quantization_cfg and isinstance(quantization_cfg['bnb_4bit_compute_dtype'],
                                                                           str):
                try:
                    quantization_cfg['bnb_4bit_compute_dtype'] = getattr(torch, quantization_cfg[
                        'bnb_4bit_compute_dtype'].split('.')[-1])
                except AttributeError:
                    logger.error(f"Invalid torch dtype string: {quantization_cfg['bnb_4bit_compute_dtype']}")
                    raise
            bnb_config = BitsAndBytesConfig(**quantization_cfg)
            model_load_kwargs["quantization_config"] = bnb_config
            logger.info(f"Applying quantization config: {quantization_cfg}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_load_kwargs
        )
        logger.info("Base model loaded successfully.")

        logger.info(f"Initializing trainer for method: {training_method}")
        # Pass the full config, model, tokenizer, and datasets to the factory
        trainer = get_trainer(config, model, tokenizer, train_dataset, eval_dataset=None)  # Eval dataset can be added

        logger.info("Starting training...")
        trainer.train()
        logger.info("--- Training Completed ---")

    elif args.mode == 'eval':
        from llm_factory.evaluation.evaluator import Evaluator
        logger.info("--- Evaluation Mode Initialized ---")
        if 'evaluation' not in config:
            logger.error("Missing 'evaluation' section in the configuration file for eval mode.")
            return

        evaluator = Evaluator(config)  # Pass the full config
        eval_results = evaluator.run()
        logger.info(f"Evaluation Results: {eval_results}")

        output_file_path = config.get('evaluation', {}).get('output_file')
        if output_file_path:
            import json
            try:
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(eval_results, f, indent=4)
                logger.info(f"Evaluation results saved to {output_file_path}")
            except Exception as e:
                logger.error(f"Failed to save evaluation results to {output_file_path}: {e}")


    elif args.mode == 'inference_api':
        from llm_factory.inference.api_server import start_server
        logger.info(f"--- Inference API Server Mode Initialized ---")

        # MODEL_PATH and TP_SIZE are primarily set by environment variables for Docker/scripts
        # but api_server.py will read them.
        # The config file passed might contain server settings like host.
        api_host = config.get('server', {}).get('host', "0.0.0.0")

        logger.info(f"Starting API server on host {api_host}, port {args.port}...")
        logger.info(f"Model for inference (from env MODEL_PATH): {os.getenv('MODEL_PATH', 'Not Set')}")
        logger.info(f"Tensor Parallel Size (from env TP_SIZE): {os.getenv('TP_SIZE', 'Not Set')}")
        start_server(host=api_host, port=args.port)


if __name__ == "__main__":
    main()