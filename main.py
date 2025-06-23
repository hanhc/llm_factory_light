import argparse
import logging  # Standard library
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data.processor import DataProcessor
from .training.trainer_factory import get_trainer
from .utils.config_loader import load_config
from .utils.logging_setup import setup_logging  # Import the setup function
from .inference.api_server import start_server

# It's good practice to get the module-level logger after setup_logging
# However, if you need to log before config is loaded (e.g. parsing args errors),
# basicConfig might be active initially, or no logging until setup_logging.
# For simplicity, we'll get it here and it will be configured by setup_logging.
logger = logging.getLogger(__name__)  # Logger for the main module


def main():
    parser = argparse.ArgumentParser(description="LLM Factory Pipeline")
    parser.add_argument("mode", choices=['train', 'eval', 'inference_api'], help="Pipeline mode to run")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")

    args = parser.parse_args()

    # Load main configuration
    config = load_config(args.config)

    # Initialize logging as early as possible
    # Pass the 'logging' section from the main config if it exists
    setup_logging(logging_config_override=config.get('logging'))

    logger.info(f"LLM-Factory started in '{args.mode}' mode.")
    logger.debug(f"Loaded configuration from: {args.config}")
    logger.debug(f"Full configuration details: {config}")

    if args.mode == 'train':
        logger.info("--- Training Mode Initialized ---")

        logger.info("Initializing DataProcessor...")
        data_processor = DataProcessor(config['data'])

        logger.info("Processing data...")
        train_dataset = data_processor.process()
        tokenizer = data_processor.tokenizer
        logger.info(f"Data processing complete. Training dataset size: {len(train_dataset)}")

        logger.info(f"Loading base model: {config['model']['name_or_path']}")
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name_or_path'],
            trust_remote_code=True
            # Add other model loading args from config if necessary
        )
        logger.info("Base model loaded successfully.")

        logger.info(f"Initializing trainer: {config['training_method']}")
        trainer = get_trainer(config, model, tokenizer, train_dataset)  # Pass full config

        logger.info("Starting training...")
        trainer.train()
        logger.info("--- Training Completed ---")

    elif args.mode == 'eval':
        logger.info("--- Evaluation Mode Initialized ---")
        # Implement evaluation logic
        # evaluator = Evaluator(config)
        # evaluator.run()
        logger.warning("Evaluation mode is not fully implemented yet.")
        pass

    elif args.mode == 'inference_api':
        logger.info("--- Inference API Server Mode Initialized ---")
        # Ensure API server components also use the configured logging
        # The VLLMEngine can get its own logger: logging.getLogger('llm_factory.inference.vllm_engine')
        logger.info("Starting API server...")
        start_server()  # This function internally might configure its own loggers or use root.


if __name__ == "__main__":
    main()