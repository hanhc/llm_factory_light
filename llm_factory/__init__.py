# llm_factory/__init__.py
# This file makes Python treat the `llm_factory` directory as a package.

VERSION = "0.1.0"

# Expose key components if desired for easier imports by users of the library
# Example:
# from .main import main
# from .data.processor import DataProcessor
# from .training.trainer_factory import get_trainer
# from .inference.vllm_engine import VLLMEngine
# from .evaluation.evaluator import Evaluator
# from .utils.logging_setup import setup_logging, get_logger
# from .utils.config_loader import load_config

# It's generally better to let the application (main.py) initialize logging
# based on its config, rather than initializing it on package import.