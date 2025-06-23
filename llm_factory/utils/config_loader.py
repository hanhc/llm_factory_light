import yaml
import logging
import os

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {config_path}")
        # Basic validation (can be expanded)
        if not isinstance(config, dict):
            logger.error(f"Configuration file {config_path} did not load as a dictionary.")
            raise ValueError(f"Configuration file {config_path} must be a YAML dictionary.")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}", exc_info=True)
        raise

# You can add more advanced features like environment variable substitution,
# schema validation (e.g., using Pydantic), or config merging here if needed.