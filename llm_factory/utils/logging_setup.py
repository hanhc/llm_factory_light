import logging
import logging.config
import yaml # Not used here for loading, but often associated
import os
from typing import Dict, Any, Optional

# Default logging configuration if not overridden by the main YAML config
DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(levelname)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO", # Default console level
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG", # Default file level
            "formatter": "standard",
            "filename": "logs/llm_factory_default.log", # Default log file
            "maxBytes": 1024 * 1024 * 100,  # 100 MB
            "backupCount": 3,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "llm_factory": { # Logger for our application
            "handlers": ["console", "file_handler"],
            "level": "DEBUG", # Capture all debug messages from our app by default
            "propagate": False,
        },
        "transformers": { # Control verbosity of transformers library
            "handlers": ["console", "file_handler"], # Can use dedicated handlers too
            "level": "WARNING", # Less verbose from libraries
            "propagate": False,
        },
        "datasets": {
            "handlers": ["console", "file_handler"],
            "level": "WARNING",
            "propagate": False,
        },
        "peft": {
            "handlers": ["console", "file_handler"],
            "level": "INFO",
            "propagate": False,
        },
        "trl": {
            "handlers": ["console", "file_handler"],
            "level": "INFO",
            "propagate": False,
        },
        "vllm": {
             "handlers": ["console", "file_handler"],
             "level": "INFO",
             "propagate": False,
        },
        "uvicorn": { # For FastAPI server logs
            "handlers": ["console", "file_handler"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": { # Important for FastAPI errors
            "handlers": ["console", "file_handler"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": { # Access logs, can be verbose
            "handlers": ["console", "file_handler"],
            "level": "WARNING", # Reduce verbosity of access logs
            "propagate": False,
        }
    },
    "root": { # Root logger: catches everything not caught by specific loggers
        "handlers": ["console"], # Default root handler to console
        "level": "WARNING", # Default level for everything else
    },
}

def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merges override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

def setup_logging(logging_config_override: Optional[Dict[str, Any]] = None):
    """
    Configures logging for the application.
    Uses a default configuration which can be overridden by a 'logging' section
    in the main YAML configuration file.
    """
    config_to_use = DEFAULT_LOGGING_CONFIG.copy()

    if logging_config_override:
        # Deep merge the override into the default config
        config_to_use = _deep_merge_dicts(config_to_use, logging_config_override)
        # print(f"Debug: Merged logging config: {json.dumps(config_to_use, indent=2)}")


    # Ensure log directory exists for file handlers
    for handler_name, handler_config in config_to_use.get("handlers", {}).items():
        if "filename" in handler_config:
            log_file_path = handler_config["filename"]
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    # print(f"Log directory created: {log_dir}") # Can be noisy
                except OSError as e:
                    # Use basic print as logger might not be fully set up or itself tries to log to file
                    print(f"Warning: Could not create log directory {log_dir}: {e}. File logging for this handler might fail.")
                    # Potentially disable this handler or log to a default location
                    pass

    try:
        logging.config.dictConfig(config_to_use)
        # Test log message to confirm setup
        # logging.getLogger("llm_factory.utils.logging_setup").info("Logging configured successfully using dictConfig.")
    except Exception as e:
        # Fallback to basicConfig if dictConfig fails for any reason
        print(f"ERROR: Failed to configure logging with dictConfig: {e}. Falling back to basicConfig.")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.getLogger("llm_factory.utils.logging_setup").warning(
            "Used basicConfig due to an error in dictConfig setup. Check logging configuration."
        )

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance. Convenience function."""
    return logging.getLogger(name)