from abc import ABC, abstractmethod
import logging
import os

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for trainers.
    """

    def __init__(self, config: dict, model, tokenizer, train_dataset, eval_dataset=None):
        """
        Initializes the BaseTrainer.

        Args:
            config (dict): The full configuration dictionary.
            model: The model instance (e.g., from Hugging Face).
            tokenizer: The tokenizer instance.
            train_dataset: The training dataset.
            eval_dataset (optional): The evaluation dataset.
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Common place for output_dir
        self.output_dir = self.config.get('training_args', {}).get('output_dir', './output/default_trained_model')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"BaseTrainer initialized. Output directory: {self.output_dir}")

    @abstractmethod
    def train(self):
        """Executes the training loop."""
        pass

    def save_model(self, sub_folder: str = None):
        """
        Saves the trained model and tokenizer to the output directory.
        Specific trainers (like SFTTrainer or PPOTrainer) might override this
        or use their internal save methods.

        Args:
            sub_folder (str, optional): A sub-folder within the main output_dir to save.
        """
        save_directory = self.output_dir
        if sub_folder:
            save_directory = os.path.join(self.output_dir, sub_folder)
            os.makedirs(save_directory, exist_ok=True)

        logger.info(f"Saving model and tokenizer to {save_directory}...")
        try:
            # Handle PEFT model saving if applicable (model might be a PeftModel)
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(save_directory)
            else:
                logger.warning(
                    "Model does not have 'save_pretrained' method. Standard saving might not work for PEFT models.")
                # Fallback or error, depending on expectation. For now, log and proceed.

            if self.tokenizer:
                self.tokenizer.save_pretrained(save_directory)
            logger.info(f"Model and tokenizer saved successfully to {save_directory}.")
        except Exception as e:
            logger.error(f"Error saving model/tokenizer to {save_directory}: {e}", exc_info=True)
            raise