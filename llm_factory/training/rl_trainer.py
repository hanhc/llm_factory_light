import logging
import torch
from transformers import AutoTokenizer  # AutoModelForCausalLM already in BaseTrainer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset  # For creating prompt dataset
import time
import os

from .base_trainer import BaseTrainer

# from ..data.processor import DataProcessor # Not directly used here, main.py calls it

logger = logging.getLogger(__name__)


class RLTrainer(BaseTrainer):
    """
    Trainer for Reinforcement Learning (RLHF) using TRL's PPOTrainer.
    Assumes SFT model is loaded as self.model, and self.train_dataset contains prompts.
    """

    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset=None):
        super().__init__(config, model, tokenizer, train_dataset, eval_dataset)
        self.rlhf_config = config['rlhf']
        self.ppo_config_args = self.rlhf_config['ppo_config']
        self.generation_kwargs = self.rlhf_config['generation_kwargs']
        self.training_args_rl = config['training_args']  # For output_dir, total_ppo_epochs, etc.
        self.output_dir = self.training_args_rl['output_dir']  # Override BaseTrainer's if different section
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_reward_function(self):
        """
        Loads or defines the reward model/function.
        This is a CRITICAL part of RLHF and often involves a separate trained model.
        """
        reward_model_name_or_path = self.rlhf_config.get("reward_model_name")
        if not reward_model_name_or_path:
            logger.error("reward_model_name not specified in rlhf_config. Cannot proceed with RLHF.")
            raise ValueError("Reward model name is required for RLHF.")

        logger.info(f"Attempting to initialize reward function/model: {reward_model_name_or_path}")

        # Example: Using a sentiment classification pipeline as a PROXY for a reward model.
        # THIS IS NOT A PROPER REWARD MODEL FOR LLM ALIGNMENT.
        # Replace with your actual reward model loading and inference logic.
        try:
            from transformers import pipeline
            # Determine device for reward model (can be different from PPO actor/critic)
            reward_device = self.ppo_config_args.get("reward_device", "cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Reward model will run on device: {reward_device}")

            # For models like 'distilbert-base-uncased-finetuned-sst-2-english', task is 'sentiment-analysis'
            # If your reward model is custom, you'll need custom loading.
            sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=reward_model_name_or_path,
                device=reward_device,
                # tokenizer=reward_model_name_or_path # Some pipelines need tokenizer explicitly
            )
            logger.info(f"Sentiment pipeline (proxy reward model) loaded: {reward_model_name_or_path}")

            def reward_fn(texts: list[str]) -> torch.Tensor:
                sentiments = sentiment_pipe(texts, truncation=True, padding=True)  # Batch process
                rewards = []
                for s in sentiments:
                    # Example: positive sentiment gets higher reward
                    # 'POSITIVE' (or 'LABEL_1' etc.) score as reward. Adjust based on your model's output.
                    if s['label'] in ['POSITIVE', 'LABEL_1', '4 stars', '5 stars']:  # Check common positive labels
                        rewards.append(torch.tensor(s['score']))
                    else:  # Negative or neutral sentiment
                        rewards.append(torch.tensor(1.0 - s['score']))  # Or a small negative value
                return torch.tensor(rewards, dtype=torch.float32).to(self.ppo_config_args.get("device", "cpu"))

            return reward_fn

        except Exception as e:
            logger.error(f"Failed to load reward model/fn '{reward_model_name_or_path}': {e}", exc_info=True)
            logger.warning("Falling back to DUMMY REWARD function (length-based). THIS IS NOT FOR PRODUCTION.")

            def dummy_reward_fn(texts: list[str]) -> torch.Tensor:
                # Longer responses get higher reward - very naive!
                return torch.tensor([float(len(t.split())) / 100.0 for t in texts], dtype=torch.float32).to(
                    self.ppo_config_args.get("device", "cpu"))

            return dummy_reward_fn

    def train(self):
        logger.info("Initializing RLHF (PPO) training process.")

        # 1. PPO Configuration from YAML
        ppo_config = PPOConfig(**self.ppo_config_args)
        logger.debug(f"PPOConfig: {ppo_config}")

        # 2. SFT Model (Actor) and ValueHead Model (Critic)
        # self.model is the SFT model. We need to wrap it with a ValueHead.
        # TRL's AutoModelForCausalLMWithValueHead can load the SFT model and add a value head.
        # The path to the SFT model is `self.rlhf_config['base_model_name']`
        sft_model_path = self.rlhf_config['base_model_name']
        logger.info(f"Loading SFT model '{sft_model_path}' and wrapping with ValueHead for PPO...")

        # Determine if the SFT model was LoRA tuned, to load adapters if needed.
        # This assumes the sft_model_path is a PEFT model if it was LoRA SFT.
        try:
            ppo_model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(
                sft_model_path,
                trust_remote_code=True,
                # peft_config might be needed if sft_model_path is base and adapter needs loading
                # For simplicity, assume sft_model_path is already a PEFT model if applicable.
                # Or, if it's a merged model, no peft_config needed here.
            )
            logger.info("SFT model wrapped with ValueHead successfully.")
        except Exception as e:
            logger.error(f"Failed to load SFT model '{sft_model_path}' with ValueHead: {e}", exc_info=True)
            raise

        # 3. Reward Function (from reward model)
        reward_function = self._get_reward_function()

        # 4. Initialize PPOTrainer
        # self.train_dataset is the prompt dataset, already loaded and processed by DataProcessor.process_for_rl()
        # It should contain 'input_ids' and 'query' (raw prompt text)

        # Simple collator for list of dicts from dataset
        def ppo_collator(data):
            return {key: [d[key] for d in data] for key in data[0]}

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=ppo_model_with_value_head,  # Model with value head
            ref_model=None,  # If None, a copy of the model is used as reference (with detached grads)
            tokenizer=self.tokenizer,
            dataset=self.train_dataset,  # Prompt dataset
            data_collator=ppo_collator,
        )
        logger.info(f"PPOTrainer initialized. Device: {ppo_trainer.accelerator.device}")

        # 5. RL Training Loop
        total_ppo_epochs = self.training_args_rl.get('total_ppo_epochs', 1)
        save_frequency = self.training_args_rl.get('save_freq', 1)  # Save every N PPO epochs

        # Ensure pad token ID is set for generation if tokenizer doesn't have it
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        current_generation_kwargs = self.generation_kwargs.copy()
        current_generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id

        for ppo_epoch in range(total_ppo_epochs):
            logger.info(f"--- Starting PPO Epoch {ppo_epoch + 1}/{total_ppo_epochs} ---")
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(ppo_trainer.dataloader):
                # 'batch' from dataloader (using ppo_collator) will be like:
                # {'input_ids': [tensor_prompt1, tensor_prompt2, ...], 'query': [str_prompt1, ...]}
                query_tensors = batch['input_ids']  # These are already tensors if collator/dataset provides them
                # Ensure they are on the correct device, PPOTrainer usually handles this internally
                # query_tensors = [qt.to(ppo_trainer.accelerator.device) for qt in query_tensors]

                # Generate responses from the policy model (actor)
                # PPOTrainer.generate takes a list of Tensors (tokenized prompts)
                logger.debug(f"Generating responses for {len(query_tensors)} prompts in batch {batch_idx + 1}")
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,  # We only want the generated part for reward
                    **current_generation_kwargs
                )

                # Detokenize responses to text for the reward model
                # response_tensors is a list of Tensors, each for a prompt
                batch['response_text'] = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in
                                          response_tensors]
                logger.debug(f"Sample generated response: {batch['response_text'][0][:100]}...")

                # Calculate reward scores for the generated responses
                # reward_function expects a list of strings
                reward_scores = reward_function(batch['response_text'])  # Returns a Tensor of rewards
                logger.debug(f"Calculated rewards for batch. Mean reward: {torch.mean(reward_scores).item():.4f}")

                # PPO optimization step
                # query_tensors: List[torch.LongTensor] -- tokenized prompts
                # response_tensors: List[torch.LongTensor] -- tokenized responses
                # reward_scores: torch.FloatTensor -- rewards for each response
                stats = ppo_trainer.step(query_tensors, response_tensors, reward_scores)

                # Log statistics
                ppo_trainer.log_stats(stats, batch, reward_scores.cpu().numpy())  # log_stats expects numpy rewards

                if (batch_idx + 1) % ppo_config.get("log_freq", 10) == 0:  # Log progress
                    logger.info(f"PPO Epoch {ppo_epoch + 1}, Batch {batch_idx + 1}/{len(ppo_trainer.dataloader)}, "
                                f"Mean Reward: {torch.mean(reward_scores).item():.4f}, "
                                f"Objective/KL: {stats.get('objective/kl', 0.0):.4f}")

            epoch_duration = time.time() - epoch_start_time
            logger.info(f"--- PPO Epoch {ppo_epoch + 1} completed in {epoch_duration:.2f} seconds ---")

            # Save model checkpoint
            if (ppo_epoch + 1) % save_frequency == 0 or (ppo_epoch + 1) == total_ppo_epochs:
                checkpoint_save_path = os.path.join(self.output_dir, f"checkpoint_epoch_{ppo_epoch + 1}")
                logger.info(f"Saving PPO model checkpoint to {checkpoint_save_path}...")
                ppo_trainer.save_pretrained(checkpoint_save_path)
                # Also save tokenizer for completeness with the checkpoint
                self.tokenizer.save_pretrained(checkpoint_save_path)
                logger.info(f"PPO Model checkpoint saved.")

        logger.info("RLHF (PPO) training finished.")
        # Final save (policy model)
        final_model_save_path = os.path.join(self.output_dir, "final_rlhf_model")
        ppo_trainer.save_pretrained(final_model_save_path)
        self.tokenizer.save_pretrained(final_model_save_path)  # Save tokenizer with final model
        logger.info(f"Final RLHF policy model saved to {final_model_save_path}")

    def save_model(self):  # Override BaseTrainer's save_model
        logger.warning("RLTrainer uses PPOTrainer's internal saving mechanism (ppo_trainer.save_pretrained()) "
                       "called within the training loop. Direct call to RLTrainer.save_model() is a no-op.")
        pass