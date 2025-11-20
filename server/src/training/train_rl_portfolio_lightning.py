"""RL Portfolio Training with PyTorch Lightning"""

import torch
import pytorch_lightning as pl
from typing import Dict, Optional, List, Tuple
import logging
import os
from collections import deque

from src.agents.portfolio_rl_agent import PortfolioRLAgent, PPOTrainer
from src.training.rl_environment import TradingEnvironment
from src.training.lightning_utils import create_callbacks, create_logger, create_trainer
from src.utils.config import load_config
from src.utils.colab_utils import setup_colab_environment

logger = logging.getLogger(__name__)


class RLPortfolioModule(pl.LightningModule):
    """
    Lightning module for RL portfolio agent training.
    Manages episode rollouts and PPO updates.
    """
    
    def __init__(
        self,
        agent: PortfolioRLAgent,
        env: TradingEnvironment,
        config: Dict,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['agent', 'env', 'config'])
        self.agent = agent
        self.env = env
        self.config = config
        
        # PPO config
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # PPO trainer
        self.ppo_trainer = PPOTrainer(agent, {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_epsilon': clip_epsilon,
            'value_coef': value_coef,
            'entropy_coef': entropy_coef,
            'max_grad_norm': max_grad_norm,
            'ppo_epochs': ppo_epochs,
            'batch_size': batch_size
        })
        
        # Episode tracking
        self.episode_buffer = deque(maxlen=100)
        self.current_episode_return = 0.0
        self.current_episode_length = 0
    
    def training_step(self, batch, batch_idx):
        """
        Training step - rolls out one episode and updates policy.
        
        Note: For RL, we don't use traditional batches. Instead, we roll out
        an episode and train on the collected experience.
        """
        # Roll out one episode
        state = self.env.reset()
        episode_return = 0.0
        rollout_buffer = []
        
        episode_length = self.config.get('episode_length', 60)
        
        for step in range(episode_length):
            # Get action from policy
            action, log_prob, entropy, value = self.agent(state, deterministic=False)
            
            # Store action metadata
            action_dict = {
                'action': action,
                'log_prob': log_prob.item() if torch.is_tensor(log_prob) else log_prob,
                'value': value.item() if torch.is_tensor(value) else value,
                'entropy': entropy.item() if torch.is_tensor(entropy) else entropy
            }
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            rollout_buffer.append((
                state,
                action_dict,
                reward,
                next_state,
                done
            ))
            
            episode_return += reward
            state = next_state
            
            if done:
                break
        
        # Train on rollout
        if rollout_buffer:
            loss_dict = self.ppo_trainer.train_step(rollout_buffer)
            
            # Log metrics
            self.log('train_loss', loss_dict.get('total_loss', 0.0), on_step=True, on_epoch=True)
            self.log('train_policy_loss', loss_dict.get('policy_loss', 0.0), on_step=False, on_epoch=True)
            self.log('train_value_loss', loss_dict.get('value_loss', 0.0), on_step=False, on_epoch=True)
            self.log('train_entropy', loss_dict.get('entropy', 0.0), on_step=False, on_epoch=True)
        
        # Log episode metrics
        self.log('episode_return', episode_return, on_step=True, on_epoch=True, prog_bar=True)
        self.log('episode_length', len(rollout_buffer), on_step=True, on_epoch=True)
        
        if info:
            self.log('portfolio_value', info.get('portfolio_value', 0.0), on_step=True, on_epoch=True)
            self.log('num_positions', info.get('num_positions', 0), on_step=True, on_epoch=True)
        
        # Store episode stats
        self.episode_buffer.append({
            'return': episode_return,
            'length': len(rollout_buffer),
            'portfolio_value': info.get('portfolio_value', 0.0) if info else 0.0
        })
        
        return loss_dict.get('total_loss', torch.tensor(0.0))
    
    def on_train_epoch_end(self):
        """Log epoch-level metrics."""
        if self.episode_buffer:
            avg_return = sum(e['return'] for e in self.episode_buffer) / len(self.episode_buffer)
            avg_length = sum(e['length'] for e in self.episode_buffer) / len(self.episode_buffer)
            avg_portfolio_value = sum(e['portfolio_value'] for e in self.episode_buffer) / len(self.episode_buffer)
            
            self.log('avg_episode_return', avg_return, on_epoch=True)
            self.log('avg_episode_length', avg_length, on_epoch=True)
            self.log('avg_portfolio_value', avg_portfolio_value, on_epoch=True)
    
    def configure_optimizers(self):
        """Configure optimizer for RL agent."""
        optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=self.learning_rate
        )
        return optimizer


def train_rl_agent(
    agent: PortfolioRLAgent,
    env: TradingEnvironment,
    config: Optional[Dict] = None,
    num_episodes: int = 1000,
    checkpoint_dir: str = 'models/rl_portfolio/'
):
    """
    Train RL agent using Lightning.
    
    Args:
        agent: Portfolio RL agent
        env: Trading environment
        config: Configuration dict
        num_episodes: Number of training episodes
        checkpoint_dir: Checkpoint directory
    """
    
    config = config or load_config()
    rl_config = config.get('rl_portfolio', {})
    ppo_config = rl_config.get('ppo', {})
    
    # Create Lightning module
    model = RLPortfolioModule(
        agent=agent,
        env=env,
        config=config,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_epsilon=ppo_config.get('clip_epsilon', 0.2),
        value_coef=ppo_config.get('value_coef', 0.5),
        entropy_coef=ppo_config.get('entropy_coef', 0.01),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        ppo_epochs=ppo_config.get('ppo_epochs', 4),
        batch_size=ppo_config.get('batch_size', 64)
    )
    
    # Get checkpoint directory
    _, models_path = setup_colab_environment(config)
    checkpoint_dir = os.path.join(models_path, 'rl_portfolio')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create callbacks and logger
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        monitor='avg_episode_return',
        mode='max',
        save_top_k=3,
        patience=rl_config.get('early_stopping_patience', 50),
        filename='rl-agent-{epoch:02d}-{avg_episode_return:.2f}'
    )
    
    logger_obj = create_logger(
        experiment_name='rl_portfolio_training',
        tracking_uri=config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns')
    )
    
    # Create trainer
    # For RL, we use num_episodes as max_steps since each step is an episode
    trainer = create_trainer(
        max_epochs=1,  # We'll use max_steps instead
        max_steps=num_episodes,
        callbacks=callbacks,
        logger=logger_obj,
        gradient_clip_val=ppo_config.get('max_grad_norm', 0.5),
        limit_train_batches=num_episodes  # Limit to num_episodes steps
    )
    
    # Create dummy dataloader (RL doesn't use traditional data)
    from torch.utils.data import DataLoader, Dataset
    
    class DummyDataset(Dataset):
        def __len__(self):
            return num_episodes
        
        def __getitem__(self, idx):
            return idx
    
    dummy_dataset = DummyDataset()
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)
    
    # Train
    logger.info("Starting RL training...")
    trainer.fit(model, dummy_dataloader)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'agent_final.pt')
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'config': config,
    }, final_checkpoint_path)
    
    logger.info(f"RL training complete. Model saved to {final_checkpoint_path}")

