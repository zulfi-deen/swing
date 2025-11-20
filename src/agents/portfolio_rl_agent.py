"""RL Portfolio Agent - PPO-based portfolio manager"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Beta
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

try:
    from torch_geometric.nn import GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logging.warning("torch_geometric not available, GATConv will use fallback")

logger = logging.getLogger(__name__)


class PortfolioRLAgent(nn.Module):
    """
    RL agent for portfolio construction using twin predictions.
    
    Architecture (as per RL Portfolio layer doc):
    1. State encoder (GNN) - aggregates 500 stock predictions
    2. Policy network - outputs stock selection + position sizing
    3. Value network - estimates state value (for PPO)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = config or {}
        
        # Hyperparameters
        self.num_stocks = self.config.get('num_stocks', 500)
        self.stock_feature_dim = 13  # 7 twin outputs + 6 features
        self.options_feature_dim = 9  # 9 key options features
        self.portfolio_feature_dim = 50
        self.macro_feature_dim = 10
        self.hidden_dim = 128
        self.max_positions = self.config.get('max_positions', 15)
        self.options_enabled = self.config.get('options_enabled', False)
        
        # === 1. STATE ENCODER ===
        
        # Encode per-stock information (twin predictions + features)
        stock_input_dim = self.stock_feature_dim
        if self.options_enabled:
            stock_input_dim += self.options_feature_dim
        
        self.stock_encoder = nn.Sequential(
            nn.Linear(stock_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Options market encoder (aggregates options signals across stocks)
        if self.options_enabled:
            self.options_market_encoder = nn.Sequential(
                nn.Linear(self.options_feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
        
        # Graph attention to aggregate stocks -> portfolio-level state
        if TORCH_GEOMETRIC_AVAILABLE:
            self.stock_aggregator = GATConv(
                in_channels=32,
                out_channels=64,
                heads=4,
                dropout=0.1,
                concat=True
            )
            gat_output_dim = 64 * 4  # 4 heads
        else:
            # Fallback: simple linear aggregation
            self.stock_aggregator = nn.Sequential(
                nn.Linear(32, 256),
                nn.ReLU()
            )
            gat_output_dim = 256
        
        # Portfolio state encoder
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(self.portfolio_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Macro encoder
        self.macro_encoder = nn.Sequential(
            nn.Linear(self.macro_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Combine all encodings
        fusion_input_dim = gat_output_dim + 64 + 32
        if self.options_enabled:
            fusion_input_dim += 16  # Options market encoding
        
        self.state_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # === 2. POLICY NETWORK (Actor) ===
        
        # Stock selection policy (attention mechanism)
        self.selection_policy = nn.Sequential(
            nn.Linear(self.hidden_dim + 32, 128),  # Global state + per-stock encoding
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Attention score per stock
        )
        
        # Position sizing policy (Beta distribution)
        # Beta(α, β) for each selected stock -> [0, 1] continuous
        self.sizing_alpha = nn.Sequential(
            nn.Linear(self.hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure α > 0
        )
        
        self.sizing_beta = nn.Sequential(
            nn.Linear(self.hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure β > 0
        )
        
        # === 3. VALUE NETWORK (Critic) ===
        
        self.value_network = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # State value V(s)
        )
    
    def encode_state(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode raw state into fixed-size embedding.
        
        Args:
            state: dict with twin_predictions, features, portfolio, macro, correlation_graph
        
        Returns:
            global_state_embedding: (batch, hidden_dim)
            per_stock_embeddings: (batch, num_stocks, 32)
        """
        
        twin_predictions = state.get('twin_predictions', {})
        features = state.get('features', {})
        portfolio = state.get('portfolio', {})
        macro = state.get('macro', {})
        correlation_graph = state.get('correlation_graph')
        tickers = state.get('tickers', [])
        
        if not tickers:
            # Empty state
            batch_size = 1
            global_state = torch.zeros((batch_size, self.hidden_dim))
            per_stock_embeddings = torch.zeros((batch_size, self.num_stocks, 32))
            return global_state, per_stock_embeddings
        
        num_stocks = len(tickers)
        
        # === Encode per-stock information ===
        
        per_stock_features = []
        options_features_list = []
        
        for ticker in tickers:
            # Concatenate twin predictions + features
            twin_pred = twin_predictions.get(ticker, {})
            feat = features.get(ticker, {})
            
            stock_vec = [
                twin_pred.get('expected_return', 0.0),
                twin_pred.get('hit_prob', 0.5),
                twin_pred.get('volatility', 0.02),
                float(twin_pred.get('regime', 0)),
                twin_pred.get('idiosyncratic_alpha', 0.0),
                twin_pred.get('quantile_10', -0.05),
                twin_pred.get('quantile_90', 0.05),
                feat.get('rsi_14', 0.5),
                feat.get('macd_signal', 0.0),
                feat.get('volume_z_score', 0.0),
                feat.get('sentiment_score', 0.0),
                feat.get('pattern_confidence', 0.0),
                feat.get('days_to_earnings', 1.0),
            ]
            
            # Add options features if enabled
            if self.options_enabled:
                options_feat = state.get('options_features', {}).get(ticker, {})
                stock_vec.extend([
                    options_feat.get('trend_signal', 0.0),
                    options_feat.get('sentiment_signal', 0.0),
                    float(options_feat.get('gamma_signal', 0)),
                    options_feat.get('pcr_zscore', 0.0),
                    options_feat.get('pcr_extreme_bullish', 0.0),
                    options_feat.get('pcr_extreme_bearish', 0.0),
                    options_feat.get('max_pain_distance_pct', 0.0),
                    options_feat.get('iv_percentile', 0.5),
                    options_feat.get('net_delta', 0.0),
                ])
                options_features_list.append(torch.tensor([
                    options_feat.get('trend_signal', 0.0),
                    options_feat.get('sentiment_signal', 0.0),
                    float(options_feat.get('gamma_signal', 0)),
                    options_feat.get('pcr_zscore', 0.0),
                    options_feat.get('pcr_extreme_bullish', 0.0),
                    options_feat.get('pcr_extreme_bearish', 0.0),
                    options_feat.get('max_pain_distance_pct', 0.0),
                    options_feat.get('iv_percentile', 0.5),
                    options_feat.get('net_delta', 0.0),
                ], dtype=torch.float32))
            
            per_stock_features.append(torch.tensor(stock_vec, dtype=torch.float32))
        
        if not per_stock_features:
            # Fallback
            per_stock_features = [torch.zeros(self.stock_feature_dim) for _ in range(num_stocks)]
        
        per_stock_features = torch.stack(per_stock_features)  # (num_stocks, 13)
        
        # Encode stocks
        per_stock_embeddings = self.stock_encoder(per_stock_features)  # (num_stocks, 32)
        
        # Aggregate with GAT (use correlation graph if available)
        if correlation_graph is not None and TORCH_GEOMETRIC_AVAILABLE:
            try:
                if hasattr(correlation_graph, 'edge_index') and correlation_graph.edge_index.numel() > 0:
                    stock_agg = self.stock_aggregator(
                        per_stock_embeddings,
                        correlation_graph.edge_index
                    )  # (num_stocks, 256)
                else:
                    # No edges, use mean pooling
                    stock_agg = per_stock_embeddings.mean(dim=0, keepdim=True).expand(num_stocks, -1)
                    stock_agg = self.stock_aggregator(stock_agg) if hasattr(self.stock_aggregator, '__call__') else stock_agg
            except Exception as e:
                logger.warning(f"Error in GAT aggregation: {e}, using mean pooling")
                stock_agg = per_stock_embeddings.mean(dim=0, keepdim=True).expand(num_stocks, -1)
                if isinstance(self.stock_aggregator, nn.Module):
                    stock_agg = self.stock_aggregator(stock_agg)
        else:
            # Fallback: mean pooling + linear
            stock_agg = per_stock_embeddings.mean(dim=0, keepdim=True)  # (1, 32)
            if isinstance(self.stock_aggregator, nn.Module):
                stock_agg = self.stock_aggregator(stock_agg)  # (1, 256)
                stock_agg = stock_agg.expand(num_stocks, -1)  # (num_stocks, 256)
            else:
                stock_agg = stock_agg.expand(num_stocks, -1)
        
        # Global pooling (mean across stocks)
        global_stock_repr = stock_agg.mean(dim=0)  # (256 or gat_output_dim,)
        
        # === Encode portfolio state ===
        
        portfolio_vec = torch.tensor([
            portfolio.get('cash', 0.35),
            portfolio.get('num_positions', 0) / self.max_positions,
            *[portfolio.get('sector_exposure', {}).get(sector, 0.0) for sector in ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 'Energy', 'Materials', 'Utilities', 'Real Estate', 'Communication', 'Consumer Staples']][:11],
            portfolio.get('days_held_avg', 0.0) / 10.0,
            # ... other portfolio features (pad to 50)
        ] + [0.0] * max(0, self.portfolio_feature_dim - 13), dtype=torch.float32)
        
        portfolio_repr = self.portfolio_encoder(portfolio_vec)  # (64,)
        
        # === Encode macro ===
        
        macro_vec = torch.tensor([
            macro.get('vix', 0.4),
            macro.get('spy_return_5d', 0.0),
            macro.get('treasury_10y', 0.4),
            1.0 if macro.get('market_regime') == 'bull' else 0.0,
            macro.get('dxy', 0.67),
            macro.get('gold_price', 0.67),
            macro.get('oil_price', 0.47),
            macro.get('credit_spread', 0.1),
            macro.get('term_spread', 0.3),
            macro.get('inflation_expectation', 0.4),
        ], dtype=torch.float32)
        
        macro_repr = self.macro_encoder(macro_vec)  # (32,)
        
        # === Encode options market (if enabled) ===
        
        options_market_repr = None
        if self.options_enabled and options_features_list:
            options_features_tensor = torch.stack(options_features_list)  # (num_stocks, 9)
            options_market_agg = options_features_tensor.mean(dim=0)  # (9,)
            options_market_repr = self.options_market_encoder(options_market_agg)  # (16,)
        elif self.options_enabled:
            options_market_repr = torch.zeros(16)
        
        # === Fuse everything ===
        
        fusion_list = [
            global_stock_repr,
            portfolio_repr,
            macro_repr
        ]
        
        if options_market_repr is not None:
            fusion_list.append(options_market_repr)
        
        global_state = torch.cat(fusion_list)
        
        # Ensure correct dimension
        if global_state.shape[0] != (gat_output_dim + 64 + 32):
            # Pad or truncate
            target_dim = gat_output_dim + 64 + 32
            if global_state.shape[0] < target_dim:
                padding = torch.zeros(target_dim - global_state.shape[0])
                global_state = torch.cat([global_state, padding])
            else:
                global_state = global_state[:target_dim]
        
        global_state_embedding = self.state_fusion(global_state.unsqueeze(0))  # (1, hidden_dim)
        
        # Expand per_stock_embeddings to batch
        per_stock_embeddings = per_stock_embeddings.unsqueeze(0)  # (1, num_stocks, 32)
        
        return global_state_embedding, per_stock_embeddings
    
    def forward(self, state: Dict, deterministic: bool = False) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: state -> action
        
        Returns:
            action: dict with stock_selection and position_sizes
            log_probs: for PPO loss
            entropy: for exploration bonus
            value: V(s) for PPO
        """
        
        # Encode state
        global_state, per_stock_embeddings = self.encode_state(state)
        
        tickers = state.get('tickers', [])
        num_stocks = len(tickers) if tickers else self.num_stocks
        
        if num_stocks == 0:
            # Empty state
            action = {
                'stock_selection': torch.zeros(self.num_stocks),
                'position_sizes': torch.zeros(self.num_stocks),
                'selected_indices': []
            }
            return action, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        # === STOCK SELECTION ===
        
        # Compute attention scores for each stock
        attention_logits = []
        
        for i in range(num_stocks):
            stock_emb = per_stock_embeddings[0, i]  # (32,)
            combined = torch.cat([global_state[0], stock_emb])  # (hidden_dim + 32,)
            
            score = self.selection_policy(combined)
            attention_logits.append(score)
        
        attention_logits = torch.cat(attention_logits)  # (num_stocks,)
        attention_weights = torch.softmax(attention_logits, dim=0)
        
        # Select top K stocks
        K = min(self.max_positions, num_stocks)
        
        if deterministic:
            top_k_indices = torch.topk(attention_weights, k=K).indices
        else:
            # Sample K stocks based on attention weights
            distribution = Categorical(probs=attention_weights)
            top_k_indices = []
            remaining_weights = attention_weights.clone()
            
            for _ in range(K):
                if remaining_weights.sum() < 1e-8:
                    break
                idx = distribution.sample()
                top_k_indices.append(idx.item())
                remaining_weights[idx] = 0
                remaining_weights = remaining_weights / (remaining_weights.sum() + 1e-8)
                distribution = Categorical(probs=remaining_weights)
            
            top_k_indices = torch.tensor(top_k_indices, dtype=torch.long)
        
        stock_selection = torch.zeros(self.num_stocks)
        if len(top_k_indices) > 0:
            # Map to actual indices if needed
            if num_stocks < self.num_stocks:
                # Pad with zeros
                pass
            stock_selection[top_k_indices] = 1.0
        
        # === POSITION SIZING ===
        
        position_sizes = torch.zeros(self.num_stocks)
        log_probs = []
        entropies = []
        
        for idx in top_k_indices:
            if idx >= num_stocks:
                continue
            
            stock_emb = per_stock_embeddings[0, idx]
            combined = torch.cat([global_state[0], stock_emb])
            
            # Beta distribution parameters
            alpha = self.sizing_alpha(combined).squeeze() + 1  # α ≥ 1
            beta = self.sizing_beta(combined).squeeze() + 1    # β ≥ 1
            
            # Sample position size from Beta(α, β)
            dist = Beta(alpha, beta)
            
            if deterministic:
                size = dist.mean  # Mode of Beta distribution
            else:
                size = dist.sample()
            
            position_sizes[idx] = size * 0.10  # Scale to max 10% per position
            
            log_probs.append(dist.log_prob(size))
            entropies.append(dist.entropy())
        
        # Normalize position sizes to sum ≤ 1.0
        total_allocation = position_sizes.sum()
        if total_allocation > 1.0:
            position_sizes = position_sizes / total_allocation
        
        # === VALUE ESTIMATE ===
        
        value = self.value_network(global_state)
        
        # === OUTPUTS ===
        
        action = {
            'stock_selection': stock_selection,
            'position_sizes': position_sizes,
            'selected_indices': top_k_indices.tolist() if len(top_k_indices) > 0 else [],
            'tickers': tickers
        }
        
        log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0)
        entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0)
        
        return action, log_prob, entropy, value.squeeze()


class PPOTrainer:
    """
    Trainer for RL portfolio agent using PPO algorithm.
    """
    
    def __init__(self, agent: PortfolioRLAgent, config: Optional[Dict] = None):
        self.agent = agent
        self.config = config or {}
        
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=self.config.get('learning_rate', 3e-4))
        
        # PPO hyperparameters
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.value_loss_coef = self.config.get('value_loss_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.gae_lambda = self.config.get('gae_lambda', 0.95)  # GAE parameter
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
    
    def train_step(self, rollout_buffer: List[Tuple]):
        """
        One PPO training step.
        
        Args:
            rollout_buffer: List of (state, action_dict, reward, next_state, done) tuples
        """
        
        if not rollout_buffer:
            return
        
        # Extract rollout data
        states = [t[0] for t in rollout_buffer]
        actions = [t[1] for t in rollout_buffer]
        rewards = torch.tensor([t[2] for t in rollout_buffer], dtype=torch.float32)
        next_states = [t[3] for t in rollout_buffer]
        dones = torch.tensor([t[4] for t in rollout_buffer], dtype=torch.float32)
        
        # Compute old log probs and values (from rollout)
        old_log_probs = torch.tensor([a.get('log_prob', 0.0) for a in actions], dtype=torch.float32)
        old_values = torch.tensor([a.get('value', 0.0) for a in actions], dtype=torch.float32)
        
        # Compute advantages using GAE
        advantages = self.compute_gae(rewards, old_values, dones)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        num_epochs = self.config.get('ppo_epochs', 4)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for i, state in enumerate(states):
                # Forward pass with current policy
                action, log_prob, entropy, value = self.agent(state, deterministic=False)
                
                # Compute ratio
                ratio = torch.exp(log_prob - old_log_probs[i])
                
                # Clipped surrogate objective
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                
                policy_loss = -torch.min(surr1, surr2)
                
                # Value loss (MSE)
                value_loss = 0.5 * (returns[i] - value).pow(2)
                
                # Entropy bonus (exploration)
                entropy_loss = -entropy
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(states)
            if epoch == 0:
                logger.debug(f"PPO epoch {epoch}: avg_loss = {avg_loss:.4f}")
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        
        advantages = []
        gae = 0.0
        
        next_value = 0.0  # Terminal state value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1] * (1 - dones[t + 1])
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)

