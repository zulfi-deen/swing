"""Loss functions for model training"""

import torch
import torch.nn as nn
from typing import Dict


class FoundationLoss(nn.Module):
    """Loss function for foundation model pre-training."""
    
    def __init__(self, return_weight: float = 1.0, prob_weight: float = 1.0):
        super().__init__()
        self.return_weight = return_weight
        self.prob_weight = prob_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
    
    def forward(self, preds: Dict, targets: Dict) -> torch.Tensor:
        """
        Compute loss for foundation model.
        
        Args:
            preds: Dict with 'return' and 'prob' predictions
            targets: Dict with 'return_5d' and 'hit_target' (optional)
        
        Returns:
            Total loss tensor
        """
        # Return prediction loss (MSE)
        return_pred = preds.get('return', torch.zeros_like(targets.get('return_5d', torch.tensor(0.0))))
        return_target = targets.get('return_5d', torch.zeros_like(return_pred))
        return_loss = self.mse(return_pred, return_target)
        
        # Hit probability loss (BCE)
        prob_pred = preds.get('prob', torch.zeros_like(return_target))
        hit_target = targets.get('hit_target', (return_target > 0).float())
        prob_loss = self.bce(prob_pred, hit_target)
        
        # Combined loss
        total_loss = self.return_weight * return_loss + self.prob_weight * prob_loss
        
        return total_loss


class TwinLoss(nn.Module):
    """Loss for digital twin fine-tuning."""
    
    def __init__(
        self,
        return_weight: float = 1.0,
        prob_weight: float = 1.0,
        regime_weight: float = 0.5,
        quantile_weight: float = 0.3
    ):
        super().__init__()
        self.return_weight = return_weight
        self.prob_weight = prob_weight
        self.regime_weight = regime_weight
        self.quantile_weight = quantile_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.quantile_loss = QuantileLoss([0.1, 0.5, 0.9])
    
    def forward(self, preds: Dict, targets: Dict) -> Dict:
        """
        Compute loss for digital twin.
        
        Args:
            preds: Dict with 'expected_return', 'hit_prob', 'regime_logits', 'quantiles'
            targets: Dict with 'return_5d', 'hit_target', 'regime' (optional)
        
        Returns:
            Dict with 'total' loss and component losses
        """
        return_target = targets.get('return_5d', torch.zeros(1))
        hit_target = targets.get('hit_target', (return_target > 0).float())
        regime_target = targets.get('regime', torch.zeros(1, dtype=torch.long))
        
        # Return loss
        expected_return = preds.get('expected_return', torch.zeros_like(return_target))
        return_loss = self.mse(expected_return, return_target)
        
        # Hit probability loss
        hit_prob = preds.get('hit_prob', torch.zeros_like(hit_target))
        prob_loss = self.bce(hit_prob, hit_target)
        
        # Regime classification loss (if regime target provided)
        regime_loss = torch.tensor(0.0, device=expected_return.device)
        if 'regime_logits' in preds and regime_target.numel() > 0:
            regime_logits = preds['regime_logits']
            if regime_logits.dim() == 1:
                regime_logits = regime_logits.unsqueeze(0)
            if regime_target.dim() == 0:
                regime_target = regime_target.unsqueeze(0)
            regime_loss = self.ce(regime_logits, regime_target)
        
        # Quantile losses
        quantile_losses = []
        quantiles = preds.get('quantiles', {})
        for q_name, q_pred in quantiles.items():
            q_val = float(q_name[1:]) / 100  # 'q10' -> 0.1
            quantile_losses.append(self.quantile_loss(q_pred, return_target, q_val))
        
        quantile_loss = torch.mean(torch.stack(quantile_losses)) if quantile_losses else torch.tensor(0.0, device=expected_return.device)
        
        # Total loss
        total = (
            self.return_weight * return_loss +
            self.prob_weight * prob_loss +
            self.regime_weight * regime_loss +
            self.quantile_weight * quantile_loss
        )
        
        return {
            'total': total,
            'return': return_loss.item() if isinstance(return_loss, torch.Tensor) else return_loss,
            'prob': prob_loss.item() if isinstance(prob_loss, torch.Tensor) else prob_loss,
            'regime': regime_loss.item() if isinstance(regime_loss, torch.Tensor) else regime_loss,
            'quantile': quantile_loss.item() if isinstance(quantile_loss, torch.Tensor) else quantile_loss
        }


class SwingTradingLoss(nn.Module):
    """Multi-task loss for TFT-GNN."""
    
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {'return': 0.4, 'prob': 0.4, 'quantile': 0.2}
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.quantile_loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    
    def forward(self, preds: Dict, targets: Dict) -> Dict:
        """
        Args:
            preds: {
                'return': (batch,),
                'prob_hit_long': (batch,),
                'quantiles': {'q10': (batch,), 'q50': (batch,), 'q90': (batch,)}
            }
            targets: {
                'return_5d': (batch,),
                'hit_target_long': (batch,)
            }
        """
        
        # 1. Return regression
        return_loss = self.mse(preds['return'], targets['return_5d'])
        
        # 2. Hit probability classification
        prob_loss = self.bce(preds['prob_hit_long'], targets['hit_target_long'])
        
        # 3. Quantile regression (for uncertainty)
        quantile_losses = []
        for q, pred_q in preds['quantiles'].items():
            q_val = float(q[1:]) / 100  # 'q10' -> 0.1
            quantile_losses.append(
                self.quantile_loss(pred_q, targets['return_5d'], q_val)
            )
        
        quantile_loss = torch.mean(torch.stack(quantile_losses))
        
        # Total loss
        total_loss = (
            self.weights['return'] * return_loss +
            self.weights['prob'] * prob_loss +
            self.weights['quantile'] * quantile_loss
        )
        
        return {
            'total': total_loss,
            'return_loss': return_loss.item(),
            'prob_loss': prob_loss.item(),
            'quantile_loss': quantile_loss.item()
        }


class QuantileLoss(nn.Module):
    """Quantile regression loss."""
    
    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        quantile: float
    ) -> torch.Tensor:
        """Pinball loss for quantile regression."""
        errors = targets - preds
        loss = torch.max((quantile - 1) * errors, quantile * errors)
        return loss.mean()

