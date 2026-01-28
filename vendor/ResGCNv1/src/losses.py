import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExpectationLoss(nn.Module):
    """
    Expectation Loss for Label Distribution Learning (LDL).

    This loss computes the expectation of the loss over the label distribution.
    For each sample, it samples multiple labels from the Gaussian distribution
    and computes the average loss.
    """

    def __init__(self, base_loss='mse', num_samples=10):
        """
        Args:
            base_loss (str): The base loss function ('mse' or 'mae')
            num_samples (int): Number of samples to draw from the distribution for expectation
        """
        super(ExpectationLoss, self).__init__()
        self.num_samples = num_samples

        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")

    def forward(self, pred, target_dist):
        """
        Args:
            pred: Predicted values (batch_size, 1)
            target_dist: Target distributions (batch_size, 2) where [:, 0] is mean, [:, 1] is std
        """
        batch_size = pred.size(0)

        # If target_dist is 1D (no LDL), fall back to standard loss
        if target_dist.dim() == 1:
            return self.base_loss(pred.squeeze(), target_dist).mean()

        # Extract mean and std from target distribution
        target_mean = target_dist[:, 0]  # (batch_size,)
        target_std = target_dist[:, 1]   # (batch_size,)

        # Sample labels from Gaussian distribution
        # pred shape: (batch_size, 1) -> squeeze to (batch_size,)
        pred = pred.squeeze()

        # Sample multiple labels for each sample in the batch
        sampled_targets = []
        for i in range(batch_size):
            # Sample from N(target_mean[i], target_std[i]^2)
            samples = torch.normal(target_mean[i], target_std[i], (self.num_samples,))
            sampled_targets.append(samples)

        sampled_targets = torch.stack(sampled_targets)  # (batch_size, num_samples)

        # Compute loss for each sample and each sampled target
        # pred: (batch_size,) -> expand to (batch_size, num_samples)
        pred_expanded = pred.unsqueeze(1).expand(-1, self.num_samples)

        # Compute base loss: (batch_size, num_samples)
        losses = self.base_loss(pred_expanded, sampled_targets)

        # Take expectation (mean) over samples for each batch item
        expectation_loss = losses.mean(dim=1)  # (batch_size,)

        # Return mean over batch
        return expectation_loss.mean()


class MSELoss(nn.Module):
    """Standard MSE Loss"""
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred.squeeze(), target)


class MAELoss(nn.Module):
    """Standard MAE Loss"""
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred.squeeze(), target)


class KLDivergenceLoss(nn.Module):
    """
    KL-Divergence Loss for Label Distribution Learning (LDL).

    This loss computes the KL divergence between predicted logits (discrete distribution)
    and target label distribution (soft labels).
    """

    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, pred_logits, target_dist):
        """
        Args:
            pred_logits: Predicted logits (batch_size, num_classes)
            target_dist: Target distributions (batch_size, num_classes) - soft labels
        """
        # Compute log softmax of predictions
        log_pred = F.log_softmax(pred_logits, dim=1)

        # KL divergence: sum over classes, mean over batch
        kl_div = F.kl_div(log_pred, target_dist, reduction='batchmean')

        return kl_div


class LDLLossWithL1(nn.Module):
    """
    Combined LDL Loss (KL Divergence + MAE/L1 Regularization).
    """

    def __init__(self, reg_weight, bin_centers):
        super(LDLLossWithL1, self).__init__()
        self.reg_weight = reg_weight
        self.register_buffer('bin_centers', bin_centers)
        
        # Calculate normalized bin centers for L1 loss scaling
        # This maps the age range to [0, 1] so L1 loss is comparable to KL loss
        min_val = bin_centers.min()
        max_val = bin_centers.max()
        range_val = max_val - min_val
        if range_val > 0:
            self.register_buffer('normalized_bin_centers', (bin_centers - min_val) / range_val)
        else:
            self.register_buffer('normalized_bin_centers', bin_centers - min_val) # Fallback if single bin
            
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_logits, target_dist):
        """
        Args:
            pred_logits: Predicted logits (batch_size, num_classes)
            target_dist: Target distributions (batch_size, num_classes) - soft labels
        """
        # KL Divergence
        log_pred = F.log_softmax(pred_logits, dim=1)
        loss_kl = F.kl_div(log_pred, target_dist, reduction='batchmean')

        # MAE Regularization
        # Calculate expected values using normalzied bin centers
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_expectation = torch.sum(pred_probs * self.normalized_bin_centers, dim=1)
        target_expectation = torch.sum(target_dist * self.normalized_bin_centers, dim=1)
        
        # Calculate L1 loss on expectations
        loss_reg = self.l1_loss(pred_expectation, target_expectation)

        # Combine losses
        total_loss = loss_kl + (self.reg_weight * loss_reg)
        
        return total_loss


class OrdinalEMDLoss(nn.Module):
    """
    Pure EMD Ordinal Loss for Label Distribution Learning (OLDL).
    
    This loss implements the Earth Mover's Distance (EMD) 
    by calculating the distance between the Cumulative Distribution Functions (CDFs)
    of the prediction and the target.
    
    Use: loss_type: ordinal_emd
    """
    def __init__(self, emd_loss_type='mse'):
        """
        Args:
            emd_loss_type (str): Loss type for EMD calculation ('mse' or 'l1')
        """
        super(OrdinalEMDLoss, self).__init__()
        if emd_loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif emd_loss_type == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type for OrdinalEMDLoss: {emd_loss_type}")

    def forward(self, pred_logits, target_dist):
        """
        Args:
            pred_logits: Predicted logits (batch_size, num_classes)
            target_dist: Target probability distributions (batch_size, num_classes) - soft labels (PDF)
        """
        # Convert logits to PDF (Probability Density Function equivalent for discrete)
        pred_probs = F.softmax(pred_logits, dim=1)
        
        # Convert PDF to CDF (Cumulative Distribution Function)
        pred_cdf = torch.cumsum(pred_probs, dim=1)
        target_cdf = torch.cumsum(target_dist, dim=1)
        
        # Calculate loss between CDFs (EMD)
        return self.loss(pred_cdf, target_cdf)


class OrdinalKLEMDLoss(nn.Module):
    """
    Ordinal-first LDL Loss (排序優先，回歸其次).
    
    Combines KL Divergence (distribution matching) with Ordinal constraint (EMD).
    
    L = KL(P_gt || P_pred) + λ * L_ordinal
    
    Where:
    - KL: Ensures predicted distribution matches target distribution shape
    - L_ordinal (EMD): Enforces ordinal relationships via CDF comparison
    
    This approach:
    1. First ensures ordering is correct (ordinal constraint)
    2. Then refines the distribution shape (KL divergence)
    
    Use: loss_type: ordinal_kl_emd
    """
    def __init__(self, ordinal_weight=1.0, emd_loss_type='l1'):
        """
        Args:
            ordinal_weight (float): Weight λ for ordinal (EMD) term. Default 1.0
            emd_loss_type (str): Loss type for EMD calculation ('mse' or 'l1')
        """
        super(OrdinalKLEMDLoss, self).__init__()
        self.ordinal_weight = ordinal_weight
        
        if emd_loss_type == 'mse':
            self.emd_loss = nn.MSELoss()
        elif emd_loss_type == 'l1':
            self.emd_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported EMD loss type: {emd_loss_type}")

    def forward(self, pred_logits, target_dist):
        """
        Args:
            pred_logits: Predicted logits (batch_size, num_classes)
            target_dist: Target probability distributions (batch_size, num_classes) - soft labels (PDF)
        
        Returns:
            Combined loss: KL + λ * EMD
        """
        # Convert logits to PDF
        pred_probs = F.softmax(pred_logits, dim=1)
        
        # === Part 1: KL Divergence (distribution shape matching) ===
        log_pred = F.log_softmax(pred_logits, dim=1)
        loss_kl = F.kl_div(log_pred, target_dist, reduction='batchmean')
        
        # === Part 2: Ordinal Constraint (EMD via CDF comparison) ===
        # Convert PDF to CDF (Cumulative Distribution Function)
        pred_cdf = torch.cumsum(pred_probs, dim=1)
        target_cdf = torch.cumsum(target_dist, dim=1)
        
        # EMD = L1/L2 distance between CDFs
        loss_ordinal = self.emd_loss(pred_cdf, target_cdf)
        
        # === Combined Loss ===
        total_loss = loss_kl + self.ordinal_weight * loss_ordinal
        
        return total_loss


# Alias for backward compatibility - OrdinalLoss now points to the combined KL+EMD version
OrdinalLoss = OrdinalKLEMDLoss
