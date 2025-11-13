"""Loss functions for Veri-Car system."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity Loss for metric learning."""
    
    def __init__(self, alpha: float = 2.0, beta: float = 50.0, lambda_val: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_val = lambda_val
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        
        # Handle edge case: batch too small
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Normalize embeddings (ensure unit length)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix (cosine similarity since normalized)
        sim_mat = torch.matmul(embeddings, embeddings.t())
        
        # Create masks for positives and negatives
        labels_expanded = labels.unsqueeze(1)
        mask_pos = (labels_expanded == labels_expanded.t()).float()
        mask_neg = (labels_expanded != labels_expanded.t()).float()
        
        # Remove diagonal (self-similarity)
        mask_pos.fill_diagonal_(0)
        
        # Collect all losses in a list, then stack
        loss_list = []
        
        for i in range(batch_size):
            # Get positive and negative similarities for anchor i
            pos_mask = mask_pos[i] > 0
            neg_mask = mask_neg[i] > 0
            
            pos_sims = sim_mat[i][pos_mask]
            neg_sims = sim_mat[i][neg_mask]
            
            # Skip if no positives or negatives
            if len(pos_sims) == 0 or len(neg_sims) == 0:
                continue
            
            # Positive loss term
            pos_exp = torch.exp(-self.alpha * (pos_sims - self.lambda_val))
            pos_loss = (1.0 / self.alpha) * torch.log(1.0 + torch.sum(pos_exp))
            
            # Negative loss term
            neg_exp = torch.exp(self.beta * (neg_sims - self.lambda_val))
            neg_loss = (1.0 / self.beta) * torch.log(1.0 + torch.sum(neg_exp))
            
            # Combine losses for this sample
            sample_loss = pos_loss + neg_loss
            loss_list.append(sample_loss)
        
        # If no valid samples, return zero loss
        if len(loss_list) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Stack and average
        total_loss = torch.stack(loss_list).mean()
        
        return total_loss


class HierarchicalMultiSimilarityLoss(nn.Module):
    """HiMS-Min: Hierarchical Multi-Similarity Loss."""
    
    def __init__(self, num_levels: int = 4, alpha: float = 2.0, 
                 beta: float = 50.0, lambda_val: float = 0.5):
        super().__init__()
        self.num_levels = num_levels
        self.base_loss = MultiSimilarityLoss(alpha, beta, lambda_val)
        
    def forward(self, embeddings: torch.Tensor, 
                hierarchical_labels: List[torch.Tensor]) -> torch.Tensor:
        
        loss_list = []
        prev_min_loss = float('inf')
        
        # Process from finest to coarsest (bottom-up)
        for level in range(len(hierarchical_labels) - 1, -1, -1):
            labels = hierarchical_labels[level]
            
            # Calculate weight: λ = exp(1/level)
            weight = torch.exp(torch.tensor(1.0 / (level + 1)))
            
            # Compute MS loss at this level
            level_loss = self.base_loss(embeddings, labels)
            
            # Use minimum of current level and previous level
            if level < len(hierarchical_labels) - 1 and prev_min_loss != float('inf'):
                level_loss = torch.min(level_loss, torch.tensor(prev_min_loss, device=embeddings.device))
            
            weighted_loss = weight * level_loss
            loss_list.append(weighted_loss)
            
            prev_min_loss = level_loss.item()
        
        # Average all level losses
        if len(loss_list) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        total_loss = torch.stack(loss_list).sum() / self.num_levels
        
        return total_loss