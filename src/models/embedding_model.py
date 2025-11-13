"""Embedding models for vehicle and color recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VehicleEmbeddingModel(nn.Module):
    """
    Vehicle embedding model using pre-trained ResNet50.
    This gives much better results than training from scratch.
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        
        # Use ResNet50 pre-trained on ImageNet
        print(f"Loading ResNet50 (pretrained={pretrained})...")
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # ResNet50 output is 2048 dimensional
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers (optional - helps prevent overfitting)
        # Uncomment these lines to freeze:
        # for param in list(self.backbone.parameters())[:-20]:
        #     param.requires_grad = False
        
        # Projection head to create embeddings
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        print(f"Model initialized with {embedding_dim}-dimensional embeddings")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        # DON'T normalize here - let the loss function handle it
        # This gives more flexibility
        return embeddings
