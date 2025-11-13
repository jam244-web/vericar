"""Optimized training for best results."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
import argparse

from src.models.embedding_model import VehicleEmbeddingModel
from src.models.loss_functions import MultiSimilarityLoss
from src.data.dataset import StanfordCarsDataset
from src.retrieval.knn_retrieval import KNNRetrieval
import numpy as np


def get_transforms():
    """Get optimized transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.05),  # Occasional grayscale
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, test_transform


def train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        embeddings = model(images)
        loss = loss_fn(embeddings, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (helps stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def extract_embeddings(model, data_loader, device):
    """Extract embeddings."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.vstack(all_embeddings), np.array(all_labels)


def evaluate_retrieval(model, test_loader, train_embeddings, train_labels, device):
    """Evaluate retrieval."""
    retriever = KNNRetrieval(k=1)
    retriever.build_database(train_embeddings, train_labels)
    
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
    
    correct = 0
    total = len(test_labels)
    
    for emb, true_label in zip(test_embeddings, test_labels):
        pred_label = retriever.predict(emb)
        if pred_label == true_label:
            correct += 1
    
    return correct / total


def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    config = {
        'data_dir': 'data/stanford_cars',
        'batch_size': 64,
        'embedding_dim': 256,  # Larger!
        'num_epochs': 150,     # More epochs!
        'base_lr': 2e-6,
        'max_lr': 2e-4,
        'weight_decay': 1e-4,  # Regularization
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Get transforms
    train_transform, test_transform = get_transforms()
    
    # Load data
    print("\nLoading data...")
    full_dataset = StanfordCarsDataset(
        config['data_dir'],
        split='train',
        transform=train_transform
    )
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Set test transform
    test_dataset.dataset.transform = test_transform
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print("\nCreating model...")
    model = VehicleEmbeddingModel(
        embedding_dim=config['embedding_dim'],
        pretrained=True
    ).to(device)
    
    # Loss (tuned hyperparameters)
    loss_fn = MultiSimilarityLoss(
        alpha=2.0,
        beta=50.0,
        lambda_val=0.5
    )
    
    # Optimizer
    optimizer = optim.AdamW(  # AdamW is better than Adam
        model.parameters(),
        lr=config['max_lr'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler (cosine annealing with warm restarts)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * 10,  # Restart every 10 epochs
        T_mult=2
    )
    
    # Training loop
    print(f"\nTraining for {config['num_epochs']} epochs...")
    best_accuracy = 0.0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("Evaluating...")
            train_embeddings, train_labels = extract_embeddings(
                model, train_loader, device
            )
            accuracy = evaluate_retrieval(
                model, test_loader, train_embeddings, train_labels, device
            )
            print(f"Accuracy: {accuracy * 100:.2f}%")
            
            # Save best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'models/best_model_optimized.pth')
                print(f"✓ Saved best model: {best_accuracy * 100:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best accuracy: {best_accuracy * 100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()