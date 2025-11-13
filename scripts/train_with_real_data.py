"""Training script with real car images."""

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
from src.data.dataset import FolderDataset, StanfordCarsDataset
from src.retrieval.knn_retrieval import KNNRetrieval
from src.models.ood_detector import KNNPlusOOD
import numpy as np


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        embeddings = model(images)
        
        # Compute loss
        loss = loss_fn(embeddings, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def extract_embeddings(model, data_loader, device):
    """Extract embeddings from dataset."""
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
    """Evaluate retrieval performance."""
    retriever = KNNRetrieval(k=1)
    retriever.build_database(train_embeddings, train_labels)
    
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
    
    correct = 0
    total = len(test_labels)
    
    for emb, true_label in zip(test_embeddings, test_labels):
        pred_label = retriever.predict(emb)
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy


def main(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    
    if args.dataset_type == 'folder':
        # Simple folder-based dataset
        dataset = FolderDataset(args.data_dir, transform=train_transform)
        
        # Split into train/test (80/20)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
    elif args.dataset_type == 'stanford':
        # Load training data only and split it
        print("Loading training data and splitting into train/val...")
        full_dataset = StanfordCarsDataset(
            args.data_dir, split='train', transform=train_transform
        )
        
        # Split: 80% train, 20% validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        print(f"Split into {train_size} train and {val_size} validation samples")
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    # Create data loaders
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2 if use_cuda else 0,  # 0 workers on CPU for Windows
        pin_memory=use_cuda  # Only pin memory if using GPU
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Creating model...")
    model = VehicleEmbeddingModel(
        embedding_dim=args.embedding_dim,
        pretrained=True  # Use ImageNet pre-trained weights
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    loss_fn = MultiSimilarityLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler (cyclic as mentioned in paper)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=2e-6, 
        max_lr=2e-4,
        step_size_up=len(train_loader) * 5,
        mode='triangular'
    )
    
    # Training loop
    print(f"\nTraining for {args.num_epochs} epochs...")
    best_accuracy = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            print("Evaluating...")
            train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
            accuracy = evaluate_retrieval(model, test_loader, train_embeddings, train_labels, device)
            print(f"Retrieval Accuracy: {accuracy * 100:.2f}%")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_path = Path(args.save_dir) / 'best_model.pth'
                model_path.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model to: {model_path}")
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy * 100:.2f}%")
    
    # Build final retrieval system
    print("\nBuilding final retrieval database...")
    model.load_state_dict(torch.load(Path(args.save_dir) / 'best_model.pth'))
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    
    # Save embeddings for later use
    np.save(Path(args.save_dir) / 'train_embeddings.npy', train_embeddings)
    np.save(Path(args.save_dir) / 'train_labels.npy', train_labels)
    
    print("Saved embeddings for retrieval system")
    print("\nAll done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Veri-Car model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--dataset_type', type=str, default='folder',
                       choices=['folder', 'stanford'],
                       help='Type of dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Dimension of embeddings')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    main(args)