"""Dataset loaders for vehicle images."""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars 196 Dataset loader.
    
    Expected directory structure:
    stanford_cars/
    ├── cars_train/
    │   ├── 00001.jpg
    │   └── ...
    ├── cars_test/
    └── labels.json  (you need to create this)
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Path to stanford_cars folder
            split: 'train' or 'test'
            transform: torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Set image directory
        if split == 'train':
            self.img_dir = self.root_dir / 'cars_train'
        else:
            self.img_dir = self.root_dir / 'cars_test'
        
        # Load labels
        labels_file = self.root_dir / f'{split}_labels.json'
        
        if not labels_file.exists():
            raise FileNotFoundError(
                f"Labels file not found: {labels_file}\n"
                "Please create it using the instructions below."
            )
        
        with open(labels_file, 'r') as f:
            self.labels_data = json.load(f)
        
        # Get list of images
        self.image_paths = sorted(list(self.img_dir.glob('*.jpg')))
        
        # Create label to index mapping
        unique_labels = sorted(set(self.labels_data.values()))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        print(f"Loaded {len(self.image_paths)} images")
        print(f"Found {len(unique_labels)} unique classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        img_name = img_path.name
        label_str = self.labels_data.get(img_name, 'unknown')
        label_idx = self.label_to_idx.get(label_str, 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx


class FolderDataset(Dataset):
    """
    Simple dataset that loads images from folders.
    Each folder is a class.
    
    Expected structure:
    data/samples/
    ├── toyota_camry/
    │   ├── car1.jpg
    │   └── car2.jpg
    ├── honda_civic/
    │   └── car1.jpg
    └── ...
    """
    
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Path to folder containing class folders
            transform: torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all class folders
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Find all images in this class folder
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class HierarchicalVehicleDataset(Dataset):
    """
    Dataset with hierarchical labels (make, type, model, year).
    
    Expected labels.json format:
    {
        "00001.jpg": {
            "make": "Toyota",
            "type": "Sedan",
            "model": "Camry",
            "year": "2019",
            "full_label": "Toyota Camry Sedan 2019"
        },
        ...
    }
    """
    
    def __init__(self, root_dir: str, labels_file: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load hierarchical labels
        with open(labels_file, 'r') as f:
            self.labels_data = json.load(f)
        
        # Get image paths
        self.image_paths = sorted(list(self.root_dir.glob('*.jpg')))
        
        # Create label mappings for each hierarchy level
        self.make_to_idx = self._create_mapping('make')
        self.type_to_idx = self._create_mapping('type')
        self.model_to_idx = self._create_mapping('model')
        self.year_to_idx = self._create_mapping('year')
        self.full_to_idx = self._create_mapping('full_label')
        
        print(f"Loaded {len(self.image_paths)} images")
        print(f"Hierarchy levels: Make={len(self.make_to_idx)}, "
              f"Type={len(self.type_to_idx)}, Model={len(self.model_to_idx)}, "
              f"Year={len(self.year_to_idx)}")
    
    def _create_mapping(self, key: str):
        """Create label to index mapping for a hierarchy level."""
        unique_labels = sorted(set(
            data[key] for data in self.labels_data.values() if key in data
        ))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get hierarchical labels
        img_name = img_path.name
        labels = self.labels_data.get(img_name, {})
        
        make_idx = self.make_to_idx.get(labels.get('make', 'unknown'), 0)
        type_idx = self.type_to_idx.get(labels.get('type', 'unknown'), 0)
        model_idx = self.model_to_idx.get(labels.get('model', 'unknown'), 0)
        year_idx = self.year_to_idx.get(labels.get('year', 'unknown'), 0)
        full_idx = self.full_to_idx.get(labels.get('full_label', 'unknown'), 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image and hierarchical labels
        return image, {
            'make': make_idx,
            'type': type_idx,
            'model': model_idx,
            'year': year_idx,
            'full': full_idx
        }