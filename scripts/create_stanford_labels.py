"""Create labels.json for Stanford Cars dataset."""

import scipy.io
import json
from pathlib import Path

def create_labels():
    # Paths
    data_dir = Path('data/stanford_cars')
    train_annos = data_dir / 'car_devkit' / 'devkit' / 'cars_train_annos.mat'
    
    # Load MATLAB file
    annos = scipy.io.loadmat(train_annos)
    annotations = annos['annotations'][0]
    
    # Extract labels
    labels = {}
    for anno in annotations:
        img_name = anno[5][0]  # filename
        class_id = anno[4][0][0]  # class
        labels[img_name] = f"class_{class_id}"
    
    # Save as JSON
    with open(data_dir / 'train_labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Created labels for {len(labels)} images")

if __name__ == '__main__':
    create_labels()