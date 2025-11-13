"""Create human-readable labels for Stanford Cars dataset."""

import scipy.io
import json
from pathlib import Path


def create_readable_labels():
    """Create labels.json with actual car names."""
    
    # Paths
    data_dir = Path('data/stanford_cars')
    devkit_dir = data_dir / 'devkit'
    
    # Check if files exist
    train_annos = devkit_dir / 'cars_train_annos.mat'
    test_annos = devkit_dir / 'cars_test_annos.mat'  
    cars_meta = devkit_dir / 'cars_meta.mat'
    
    if not train_annos.exists():
        print(f"ERROR: Cannot find {train_annos}")
        print("\nExpected structure:")
        print("data/stanford_cars/")
        print("├── cars_train/")
        print("│   ├── 00001.jpg")
        print("│   └── ...")
        print("├── cars_test/")
        print("│   ├── 00001.jpg")
        print("│   └── ...")
        print("└── devkit/")
        print("    ├── cars_train_annos.mat")
        print("    ├── cars_test_annos.mat")
        print("    └── cars_meta.mat")
        return
    
    print("Loading Stanford Cars metadata...")
    
    # Load class names (meta data)
    meta = scipy.io.loadmat(cars_meta)
    class_names = meta['class_names'][0]
    
    # Convert to readable format
    # class_names is array of arrays, extract strings
    class_id_to_name = {}
    for i, name_array in enumerate(class_names):
        class_name = name_array[0]
        class_id_to_name[i + 1] = class_name  # Classes are 1-indexed
    
    print(f"Loaded {len(class_id_to_name)} class names")
    print("\nExample classes:")
    for i in range(1, 6):
        print(f"  Class {i}: {class_id_to_name[i]}")
    
    # Process training annotations
    print("\nProcessing training annotations...")
    train_labels = process_annotations(train_annos, class_id_to_name)
    
    # Save training labels
    train_labels_file = data_dir / 'train_labels.json'
    with open(train_labels_file, 'w') as f:
        json.dump(train_labels, f, indent=2)
    print(f"✓ Saved {len(train_labels)} training labels to: {train_labels_file}")
    
    # Process test annotations (if available)
    if test_annos.exists():
        print("\nProcessing test annotations...")
        test_labels = process_annotations(test_annos, class_id_to_name)
        
        test_labels_file = data_dir / 'test_labels.json'
        with open(test_labels_file, 'w') as f:
            json.dump(test_labels, f, indent=2)
        print(f"✓ Saved {len(test_labels)} test labels to: {test_labels_file}")
    
    # Also create hierarchical labels (for advanced training)
    print("\nCreating hierarchical labels...")
    hierarchical_labels = create_hierarchical_labels(train_labels, class_id_to_name)
    
    hierarchical_file = data_dir / 'train_hierarchical_labels.json'
    with open(hierarchical_file, 'w') as f:
        json.dump(hierarchical_labels, f, indent=2)
    print(f"✓ Saved hierarchical labels to: {hierarchical_file}")
    
    print("\n" + "="*60)
    print("✓ All done! Labels created successfully.")
    print("="*60)


def process_annotations(annos_file, class_id_to_name):
    """Process MATLAB annotations file."""
    annos = scipy.io.loadmat(annos_file)
    annotations = annos['annotations'][0]
    
    labels = {}
    for anno in annotations:
        # MATLAB structure varies between train and test
        # Try to handle both formats
        try:
            # Format 1: [bbox_x1, bbox_y1, bbox_x2, bbox_y2, class, fname]
            if len(anno) >= 6:
                img_name = anno[5][0]  # filename
                class_id = int(anno[4][0][0])  # class
            # Format 2: Different structure
            else:
                img_name = anno[-1][0]  # filename (last field)
                class_id = int(anno[-2][0][0])  # class (second to last)
        except:
            print(f"Warning: Could not parse annotation: {anno}")
            continue
        
        # Get readable class name
        class_name = class_id_to_name.get(class_id, f"unknown_class_{class_id}")
        labels[img_name] = class_name
    
    return labels


def create_hierarchical_labels(labels, class_id_to_name):
    """
    Create hierarchical labels (make, model, type, year).
    
    Stanford Cars class names are formatted like:
    "AM General Hummer SUV 2000"
    "Acura Integra Type R 2001"
    "Audi R8 Coupe 2012"
    """
    hierarchical = {}
    
    for img_name, full_label in labels.items():
        # Parse the label
        # Format: "Make Model Type Year" or "Make Model Year"
        parts = full_label.split()
        
        if len(parts) >= 3:
            # Try to extract components
            year = parts[-1]  # Last part is year
            
            # Type might be: Coupe, Sedan, SUV, Convertible, etc.
            vehicle_types = ['Coupe', 'Sedan', 'SUV', 'Convertible', 'Wagon', 
                           'Hatchback', 'Minivan', 'Pickup', 'Van', 'Cab']
            
            vehicle_type = 'Unknown'
            for vtype in vehicle_types:
                if vtype in parts:
                    vehicle_type = vtype
                    break
            
            # Make is usually first 1-2 words
            if parts[0] == 'AM' and parts[1] == 'General':
                make = 'AM General'
                model_parts = parts[2:-1]
            elif parts[0] == 'Aston' and parts[1] == 'Martin':
                make = 'Aston Martin'
                model_parts = parts[2:-1]
            else:
                make = parts[0]
                model_parts = parts[1:-1]
            
            # Model is everything between make and year (excluding type)
            model = ' '.join([p for p in model_parts if p not in vehicle_types])
            
            hierarchical[img_name] = {
                'make': make,
                'type': vehicle_type,
                'model': model,
                'year': year,
                'full_label': full_label
            }
        else:
            # Fallback for unusual formats
            hierarchical[img_name] = {
                'make': 'Unknown',
                'type': 'Unknown',
                'model': 'Unknown',
                'year': 'Unknown',
                'full_label': full_label
            }
    
    return hierarchical


def show_sample_labels():
    """Show sample of created labels."""
    data_dir = Path('data/stanford_cars')
    
    # Load and display samples
    files_to_check = [
        ('train_labels.json', 'Training Labels'),
        ('train_hierarchical_labels.json', 'Hierarchical Labels')
    ]
    
    for filename, title in files_to_check:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\n{title} Sample:")
            print("-" * 60)
            with open(filepath, 'r') as f:
                labels = json.load(f)
                # Show first 5 entries
                for i, (img_name, label) in enumerate(list(labels.items())[:5]):
                    print(f"{img_name}: {label}")
                    if i >= 4:
                        break
            print(f"... ({len(labels)} total entries)")


if __name__ == '__main__':
    create_readable_labels()
    show_sample_labels()
