"""
Data loading utilities for handwriting recognition datasets.
Supports loading from HuggingFace datasets and custom image datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import io
from pathlib import Path


class HandwritingDataset(Dataset):
    """Custom dataset class for handwriting recognition."""
    
    def __init__(
        self, 
        images: List[Image.Image], 
        labels: List[str], 
        transform: Optional[transforms.Compose] = None,
        label_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            images: List of PIL Images
            labels: List of corresponding labels
            transform: Optional transforms to apply
            label_to_idx: Optional mapping from labels to indices
        """
        self.images = images
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Create label mapping if not provided
        if label_to_idx is None:
            unique_labels = sorted(list(set(labels)))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.label_to_idx[label]
        return image, label_idx


class HandwritingDataLoader:
    """Main class for loading and preprocessing handwriting datasets."""
    
    def __init__(self, image_size: Tuple[int, int] = (32, 32)):
        """
        Initialize the data loader.
        
        Args:
            image_size: Target size for images (height, width)
        """
        self.image_size = image_size
        self.transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.transform_val = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def load_huggingface_dataset(
        self, 
        dataset_name: str = "timbrooks/instruct-pix2pix",
        subset: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> Tuple[HandwritingDataset, Dict[str, int]]:
        """
        Load a dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            subset: Optional subset of the dataset
            split: Dataset split to load
            max_samples: Maximum number of samples to load
            
        Returns:
            Tuple of (dataset, label_to_idx mapping)
        """
        print(f"Loading dataset: {dataset_name}")
        
        # Load the dataset
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        # Convert to our format
        images = []
        labels = []
        
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
                
            # Handle different dataset formats
            if 'image' in item:
                image = item['image']
                if isinstance(image, dict) and 'bytes' in image:
                    # Handle datasets with byte data
                    image = Image.open(io.BytesIO(image['bytes']))
                elif not isinstance(image, Image.Image):
                    image = Image.fromarray(np.array(image))
                
                # Convert to grayscale if needed
                if image.mode != 'L':
                    image = image.convert('L')
                
                images.append(image)
                
                # Extract label
                if 'label' in item:
                    labels.append(str(item['label']))
                elif 'text' in item:
                    labels.append(str(item['text']))
                else:
                    labels.append(f"class_{i % 10}")  # Fallback labels
        
        # Create dataset
        handwriting_dataset = HandwritingDataset(
            images=images,
            labels=labels,
            transform=self.transform_val
        )
        
        print(f"Loaded {len(handwriting_dataset)} samples with {handwriting_dataset.num_classes} classes")
        return handwriting_dataset, handwriting_dataset.label_to_idx
    
    def load_custom_dataset(
        self, 
        data_dir: str,
        label_file: Optional[str] = None
    ) -> Tuple[HandwritingDataset, Dict[str, int]]:
        """
        Load a custom dataset from a directory.
        
        Args:
            data_dir: Directory containing images
            label_file: Optional CSV file with image paths and labels
            
        Returns:
            Tuple of (dataset, label_to_idx mapping)
        """
        data_path = Path(data_dir)
        images = []
        labels = []
        
        if label_file:
            # Load from CSV file
            import pandas as pd
            df = pd.read_csv(label_file)
            
            for _, row in df.iterrows():
                image_path = data_path / row['image_path']
                if image_path.exists():
                    image = Image.open(image_path).convert('L')
                    images.append(image)
                    labels.append(str(row['label']))
        else:
            # Load from directory structure (class folders)
            for class_dir in data_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for image_file in class_dir.glob('*.png'):
                        image = Image.open(image_file).convert('L')
                        images.append(image)
                        labels.append(class_name)
        
        # Create dataset
        handwriting_dataset = HandwritingDataset(
            images=images,
            labels=labels,
            transform=self.transform_val
        )
        
        print(f"Loaded {len(handwriting_dataset)} samples with {handwriting_dataset.num_classes} classes")
        return handwriting_dataset, handwriting_dataset.label_to_idx
    
    def create_data_loaders(
        self, 
        dataset: HandwritingDataset,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            dataset: The dataset to split
            batch_size: Batch size for data loaders
            train_split: Fraction for training set
            val_split: Fraction for validation set
            test_split: Fraction for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Apply different transforms to train set
        train_dataset.dataset.transform = self.transform_train
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Created data loaders:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def get_sample_batch(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch from a data loader."""
        for batch in data_loader:
            return batch
        raise ValueError("Data loader is empty")


# Example usage and popular handwriting datasets
POPULAR_DATASETS = {
    "emnist": "timbrooks/instruct-pix2pix",  # Example - replace with actual EMNIST dataset
    "iam_handwriting": "timbrooks/instruct-pix2pix",  # Example - replace with actual IAM dataset
    "cvl_database": "timbrooks/instruct-pix2pix",  # Example - replace with actual CVL dataset
}


def load_sample_dataset() -> Tuple[HandwritingDataset, Dict[str, int]]:
    """Load a sample dataset for testing purposes."""
    data_loader = HandwritingDataLoader()
    
    # For demo purposes, we'll create a simple synthetic dataset
    # In practice, you would load from HuggingFace or custom data
    images = []
    labels = []
    
    # Create some synthetic images (in practice, load real handwriting images)
    for i in range(100):
        # Create a simple synthetic image
        img_array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        image = Image.fromarray(img_array, mode='L')
        images.append(image)
        labels.append(f"class_{i % 10}")
    
    dataset = HandwritingDataset(images, labels)
    return dataset, dataset.label_to_idx
