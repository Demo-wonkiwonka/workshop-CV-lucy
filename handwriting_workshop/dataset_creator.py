"""
Custom dataset creation tool for handwriting recognition.
Allows users to create datasets from handwriting images and organize them properly.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    output_dir: str
    image_size: Tuple[int, int] = (32, 32)
    num_samples_per_class: int = 100
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    background_colors: List[Tuple[int, int, int]] = None
    font_sizes: List[int] = None
    rotation_range: Tuple[float, float] = (-15, 15)
    noise_level: float = 0.1
    
    def __post_init__(self):
        if self.background_colors is None:
            self.background_colors = [(255, 255, 255), (240, 240, 240), (250, 250, 250)]
        if self.font_sizes is None:
            self.font_sizes = [16, 20, 24, 28, 32]


class HandwritingDatasetCreator:
    """Tool for creating custom handwriting datasets."""
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the dataset creator.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Define character sets
        self.character_sets = {
            'uppercase_letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'lowercase_letters': 'abcdefghijklmnopqrstuvwxyz',
            'numbers': '0123456789',
            'symbols': '!@#$%^&*()_+-=[]{}|;:,.<>?',
            'mixed': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        }
        
        self.dataset_info = {
            'config': config.__dict__,
            'classes': [],
            'samples_per_class': {},
            'total_samples': 0
        }
    
    def create_synthetic_dataset(
        self, 
        character_set: str = 'mixed',
        custom_characters: Optional[List[str]] = None
    ) -> None:
        """
        Create a synthetic handwriting dataset.
        
        Args:
            character_set: Which character set to use
            custom_characters: Custom list of characters to use
        """
        if custom_characters:
            characters = custom_characters
        else:
            characters = list(self.character_sets.get(character_set, self.character_sets['mixed']))
        
        print(f"Creating synthetic dataset with {len(characters)} characters...")
        
        # Create class directories
        for char in characters:
            for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
                (split_dir / char).mkdir(exist_ok=True)
        
        # Calculate samples per split
        train_samples = int(self.config.num_samples_per_class * self.config.train_split)
        val_samples = int(self.config.num_samples_per_class * self.config.val_split)
        test_samples = self.config.num_samples_per_class - train_samples - val_samples
        
        # Generate samples
        for char in tqdm(characters, desc="Generating characters"):
            # Generate training samples
            for i in range(train_samples):
                image = self._generate_character_image(char)
                image_path = self.train_dir / char / f"{char}_{i:04d}.png"
                image.save(image_path)
            
            # Generate validation samples
            for i in range(val_samples):
                image = self._generate_character_image(char)
                image_path = self.val_dir / char / f"{char}_{i:04d}.png"
                image.save(image_path)
            
            # Generate test samples
            for i in range(test_samples):
                image = self._generate_character_image(char)
                image_path = self.test_dir / char / f"{char}_{i:04d}.png"
                image.save(image_path)
            
            # Update dataset info
            self.dataset_info['classes'].append(char)
            self.dataset_info['samples_per_class'][char] = self.config.num_samples_per_class
            self.dataset_info['total_samples'] += self.config.num_samples_per_class
        
        # Save dataset info
        self._save_dataset_info()
        print(f"Dataset created successfully in {self.output_dir}")
        print(f"Total samples: {self.dataset_info['total_samples']}")
        print(f"Classes: {len(self.dataset_info['classes'])}")
    
    def _generate_character_image(self, character: str) -> Image.Image:
        """Generate a synthetic character image."""
        # Create image with random background
        bg_color = random.choice(self.config.background_colors)
        image = Image.new('RGB', (64, 64), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Try to use a handwriting-like font
        try:
            # Try different font paths
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "C:/Windows/Fonts/arial.ttf",  # Windows
                "C:/Windows/Fonts/calibri.ttf",  # Windows
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font_size = random.choice(self.config.font_sizes)
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get text size and position
        bbox = draw.textbbox((0, 0), character, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (64 - text_width) // 2
        y = (64 - text_height) // 2
        
        # Add some random offset
        x += random.randint(-5, 5)
        y += random.randint(-5, 5)
        
        # Draw the character
        draw.text((x, y), character, fill=(0, 0, 0), font=font)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to target size
        image = image.resize(self.config.image_size, Image.LANCZOS)
        
        # Add some noise
        if self.config.noise_level > 0:
            image = self._add_noise(image)
        
        # Apply rotation
        if self.config.rotation_range[1] > self.config.rotation_range[0]:
            angle = random.uniform(*self.config.rotation_range)
            image = image.rotate(angle, fillcolor=255, expand=False)
        
        return image
    
    def _add_noise(self, image: Image.Image) -> Image.Image:
        """Add noise to the image."""
        img_array = np.array(image)
        noise = np.random.normal(0, self.config.noise_level * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def create_from_images(
        self, 
        images_dir: str,
        labels_file: Optional[str] = None
    ) -> None:
        """
        Create dataset from existing images.
        
        Args:
            images_dir: Directory containing images
            labels_file: Optional CSV file with image paths and labels
        """
        images_path = Path(images_dir)
        
        if labels_file:
            # Load from CSV file
            df = pd.read_csv(labels_file)
            self._process_csv_dataset(df, images_path)
        else:
            # Load from directory structure
            self._process_directory_dataset(images_path)
        
        # Save dataset info
        self._save_dataset_info()
        print(f"Dataset created successfully from {images_dir}")
    
    def _process_csv_dataset(self, df: pd.DataFrame, images_path: Path) -> None:
        """Process dataset from CSV file."""
        # Shuffle the dataframe
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        total_samples = len(df)
        train_end = int(total_samples * self.config.train_split)
        val_end = train_end + int(total_samples * self.config.val_split)
        
        # Process each split
        splits = [
            ('train', df.iloc[:train_end]),
            ('val', df.iloc[train_end:val_end]),
            ('test', df.iloc[val_end:])
        ]
        
        for split_name, split_df in splits:
            split_dir = getattr(self, f"{split_name}_dir")
            
            for _, row in tqdm(split_df.iterrows(), desc=f"Processing {split_name}", total=len(split_df)):
                image_path = images_path / row['image_path']
                label = str(row['label'])
                
                if image_path.exists():
                    # Create class directory
                    (split_dir / label).mkdir(exist_ok=True)
                    
                    # Load and process image
                    image = Image.open(image_path).convert('L')
                    image = image.resize(self.config.image_size, Image.LANCZOS)
                    
                    # Save processed image
                    output_path = split_dir / label / image_path.name
                    image.save(output_path)
                    
                    # Update dataset info
                    if label not in self.dataset_info['classes']:
                        self.dataset_info['classes'].append(label)
                        self.dataset_info['samples_per_class'][label] = 0
                    
                    self.dataset_info['samples_per_class'][label] += 1
                    self.dataset_info['total_samples'] += 1
    
    def _process_directory_dataset(self, images_path: Path) -> None:
        """Process dataset from directory structure."""
        # Find all class directories
        class_dirs = [d for d in images_path.iterdir() if d.is_dir()]
        
        for class_dir in tqdm(class_dirs, desc="Processing classes"):
            class_name = class_dir.name
            
            # Get all images in the class directory
            image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg'))
            
            # Shuffle images
            random.shuffle(image_files)
            
            # Calculate split indices
            total_images = len(image_files)
            train_end = int(total_images * self.config.train_split)
            val_end = train_end + int(total_images * self.config.val_split)
            
            # Process each split
            splits = [
                ('train', image_files[:train_end]),
                ('val', image_files[train_end:val_end]),
                ('test', image_files[val_end:])
            ]
            
            for split_name, split_images in splits:
                split_dir = getattr(self, f"{split_name}_dir")
                (split_dir / class_name).mkdir(exist_ok=True)
                
                for image_file in split_images:
                    # Load and process image
                    image = Image.open(image_file).convert('L')
                    image = image.resize(self.config.image_size, Image.LANCZOS)
                    
                    # Save processed image
                    output_path = split_dir / class_name / image_file.name
                    image.save(output_path)
                    
                    # Update dataset info
                    if class_name not in self.dataset_info['classes']:
                        self.dataset_info['classes'].append(class_name)
                        self.dataset_info['samples_per_class'][class_name] = 0
                    
                    self.dataset_info['samples_per_class'][class_name] += 1
                    self.dataset_info['total_samples'] += 1
    
    def _save_dataset_info(self) -> None:
        """Save dataset information to JSON file."""
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
    
    def create_sample_sheet(
        self, 
        output_path: str = "handwriting_sample_sheet.png",
        characters: Optional[List[str]] = None
    ) -> None:
        """
        Create a sample sheet for handwriting data collection.
        
        Args:
            output_path: Path to save the sample sheet
            characters: Characters to include in the sheet
        """
        if characters is None:
            characters = list(self.character_sets['mixed'])
        
        # Create a large image for the sample sheet
        sheet_width = 1200
        sheet_height = 1600
        sheet = Image.new('RGB', (sheet_width, sheet_height), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        
        # Try to load a font
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_medium = ImageFont.truetype("arial.ttf", 36)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Title
        title = "Handwriting Sample Collection Sheet"
        title_bbox = draw.textbbox((0, 0), title, font=font_large)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((sheet_width - title_width) // 2, 50), title, fill=(0, 0, 0), font=font_large)
        
        # Instructions
        instructions = [
            "Instructions:",
            "1. Write each character in the space provided",
            "2. Use your natural handwriting style",
            "3. Make sure characters are clearly visible",
            "4. Take a photo of this sheet when complete"
        ]
        
        y_pos = 150
        for instruction in instructions:
            draw.text((50, y_pos), instruction, fill=(0, 0, 0), font=font_medium)
            y_pos += 40
        
        # Character grid
        y_pos += 50
        chars_per_row = 10
        char_size = 80
        spacing = 20
        
        for i, char in enumerate(characters):
            row = i // chars_per_row
            col = i % chars_per_row
            
            x = 50 + col * (char_size + spacing)
            y = y_pos + row * (char_size + spacing)
            
            # Draw character label
            char_bbox = draw.textbbox((0, 0), char, font=font_medium)
            char_width = char_bbox[2] - char_bbox[0]
            draw.text((x + (char_size - char_width) // 2, y - 30), char, fill=(0, 0, 0), font=font_medium)
            
            # Draw box for handwriting
            draw.rectangle([x, y, x + char_size, y + char_size], outline=(0, 0, 0), width=2)
        
        # Save the sheet
        sheet.save(output_path)
        print(f"Sample sheet saved to {output_path}")
    
    def visualize_dataset(self, num_samples: int = 16) -> None:
        """Visualize samples from the created dataset."""
        # Get a sample from each split
        splits = ['train', 'val', 'test']
        fig, axes = plt.subplots(len(splits), 1, figsize=(12, 8))
        
        if len(splits) == 1:
            axes = [axes]
        
        for i, split in enumerate(splits):
            split_dir = getattr(self, f"{split}_dir")
            
            # Get all class directories
            class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            
            if not class_dirs:
                continue
            
            # Get samples from each class
            samples = []
            labels = []
            
            for class_dir in class_dirs[:4]:  # Limit to 4 classes for visualization
                class_name = class_dir.name
                image_files = list(class_dir.glob('*.png'))[:4]  # 4 samples per class
                
                for img_file in image_files:
                    img = Image.open(img_file)
                    samples.append(np.array(img))
                    labels.append(class_name)
            
            # Create subplot
            cols = 4
            rows = len(samples) // cols + (1 if len(samples) % cols > 0 else 0)
            
            for j, (img, label) in enumerate(zip(samples, labels)):
                if j < 16:  # Limit to 16 samples
                    row = j // cols
                    col = j % cols
                    
                    if row < rows:
                        ax = axes[i].inset_axes([col * 0.25, 1 - (row + 1) * 0.25, 0.2, 0.2])
                        ax.imshow(img, cmap='gray')
                        ax.set_title(label, fontsize=8)
                        ax.axis('off')
            
            axes[i].set_title(f'{split.capitalize()} Split ({len(samples)} samples)')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the created dataset."""
        stats = {
            'total_samples': self.dataset_info['total_samples'],
            'num_classes': len(self.dataset_info['classes']),
            'classes': self.dataset_info['classes'],
            'samples_per_class': self.dataset_info['samples_per_class'],
            'splits': {}
        }
        
        # Count samples in each split
        for split in ['train', 'val', 'test']:
            split_dir = getattr(self, f"{split}_dir")
            split_samples = 0
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    split_samples += len(list(class_dir.glob('*.png')))
            
            stats['splits'][split] = split_samples
        
        return stats


def create_handwriting_dataset(
    output_dir: str,
    dataset_type: str = "synthetic",
    **kwargs
) -> HandwritingDatasetCreator:
    """
    Convenience function to create a handwriting dataset.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_type: Type of dataset ("synthetic" or "from_images")
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        HandwritingDatasetCreator instance
    """
    config = DatasetConfig(output_dir=output_dir, **kwargs)
    creator = HandwritingDatasetCreator(config)
    
    if dataset_type == "synthetic":
        character_set = kwargs.get('character_set', 'mixed')
        custom_characters = kwargs.get('custom_characters', None)
        creator.create_synthetic_dataset(character_set, custom_characters)
    elif dataset_type == "from_images":
        images_dir = kwargs.get('images_dir')
        labels_file = kwargs.get('labels_file')
        if images_dir:
            creator.create_from_images(images_dir, labels_file)
        else:
            raise ValueError("images_dir must be provided for 'from_images' dataset type")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return creator


# Example usage and utilities
def create_sample_workshop_dataset() -> HandwritingDatasetCreator:
    """Create a sample dataset for workshop demonstration."""
    config = DatasetConfig(
        output_dir="workshop_dataset",
        image_size=(32, 32),
        num_samples_per_class=50,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    creator = HandwritingDatasetCreator(config)
    
    # Create dataset with basic characters
    basic_characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    creator.create_synthetic_dataset(custom_characters=basic_characters)
    
    # Create sample sheet
    creator.create_sample_sheet("workshop_sample_sheet.png", basic_characters)
    
    return creator
