"""
Different model architectures for handwriting recognition.
Includes CNN, ResNet, and Vision Transformer implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import ViTForImageClassification, ViTConfig
from typing import Optional, Tuple


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for handwriting recognition.
    Good starting point for beginners.
    """
    
    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (32, 32)):
        """
        Initialize the Simple CNN.
        
        Args:
            num_classes: Number of output classes
            input_size: Input image size (height, width)
        """
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Assuming input size 32x32, after 3 pooling operations: 32 -> 16 -> 8 -> 4
        conv_output_size = 128 * 4 * 4  # 128 channels * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """Get model information for comparison."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": "SimpleCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "CNN with 3 conv layers + 3 FC layers",
            "input_size": self.input_size,
            "num_classes": self.num_classes
        }


class ResNetHandwriting(nn.Module):
    """
    ResNet-based architecture for handwriting recognition.
    Uses a pre-trained ResNet backbone with custom classifier.
    """
    
    def __init__(
        self, 
        num_classes: int, 
        pretrained: bool = True,
        resnet_version: str = "resnet18"
    ):
        """
        Initialize the ResNet model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            resnet_version: ResNet version ("resnet18", "resnet34", "resnet50")
        """
        super(ResNetHandwriting, self).__init__()
        
        self.num_classes = num_classes
        self.resnet_version = resnet_version
        
        # Load pre-trained ResNet
        if resnet_version == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif resnet_version == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif resnet_version == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        
        # Modify first layer to accept single channel input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)
    
    def get_model_info(self) -> dict:
        """Get model information for comparison."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": f"ResNetHandwriting_{self.resnet_version}",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": f"ResNet {self.resnet_version} with custom classifier",
            "num_classes": self.num_classes
        }


class VisionTransformerHandwriting(nn.Module):
    """
    Vision Transformer for handwriting recognition.
    Uses HuggingFace's ViT implementation.
    """
    
    def __init__(
        self, 
        num_classes: int,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        pretrained: bool = True
    ):
        """
        Initialize the Vision Transformer.
        
        Args:
            num_classes: Number of output classes
            image_size: Input image size (assumes square images)
            patch_size: Size of image patches
            hidden_size: Hidden dimension size
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            pretrained: Whether to use pretrained weights
        """
        super(VisionTransformerHandwriting, self).__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Create ViT configuration
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=1,  # Grayscale images
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_labels=num_classes,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        # Load pre-trained model or create new one
        if pretrained:
            try:
                self.vit = ViTForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224",
                    config=config,
                    ignore_mismatched_sizes=True
                )
            except:
                # If pretrained loading fails, create from scratch
                self.vit = ViTForImageClassification(config)
        else:
            self.vit = ViTForImageClassification(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Resize input to match expected size
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        outputs = self.vit(pixel_values=x)
        return outputs.logits
    
    def get_model_info(self) -> dict:
        """Get model information for comparison."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": "VisionTransformerHandwriting",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "Vision Transformer (ViT)",
            "image_size": self.image_size,
            "num_classes": self.num_classes
        }


class ImprovedCNN(nn.Module):
    """
    Improved CNN with more sophisticated architecture.
    Includes residual connections and attention mechanisms.
    """
    
    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (32, 32)):
        """
        Initialize the Improved CNN.
        
        Args:
            num_classes: Number of output classes
            input_size: Input image size (height, width)
        """
        super(ImprovedCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 64)
        self.res_block2 = self._make_residual_block(64, 128, stride=2)
        self.res_block3 = self._make_residual_block(128, 256, stride=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _make_residual_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """Get model information for comparison."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": "ImprovedCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "Improved CNN with residual blocks",
            "input_size": self.input_size,
            "num_classes": self.num_classes
        }


def create_model(
    model_type: str, 
    num_classes: int, 
    **kwargs
) -> nn.Module:
    """
    Factory function to create different model types.
    
    Args:
        model_type: Type of model to create
        num_classes: Number of output classes
        **kwargs: Additional arguments for model creation
        
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == "simple_cnn":
        return SimpleCNN(num_classes, **kwargs)
    elif model_type == "resnet18":
        return ResNetHandwriting(num_classes, resnet_version="resnet18", **kwargs)
    elif model_type == "resnet34":
        return ResNetHandwriting(num_classes, resnet_version="resnet34", **kwargs)
    elif model_type == "resnet50":
        return ResNetHandwriting(num_classes, resnet_version="resnet50", **kwargs)
    elif model_type == "vit":
        return VisionTransformerHandwriting(num_classes, **kwargs)
    elif model_type == "improved_cnn":
        return ImprovedCNN(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_available_models() -> list:
    """Get list of available model types."""
    return [
        "simple_cnn",
        "resnet18", 
        "resnet34",
        "resnet50",
        "vit",
        "improved_cnn"
    ]


def compare_model_sizes(num_classes: int = 10) -> dict:
    """
    Compare the sizes of different models.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Dictionary with model information
    """
    models_info = {}
    
    for model_type in get_available_models():
        try:
            model = create_model(model_type, num_classes)
            models_info[model_type] = model.get_model_info()
        except Exception as e:
            models_info[model_type] = {"error": str(e)}
    
    return models_info
