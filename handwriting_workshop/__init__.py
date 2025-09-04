"""
Handwriting Recognition Workshop
A comprehensive workshop for training and comparing different vision models
to recognize handwritten text and symbols.
"""

__version__ = "1.0.0"
__author__ = "Workshop Instructor"

from .data_loader import HandwritingDataLoader
from .models import SimpleCNN, ResNetHandwriting, VisionTransformerHandwriting
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .dataset_creator import HandwritingDatasetCreator

__all__ = [
    "HandwritingDataLoader",
    "SimpleCNN", 
    "ResNetHandwriting",
    "VisionTransformerHandwriting",
    "ModelTrainer",
    "ModelEvaluator",
    "HandwritingDatasetCreator"
]
