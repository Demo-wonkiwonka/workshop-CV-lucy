"""
Main workshop notebook for handwriting recognition.
This script demonstrates the complete workflow from dataset creation to model comparison.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from .data_loader import HandwritingDataLoader, load_sample_dataset
from .models import create_model, get_available_models, compare_model_sizes
from .trainer import ModelTrainer, MultiModelTrainer
from .evaluator import ModelEvaluator, ModelComparison
from .dataset_creator import HandwritingDatasetCreator, DatasetConfig, create_handwriting_dataset


class HandwritingWorkshop:
    """Main workshop class that orchestrates the entire workflow."""
    
    def __init__(self, workshop_dir: str = "handwriting_workshop_results"):
        """
        Initialize the workshop.
        
        Args:
            workshop_dir: Directory to save workshop results
        """
        self.workshop_dir = Path(workshop_dir)
        self.workshop_dir.mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Workshop initialized on device: {self.device}")
        
        # Workshop state
        self.dataset = None
        self.data_loaders = None
        self.models = {}
        self.trainers = {}
        self.evaluators = {}
        self.results = {}
    
    def step1_create_dataset(self, dataset_type: str = "synthetic") -> None:
        """
        Step 1: Create or load a dataset.
        
        Args:
            dataset_type: Type of dataset to create ("synthetic" or "sample")
        """
        print("\n" + "="*60)
        print("STEP 1: CREATING DATASET")
        print("="*60)
        
        if dataset_type == "synthetic":
            # Create a synthetic dataset
            print("Creating synthetic handwriting dataset...")
            creator = create_handwriting_dataset(
                output_dir=str(self.workshop_dir / "synthetic_dataset"),
                dataset_type="synthetic",
                character_set="mixed",
                num_samples_per_class=100,
                image_size=(32, 32)
            )
            
            # Load the created dataset
            data_loader = HandwritingDataLoader(image_size=(32, 32))
            self.dataset, self.label_to_idx = data_loader.load_custom_dataset(
                str(self.workshop_dir / "synthetic_dataset")
            )
            
        elif dataset_type == "sample":
            # Load sample dataset for demonstration
            print("Loading sample dataset...")
            self.dataset, self.label_to_idx = load_sample_dataset()
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        print(f"Dataset loaded successfully!")
        print(f"Number of classes: {self.dataset.num_classes}")
        print(f"Number of samples: {len(self.dataset)}")
        print(f"Classes: {list(self.label_to_idx.keys())}")
    
    def step2_prepare_data(self, batch_size: int = 32) -> None:
        """
        Step 2: Prepare data loaders.
        
        Args:
            batch_size: Batch size for data loaders
        """
        print("\n" + "="*60)
        print("STEP 2: PREPARING DATA LOADERS")
        print("="*60)
        
        data_loader = HandwritingDataLoader(image_size=(32, 32))
        self.data_loaders = data_loader.create_data_loaders(
            self.dataset,
            batch_size=batch_size,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15
        )
        
        train_loader, val_loader, test_loader = self.data_loaders
        
        print(f"Data loaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
    
    def step3_explore_models(self) -> None:
        """
        Step 3: Explore available model architectures.
        """
        print("\n" + "="*60)
        print("STEP 3: EXPLORING MODEL ARCHITECTURES")
        print("="*60)
        
        # Get available models
        available_models = get_available_models()
        print(f"Available model types: {available_models}")
        
        # Compare model sizes
        print("\nModel size comparison:")
        model_info = compare_model_sizes(num_classes=self.dataset.num_classes)
        
        for model_type, info in model_info.items():
            if 'error' not in info:
                print(f"{model_type:15} | Parameters: {info['total_parameters']:>8,} | "
                      f"Architecture: {info['architecture']}")
            else:
                print(f"{model_type:15} | Error: {info['error']}")
    
    def step4_train_models(
        self, 
        model_types: list = None,
        epochs: int = 5,
        learning_rate: float = 0.001
    ) -> None:
        """
        Step 4: Train multiple models.
        
        Args:
            model_types: List of model types to train
            epochs: Number of epochs to train
            learning_rate: Learning rate for training
        """
        print("\n" + "="*60)
        print("STEP 4: TRAINING MODELS")
        print("="*60)
        
        if model_types is None:
            model_types = ["simple_cnn", "resnet18", "improved_cnn"]
        
        train_loader, val_loader, test_loader = self.data_loaders
        
        # Create model configurations
        models_config = []
        for model_type in model_types:
            models_config.append({
                'name': model_type,
                'type': model_type,
                'num_classes': self.dataset.num_classes,
                'training_kwargs': {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'save_best': True,
                    'early_stopping_patience': 5
                }
            })
        
        # Train models using MultiModelTrainer
        multi_trainer = MultiModelTrainer(
            save_dir=str(self.workshop_dir / "trained_models")
        )
        
        self.results = multi_trainer.train_multiple_models(
            models_config=models_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Store trainers for later use
        self.multi_trainer = multi_trainer
        
        print("\nTraining completed for all models!")
        multi_trainer.compare_models()
    
    def step5_evaluate_models(self) -> None:
        """
        Step 5: Comprehensive model evaluation.
        """
        print("\n" + "="*60)
        print("STEP 5: EVALUATING MODELS")
        print("="*60)
        
        train_loader, val_loader, test_loader = self.data_loaders
        
        # Create model comparison
        model_comparison = ModelComparison()
        
        # Evaluate each model
        for model_name, results in self.results.items():
            print(f"\nEvaluating {model_name}...")
            
            # Load the best model
            model_path = self.workshop_dir / "trained_models" / model_name / "best_model_epoch_*.pt"
            model_files = list(self.workshop_dir.glob(f"trained_models/{model_name}/best_model_*.pt"))
            
            if model_files:
                # Create model
                model = create_model(
                    model_name.split('_')[0] if '_' in model_name else model_name,
                    self.dataset.num_classes
                )
                
                # Create trainer and load model
                trainer = ModelTrainer(model, save_dir=str(self.workshop_dir / "trained_models" / model_name))
                trainer.load_model(model_files[0])
                
                # Create evaluator
                evaluator = ModelEvaluator(model)
                evaluator.evaluate_dataset(test_loader, list(self.label_to_idx.keys()))
                
                # Add to comparison
                model_comparison.add_model_results(model_name, evaluator)
                
                # Generate individual reports
                report_path = self.workshop_dir / f"{model_name}_evaluation_report.txt"
                report = evaluator.generate_report(report_path)
                
                # Plot confusion matrix
                cm_path = self.workshop_dir / f"{model_name}_confusion_matrix.png"
                evaluator.plot_confusion_matrix(cm_path)
                
                print(f"  Test Accuracy: {evaluator._calculate_metrics()['accuracy']:.4f}")
        
        # Generate comparison report
        comparison_report_path = self.workshop_dir / "model_comparison_report.txt"
        comparison_report = model_comparison.generate_comparison_report(comparison_report_path)
        
        # Plot comparison
        comparison_plot_path = self.workshop_dir / "model_comparison_plot.png"
        model_comparison.plot_comparison(comparison_plot_path)
        
        print("\nModel evaluation completed!")
        print(f"Results saved to: {self.workshop_dir}")
    
    def step6_create_custom_dataset(self) -> None:
        """
        Step 6: Demonstrate custom dataset creation.
        """
        print("\n" + "="*60)
        print("STEP 6: CREATING CUSTOM DATASET")
        print("="*60)
        
        # Create a custom dataset creator
        config = DatasetConfig(
            output_dir=str(self.workshop_dir / "custom_dataset"),
            image_size=(32, 32),
            num_samples_per_class=50,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15
        )
        
        creator = HandwritingDatasetCreator(config)
        
        # Create synthetic dataset with custom characters
        custom_characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        creator.create_synthetic_dataset(custom_characters=custom_characters)
        
        # Create sample sheet for handwriting collection
        sample_sheet_path = self.workshop_dir / "handwriting_sample_sheet.png"
        creator.create_sample_sheet(str(sample_sheet_path), custom_characters)
        
        # Visualize the dataset
        creator.visualize_dataset()
        
        # Get dataset statistics
        stats = creator.get_dataset_stats()
        print(f"\nCustom dataset statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Number of classes: {stats['num_classes']}")
        print(f"  Train samples: {stats['splits']['train']}")
        print(f"  Validation samples: {stats['splits']['val']}")
        print(f"  Test samples: {stats['splits']['test']}")
        
        print(f"\nSample sheet created at: {sample_sheet_path}")
        print("You can print this sheet and collect real handwriting samples!")
    
    def run_complete_workshop(self) -> None:
        """Run the complete workshop workflow."""
        print("HANDWRITING RECOGNITION WORKSHOP")
        print("="*60)
        print("This workshop will guide you through:")
        print("1. Creating/loading a dataset")
        print("2. Preparing data loaders")
        print("3. Exploring model architectures")
        print("4. Training multiple models")
        print("5. Evaluating and comparing models")
        print("6. Creating custom datasets")
        print("="*60)
        
        try:
            # Step 1: Create dataset
            self.step1_create_dataset("synthetic")
            
            # Step 2: Prepare data
            self.step2_prepare_data()
            
            # Step 3: Explore models
            self.step3_explore_models()
            
            # Step 4: Train models
            self.step4_train_models(epochs=3)  # Reduced epochs for demo
            
            # Step 5: Evaluate models
            self.step5_evaluate_models()
            
            # Step 6: Create custom dataset
            self.step6_create_custom_dataset()
            
            print("\n" + "="*60)
            print("WORKSHOP COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"All results saved to: {self.workshop_dir}")
            print("\nNext steps:")
            print("1. Review the model comparison results")
            print("2. Print the sample sheet and collect real handwriting")
            print("3. Experiment with different model architectures")
            print("4. Try different hyperparameters")
            print("5. Create your own custom datasets")
            
        except Exception as e:
            print(f"\nWorkshop encountered an error: {e}")
            print("Please check the error and try again.")


def run_quick_demo() -> None:
    """Run a quick demonstration of the workshop."""
    print("HANDWRITING RECOGNITION - QUICK DEMO")
    print("="*50)
    
    # Create workshop instance
    workshop = HandwritingWorkshop("quick_demo_results")
    
    # Run a simplified version
    workshop.step1_create_dataset("sample")
    workshop.step2_prepare_data(batch_size=16)
    workshop.step3_explore_models()
    
    # Train just one model for demo
    workshop.step4_train_models(model_types=["simple_cnn"], epochs=2)
    
    print("\nQuick demo completed!")
    print("Run workshop.run_complete_workshop() for the full experience.")


if __name__ == "__main__":
    # Run the workshop
    workshop = HandwritingWorkshop()
    workshop.run_complete_workshop()
