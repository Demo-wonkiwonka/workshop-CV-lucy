"""
Example usage of the Handwriting Recognition Workshop.
This script demonstrates how to use the workshop components individually.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import workshop components
from handwriting_workshop.data_loader import HandwritingDataLoader, load_sample_dataset
from handwriting_workshop.models import create_model, get_available_models
from handwriting_workshop.trainer import ModelTrainer
from handwriting_workshop.evaluator import ModelEvaluator
from handwriting_workshop.dataset_creator import create_handwriting_dataset


def example_1_basic_usage():
    """Example 1: Basic usage with sample dataset."""
    print("="*60)
    print("EXAMPLE 1: BASIC USAGE")
    print("="*60)
    
    # Load sample dataset
    dataset, label_to_idx = load_sample_dataset()
    print(f"Loaded dataset with {len(dataset)} samples and {dataset.num_classes} classes")
    
    # Create data loaders
    data_loader = HandwritingDataLoader()
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(
        dataset, batch_size=16, train_split=0.8, val_split=0.1, test_split=0.1
    )
    
    # Create and train a simple model
    model = create_model("simple_cnn", num_classes=dataset.num_classes)
    trainer = ModelTrainer(model)
    
    # Train for a few epochs
    history = trainer.train(train_loader, val_loader, epochs=3)
    
    # Evaluate the model
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate_dataset(test_loader, list(label_to_idx.keys()))
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    # Plot training history
    trainer.plot_training_history()


def example_2_model_comparison():
    """Example 2: Compare different model architectures."""
    print("\n" + "="*60)
    print("EXAMPLE 2: MODEL COMPARISON")
    print("="*60)
    
    # Load dataset
    dataset, label_to_idx = load_sample_dataset()
    data_loader = HandwritingDataLoader()
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(dataset, batch_size=16)
    
    # Compare different models
    model_types = ["simple_cnn", "improved_cnn"]
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        
        # Create and train model
        model = create_model(model_type, num_classes=dataset.num_classes)
        trainer = ModelTrainer(model)
        trainer.train(train_loader, val_loader, epochs=2)
        
        # Evaluate model
        evaluator = ModelEvaluator(model)
        test_results = evaluator.evaluate_dataset(test_loader, list(label_to_idx.keys()))
        results[model_type] = test_results['accuracy']
        
        print(f"{model_type} Test Accuracy: {test_results['accuracy']:.4f}")
    
    # Print comparison
    print(f"\nModel Comparison:")
    for model_type, accuracy in results.items():
        print(f"  {model_type}: {accuracy:.4f}")


def example_3_custom_dataset():
    """Example 3: Create and use a custom dataset."""
    print("\n" + "="*60)
    print("EXAMPLE 3: CUSTOM DATASET CREATION")
    print("="*60)
    
    # Create a custom synthetic dataset
    creator = create_handwriting_dataset(
        output_dir="example_custom_dataset",
        dataset_type="synthetic",
        character_set="numbers",  # Just numbers for this example
        num_samples_per_class=50,
        image_size=(32, 32)
    )
    
    # Create a sample sheet
    creator.create_sample_sheet(
        "example_sample_sheet.png",
        characters=list("0123456789")
    )
    
    print("Custom dataset created!")
    print("Check 'example_custom_dataset' folder for the dataset")
    print("Check 'example_sample_sheet.png' for the sample sheet")
    
    # Visualize the dataset
    creator.visualize_dataset()


def example_4_advanced_evaluation():
    """Example 4: Advanced model evaluation."""
    print("\n" + "="*60)
    print("EXAMPLE 4: ADVANCED EVALUATION")
    print("="*60)
    
    # Load dataset and train a model
    dataset, label_to_idx = load_sample_dataset()
    data_loader = HandwritingDataLoader()
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(dataset, batch_size=16)
    
    # Train model
    model = create_model("simple_cnn", num_classes=dataset.num_classes)
    trainer = ModelTrainer(model)
    trainer.train(train_loader, val_loader, epochs=3)
    
    # Comprehensive evaluation
    evaluator = ModelEvaluator(model)
    evaluator.evaluate_dataset(test_loader, list(label_to_idx.keys()))
    
    # Generate various plots and reports
    print("Generating evaluation plots...")
    
    # Confusion matrix
    evaluator.plot_confusion_matrix()
    
    # Per-class performance
    evaluator.plot_class_performance()
    
    # ROC curves
    evaluator.plot_roc_curves()
    
    # Generate text report
    report = evaluator.generate_report()
    print("\nEvaluation Report:")
    print(report)
    
    # Analyze misclassifications
    misclassifications = evaluator.analyze_misclassifications(test_loader, num_examples=5)
    if misclassifications:
        print(f"\nFound {len(misclassifications['indices'])} misclassified examples")
        for i, (true_label, pred_label) in enumerate(zip(
            misclassifications['true_labels'], 
            misclassifications['predicted_labels']
        )):
            print(f"  Example {i+1}: True={true_label}, Predicted={pred_label}")


def example_5_model_info():
    """Example 5: Explore model information and sizes."""
    print("\n" + "="*60)
    print("EXAMPLE 5: MODEL INFORMATION")
    print("="*60)
    
    # Get available models
    available_models = get_available_models()
    print(f"Available models: {available_models}")
    
    # Compare model sizes
    print("\nModel size comparison:")
    model_info = create_model("simple_cnn", num_classes=10).get_model_info()
    print(f"Simple CNN: {model_info['total_parameters']:,} parameters")
    
    # Try different models
    for model_type in ["simple_cnn", "improved_cnn"]:
        try:
            model = create_model(model_type, num_classes=10)
            info = model.get_model_info()
            print(f"{model_type}: {info['total_parameters']:,} parameters")
        except Exception as e:
            print(f"{model_type}: Error - {e}")


def run_all_examples():
    """Run all examples."""
    print("HANDWRITING RECOGNITION WORKSHOP - EXAMPLES")
    print("="*60)
    print("This script demonstrates various features of the workshop.")
    print("Each example is independent and can be run separately.")
    print("="*60)
    
    try:
        example_1_basic_usage()
        example_2_model_comparison()
        example_3_custom_dataset()
        example_4_advanced_evaluation()
        example_5_model_info()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the generated files and plots.")
        print("You can now explore the workshop components in more detail.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This might be due to missing dependencies or system requirements.")
        print("Please check the installation and try again.")


if __name__ == "__main__":
    # Run all examples
    run_all_examples()
    
    # Or run individual examples:
    # example_1_basic_usage()
    # example_2_model_comparison()
    # example_3_custom_dataset()
    # example_4_advanced_evaluation()
    # example_5_model_info()
