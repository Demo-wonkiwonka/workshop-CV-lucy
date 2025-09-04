"""
Model evaluation and visualization tools for handwriting recognition.
Includes confusion matrices, classification reports, and detailed analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: The model to evaluate
            device: Device to use for evaluation
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.class_names = []
    
    def evaluate_dataset(
        self, 
        data_loader: DataLoader, 
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            class_names: Optional list of class names
            
        Returns:
            Dictionary with evaluation results
        """
        self.class_names = class_names or [f"Class_{i}" for i in range(self.model.num_classes)]
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        self.predictions = np.array(all_predictions)
        self.targets = np.array(all_targets)
        self.probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        results = self._calculate_metrics()
        
        return results
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        # Basic accuracy
        accuracy = np.mean(self.predictions == self.targets)
        
        # Classification report
        report = classification_report(
            self.targets, 
            self.predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.targets, self.predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.targets, self.predictions, average=None
        )
        
        # Macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average='macro'
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average='micro'
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average='weighted'
        )
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'per_class_metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            },
            'macro_averages': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'micro_averages': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1_score': micro_f1
            },
            'weighted_averages': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            }
        }
        
        return results
    
    def plot_confusion_matrix(
        self, 
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Plot confusion matrix."""
        if len(self.predictions) == 0:
            print("No predictions available. Run evaluate_dataset first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Get confusion matrix from metrics
        metrics = self._calculate_metrics()
        cm = metrics['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_performance(
        self, 
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """Plot per-class performance metrics."""
        if len(self.predictions) == 0:
            print("No predictions available. Run evaluate_dataset first.")
            return
        
        metrics = self._calculate_metrics()
        per_class = metrics['per_class_metrics']
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': per_class['precision'],
            'Recall': per_class['recall'],
            'F1-Score': per_class['f1_score'],
            'Support': per_class['support']
        })
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Precision
        axes[0, 0].bar(df['Class'], df['Precision'], color='skyblue')
        axes[0, 0].set_title('Precision by Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[0, 1].bar(df['Class'], df['Recall'], color='lightcoral')
        axes[0, 1].set_title('Recall by Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1, 0].bar(df['Class'], df['F1-Score'], color='lightgreen')
        axes[1, 0].set_title('F1-Score by Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Support
        axes[1, 1].bar(df['Class'], df['Support'], color='gold')
        axes[1, 1].set_title('Support by Class')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(
        self, 
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Plot ROC curves for multi-class classification."""
        if len(self.predictions) == 0:
            print("No predictions available. Run evaluate_dataset first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Convert to one-hot encoding for ROC calculation
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(self.targets, classes=range(len(self.class_names)))
        
        # Calculate ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(self.class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        
        for i, color in zip(range(len(self.class_names)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_misclassifications(
        self, 
        data_loader: DataLoader,
        num_examples: int = 10
    ) -> Dict[str, List[Any]]:
        """
        Analyze misclassified examples.
        
        Args:
            data_loader: Data loader containing the data
            num_examples: Number of misclassified examples to analyze
            
        Returns:
            Dictionary with misclassified examples
        """
        if len(self.predictions) == 0:
            print("No predictions available. Run evaluate_dataset first.")
            return {}
        
        # Find misclassified examples
        misclassified_indices = np.where(self.predictions != self.targets)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return {}
        
        # Get random sample of misclassifications
        sample_indices = np.random.choice(
            misclassified_indices, 
            min(num_examples, len(misclassified_indices)), 
            replace=False
        )
        
        misclassifications = {
            'indices': sample_indices.tolist(),
            'true_labels': [self.class_names[self.targets[i]] for i in sample_indices],
            'predicted_labels': [self.class_names[self.predictions[i]] for i in sample_indices],
            'probabilities': [self.probabilities[i].tolist() for i in sample_indices]
        }
        
        return misclassifications
    
    def generate_report(
        self, 
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        if len(self.predictions) == 0:
            return "No predictions available. Run evaluate_dataset first."
        
        metrics = self._calculate_metrics()
        
        report = f"""
HANDWRITING RECOGNITION MODEL EVALUATION REPORT
{'='*50}

OVERALL PERFORMANCE:
- Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)

MACRO AVERAGES:
- Precision: {metrics['macro_averages']['precision']:.4f}
- Recall: {metrics['macro_averages']['recall']:.4f}
- F1-Score: {metrics['macro_averages']['f1_score']:.4f}

MICRO AVERAGES:
- Precision: {metrics['micro_averages']['precision']:.4f}
- Recall: {metrics['micro_averages']['recall']:.4f}
- F1-Score: {metrics['micro_averages']['f1_score']:.4f}

WEIGHTED AVERAGES:
- Precision: {metrics['weighted_averages']['precision']:.4f}
- Recall: {metrics['weighted_averages']['recall']:.4f}
- F1-Score: {metrics['weighted_averages']['f1_score']:.4f}

PER-CLASS PERFORMANCE:
"""
        
        per_class = metrics['per_class_metrics']
        for i, class_name in enumerate(self.class_names):
            report += f"""
{class_name}:
  - Precision: {per_class['precision'][i]:.4f}
  - Recall: {per_class['recall'][i]:.4f}
  - F1-Score: {per_class['f1_score'][i]:.4f}
  - Support: {per_class['support'][i]}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def compare_with_baseline(
        self, 
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare current model results with baseline results.
        
        Args:
            baseline_results: Results from baseline model
            
        Returns:
            Comparison results
        """
        if len(self.predictions) == 0:
            print("No predictions available. Run evaluate_dataset first.")
            return {}
        
        current_metrics = self._calculate_metrics()
        
        comparison = {
            'accuracy_improvement': current_metrics['accuracy'] - baseline_results['accuracy'],
            'macro_f1_improvement': (
                current_metrics['macro_averages']['f1_score'] - 
                baseline_results['macro_averages']['f1_score']
            ),
            'weighted_f1_improvement': (
                current_metrics['weighted_averages']['f1_score'] - 
                baseline_results['weighted_averages']['f1_score']
            )
        }
        
        return comparison


class ModelComparison:
    """Compare multiple models side by side."""
    
    def __init__(self):
        """Initialize the model comparison."""
        self.results = {}
    
    def add_model_results(
        self, 
        model_name: str, 
        evaluator: ModelEvaluator
    ) -> None:
        """
        Add results from a model evaluator.
        
        Args:
            model_name: Name of the model
            evaluator: ModelEvaluator instance with results
        """
        if len(evaluator.predictions) == 0:
            print(f"No predictions available for {model_name}")
            return
        
        self.results[model_name] = evaluator._calculate_metrics()
    
    def plot_comparison(
        self, 
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """Plot comparison of all models."""
        if not self.results:
            print("No model results available for comparison.")
            return
        
        model_names = list(self.results.keys())
        
        # Extract metrics
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        macro_f1s = [self.results[name]['macro_averages']['f1_score'] for name in model_names]
        weighted_f1s = [self.results[name]['weighted_averages']['f1_score'] for name in model_names]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Macro F1 comparison
        axes[0, 1].bar(model_names, macro_f1s, color='lightcoral')
        axes[0, 1].set_title('Macro F1-Score Comparison')
        axes[0, 1].set_ylabel('Macro F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Weighted F1 comparison
        axes[1, 0].bar(model_names, weighted_f1s, color='lightgreen')
        axes[1, 0].set_title('Weighted F1-Score Comparison')
        axes[1, 0].set_ylabel('Weighted F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Combined metrics
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[1, 1].bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
        axes[1, 1].bar(x, macro_f1s, width, label='Macro F1', color='lightcoral')
        axes[1, 1].bar(x + width, weighted_f1s, width, label='Weighted F1', color='lightgreen')
        
        axes[1, 1].set_title('Combined Metrics Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_comparison_report(
        self, 
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate a comparison report for all models.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Comparison report as string
        """
        if not self.results:
            return "No model results available for comparison."
        
        report = f"""
MODEL COMPARISON REPORT
{'='*50}

"""
        
        # Create comparison table
        model_names = list(self.results.keys())
        
        report += f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12}\n"
        report += "-" * 60 + "\n"
        
        for name in model_names:
            results = self.results[name]
            report += f"{name:<20} {results['accuracy']:<10.4f} "
            report += f"{results['macro_averages']['f1_score']:<10.4f} "
            report += f"{results['weighted_averages']['f1_score']:<12.4f}\n"
        
        # Find best model
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_macro_f1 = max(self.results.items(), key=lambda x: x[1]['macro_averages']['f1_score'])
        best_weighted_f1 = max(self.results.items(), key=lambda x: x[1]['weighted_averages']['f1_score'])
        
        report += f"""
BEST PERFORMING MODELS:
- Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})
- Best Macro F1: {best_macro_f1[0]} ({best_macro_f1[1]['macro_averages']['f1_score']:.4f})
- Best Weighted F1: {best_weighted_f1[0]} ({best_weighted_f1[1]['weighted_averages']['f1_score']:.4f})
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
