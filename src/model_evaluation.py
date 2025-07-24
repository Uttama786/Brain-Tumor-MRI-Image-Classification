"""
Model evaluation and comparison for Brain Tumor MRI Classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

from utils import DataLoader, plot_confusion_matrix, create_model_comparison_plot

class ModelEvaluator:
    """Class for evaluating and comparing multiple models."""
    
    def __init__(self, data_dir='.', img_size=(224, 224), batch_size=32):
        """
        Initialize ModelEvaluator.
        
        Args:
            data_dir (str): Path to data directory
            img_size (tuple): Image size for preprocessing
            batch_size (int): Batch size for evaluation
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_loader = DataLoader(data_dir, img_size, batch_size)
        self.class_names = self.data_loader.class_names
        self.results = {}
        
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            tensorflow.keras.Model: Loaded model
        """
        try:
            model = load_model(model_path)
            print(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    
    def evaluate_single_model(self, model, model_name, test_gen):
        """
        Evaluate a single model.
        
        Args:
            model: Keras model to evaluate
            model_name (str): Name of the model
            test_gen: Test data generator
            
        Returns:
            dict: Evaluation results
        """
        print(f"\nEvaluating {model_name}...")
        
        # Reset generator
        test_gen.reset()
        
        # Get predictions
        y_pred = model.predict(test_gen)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_gen.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='weighted')
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_classes': y_pred_classes,
            'y_true': y_true
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Print detailed classification report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_true, y_pred_classes, target_names=self.class_names))
        
        return results
    
    def evaluate_all_models(self, model_paths):
        """
        Evaluate all models in the given paths.
        
        Args:
            model_paths (dict): Dictionary mapping model names to file paths
        """
        # Get test data generator
        _, _, test_gen = self.data_loader.get_data_generators()
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                model = self.load_model(model_path)
                if model is not None:
                    self.evaluate_single_model(model, model_name, test_gen)
            else:
                print(f"Model file not found: {model_path}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all evaluated models."""
        for model_name, results in self.results.items():
            save_path = f'results/confusion_matrices/{model_name}_confusion_matrix.png'
            plot_confusion_matrix(
                results['y_true'], 
                results['y_pred_classes'], 
                self.class_names, 
                save_path
            )
    
    def create_comparison_plot(self):
        """Create comparison plot for all models."""
        # Prepare data for plotting
        comparison_data = {}
        for model_name, results in self.results.items():
            comparison_data[model_name] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }
        
        # Create comparison plot
        create_model_comparison_plot(
            comparison_data, 
            save_path='results/model_comparison.png'
        )
    
    def save_results(self, output_path='results/model_evaluation_results.json'):
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path (str): Path to save the results
        """
        # Prepare results for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            }
        
        # Save to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"Results saved to {output_path}")
    
    def create_results_table(self, output_path='results/model_comparison_table.csv'):
        """
        Create a CSV table with all model results.
        
        Args:
            output_path (str): Path to save the CSV table
        """
        # Create DataFrame
        data = []
        for model_name, results in self.results.items():
            data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"Results table saved to {output_path}")
        print("\nModel Comparison Table:")
        print(df.to_string(index=False))
        
        return df
    
    def find_best_model(self, metric='accuracy'):
        """
        Find the best model based on a specific metric.
        
        Args:
            metric (str): Metric to use for comparison ('accuracy', 'precision', 'recall', 'f1_score')
            
        Returns:
            tuple: (best_model_name, best_score)
        """
        if not self.results:
            print("No models evaluated yet.")
            return None, None
        
        best_model = None
        best_score = -1
        
        for model_name, results in self.results.items():
            score = results[metric]
            if score > best_score:
                best_score = score
                best_model = model_name
        
        print(f"\nBest model based on {metric}: {best_model} ({best_score:.4f})")
        return best_model, best_score

def main():
    """Main function to evaluate all models."""
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Define model paths
    model_paths = {
        'Custom CNN': 'models/custom_cnn_model.h5',
        'ResNet50': 'models/resnet50_model_final.h5',
        'EfficientNet': 'models/efficientnet_model_final.h5',
        'MobileNet': 'models/mobilenet_model_final.h5'
    }
    
    print("="*60)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*60)
    
    # Evaluate all models
    evaluator.evaluate_all_models(model_paths)
    
    # Create visualizations
    print("\nCreating visualizations...")
    evaluator.plot_confusion_matrices()
    evaluator.create_comparison_plot()
    
    # Save results
    evaluator.save_results()
    evaluator.create_results_table()
    
    # Find best model
    print("\n" + "="*40)
    print("BEST MODEL ANALYSIS")
    print("="*40)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        evaluator.find_best_model(metric)
    
    print("\nModel evaluation completed!")

if __name__ == "__main__":
    main() 