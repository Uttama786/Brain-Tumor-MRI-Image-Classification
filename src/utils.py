"""
Utility functions for Brain Tumor MRI Classification project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataLoader:
    """Class for loading and preprocessing MRI image data."""
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize DataLoader.
        
        Args:
            data_dir (str): Path to the data directory
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.num_classes = len(self.class_names)
        
    def load_data(self, split='train'):
        """
        Load data from specified split.
        
        Args:
            split (str): Data split ('train', 'valid', 'test')
            
        Returns:
            tuple: (images, labels)
        """
        data_path = os.path.join(self.data_dir, 'data', split)
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            img = img / 255.0  # Normalize to [0, 1]
                            
                            images.append(img)
                            labels.append(class_idx)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def get_data_generators(self):
        """
        Create data generators for training, validation, and testing.
        
        Returns:
            tuple: (train_gen, valid_gen, test_gen)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_gen = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'data', 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        valid_gen = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'data', 'valid'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_gen = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'data', 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, valid_gen, test_gen

def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Keras history object
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution(labels, class_names, save_path=None):
    """
    Plot class distribution.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path (str): Path to save the plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts)
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                str(count), ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_sample_images(images, labels, class_names, num_samples=4, save_path=None):
    """
    Plot sample images from each class.
    
    Args:
        images: Array of images
        labels: Array of labels
        class_names: List of class names
        num_samples (int): Number of samples per class
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=(15, 12))
    
    for i, class_name in enumerate(class_names):
        class_indices = np.where(labels == i)[0]
        sample_indices = np.random.choice(class_indices, num_samples, replace=False)
        
        for j, idx in enumerate(sample_indices):
            axes[i, j].imshow(images[idx])
            axes[i, j].set_title(f'{class_name}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_gen, class_names, save_path=None):
    """
    Evaluate model and generate comprehensive report.
    
    Args:
        model: Trained Keras model
        test_gen: Test data generator
        class_names: List of class names
        save_path (str): Path to save confusion matrix
    """
    # Get predictions
    test_gen.reset()
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, class_names, save_path)
    
    return y_pred, y_pred_classes, y_true

def create_model_comparison_plot(results, save_path=None):
    """
    Create comparison plot for different models.
    
    Args:
        results (dict): Dictionary with model results
        save_path (str): Path to save the plot
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def preprocess_image_for_prediction(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size for the image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Load and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img 