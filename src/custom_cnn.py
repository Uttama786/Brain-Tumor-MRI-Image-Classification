"""
Custom CNN model for Brain Tumor MRI Classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

from utils import DataLoader, plot_training_history, evaluate_model

class CustomCNN:
    """Custom Convolutional Neural Network for brain tumor classification."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        """
        Initialize CustomCNN.
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """
        Build the custom CNN architecture.
        
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=self.input_shape, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Global Average Pooling
            GlobalAveragePooling2D(),
            
            # Dense Layers
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output Layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, model_save_path):
        """
        Get training callbacks.
        
        Args:
            model_save_path (str): Path to save the best model
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_gen, valid_gen, epochs=50, model_save_path='models/custom_cnn_model.h5'):
        """
        Train the custom CNN model.
        
        Args:
            train_gen: Training data generator
            valid_gen: Validation data generator
            epochs (int): Number of training epochs
            model_save_path (str): Path to save the trained model
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Build the model
        model = self.build_model()
        
        # Print model summary
        print("Custom CNN Model Architecture:")
        model.summary()
        
        # Get callbacks
        callbacks = self.get_callbacks(model_save_path)
        
        # Train the model
        print("\nStarting training...")
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=valid_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        model.save(model_save_path)
        print(f"\nModel saved to {model_save_path}")
        
        return history
    
    def evaluate(self, test_gen, class_names):
        """
        Evaluate the trained model.
        
        Args:
            test_gen: Test data generator
            class_names: List of class names
        """
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return
        
        print("\nEvaluating Custom CNN Model...")
        y_pred, y_pred_classes, y_true = evaluate_model(
            self.model, test_gen, class_names,
            save_path='results/confusion_matrices/custom_cnn_confusion_matrix.png'
        )
        
        return y_pred, y_pred_classes, y_true

def main():
    """Main function to train and evaluate the custom CNN model."""
    
    # Initialize data loader
    data_loader = DataLoader(data_dir='.', img_size=(224, 224), batch_size=32)
    
    # Get data generators
    train_gen, valid_gen, test_gen = data_loader.get_data_generators()
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {valid_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Number of classes: {len(data_loader.class_names)}")
    print(f"Class names: {data_loader.class_names}")
    
    # Initialize and train custom CNN
    custom_cnn = CustomCNN(input_shape=(224, 224, 3), num_classes=4)
    
    # Train the model
    history = custom_cnn.train(train_gen, valid_gen, epochs=50)
    
    # Plot training history
    plot_training_history(history, save_path='results/custom_cnn_training_history.png')
    
    # Evaluate the model
    custom_cnn.evaluate(test_gen, data_loader.class_names)
    
    print("\nCustom CNN training and evaluation completed!")

if __name__ == "__main__":
    main() 