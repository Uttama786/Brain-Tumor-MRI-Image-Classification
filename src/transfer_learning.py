"""
Transfer Learning models for Brain Tumor MRI Classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, EfficientNetB0, MobileNetV2, InceptionV3
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

from utils import DataLoader, plot_training_history, evaluate_model

class TransferLearningModel:
    """Base class for transfer learning models."""
    
    def __init__(self, base_model_name, input_shape=(224, 224, 3), num_classes=4):
        """
        Initialize TransferLearningModel.
        
        Args:
            base_model_name (str): Name of the base model
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
        """
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.base_model = None
        
    def get_base_model(self):
        """
        Get the base pretrained model.
        
        Returns:
            tensorflow.keras.Model: Base model
        """
        if self.base_model_name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'inception':
            base_model = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        return base_model
    
    def build_model(self, fine_tune_layers=0):
        """
        Build the transfer learning model.
        
        Args:
            fine_tune_layers (int): Number of layers to fine-tune
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        # Get base model
        self.base_model = self.get_base_model()
        
        # Freeze base model layers
        self.base_model.trainable = False
        
        # Create new model
        model = tf.keras.Sequential([
            self.base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fine_tune(self, fine_tune_layers=10):
        """
        Fine-tune the model by unfreezing some layers.
        
        Args:
            fine_tune_layers (int): Number of layers to fine-tune
        """
        if self.base_model is None:
            print("Model not built yet. Please build the model first.")
            return
        
        # Unfreeze the last few layers
        self.base_model.trainable = True
        
        # Freeze all layers except the last few
        for layer in self.base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuned last {fine_tune_layers} layers of {self.base_model_name}")
    
    def get_callbacks(self, model_save_path):
        """
        Get training callbacks.
        
        Args:
            model_save_path (str): Path to save the best model
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_gen, valid_gen, epochs=30, fine_tune_epochs=10, 
              model_save_path=None):
        """
        Train the transfer learning model.
        
        Args:
            train_gen: Training data generator
            valid_gen: Validation data generator
            epochs (int): Number of epochs for initial training
            fine_tune_epochs (int): Number of epochs for fine-tuning
            model_save_path (str): Path to save the model
            
        Returns:
            tuple: (initial_history, fine_tune_history)
        """
        if model_save_path is None:
            model_save_path = f'models/{self.base_model_name}_model.h5'
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Build the model
        model = self.build_model()
        
        # Print model summary
        print(f"{self.base_model_name.upper()} Model Architecture:")
        model.summary()
        
        # Get callbacks
        callbacks = self.get_callbacks(model_save_path)
        
        # Initial training with frozen base model
        print(f"\nStarting initial training with frozen {self.base_model_name}...")
        initial_history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=valid_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning
        print(f"\nStarting fine-tuning of {self.base_model_name}...")
        self.fine_tune(fine_tune_layers=10)
        
        # Update callbacks for fine-tuning
        fine_tune_callbacks = self.get_callbacks(model_save_path.replace('.h5', '_fine_tuned.h5'))
        
        fine_tune_history = model.fit(
            train_gen,
            epochs=fine_tune_epochs,
            validation_data=valid_gen,
            callbacks=fine_tune_callbacks,
            verbose=1
        )
        
        # Save the final model
        model.save(model_save_path.replace('.h5', '_final.h5'))
        print(f"\nModel saved to {model_save_path.replace('.h5', '_final.h5')}")
        
        return initial_history, fine_tune_history
    
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
        
        print(f"\nEvaluating {self.base_model_name.upper()} Model...")
        y_pred, y_pred_classes, y_true = evaluate_model(
            self.model, test_gen, class_names,
            save_path=f'results/confusion_matrices/{self.base_model_name}_confusion_matrix.png'
        )
        
        return y_pred, y_pred_classes, y_true

def train_resnet50(train_gen, valid_gen, test_gen, class_names):
    """Train ResNet50 model."""
    print("\n" + "="*50)
    print("TRAINING RESNET50 MODEL")
    print("="*50)
    
    resnet_model = TransferLearningModel('resnet50', input_shape=(224, 224, 3), num_classes=4)
    initial_history, fine_tune_history = resnet_model.train(
        train_gen, valid_gen, epochs=20, fine_tune_epochs=10,
        model_save_path='models/resnet50_model.h5'
    )
    
    # Plot training history
    plot_training_history(initial_history, save_path='results/resnet50_initial_training.png')
    plot_training_history(fine_tune_history, save_path='results/resnet50_fine_tune_training.png')
    
    # Evaluate
    resnet_model.evaluate(test_gen, class_names)
    
    return resnet_model

def train_efficientnet(train_gen, valid_gen, test_gen, class_names):
    """Train EfficientNet model."""
    print("\n" + "="*50)
    print("TRAINING EFFICIENTNET MODEL")
    print("="*50)
    
    efficientnet_model = TransferLearningModel('efficientnet', input_shape=(224, 224, 3), num_classes=4)
    initial_history, fine_tune_history = efficientnet_model.train(
        train_gen, valid_gen, epochs=20, fine_tune_epochs=10,
        model_save_path='models/efficientnet_model.h5'
    )
    
    # Plot training history
    plot_training_history(initial_history, save_path='results/efficientnet_initial_training.png')
    plot_training_history(fine_tune_history, save_path='results/efficientnet_fine_tune_training.png')
    
    # Evaluate
    efficientnet_model.evaluate(test_gen, class_names)
    
    return efficientnet_model

def train_mobilenet(train_gen, valid_gen, test_gen, class_names):
    """Train MobileNet model."""
    print("\n" + "="*50)
    print("TRAINING MOBILENET MODEL")
    print("="*50)
    
    mobilenet_model = TransferLearningModel('mobilenet', input_shape=(224, 224, 3), num_classes=4)
    initial_history, fine_tune_history = mobilenet_model.train(
        train_gen, valid_gen, epochs=20, fine_tune_epochs=10,
        model_save_path='models/mobilenet_model.h5'
    )
    
    # Plot training history
    plot_training_history(initial_history, save_path='results/mobilenet_initial_training.png')
    plot_training_history(fine_tune_history, save_path='results/mobilenet_fine_tune_training.png')
    
    # Evaluate
    mobilenet_model.evaluate(test_gen, class_names)
    
    return mobilenet_model

def main():
    """Main function to train all transfer learning models."""
    
    # Initialize data loader
    data_loader = DataLoader(data_dir='.', img_size=(224, 224), batch_size=32)
    
    # Get data generators
    train_gen, valid_gen, test_gen = data_loader.get_data_generators()
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {valid_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Number of classes: {len(data_loader.class_names)}")
    print(f"Class names: {data_loader.class_names}")
    
    # Train different transfer learning models
    models = {}
    
    # Train ResNet50
    models['resnet50'] = train_resnet50(train_gen, valid_gen, test_gen, data_loader.class_names)
    
    # Train EfficientNet
    models['efficientnet'] = train_efficientnet(train_gen, valid_gen, test_gen, data_loader.class_names)
    
    # Train MobileNet
    models['mobilenet'] = train_mobilenet(train_gen, valid_gen, test_gen, data_loader.class_names)
    
    print("\nAll transfer learning models training and evaluation completed!")
    
    return models

if __name__ == "__main__":
    main() 