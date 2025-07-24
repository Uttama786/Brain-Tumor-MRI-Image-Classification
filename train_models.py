"""
Main training script for Brain Tumor MRI Classification.
This script trains all models and generates comprehensive results.
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from src.custom_cnn import CustomCNN
from src.transfer_learning import (
    train_resnet50, train_efficientnet, train_mobilenet
)
from src.model_evaluation import ModelEvaluator
from src.utils import DataLoader, plot_training_history

def main():
    """Main training function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Models')
    parser.add_argument('--custom-cnn', action='store_true', help='Train Custom CNN model')
    parser.add_argument('--resnet50', action='store_true', help='Train ResNet50 model')
    parser.add_argument('--efficientnet', action='store_true', help='Train EfficientNet model')
    parser.add_argument('--mobilenet', action='store_true', help='Train MobileNet model')
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate all trained models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    # If no specific model is selected, train all
    if not any([args.custom_cnn, args.resnet50, args.efficientnet, args.mobilenet, args.all]):
        args.all = True
    
    print("="*70)
    print("🧠 BRAIN TUMOR MRI CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Arguments: {vars(args)}")
    print("="*70)
    
    # Initialize data loader
    print("\n📊 Loading dataset...")
    data_loader = DataLoader(data_dir='.', img_size=(224, 224), batch_size=args.batch_size)
    train_gen, valid_gen, test_gen = data_loader.get_data_generators()
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {valid_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Number of classes: {len(data_loader.class_names)}")
    print(f"Class names: {data_loader.class_names}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/confusion_matrices', exist_ok=True)
    
    # Train Custom CNN
    if args.custom_cnn or args.all:
        print("\n" + "="*50)
        print("🏗️ TRAINING CUSTOM CNN MODEL")
        print("="*50)
        
        start_time = time.time()
        custom_cnn = CustomCNN(input_shape=(224, 224, 3), num_classes=4)
        history = custom_cnn.train(train_gen, valid_gen, epochs=args.epochs)
        
        # Plot training history
        plot_training_history(history, save_path='results/custom_cnn_training_history.png')
        
        # Evaluate
        custom_cnn.evaluate(test_gen, data_loader.class_names)
        
        training_time = time.time() - start_time
        print(f"Custom CNN training completed in {training_time:.2f} seconds")
    
    # Train Transfer Learning Models
    if args.resnet50 or args.all:
        print("\n" + "="*50)
        print("🔄 TRAINING RESNET50 MODEL")
        print("="*50)
        
        start_time = time.time()
        resnet_model = train_resnet50(train_gen, valid_gen, test_gen, data_loader.class_names)
        training_time = time.time() - start_time
        print(f"ResNet50 training completed in {training_time:.2f} seconds")
    
    if args.efficientnet or args.all:
        print("\n" + "="*50)
        print("⚡ TRAINING EFFICIENTNET MODEL")
        print("="*50)
        
        start_time = time.time()
        efficientnet_model = train_efficientnet(train_gen, valid_gen, test_gen, data_loader.class_names)
        training_time = time.time() - start_time
        print(f"EfficientNet training completed in {training_time:.2f} seconds")
    
    if args.mobilenet or args.all:
        print("\n" + "="*50)
        print("📱 TRAINING MOBILENET MODEL")
        print("="*50)
        
        start_time = time.time()
        mobilenet_model = train_mobilenet(train_gen, valid_gen, test_gen, data_loader.class_names)
        training_time = time.time() - start_time
        print(f"MobileNet training completed in {training_time:.2f} seconds")
    
    # Evaluate all models
    if args.evaluate or args.all:
        print("\n" + "="*50)
        print("📈 MODEL EVALUATION AND COMPARISON")
        print("="*50)
        
        evaluator = ModelEvaluator()
        
        # Define model paths
        model_paths = {
            'Custom CNN': 'models/custom_cnn_model.h5',
            'ResNet50': 'models/resnet50_model_final.h5',
            'EfficientNet': 'models/efficientnet_model_final.h5',
            'MobileNet': 'models/mobilenet_model_final.h5'
        }
        
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
        print("🏆 BEST MODEL ANALYSIS")
        print("="*40)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            evaluator.find_best_model(metric)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 Generated files:")
    print("  - models/: Trained model files (.h5)")
    print("  - results/: Training plots and evaluation results")
    print("  - results/confusion_matrices/: Confusion matrices for each model")
    print("\n🚀 Next steps:")
    print("  1. Run the Streamlit app: streamlit run app/streamlit_app.py")
    print("  2. Check the results/ directory for detailed analysis")
    print("  3. Review model_comparison_table.csv for performance comparison")
    print("="*70)

if __name__ == "__main__":
    main() 