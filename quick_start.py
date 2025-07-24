"""
Quick Start Script for Brain Tumor MRI Classification Project.
This script provides an easy way to get started with the project.
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print project banner."""
    print("="*70)
    print("🧠 BRAIN TUMOR MRI CLASSIFICATION PROJECT")
    print("="*70)
    print("Quick Start Script")
    print("="*70)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'tensorflow', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'scikit-learn', 'opencv-python', 
        'Pillow', 'streamlit', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    print("✅ All dependencies are available!")
    return True

def check_dataset():
    """Check if dataset is properly structured."""
    print("\n📊 Checking dataset structure...")
    
    required_dirs = [
        'data/train/glioma',
        'data/train/meningioma', 
        'data/train/no_tumor',
        'data/train/pituitary',
        'data/valid/glioma',
        'data/valid/meningioma',
        'data/valid/no_tumor', 
        'data/valid/pituitary',
        'data/test/glioma',
        'data/test/meningioma',
        'data/test/no_tumor',
        'data/test/pituitary'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ {dir_path} ({file_count} images)")
        else:
            print(f"❌ {dir_path} - Missing")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️ Missing directories: {len(missing_dirs)}")
        print("Please ensure the dataset is properly organized.")
        return False
    
    print("✅ Dataset structure is correct!")
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating project directories...")
    
    directories = [
        'models',
        'results',
        'results/confusion_matrices',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")

def run_data_exploration():
    """Run data exploration."""
    print("\n🔬 Running data exploration...")
    
    try:
        # Import and run data exploration
        sys.path.append('src')
        from src.utils import DataLoader, plot_class_distribution, plot_sample_images
        
        # Load data
        data_loader = DataLoader(data_dir='.', img_size=(224, 224), batch_size=32)
        train_images, train_labels = data_loader.load_data('train')
        
        # Create visualizations
        plot_class_distribution(train_labels, data_loader.class_names, 
                               save_path='results/class_distribution.png')
        plot_sample_images(train_images, train_labels, data_loader.class_names, 
                          num_samples=4, save_path='results/sample_images.png')
        
        print("✅ Data exploration completed!")
        print("📊 Check results/class_distribution.png and results/sample_images.png")
        
    except Exception as e:
        print(f"❌ Error during data exploration: {e}")

def train_models():
    """Train models with reduced epochs for quick testing."""
    print("\n🏋️ Training models (quick test with 5 epochs)...")
    
    try:
        # Run training with reduced epochs
        subprocess.run([
            sys.executable, "train_models.py", 
            "--custom-cnn", "--epochs", "5"
        ], check=True)
        
        print("✅ Quick training completed!")
        print("📁 Check models/ directory for trained models")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during training: {e}")

def run_streamlit_app():
    """Run the Streamlit application."""
    print("\n🚀 Starting Streamlit application...")
    print("📱 The app will open in your browser automatically.")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")

def main():
    """Main function."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages manually.")
        return
    
    # Check dataset
    if not check_dataset():
        print("❌ Dataset check failed. Please ensure the dataset is properly organized.")
        return
    
    # Create directories
    create_directories()
    
    # Show menu
    while True:
        print("\n" + "="*50)
        print("🎯 QUICK START MENU")
        print("="*50)
        print("1. 🔬 Run Data Exploration")
        print("2. 🏋️ Train Models (Quick Test)")
        print("3. 🚀 Launch Streamlit App")
        print("4. 📊 Full Training (All Models)")
        print("5. 📈 Evaluate Models")
        print("6. ❌ Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            run_data_exploration()
        elif choice == '2':
            train_models()
        elif choice == '3':
            run_streamlit_app()
        elif choice == '4':
            print("\n🏋️ Starting full training (this may take several hours)...")
            try:
                subprocess.run([sys.executable, "train_models.py", "--all"], check=True)
                print("✅ Full training completed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error during full training: {e}")
        elif choice == '5':
            print("\n📈 Evaluating models...")
            try:
                subprocess.run([sys.executable, "src/model_evaluation.py"], check=True)
                print("✅ Model evaluation completed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error during evaluation: {e}")
        elif choice == '6':
            print("\n👋 Thank you for using Brain Tumor MRI Classification!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 