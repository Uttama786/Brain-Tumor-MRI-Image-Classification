"""
Test script to verify model loading with corrected paths.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_model_loading():
    """Test if models can be loaded with the corrected paths."""
    print("🧠 Testing Model Loading...")
    print("="*50)
    
    model_paths = {
        'Custom CNN': 'models/custom_cnn_model.h5',
        'ResNet50': 'models/resnet50_model_final.h5',
        'ResNet50 (Fine-tuned)': 'models/resnet50_model_fine_tuned.h5',
        'ResNet50 (Initial)': 'models/resnet50_model.h5'
    }
    
    loaded_models = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n🔍 Testing {model_name}...")
        
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                loaded_models[model_name] = model
                print(f"✅ {model_name} loaded successfully!")
                print(f"   📁 Path: {model_path}")
                print(f"   📊 Model summary: {len(model.layers)} layers")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {str(e)}")
        else:
            print(f"❌ {model_name} not found at {model_path}")
    
    print("\n" + "="*50)
    print(f"📈 Summary: {len(loaded_models)}/{len(model_paths)} models loaded successfully")
    
    if loaded_models:
        print("🎉 All available models are ready for use!")
        return True
    else:
        print("❌ No models could be loaded.")
        return False

if __name__ == "__main__":
    test_model_loading() 