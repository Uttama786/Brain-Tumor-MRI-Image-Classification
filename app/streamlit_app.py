"""
Streamlit Web Application for Brain Tumor MRI Classification.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Lottie animation support
try:
    from streamlit_lottie import st_lottie
    import requests
except ImportError:
    st.warning("Install streamlit-lottie for animated icons: pip install streamlit-lottie requests")
    st_lottie = None
    requests = None

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import preprocess_image_for_prediction

# Set page config
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@700&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }
    .main .block-container {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        color: #fff;
        min-height: 100vh;
    }
    .css-1d391kg {
        background: linear-gradient(135deg, #16213e 0%, #314755 100%);
    }
    .main-header {
        font-family: 'Lato', sans-serif;
        font-size: 3rem;
        color: #ff6e7f;
        background: linear-gradient(90deg, #ff6e7f 0%, #bfe9ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        letter-spacing: 2px;
        animation: fadeInDown 1s;
    }
    .sub-header {
        font-family: 'Lato', sans-serif;
        font-size: 1.5rem;
        color: #bfe9ff;
        margin-bottom: 1rem;
        letter-spacing: 1px;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #fff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6e7f;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .prediction-box {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #fff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #ff6e7f;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(255,110,127,0.15);
        animation: fadeIn 1.2s;
    }
    .upload-section {
        background: linear-gradient(135deg, #0f3460 0%, #314755 100%);
        color: #fff;
        padding: 2rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1.5px solid #bfe9ff;
        transition: box-shadow 0.3s;
    }
    .upload-section:hover {
        box-shadow: 0 6px 24px rgba(191,233,255,0.15);
        border-color: #ff6e7f;
    }
    .stMarkdown {
        color: #fff;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff6e7f 0%, #bfe9ff 100%);
        color: #232526;
        border: none;
        border-radius: 0.5rem;
        padding: 0.7rem 1.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(255,110,127,0.10);
        transition: background 0.3s, transform 0.2s;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #bfe9ff 0%, #ff6e7f 100%);
        color: #16213e;
        transform: scale(1.05);
        box-shadow: 0 4px 16px rgba(255,110,127,0.18);
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-40px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

class BrainTumorClassifier:
    """Brain Tumor Classification App."""
    
    def __init__(self):
        """Initialize the classifier."""
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        model_paths = {
            'Custom CNN': 'models/custom_cnn_model.h5',
            'ResNet50': 'models/resnet50_model_final.h5',
            'EfficientNet': 'models/efficientnet_model_final.h5',
            'MobileNet': 'models/mobilenet_model_final.h5'
        }
        
        for model_name, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(model_path)
                    st.sidebar.success(f"✅ {model_name} loaded successfully")
                else:
                    st.sidebar.warning(f"⚠️ {model_name} not found at {model_path}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {model_name}: {str(e)}")
    
    def preprocess_image(self, image):
        """
        Preprocess uploaded image.
        
        Args:
            image: PIL Image object
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert RGB to BGR if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize to 224x224
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_normalized = img_rgb / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_rgb
    
    def predict(self, image, model_name):
        """
        Make prediction using specified model.
        
        Args:
            image: Preprocessed image
            model_name: Name of the model to use
            
        Returns:
            tuple: (predicted_class, confidence_scores)
        """
        if model_name not in self.models:
            return None, None
        
        model = self.models[model_name]
        predictions = model.predict(image)
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence_scores = predictions[0]
        
        return predicted_class, confidence_scores
    
    def create_confidence_chart(self, confidence_scores, model_name):
        """
        Create confidence scores visualization.
        
        Args:
            confidence_scores: Array of confidence scores
            model_name: Name of the model
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        fig = go.Figure(data=[
            go.Bar(
                x=self.class_names,
                y=confidence_scores,
                marker_color=['#1f77b4' if i == np.argmax(confidence_scores) else '#d3d3d3' 
                             for i in range(len(confidence_scores))],
                text=[f'{score:.3f}' for score in confidence_scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f'Confidence Scores - {model_name}',
            xaxis_title='Tumor Types',
            yaxis_title='Confidence',
            yaxis_range=[0, 1],
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_model_comparison(self, image):
        """
        Compare predictions from all available models.
        
        Args:
            image: Preprocessed image
            
        Returns:
            plotly.graph_objects.Figure: Comparison chart
        """
        results = []
        
        for model_name in self.models.keys():
            predicted_class, confidence_scores = self.predict(image, model_name)
            if predicted_class is not None:
                max_confidence = np.max(confidence_scores)
                results.append({
                    'Model': model_name,
                    'Predicted Class': predicted_class,
                    'Confidence': max_confidence
                })
        
        if not results:
            return None
        
        df = pd.DataFrame(results)
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['Model'],
                y=df['Confidence'],
                text=[f'{row["Predicted Class"]}<br>{row["Confidence"]:.3f}' 
                      for _, row in df.iterrows()],
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title='Model Comparison - Confidence Scores',
            xaxis_title='Models',
            yaxis_title='Confidence',
            yaxis_range=[0, 1],
            showlegend=False,
            height=400
        )
        
        return fig, df

# Lottie animation loader

def load_lottieurl(url):
    if requests is None:
        return None
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URLs
LOTTIE_BRAIN = "https://assets2.lottiefiles.com/packages/lf20_2ks3pjua.json"
LOTTIE_PREDICT = "https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"
LOTTIE_UPLOAD = "https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json"

def main():
    """Main Streamlit application."""
    # Header with animated icon
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if st_lottie:
            st_lottie(load_lottieurl(LOTTIE_BRAIN), height=90, key="header-brain")
    with col_title:
        st.markdown('<h1 class="main-header">🧠 Brain Tumor MRI Classifier</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Brain Tumor MRI Classification System</p>', unsafe_allow_html=True)

    # Initialize classifier
    classifier = BrainTumorClassifier()

    # Sidebar
    st.sidebar.markdown("## 🎛️ Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose a model for prediction:",
        list(classifier.models.keys()) if classifier.models else ["No models available"]
    )

    st.sidebar.markdown("## 📊 About")
    st.sidebar.markdown("""
    This application uses deep learning models to classify brain MRI images into four categories:
    
    - **Glioma**: Tumors in glial cells
    - **Meningioma**: Tumors in the meninges  
    - **Pituitary**: Tumors in the pituitary gland
    - **No Tumor**: Normal brain MRI scans
    
    **Note**: This is a research tool and should not be used for clinical diagnosis without proper medical validation.
    """)

    st.sidebar.markdown("""
## 📞 Contact

<div style="display: flex; align-items: center; gap: 10px; margin-top: 10px;">
    <a href="https://www.instagram.com/__u_t_t_a_m_a__/" target="_blank" style="display: flex; align-items: center; text-decoration: none;">
        <img src="https://img.icons8.com/ios-filled/50/ffffff/instagram-new.png" 
             style="width: 32px; height: 32px; border-radius: 50%; margin-right: 8px; transition: transform 0.2s;" 
             alt="Instagram"/>
        <span style="color: #fff; font-size: 1.1rem; font-weight: 500;">Uttam Bhise</span>
    </a>
</div>
<div style="display: flex; align-items: center; gap: 10px; margin-top: 10px;">
    <a href="mailto:uttamabhise@gmail.com" style="display: flex; align-items: center; text-decoration: none;">
        <img src="https://img.icons8.com/ios-filled/50/ffffff/new-post.png" 
             style="width: 28px; height: 28px; border-radius: 50%; margin-right: 8px; transition: transform 0.2s;" 
             alt="Email"/>
        <span style="color: #fff; font-size: 1.05rem; font-weight: 400;">uttamabhise@gmail.com</span>
    </a>
</div>
<div style="display: flex; align-items: center; gap: 10px; margin-top: 10px;">
    <img src="https://img.icons8.com/ios-filled/50/ffffff/brain.png" 
         style="width: 28px; height: 28px; border-radius: 50%; margin-right: 8px;" 
         alt="Project"/>
    <span style="color: #fff; font-size: 1.05rem; font-weight: 400;">Brain Tumor MRI Classification System</span>
</div>
""", unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload MRI Image</h2>', unsafe_allow_html=True)
        if st_lottie:
            st_lottie(load_lottieurl(LOTTIE_UPLOAD), height=120, key="upload-anim")
        upload_area = st.empty()
        with upload_area.container():
            uploaded_file = st.file_uploader(
                "Choose an MRI image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a brain MRI image in JPG, JPEG, or PNG format"
            )
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)
            # Preprocess image
            processed_image, display_image = classifier.preprocess_image(image)
            # Display preprocessed image
            st.markdown("### Preprocessed Image")
            st.image(display_image, caption="Preprocessed Image (224x224)", use_column_width=True)

    with col2:
        if uploaded_file is not None and selected_model in classifier.models:
            st.markdown('<h2 class="sub-header">🔍 Prediction Results</h2>', unsafe_allow_html=True)
            # Animated spinner during prediction
            with st.spinner('Predicting...'):
                if st_lottie:
                    st_lottie(load_lottieurl(LOTTIE_PREDICT), height=90, key="predict-anim")
                predicted_class, confidence_scores = classifier.predict(processed_image, selected_model)
            if predicted_class is not None:
                # Display prediction
                max_confidence = np.max(confidence_scores)
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Prediction: {predicted_class}</h3>
                    <p><strong>Confidence:</strong> {max_confidence:.3f} ({max_confidence:.1%})</p>
                    <p><strong>Model:</strong> {selected_model}</p>
                </div>
                """, unsafe_allow_html=True)
                # Confidence chart
                confidence_fig = classifier.create_confidence_chart(confidence_scores, selected_model)
                st.plotly_chart(confidence_fig, use_container_width=True)
                # Detailed confidence scores
                st.markdown("### Detailed Confidence Scores")
                confidence_df = pd.DataFrame({
                    'Tumor Type': classifier.class_names,
                    'Confidence': confidence_scores
                }).sort_values('Confidence', ascending=False)
                st.dataframe(confidence_df, use_container_width=True)
    
    # Model comparison section
    if uploaded_file is not None and len(classifier.models) > 1:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
        
        comparison_fig, comparison_df = classifier.create_model_comparison(processed_image)
        
        if comparison_fig is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(comparison_fig, use_container_width=True)
            
            with col2:
                st.markdown("### Comparison Table")
                st.dataframe(comparison_df, use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">ℹ️ Information</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This tool assists medical professionals in the preliminary classification of brain MRI images for tumor detection.
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Disclaimer
        This application is for research and educational purposes only. It should not replace professional medical diagnosis.
        """)
    
    with col3:
        st.markdown("""
        ### 🔬 Technology
        Built using deep learning models including Custom CNN, ResNet50, EfficientNet, and MobileNet architectures.
        """)

    # =====================
    # Model Evaluation Section
    # =====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📝 Model Evaluation</h2>', unsafe_allow_html=True)
    import json
    eval_json_path = "results/model_evaluation_results.json"
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            eval_results = json.load(f)
        eval_df = pd.DataFrame(eval_results).T  # Transpose for better display
        st.dataframe(eval_df, use_container_width=True)
    else:
        st.info("Model evaluation results not found.")

    # Show confusion matrices
    st.markdown("### Confusion Matrices")
    conf_matrix_dir = "results/confusion_matrices/"
    if os.path.exists(conf_matrix_dir):
        for model_name in (eval_results.keys() if 'eval_results' in locals() else []):
            img_path = f"{conf_matrix_dir}{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{model_name} Confusion Matrix", width=350)
    else:
        st.info("Confusion matrices not found.")

    # Show training history plots
    st.markdown("### Training History")
    history_imgs = [
        "results/custom_cnn_training_history.png",
        "results/efficientnet_initial_training.png",
        "results/efficientnet_fine_tune_training.png",
        "results/mobilenet_initial_training.png",
        "results/mobilenet_fine_tune_training.png",
        "results/resnet50_initial_training.png",
        "results/resnet50_fine_tune_training.png"
    ]
    for img in history_imgs:
        if os.path.exists(img):
            st.image(img, caption=os.path.basename(img).replace('_', ' ').replace('.png', '').title(), use_column_width=True)

    # =====================
    # Model Comparison Section
    # =====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🏆 Model Comparison</h2>', unsafe_allow_html=True)

    # Show comparison table
    comp_csv_path = "results/model_comparison_table.csv"
    if os.path.exists(comp_csv_path):
        comp_df = pd.read_csv(comp_csv_path)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Model comparison table not found.")

    # Show comparison plot
    comp_img_path = "results/model_comparison.png"
    if os.path.exists(comp_img_path):
        st.image(comp_img_path, caption="Model Comparison", use_column_width=True)
    else:
        st.info("Model comparison plot not found.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🧠 Brain Tumor MRI Classification System | Built with Streamlit and TensorFlow</p>
        <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 