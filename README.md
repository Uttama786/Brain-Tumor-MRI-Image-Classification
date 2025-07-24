# 🧠 Brain Tumor MRI Image Classification

## Overview
This project implements a deep learning-based solution for classifying brain MRI images into multiple tumor categories. It supports both custom CNN and transfer learning models (ResNet50, EfficientNet, MobileNet) and provides a user-friendly Streamlit web application for real-time predictions.

## Dataset
- **Source:** Labeled MRI Brain Tumor Dataset ([Roboflow Universe](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset))
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Total Images:** 2,443
- **Splits:**
  - Train: 1,695
  - Validation: 502
  - Test: 246
- **Structure:**
  - `data/train/<class>/`
  - `data/valid/<class>/`
  - `data/test/<class>/`

## Project Structure
```
Brain Tumor MRI Image Classification/
├── app/                  # Streamlit web app
├── models/               # Trained models (.h5)
├── results/              # Training plots, confusion matrices
├── src/                  # Core Python modules
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── ...
```

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure your data is organized as:**
   ```
   data/train/glioma/
   data/valid/glioma/
   data/test/glioma/
   ...
   ```

## Training
Train a model (e.g., Custom CNN):
```bash
python train_models.py --custom-cnn --epochs 5
```
Or train all models:
```bash
python train_models.py --all --epochs 10
```

## Evaluation
Evaluate all trained models and generate comparison plots:
```bash
python train_models.py --evaluate
```

## Running the Web App
Launch the Streamlit web application:
```bash
python -m streamlit run app/streamlit_app.py
```
Then open your browser to [http://localhost:8501](http://localhost:8501)

## Usage
- **Upload MRI images** via the web app
- **Get predictions** for tumor type and confidence scores
- **Compare models** if multiple are trained

## Results
- Training history and confusion matrices are saved in the `results/` folder
- Trained models are saved in the `models/` folder

## Troubleshooting
- If you see errors about missing models, train a model first
- For data issues, ensure your folders match the structure above

## License
Dataset: CC BY 4.0 (see dataset source)
Project code: MIT License

---
**For research and educational use only. Not for clinical diagnosis.** 