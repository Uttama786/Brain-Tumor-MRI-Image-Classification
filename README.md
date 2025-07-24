# Brain Tumor MRI Image Classification

## Overview
This project provides a deep learning-based solution for classifying brain MRI images into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. It features a user-friendly Streamlit web application for uploading MRI images, visualizing predictions, and comparing the performance of multiple neural network architectures (Custom CNN, ResNet50, EfficientNet, and MobileNet).

## Data
- **Download the dataset:** [Google Drive Link]([https://drive.google.com/open?id=1DAyorv1mwSdblsM1N67erQpa3rjAJfgl&usp=drive_fs](https://drive.google.com/file/d/1DAyorv1mwSdblsM1N67erQpa3rjAJfgl/view?usp=sharing))
- After downloading, extract the dataset and place it in the `data/` directory, preserving the folder structure:
  - `data/train/<class>/`
  - `data/valid/<class>/`
  - `data/test/<class>/`

## Project Structure
```
Brain Tumor MRI Image Classification/
├── app/
│   └── streamlit_app.py            # Streamlit web application
├── data/
│   ├── train/                      # Training images (by class)
│   ├── valid/                      # Validation images (by class)
│   └── test/                       # Test images (by class)
├── models/
│   ├── 
├── notebooks/                      # (Optional) Jupyter notebooks for exploration
├── quick_start.py                  # Quick start script (if available)
├── requirements.txt                # Python dependencies
├── results/
│   ├── confusion_matrices/
├── src/
│   ├── custom_cnn.py               # Custom CNN model definition
│   ├── model_evaluation.py         # Model evaluation utilities
│   ├── transfer_learning.py        # Transfer learning model training
│   └── utils.py                    # Data loading and preprocessing utilities
├── test_models.py                  # Script to test models
├── train_models.py                 # Script to train models
└── README.md                       # Project documentation (this file)
```

## Setup & Usage
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Download and extract the dataset** to the `data/` directory as described above.
3. **Train models:**
   - To train all models and evaluate:
     ```sh
     python train_models.py --all --evaluate
     ```
   - To train a specific model (e.g., Custom CNN):
     ```sh
     python train_models.py --custom-cnn
     ```
4. **Run the web app:**
   ```sh
   streamlit run app/streamlit_app.py
   ```

## Main Components
- **app/streamlit_app.py**: The main web application for image upload, prediction, and visualization.
- **src/**: Contains model definitions, training, evaluation, and utility scripts.
- **models/**: Stores trained model weights.
- **results/**: Stores evaluation results, plots, and confusion matrices.
- **data/**: Place your dataset here (see above).

## Citation & Contact
If you use this project or dataset, please cite the original authors and sources as appropriate.

For questions or contributions, contact: **Uttam Bhise** (uttamabhise@gmail.com) 
