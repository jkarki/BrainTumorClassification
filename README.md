# Brain Cancer Classification
Jay Karki | [LinkedIn](https://www.linkedin.com/in/jaykarki/)

Group project for COSC 525 at the University of Tennessee Knoxville. Built and evaluated multiple machine learning models to classify brain MRI images as tumor or no tumor, and by tumor type.

**Collaborators:** Grant Alderson, Utkarsh Karki, David Musili

## My Contributions

I was responsible for two parts of the pipeline:

**Base & Improved Logistic Regression (Sections 1–5b)**
- Loaded and preprocessed MRI image data (64x64 grayscale, flattened and normalized)
- Trained a baseline logistic regression model on raw pixel values
- Improved it by adding StandardScaler and class weight balancing to handle dataset imbalance
- Evaluated both models using accuracy, precision, recall, and F1

**ResNet18 Feature Extraction + Logistic Regression (Section 7a)**
- Used a pretrained ResNet18 model as a feature extractor to pull 512-dimensional deep features from each image
- Trained a logistic regression classifier on top of those features using 5-fold cross-validation
- This approach tested whether classical ML on deep features could compete with full fine-tuning

## Models & Results (YES/NO Tumor Detection)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Base Logistic Regression | 84.3% | 84.9% | 90.3% | 87.5% |
| Improved Logistic Regression | 86.3% | 87.5% | 90.3% | 88.9% |
| ResNet18 Features + LogReg | 80.3% | — | — | — |
| CNN (teammates) | 56.8% binary | — | — | — |
| Fine-Tuned ResNet18 CNN (teammates) | 92.1% | 93.5% | 93.5% | 93.5% |
| 4-Class ResNet18 (teammates) | 45.1% binary | — | — | — |

The Fine-Tuned ResNet18 CNN performed best overall. My Improved Logistic Regression was the strongest classical ML model and the second best overall.

## Data

MRI brain scan images categorized into 4 classes:
- No Tumor
- Glioma
- Meningioma
- Pituitary Tumor

Data was loaded and processed in Google Colab — not included in this repo due to size.

## Tools

- **Python:** NumPy, scikit-learn, PyTorch, TensorFlow/Keras, Matplotlib
- **Environment:** Google Colab
