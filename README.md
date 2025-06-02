# 🐱🐶 Cat vs Dog Image Classifier using CNN

A deep learning project that classifies images of cats and dogs using Convolutional Neural Networks (CNNs) in TensorFlow and Keras.

## 🚀 Project Overview

This project uses the popular **Dogs vs Cats** dataset from Kaggle and builds a CNN model from scratch. It includes:
- Dataset loading & preprocessing
- Data normalization
- Custom CNN architecture
- Model training & validation
- Prediction on custom test images

## 🧠 Technologies Used

- Python
- TensorFlow & Keras
- OpenCV (for image processing)
- Matplotlib (for image display)
- Kaggle API (to fetch dataset)

---

## 📂 Dataset

- **Source**: [Kaggle - Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **Structure**:
/train/ → Cat and Dog training images
/test/ → Cat and Dog validation images

---

## 📸 Model Architecture (CNN)

python
Conv2D (32 filters) → BatchNorm → MaxPooling
Conv2D (64 filters) → BatchNorm → MaxPooling
Conv2D (128 filters) → BatchNorm → MaxPooling
Flatten → Dense(128) → Dropout(0.1)
→ Dense(64) → Dropout(0.1)
→ Dense(1, activation='sigmoid')
