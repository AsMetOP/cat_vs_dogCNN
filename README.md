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
✅ ReLU Activation is used in all Conv layers
✅ Sigmoid Activation is used in the final layer for binary classification


🧪 Model Compilation & Training
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset,
          epochs=10,
          validation_data=validation_dataset)


🔎 Predictions
test_image = cv2.imread('catty1.jpeg')
test_image = cv2.resize(test_image, (256, 256))
test_input = test_image.reshape((1, 256, 256, 3))
prediction = model.predict(test_input)
print("Dog" if prediction > 0.5 else "Cat")


📈 Results
Achieved ~85–90% accuracy after 10 epochs
Accurate prediction on new unseen images
Stable training using Batch Normalization

📌 Concepts Covered
CNN (Convolutional Neural Networks)
ReLU and Sigmoid Activations
Batch Normalization
Data Normalization (0-255 to 0-1)
Image resizing and reshaping
Overfitting prevention with Dropout
Model evaluation with validation set
