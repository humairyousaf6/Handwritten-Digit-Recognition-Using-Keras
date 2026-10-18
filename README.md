# 🔢 Handwritten Digit Recognition Using Deep Learning and Keras

A deep learning project that builds a **handwritten digit recognition system** using Python, TensorFlow, and Keras.

The project trains a fully connected Artificial Neural Network (ANN) on the MNIST handwritten digit dataset to classify grayscale images into 10 categories, representing digits from **0 to 9**.

The notebook demonstrates the complete workflow, including data preprocessing, neural network construction, model training, performance evaluation, model serialization, and prediction on an external handwritten digit image.

## ✨ Features

- 🔢 Handwritten digit classification (0–9)
- 🧠 Artificial Neural Network (ANN) using Keras
- 📊 MNIST dataset containing 70,000 handwritten digit images
- 🖼️ Grayscale image preprocessing and normalization
- ⚡ Dense neural network with ReLU and Softmax activations
- 📈 Model training and accuracy evaluation
- 💾 Save and reload trained models using HDF5
- 🔍 Predict handwritten digits from external images
- 🎯 Convert prediction probabilities into readable digit labels

## 🎯 Project Objective

Handwritten digit recognition is a fundamental computer vision problem that involves identifying numerical characters from images.

The objective of this project is to develop a neural network capable of recognizing handwritten digits and demonstrate how deep learning models can learn patterns from image pixel data.

The project explores:

- How handwritten images are prepared for neural network training
- How fully connected neural networks classify numerical characters
- How Softmax probabilities are converted into predicted classes
- How model performance is evaluated using unseen test images
- How trained models can be saved and reused for image prediction

## 📁 Dataset

### MNIST Handwritten Digit Dataset

The project uses the MNIST dataset available through Keras.

MNIST contains **70,000 grayscale images** of handwritten digits from 0 to 9.

| Dataset | Images | Image Size |
|---|---:|---|
| Training | 60,000 | 28 × 28 |
| Testing | 10,000 | 28 × 28 |
| Total | 70,000 | 28 × 28 |

Each image represents one handwritten numerical character.

### Loading the Dataset

```python
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

## 🏗️ Neural Network Architecture

The notebook implements a fully connected feedforward neural network using the Keras Sequential API.

```text
MNIST Handwritten Digit Image
          │
          ▼
    Grayscale Image
        28 × 28
          │
          ▼
   Flatten / Reshape
      784 Features
          │
          ▼
     Dense Layer
      512 Neurons
    ReLU Activation
          │
          ▼
     Output Layer
       10 Neurons
   Softmax Activation
          │
          ▼
  Prediction Probabilities
       (Digits 0–9)
          │
          ▼
     Predicted Digit
```

### Model Implementation

```python
from keras import models
from keras import layers

network = models.Sequential()

network.add(
    layers.Dense(512, activation='relu', input_shape=(28 * 28,))
)

network.add(
    layers.Dense(10, activation='softmax')
)
```

### Architecture Details

| Component | Configuration |
|---|---|
| Model | Sequential Neural Network |
| Input | 784 pixel values |
| Hidden Layer | 512 neurons |
| Hidden Activation | ReLU |
| Output Layer | 10 neurons |
| Output Activation | Softmax |
| Optimizer | RMSprop |
| Loss Function | Categorical Crossentropy |
| Evaluation Metric | Accuracy |

**Note:** This project uses a fully connected neural network, not a Convolutional Neural Network (CNN).

## 🧠 Methodology

### 1. Data Loading

Load the MNIST training and testing datasets through Keras.

### 2. Data Preprocessing

Each 28 × 28 image is converted into a one-dimensional vector of 784 features.

Pixel values are normalized to the range 0–1.

```python
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
```

### 3. Label Encoding

Convert digit labels into one-hot encoded categorical vectors.

```python
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

### 4. Model Compilation

```python
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc']
)
```

### 5. Model Training

Train the network using 60,000 images.

```python
Network_History = network.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=128
)
```

### 6. Model Evaluation

Evaluate the trained model on the 10,000-image MNIST test set.

```python
test_loss, test_acc = network.evaluate(
    test_images,
    test_labels
)

print('test_acc:', test_acc)
```

### 7. Model Saving

Save the trained model for future use.

```python
network.save('HandDigitRecognition.h5')
```

### 8. Model Loading

Reload the saved model without retraining.

```python
from keras.models import load_model

network = load_model('HandDigitRecognition.h5')
```

### 9. Handwritten Digit Prediction

Load a handwritten digit image, preprocess it, and generate prediction probabilities.

```python
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

img = load_img('digit5.jpeg', color_mode='grayscale')

newimg_arr = img_to_array(img)
newimg_arr = newimg_arr.reshape((1, 28 * 28))
newimg_arr = newimg_arr.astype('float32') / 255

prediction = network.predict(newimg_arr)

NamesOfClasses = [
    'Zero', 'One', 'Two', 'Three', 'Four',
    'Five', 'Six', 'Seven', 'Eight', 'Nine'
]

PredictedDigit = NamesOfClasses[np.argmax(prediction)]

print(PredictedDigit)
```

## 📈 Experimental Results

The notebook reports the following training performance:

| Epoch | Training Loss | Training Accuracy |
|---|---:|---:|
| 1 | 0.2544 | 92.61% |
| 2 | 0.1044 | 96.90% |
| 3 | 0.0691 | 97.93% |
| 4 | 0.0510 | 98.53% |
| 5 | 0.0379 | 98.86% |

### Final Evaluation

| Metric | Result |
|---|---:|
| Final Training Accuracy | 98.86% |
| Test Accuracy | 8.92% |
| Test Loss | 2.2887 |
| Training Epochs | 5 |
| Batch Size | 128 |

### Result Interpretation

The network achieves high accuracy on the training dataset. However, the test accuracy is only **8.92%**, indicating that the model is not generalizing successfully in the notebook's recorded evaluation.

The discrepancy requires investigation before the model can be considered reliable for unseen handwritten digits.

Possible areas to investigate include data preprocessing, image-label alignment, training and evaluation consistency, and class prediction behavior.

### External Image Prediction

The notebook also tests a separate image named `digit5.jpeg`.

**Recorded prediction:**

```text
Predicted Digit: Five
```

The model assigns approximately **99.86% probability** to the digit Five for this individual image.

This single prediction does not establish reliable classification performance across unseen images.

## 🛠️ Tech Stack

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras
- Sequential Neural Network
- Dense Layers
- ReLU Activation
- Softmax Classification

### Data Processing
- NumPy
- Keras Image Preprocessing

### Development Environment
- Jupyter Notebook

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Handwritten-Digit-Recognition-Using-Keras.git
cd Handwritten-Digit-Recognition-Using-Keras
```

Replace `<your-username>` with your GitHub username.

### 2. Install Dependencies

```bash
pip install tensorflow keras numpy notebook pillow
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Hand_Digit_Recognition_System.ipynb
```

### 4. Run the Notebook

Execute the notebook cells sequentially:

1. Import libraries
2. Load the MNIST dataset
3. Preprocess images and labels
4. Construct the neural network
5. Train the model
6. Evaluate performance
7. Save the trained model
8. Reload the model
9. Load an external digit image
10. Generate a handwritten digit prediction

**Note:** To run the external image prediction cells, provide a suitable 28 × 28 grayscale image named `digit5.jpeg`, or update the image path. The notebook assumes the image already has compatible dimensions.

## 📂 Suggested Project Structure

```text
Handwritten-Digit-Recognition-Using-Keras/
│
├── Hand_Digit_Recognition_System.ipynb
│
├── digit5.jpeg
│
├── HandDigitRecognition.h5
│
└── README.md
```

The JPEG image and HDF5 model are required or generated artifacts respectively. They may need to be added separately to the repository.

The MNIST dataset is downloaded automatically using Keras.

## ⚠️ Limitations

- The recorded test accuracy is substantially lower than the training accuracy.
- The current evaluation does not demonstrate successful generalization.
- The model uses fully connected layers rather than convolutional image feature extraction.
- The notebook does not include a confusion matrix or per-class evaluation.
- External prediction requires an appropriately sized grayscale image.
- No validation split, early stopping, or model checkpoint selection is implemented.
- A single correct image prediction cannot establish model reliability.
- The causes of the low test accuracy have not been determined.

## 🔮 Future Improvements

- Investigate and resolve the low MNIST test accuracy.
- Verify preprocessing and label consistency across training and testing.
- Implement a Convolutional Neural Network (CNN).
- Add data augmentation for improved robustness.
- Introduce a validation split and early stopping.
- Evaluate precision, recall, F1-score, and confusion matrices.
- Visualize training and validation accuracy curves.
- Support automatic resizing and preprocessing of external images.
- Build a simple interface for uploading handwritten digit images.
- Export models using the modern Keras model format.

## ⭐ Project Highlights

- 🔢 Ten-class handwritten digit classification
- 🧠 Deep learning with TensorFlow and Keras
- 📊 MNIST image classification
- ⚙️ Neural network training and evaluation
- 💾 Model saving and reloading
- 🖼️ External image inference
- 📈 Experimental performance analysis
- 🔍 Practical introduction to computer vision

## 📌 Conclusion

This project demonstrates the implementation of a handwritten digit recognition pipeline using a fully connected neural network.

It covers the core stages of a supervised deep learning workflow, from dataset preparation and neural network training to model persistence and individual image prediction.

While the network achieves high training accuracy, the low recorded test accuracy highlights the importance of validating model generalization before using predictions in real-world applications.
