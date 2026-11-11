# SignFrame AI

### Real-Time ASL Fingerspelling Recognition Using Deep Learning and Computer Vision

**SignFrame AI** is a computer vision prototype that recognizes American Sign Language (ASL) fingerspelling handshapes using a webcam and converts confirmed letter predictions into text.

The project combines a custom Convolutional Neural Network (CNN) trained using PyTorch, MediaPipe hand tracking, and OpenCV to build an interactive, real-time hand gesture recognition application.

The goal is to explore how deep learning and computer vision can contribute to accessible communication technologies.

**Project Status:** Functional Prototype — Core Implementation Complete

**Technologies:** Python · PyTorch · OpenCV · MediaPipe · NumPy · scikit-learn

---

## 1. Project Overview

SignFrame AI combines two computer vision components:

- **MediaPipe Hand Landmarker:** Detects hands and tracks 21 hand landmarks in real time.
- **Custom PyTorch CNN:** Classifies cropped hand images into one of 24 static ASL fingerspelling letters.

The application displays live predictions, confidence scores, and hand landmarks. Users can confirm predicted letters to compose text interactively.

The system operates locally without requiring a cloud inference service.

**Important:** SignFrame AI is an experimental fingerspelling recognition prototype, not a complete or dependable sign language translation system.

## 2. Implemented Features

### Real-Time Hand Tracking

- Live webcam capture using OpenCV.
- Hand detection using MediaPipe.
- Green bounding box around the detected hand.
- Visualization of 21 hand landmarks using red dots.
- Green skeleton connections between hand landmarks.
- Continuous hand tracking during webcam operation.

### CNN-Based Fingerspelling Recognition

- Custom CNN developed using PyTorch.
- Classification of 24 static ASL letters.
- Real-time inference on webcam frames.
- Confidence score displayed for each prediction.
- Prediction stabilization to reduce rapidly changing outputs.
- Apple Silicon MPS acceleration when available.

### Interactive Text Composition

- Keyboard confirmation of stabilized predictions.
- Compose words and sentences letter by letter.
- Insert spaces between words.
- Delete previously confirmed characters.
- Clear the composed text.
- Display the final message when the application closes.

### Model Development and Evaluation

- Training on Sign Language MNIST.
- Evaluation on a held-out test dataset.
- Classification metrics and confusion matrix.
- Separate experimental webcam evaluation.
- Investigation of differences between dataset accuracy and real-world webcam recognition.

---

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| PyTorch | CNN architecture, training, and inference |
| OpenCV | Webcam capture and visualization |
| MediaPipe | Hand detection and landmark tracking |
| NumPy | Image preprocessing |
| pandas | Dataset loading and manipulation |
| scikit-learn | Classification metrics and evaluation |
| Matplotlib | Model evaluation visualizations |
| Pillow | Image processing |

## 4. Dataset

The CNN was trained using the **Sign Language MNIST** dataset.

**Dataset source:** https://www.kaggle.com/datasets/datamunge/sign-language-mnist

### Dataset Characteristics

| Property | Value |
|---|---|
| Training images | 27,455 |
| Test images | 7,172 |
| Image resolution | 28 × 28 pixels |
| Image format | Grayscale |
| Classes | 24 |
| Excluded letters | J and Z |

The dataset contains labeled images representing static ASL fingerspelling handshapes.

Letters J and Z are excluded because they involve movement and cannot be represented completely using a single static image.

### Preprocessing

The original dataset images are:

1. Loaded from CSV files.
2. Converted into 28 × 28 grayscale arrays.
3. Normalized to the range [0, 1].
4. Converted into PyTorch tensors.
5. Assigned class indices from 0 to 23.

For webcam inference, MediaPipe detects the hand, and the application crops the corresponding image region, converts it to grayscale, resizes it to 28 × 28 pixels, and passes it to the CNN.

---

## 5. System Architecture

### CNN Training Pipeline

Dataset → Image Preprocessing → CNN Training → Saved Model → Test Evaluation

### Real-Time Recognition Pipeline

Webcam → MediaPipe Hand Detection → Hand Cropping → Grayscale Preprocessing → CNN Inference → Prediction Stabilization → Text Composition

MediaPipe is responsible for hand detection and landmark tracking.

The CNN performs letter classification using the processed image rather than the landmark coordinates.

## 6. CNN Architecture

The model is a custom Convolutional Neural Network implemented using PyTorch.

| Layer | Configuration |
|---|---|
| Input | 1 × 28 × 28 grayscale image |
| Convolution 1 | 32 filters, 3 × 3 kernel, ReLU |
| Max Pooling 1 | 2 × 2 |
| Convolution 2 | 64 filters, 3 × 3 kernel, ReLU |
| Max Pooling 2 | 2 × 2 |
| Flatten | 3,136 features |
| Fully Connected | 128 neurons, ReLU |
| Dropout | 0.3 |
| Output | 24 classes |

The network learns visual features from grayscale hand images and produces class scores for the supported ASL letters.

Softmax is applied during inference to display prediction confidence.

---

## 7. Model Evaluation

### Sign Language MNIST Evaluation

The original CNN achieved:

**Test Accuracy: 92.81%**

This result was measured on the held-out Sign Language MNIST test dataset containing 7,172 images.

The result demonstrates classification performance on the original dataset. It does not represent real-time webcam recognition accuracy.

### Webcam Evaluation

A separate experimental dataset was collected using a webcam.

| Dataset | Images |
|---|---:|
| Webcam development dataset | 600 |
| Webcam test dataset | 120 |
| Supported letters | 24 |

The webcam images were collected from a single signer using the application's preprocessing pipeline.

The original CNN was evaluated on the 120-image webcam test set.

**Original CNN Webcam Accuracy: 13.33%**

An experimental fine-tuning run was also conducted:

| Model | Webcam Test Accuracy |
|---|---:|
| Original CNN | 13.33% |
| Experimental fine-tuned CNN | 20.83% |

Fine-tuning improved accuracy by 7.50 percentage points on this small test set. However, recognition remained unreliable.

The fine-tuned model was not adopted for the main application. SignFrame AI currently uses the original CNN checkpoint.

### Key Finding: Domain Shift

Although the CNN achieved 92.81% accuracy on Sign Language MNIST, webcam performance was substantially lower.

Potential contributing factors include:

- Differences in lighting and background.
- Hand position and orientation.
- Image cropping and scaling.
- Differences between dataset images and webcam images.
- Limited diversity in the experimental webcam dataset.

This demonstrates a common machine learning challenge: **a model that performs well on a benchmark dataset may not generalize to real-world inputs.**

The webcam test results are preliminary because the dataset contains only 120 images from one signer.

---

## 8. Project Structure

```text
signframe-ai/
├── assets/
│   └── hand_landmarker.task
├── data/
│   └── raw/
│       ├── sign_mnist_train.csv
│       └── sign_mnist_test.csv
├── models/
│   └── best_cnn.pth
├── src/
│   ├── camera_test.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── hand_tracker.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── stabilizer.py
│   ├── text_composer.py
│   ├── train.py
│   ├── visualize_dataset.py
│   ├── visualizer.py
│   └── webcam_app.py
├── results/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore
```

The model checkpoint, downloaded MediaPipe asset, and datasets are stored locally and excluded from Git tracking.

---

## 9. Installation and Setup

### Prerequisites

- Python 3.12
- A working webcam
- macOS, Windows, or Linux
- Git

The application was developed and tested on an Apple Silicon Mac.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/tayyabastack/signframe-ai.git
cd signframe-ai
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it.

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Dataset

Download Sign Language MNIST from:

https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Extract the CSV files and place them at:

```text
data/raw/sign_mnist_train.csv
data/raw/sign_mnist_test.csv
```

Create the directories if they do not already exist.

### Step 5 — Prepare the CNN Checkpoint

The trained model checkpoint is not currently distributed through this repository.

To reproduce it locally, run:

```bash
python -m src.train
```

The application expects the trained checkpoint at:

```text
models/best_cnn.pth
```

### Step 6 — Download the MediaPipe Hand Landmarker

Download the compatible MediaPipe Hand Landmarker model from the official documentation:

https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

Save the model asset as:

```text
assets/hand_landmarker.task
```

### Step 7 — Run SignFrame AI

```bash
python -m src.webcam_app
```

Allow camera access if prompted by your operating system.

The application opens a webcam window displaying hand tracking, predictions, and composed text.

**Note:** The application requires both the CNN checkpoint and the MediaPipe model asset to be available locally. The setup has been tested in the original development environment; fresh-clone installation has not yet been independently verified.

---

## 10. Application Controls

| Key | Action |
|---|---|
| Enter | Confirm and append the stable predicted letter |
| Space | Insert a space |
| Backspace | Delete the previous character |
| C | Clear composed text |
| Q or Esc | Exit the application |

The application requires a sufficiently stable prediction before a letter can be confirmed.

The user must focus the webcam window for keyboard input to be detected.

### Example Interaction

1. Position your hand in front of the webcam.
2. Form a supported ASL fingerspelling handshape.
3. Wait for a stable prediction.
4. Press Enter to confirm the letter.
5. Repeat to compose a word.
6. Press Space to separate words.
7. Press Q to exit and print the final text.

An end-to-end prototype demonstration successfully composed the text:

**HELLO**

This demonstration confirms that the application components work together, but it does not establish reliable recognition across all supported letters.

---

## 11. Current Limitations

SignFrame AI is an experimental computer vision application with the following limitations:

- Only 24 static ASL fingerspelling letters are supported.
- Dynamic letters J and Z are not recognized.
- The CNN was trained primarily on Sign Language MNIST images.
- Real-world webcam recognition accuracy remains limited.
- Predictions are sensitive to image conditions and hand orientation.
- Only one hand is processed for recognition.
- Users must manually confirm recognized letters.
- The system does not recognize complete ASL words, grammar, facial expressions, or continuous signing.
- The system is not suitable for dependable accessibility-critical communication.

The application should be considered a research and portfolio prototype rather than a production-ready assistive communication tool.

---

## 12. Future Improvements

### Accessibility and Communication

- Text-to-speech for composed messages.
- Full-screen message display for face-to-face communication.
- Message editing and correction before speaking.
- A modern, accessible application dashboard.

### Machine Learning

- Improved training data diversity.
- Better webcam preprocessing.
- Recognition across multiple users and hand orientations.
- More robust evaluation on independent signers.
- Improved prediction confidence calibration.
- Exploration of landmark-based or hybrid classification models.

### Extended Recognition

- Dynamic recognition of J and Z.
- Continuous fingerspelling recognition.
- Word-level gesture recognition.

These features are planned and are not part of the current implementation.

---

## 13. Project Motivation

SignFrame AI was developed as a practical exploration of deep learning, real-time computer vision, and human-computer interaction.

The project covers the complete machine learning development process, including dataset preparation, CNN implementation, model training, evaluation, webcam integration, and interactive inference.

A key learning outcome was understanding the difference between benchmark performance and real-world generalization.

The long-term motivation is to explore how computer vision systems can support more accessible communication experiences.

---

## 14. Author

**Tayyaba Ilyas**

MSc Computer Science — Philipps-Universität Marburg

GitHub: https://github.com/tayyabastack

**Project Repository:** https://github.com/tayyabastack/signframe-ai

---

*SignFrame AI is an independent educational and portfolio project focused on deep learning, computer vision, and accessibility-oriented human-computer interaction.*