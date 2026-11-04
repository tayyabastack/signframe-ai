
# SignFrame AI

### Real-Time ASL Fingerspelling Recognition Using Deep Learning and Computer Vision

SignFrame AI is a real-time computer vision application designed to recognize American Sign Language (ASL) fingerspelling handshapes through a webcam and convert recognized letters into text.

The project combines a custom-trained Convolutional Neural Network (CNN), MediaPipe hand tracking, OpenCV, and PyTorch to create an interactive hand recognition system.

> **Project Status:** In Development

## Project Objectives

- Train a custom CNN from scratch using PyTorch.
- Recognize 24 static ASL alphabet handshapes (excluding J and Z).
- Detect and track hands in real time using MediaPipe.
- Display a green bounding box around the detected hand.
- Visualize 21 hand landmarks using red dots and green skeleton connections.
- Predict ASL letters using the trained CNN.
- Display prediction confidence and stability.
- Convert confirmed letters into text.
- Evaluate model performance on both test data and real webcam images.

## Technologies

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| PyTorch | CNN training and inference |
| OpenCV | Webcam capture and visualization |
| MediaPipe | Hand detection and landmark tracking |
| NumPy | Image preprocessing |
| pandas | Dataset loading |
| scikit-learn | Model evaluation |
| Matplotlib | Training and evaluation graphs |

## Dataset

The project uses the Sign Language MNIST dataset.

Dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

- 27,455 training images
- 7,172 testing images
- 28 × 28 grayscale images
- 24 ASL alphabet classes
- J and Z excluded because they require movement

## System Architecture

### Model Training

Dataset → Preprocessing → CNN Training → Validation → Saved Model → Evaluation

### Real-Time Recognition

Webcam → MediaPipe Hand Detection → Hand Cropping → Image Preprocessing → CNN Prediction → Prediction Stabilization → Text Composition

MediaPipe provides the hand landmarks and tracking information. The custom CNN is responsible for recognizing ASL letters.

## CNN Architecture

1. Input: 28 × 28 grayscale image
2. Convolutional layer: 32 filters + ReLU
3. Max pooling
4. Convolutional layer: 64 filters + ReLU
5. Max pooling
6. Flatten
7. Fully connected layer: 128 neurons + ReLU
8. Dropout
9. Output: 24 ASL letter classes

## Planned Features

- Live webcam recognition
- Moving green hand bounding box
- Green hand skeleton with 21 red landmark points
- CNN-based ASL letter prediction
- Confidence percentage
- Prediction stabilization
- Interactive text composition
- Keyboard controls for editing text
- Offline operation

## Project Structure

```text
signframe-ai/
├── data/
│   ├── raw/
│   ├── webcam_dev/
│   └── webcam_test/
├── assets/
├── src/
├── models/
├── results/
├── tests/
├── docs/
├── demo/
├── README.md
├── requirements.txt
└── .gitignore/
```

## Evaluation

The project will evaluate:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training and validation loss
- Real-world webcam recognition performance

Results will be added after model training and testing.

## Limitations

- Only 24 static ASL letters are supported.
- Dynamic letters J and Z are excluded.
- This is not a complete sign language translation system.
- Webcam performance may differ from dataset performance due to lighting, background, and hand positioning.

## Future Improvements

- Dynamic gesture recognition
- Hands-free letter confirmation
- Improved recognition across different users
- Word-level sign recognition

## Author

Personal AI and Computer Vision Portfolio Project