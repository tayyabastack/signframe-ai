
# SignFrame AI

### Real-Time ASL Fingerspelling Recognition with Deep Learning

SignFrame AI is a computer vision application that recognizes American Sign Language (ASL) fingerspelling letters through a webcam and converts predictions into text and speech.

Built with **PyTorch, MediaPipe, and OpenCV**, the application combines a custom CNN, real-time hand tracking, prediction stabilization, and an interactive communication interface.

The project focuses on real-world gesture recognition, model adaptation, and human-computer interaction.

## Features

- Real-time recognition of 24 static ASL letters (excluding J and Z).
- Hand detection using MediaPipe's 21 hand landmarks.
- Custom PyTorch CNN for fingerspelling classification.
- Live predictions with confidence scores.
- Prediction stabilization before letter confirmation.
- Text composition and message editing.
- Text-to-speech for composed messages.
- Full-screen-style communication display.
- Local inference without cloud APIs.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| PyTorch | CNN training and inference |
| OpenCV | Webcam capture and interface |
| MediaPipe | Hand tracking and localization |
| NumPy | Image preprocessing |
| scikit-learn | Model evaluation |
| pandas | Dataset processing |
| Matplotlib | Evaluation visualizations |

Developed and tested on macOS with Apple Silicon (MPS).

## How It Works

The application uses MediaPipe to locate the hand in each webcam frame. The detected region is cropped, converted to grayscale, and resized to match the CNN input.

The CNN predicts one of 24 fingerspelling classes. Predictions are stabilized across frames before users confirm letters to build messages.

### Recognition Pipeline

```text
Webcam
   |
   v
MediaPipe Hand Detection
   |
   v
Hand Cropping & Preprocessing
   |
   v
PyTorch CNN (24 Classes)
   |
   v
Prediction Stabilization
   |
   v
Text Composition
   |
   +-- Message Editing
   +-- Text-to-Speech
   +-- Communication Display
```

MediaPipe landmarks are used for hand localization and visualization. The CNN classifies grayscale hand images rather than landmark coordinates.

## Model Architecture

The recognition model is a custom Convolutional Neural Network implemented in PyTorch.

| Layer | Configuration |
|---|---|
| Input | 28 × 28 grayscale |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Flatten | 3,136 features |
| Dense | 128 neurons, ReLU |
| Dropout | 0.3 |
| Output | 24 classes |

Softmax is applied during inference to obtain class probabilities.

The application loads the feedback-adapted checkpoint from `models/feedback_candidate.pth`.

## Dataset & Training

The baseline CNN was trained on [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

| Dataset | Images |
|---|---:|
| Training | 27,455 |
| Testing | 7,172 |
| Image size | 28 × 28 |
| Classes | 24 |

Letters J and Z are excluded because they require movement.

### Model Adaptation

Although the baseline model performed well on Sign Language MNIST, initial webcam testing showed poor generalization to real-world inputs.

A human-in-the-loop feedback workflow was developed to collect labeled webcam examples through confirmed and corrected predictions.

The feedback dataset eventually contained **611 labeled images** across the 24 supported letters.

These examples were used to adapt the CNN to webcam inputs. The final application uses a pretrained checkpoint; feedback collection and retraining controls are not exposed to users.

## Model Evaluation

| Model / Evaluation | Accuracy |
|---|---:|
| Baseline CNN — MNIST test | 92.81% |
| Baseline CNN — webcam test (120 images) | 13.33% |
| Early fine-tuned CNN — webcam test | 20.83% |
| Feedback CNN — validation (31 images) | 77.42% |
| Feedback CNN — MNIST test | 91.22% |

The feedback CNN results above are from an earlier experiment using 146 labeled webcam examples, not the final 611-image training run.

The model was subsequently retrained with additional feedback data. Independent webcam accuracy for the final checkpoint has not yet been established.

**Key observation:** High benchmark accuracy does not necessarily translate to reliable webcam recognition. Lighting, background, hand orientation, and preprocessing differences can significantly affect performance.

## Project Structure

```text
signframe-ai/
├── assets/
│   └── hand_landmarker.task
├── data/
├── models/
│   ├── best_cnn.pth
│   └── feedback_candidate.pth
├── src/
│   ├── communication.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── hand_tracker.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── stabilizer.py
│   ├── text_composer.py
│   ├── train.py
│   ├── visualizer.py
│   └── webcam_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

**Requirements:** Python 3.12, a webcam, and Git.

### 1. Clone the repository

```bash
git clone https://github.com/tayyabastack/signframe-ai.git
cd signframe-ai
```

### 2. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment using `.venv\Scripts\activate`.

### 3. Prepare the model files

The application requires:

- `models/feedback_candidate.pth` — trained CNN checkpoint.
- `assets/hand_landmarker.task` — MediaPipe Hand Landmarker model.

The MediaPipe asset is available from the [official documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

The CNN checkpoint must be obtained separately if it is not included in the repository. Running the baseline training script alone will not reproduce the feedback-adapted model.

### 4. Run the application

```bash
python -m src.webcam_app
```

Allow webcam access when prompted.

## Keyboard Controls

| Key | Action |
|---|---|
| Enter | Confirm predicted letter |
| Space | Insert space |
| Backspace | Delete character |
| E | Edit message |
| V | Speak message |
| F | Open communication display |
| C | Clear message |
| Q / Esc | Quit |

In editing mode, Enter saves changes and Esc cancels. In communication display mode, F or Esc returns to the dashboard.

## Limitations

- Supports static fingerspelling only; J and Z are not recognized.
- Does not recognize continuous ASL signing or grammar.
- Performance varies with lighting, background, and hand orientation.
- Feedback training data came from a single signer.
- Final-model accuracy on independent signers remains untested.
- Users must manually confirm predicted letters.

SignFrame AI is an experimental prototype, not a dependable sign language translation or accessibility-critical communication system.

## Future Work

- Improve recognition across different users and environments.
- Explore landmark-based and hybrid CNN architectures.
- Support dynamic letters J and Z.
- Investigate continuous fingerspelling recognition.
- Evaluate real-time latency and cross-platform compatibility.

## Author

**Tayyaba Ilyas**  

[GitHub](https://github.com/tayyabastack) | [Project Repository](https://github.com/tayyabastack/signframe-ai)
