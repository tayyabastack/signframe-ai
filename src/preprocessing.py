
import cv2
import numpy as np
import torch


IMAGE_SIZE = 28


def get_hand_bbox(landmarks, frame_shape, padding=35):
    height, width = frame_shape[:2]

    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # Center of the detected hand
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    hand_width = max_x - min_x
    hand_height = max_y - min_y

    # Create a square crop with proportional padding
    box_size = int(
        max(hand_width, hand_height) * 1.25
    )

    box_size = max(box_size, 80)

    x1 = int(center_x - box_size / 2)
    y1 = int(center_y - box_size / 2)
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Keep crop inside webcam boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return x1, y1, x2, y2


def preprocess_hand(frame, bbox):
    x1, y1, x2, y2 = bbox

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to 28x28
    resized = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Normalize to [0, 1]
    normalized = (
        resized.astype(np.float32) / 255.0
    )

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    return tensor
