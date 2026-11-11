
import cv2


GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# Connections between MediaPipe's 21 hand landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand(frame, landmarks, bbox):
    height, width = frame.shape[:2]

    points = [
        (int(p.x * width), int(p.y * height))
        for p in landmarks
    ]

    # Green skeleton.
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            GREEN,
            2
        )

    # Red landmark dots.
    for point in points:
        cv2.circle(frame, point, 4, RED, -1)

    # Green bounding box.
    x1, y1, x2, y2 = bbox
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        GREEN,
        2
    )


def draw_interface(
    frame,
    predicted_letter,
    confidence,
    stable_letter,
    composed_text
):
    height, width = frame.shape[:2]

    # Top information panel.
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 105),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "SignFrame AI",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        GREEN,
        2
    )

    prediction = (
        f"Prediction: {predicted_letter} "
        f"({confidence * 100:.1f}%)"
        if predicted_letter
        else "Prediction: No hand"
    )

    cv2.putText(
        frame,
        prediction,
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1
    )

    stable_text = stable_letter or "Uncertain"

    cv2.putText(
        frame,
        f"Stable letter: {stable_text}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GREEN if stable_letter else WHITE,
        1
    )

    # Bottom text panel.
    cv2.rectangle(
        frame,
        (0, height - 100),
        (width, height),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "Text: " + composed_text[-35:],
        (15, height - 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        "Enter: Add | Space | Backspace | C: Clear | Q: Quit",
        (15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        GREEN,
        1
    )
