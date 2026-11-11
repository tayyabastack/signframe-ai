
import cv2

from src.hand_tracker import HandTracker
from src.preprocessing import (
    get_hand_bbox,
    preprocess_hand
)
from src.inference import SignPredictor
from src.stabilizer import PredictionStabilizer
from src.text_composer import TextComposer
from src.visualizer import (
    draw_hand,
    draw_interface
)


WINDOW_NAME = "SignFrame AI"


def main():
    print("\n" + "=" * 45)
    print("SignFrame AI — Real-Time ASL Recognition")
    print("=" * 45)

    tracker = HandTracker()
    predictor = SignPredictor()
    stabilizer = PredictionStabilizer()
    composer = TextComposer()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        tracker.close()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    print("\nWebcam ready!")
    print("\nKeyboard Controls:")
    print("ENTER     = Append stable predicted letter")
    print("SPACE     = Insert space")
    print("BACKSPACE = Delete last character")
    print("C         = Clear text")
    print("Q / ESC   = Quit application")
    print("\nShow your hand to the webcam.\n")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            # Mirror webcam for a natural preview.
            frame = cv2.flip(frame, 1)

            # Preserve a clean image for CNN inference.
            clean_frame = frame.copy()

            predicted_letter = None
            confidence = 0.0
            stable_letter = None

            # Detect 21 hand landmarks using MediaPipe.
            landmarks = tracker.detect(clean_frame)

            if landmarks:
                bbox = get_hand_bbox(
                    landmarks,
                    clean_frame.shape
                )

                image_tensor = preprocess_hand(
                    clean_frame,
                    bbox
                )

                if image_tensor is not None:
                    predicted_letter, confidence = (
                        predictor.predict(image_tensor)
                    )

                    stable_letter = stabilizer.update(
                        predicted_letter,
                        confidence
                    )
                else:
                    stabilizer.reset()

                # Draw green skeleton, red dots and bounding box.
                draw_hand(
                    frame,
                    landmarks,
                    bbox
                )

            else:
                stabilizer.reset()

            # Draw prediction and text composer interface.
            draw_interface(
                frame,
                predicted_letter,
                confidence,
                stable_letter,
                composer.text
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit application.
            if key in (ord("q"), 27):
                break

            # ENTER: Confirm stable predicted letter.
            elif key in (10, 13):
                if stable_letter is not None:
                    composer.append_letter(stable_letter)
                    print("Added letter:", stable_letter)
                    stabilizer.reset()
                else:
                    print(
                        "No stable letter available to confirm."
                    )

            # SPACE: Add space.
            elif key == 32:
                composer.add_space()

            # BACKSPACE: Delete previous character.
            elif key in (8, 127):
                composer.backspace()

            # C: Clear composed text.
            elif key == ord("c"):
                composer.clear()
                stabilizer.reset()
                print("Text cleared.")

    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

    print("\n" + "=" * 45)
    print("Final composed text:")
    print(composer.text)
    print("=" * 45)


if __name__ == "__main__":
    main()
