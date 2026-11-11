
import cv2

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Camera opened successfully!")
print("Press Q to quit.")

while True:
    # Read one frame from the camera
    success, frame = cap.read()

    if not success:
        print("Failed to read camera frame.")
        break

    # Display the live video
    cv2.imshow("SignFrame AI - Camera Test", frame)

    # Quit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera resources
cap.release()
cv2.destroyAllWindows()
