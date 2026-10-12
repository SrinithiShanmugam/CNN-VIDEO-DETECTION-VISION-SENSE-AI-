import cv2
from ultralytics import YOLO

# ==========================================
# Load YOLO model
# ==========================================

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")
print("Starting camera...")


# ==========================================
# Open laptop camera
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# ==========================================
# Live Object Detection
# ==========================================

while True:

    # Read frame from camera
    success, frame = camera.read()

    if not success:
        print("ERROR: Cannot read camera.")
        break

    # Detect objects
    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    # Draw bounding boxes and names
    detected_frame = results[0].plot()

    # Show camera
    cv2.imshow(
        "Live Object Detection",
        detected_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Close camera
# ==========================================

camera.release()
cv2.destroyAllWindows()

print("Camera closed.")


