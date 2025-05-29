from pathlib import Path

# Create the script content
script_content = """
import cv2
from ultralytics import YOLO
import numpy as np

# === Configuration ===
video_path = 'input.mp4'  # Replace with your video file
output_path = 'output_detected.mp4'

# ROI for the fridge (adjusted to the screenshot you provided)
fridge_roi = (900, 100, 1150, 650)  # (x1, y1, x2, y2)
frame_threshold = 30  # 1 second at 30 FPS

# Load model (person detection)
model = YOLO('yolov8n.pt')  # lightweight and fast

# Setup video capture and output
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

stay_counter = 0
fridge_open = False
prev_roi_brightness = None
brightness_threshold = 15  # Adjust for sensitivity

def calculate_brightness(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people
    results = model(frame)[0]
    boxes = results.boxes
    person_in_roi = False

    # Draw fridge ROI
    x1, y1, x2, y2 = fridge_roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "Fridge", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # person
            continue
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            person_in_roi = True
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    if person_in_roi:
        stay_counter += 1
    else:
        stay_counter = 0

    if stay_counter >= frame_threshold:
        cv2.putText(frame, "Person near fridge > 1s", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Fridge state detection via brightness change
    fridge_crop = frame[y1:y2, x1:x2]
    curr_brightness = calculate_brightness(fridge_crop)

    if prev_roi_brightness is not None:
        delta = abs(curr_brightness - prev_roi_brightness)
        if delta > brightness_threshold:
            fridge_open = not fridge_open  # toggle state
            state = "OPEN" if fridge_open else "CLOSED"
            cv2.putText(frame, f"Fridge is now {state}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 165, 255) if fridge_open else (0, 255, 255), 2)

    prev_roi_brightness = curr_brightness

    # Show and save
    out.write(frame)
    cv2.imshow("Fridge Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
"""

# Save the script to a file
script_path = Path("fridge_monitor.py")  # Save in the current directory
script_path.write_text(script_content)

script_path.name  # Return just the filename for download link
