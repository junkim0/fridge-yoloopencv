import cv2
from ultralytics import YOLO
import numpy as np

# === Configuration ===
video_path = 'input.mp4'  # Replace with your video file
output_path = 'output_detected.mp4'

# ROI for the fridge (x=785, y=55, width=185, height=470)
fridge_roi = (785, 55, 970, 525)  # (x1, y1, x2, y2) where x2=x+width, y2=y+height
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

def draw_info_panel(frame, info_dict, start_y=30):
    # Draw semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, start_y + 30*len(info_dict)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw text
    y = start_y
    for label, value in info_dict.items():
        text = f"{label}: {value}"
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people
    results = model(frame)[0]
    boxes = results.boxes
    person_in_roi = False
    person_count = 0
    person_near_fridge = 0

    # Draw fridge ROI
    x1, y1, x2, y2 = fridge_roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "Fridge ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Process person detections
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id != 0:  # person
            continue
        person_count += 1
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        
        # Draw person detection with confidence
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (bx1, by1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            person_in_roi = True
            person_near_fridge += 1
            # Draw line from person to fridge
            cv2.line(frame, (cx, cy), ((x1 + x2)//2, (y1 + y2)//2), 
                    (0, 255, 255), 2)

    # Update fridge state based on person presence
    if person_in_roi:
        stay_counter += 1
        if stay_counter >= frame_threshold and not fridge_open:
            fridge_open = True
            cv2.putText(frame, "* FRIDGE OPENED *", (width//2 - 200, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
    else:
        stay_counter = 0
        if fridge_open:
            fridge_open = False
            cv2.putText(frame, "* FRIDGE CLOSED *", (width//2 - 200, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # Prepare info dictionary
    info = {
        "Total People": person_count,
        "People near Fridge": person_near_fridge,
        "Time near Fridge": f"{stay_counter/fps:.1f}s",
        "Fridge State": "OPEN" if fridge_open else "CLOSED"
    }
    
    # Draw information panel
    draw_info_panel(frame, info)

    # Show and save
    out.write(frame)
    cv2.imshow("Fridge Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
