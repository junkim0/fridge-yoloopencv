# Fridge Monitor

A Python application that uses computer vision to monitor fridge usage by detecting when people interact with the fridge.

## Features

- Real-time person detection using YOLOv8
- Tracks when people are near the fridge
- Determines fridge state (OPEN/CLOSED) based on person proximity
- Visual indicators for person detection and fridge state
- Saves processed video with annotations

## Development Journey

This project went through two main approaches for detecting fridge state:

### Current Version: Person Proximity Detection
The current version uses a more reliable approach based on person proximity:
- Detects when a person stays near the fridge for 1 second
- Assumes fridge is opened when person remains in the area
- Closes when person leaves
- More reliable as it's based on actual human interaction
- Eliminates false positives from lighting changes

### Previous Version: Brightness-Based Detection
The previous version (available in `old_versions/fridge_monitor_brightness.py`) used a different approach:
- Monitored brightness changes in the fridge area
- Assumed fridge was opened when brightness changed significantly
- Included real-time brightness visualization
- Had issues with:
  - False positives from ambient lighting changes
  - Missed detections when lighting change was gradual
  - Inconsistent behavior in different lighting conditions
  - Multiple state toggles during a single open/close action

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/junkim0/fridge-yoloopencv.git
cd fridge-yoloopencv
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. To use with your own video:
   - Either rename your video file to `input.mp4`
   - Or modify the `video_path` variable in `fridge_monitor.py`

2. Run the script:
```bash
python fridge_monitor.py
```

3. The processed video will be saved as `output_detected.mp4`

4. Press 'q' to quit the program

## How it Works

- The script uses YOLOv8 for person detection
- A region of interest (ROI) is defined around the fridge area
- When a person stays near the fridge for 1 second, it's considered opened
- When they leave the area, it's considered closed
- Visual indicators show:
  - Blue rectangle: Fridge area
  - Green boxes: Detected people with confidence scores
  - Yellow lines: Connection between person and fridge when nearby
  - Information panel: People count, fridge state, and timing

## Trying Different Approaches

If you want to experiment with the brightness-based detection:
1. Copy `old_versions/fridge_monitor_brightness.py` to your working directory
2. Run it the same way as the main script
3. You'll see additional visualizations:
   - Real-time brightness graph
   - Brightness threshold line
   - Brightness delta values 