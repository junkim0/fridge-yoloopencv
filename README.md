# Fridge Monitor

A Python application that uses computer vision to monitor fridge usage by detecting when people interact with the fridge.

## Features

- Real-time person detection using YOLOv8
- Tracks when people are near the fridge
- Determines fridge state (OPEN/CLOSED) based on person proximity
- Visual indicators for person detection and fridge state
- Saves processed video with annotations

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy

## Installation

1. Clone this repository:
```bash
git clone [your-repo-url]
cd fridge-monitor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input video file in the project directory and rename it to `input.mp4`, or modify the `video_path` variable in `fridge_monitor.py`

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
  - Green boxes: Detected people
  - Yellow lines: Connection between person and fridge when nearby
  - Information panel: People count, fridge state, and timing 