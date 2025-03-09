# Tennis-Tracker

A computer vision application for tennis ball tracking and speed analysis.

## Overview

This project uses computer vision and machine learning techniques to:
- Detect and track tennis balls in video footage
- Identify tennis court lines and transform the perspective
- Calculate and display the ball speed
- Generate output videos with tracking visualization

## Features

- Tennis ball detection using YOLO
- Court line detection using a custom ResNet50-based model
- Perspective transformation for top-down court view
- Real-time ball speed calculation
- Side-by-side video output showing tracking results

## Project Structure

```
Tennis-Tracker/
│
├── main.py                             # Main execution script
├── step_by_step_implementation.ipynb   # Jupyter notebook with implementation details
├── court_line_detector.py              # Court detection functionality
├── utils.py                            # Utility functions
├── tennis_court.jpg                    # Reference court image for visualization
│
├── models/                             # Pre-trained models
│   ├── keypoints_model.pth             # Model for court line detection
│   └── best.pt                         # YOLO model for ball tracking
│
└── input_videos/                       # Directory for input videos
    └── input_video_1.mp4               # Example input video
```

## Requirements

- Python
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Tennis-Tracker.git
   cd Tennis-Tracker
   ```

2. Install the required packages:
   ```
   pip install torch torchvision opencv-python ultralytics numpy matplotlib
   ```

## Usage

1. Place your input tennis videos in the `input_videos/` folder
2. Run the main script changing video_name variable:
    ```
    python main.py
    ```
3. The output video will be saved in the project directory

## Demo


<div align="center">
  <a href="https://www.youtube.com/watch?v=n8LUPDluGwQ">
     <div style="position: relative; display: inline-block;">
        <img src="https://img.youtube.com/vi/n8LUPDluGwQ/0.jpg" alt="Tennis-Tracker Demo" width="600">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
          <img src="https://img.shields.io/badge/▶%20Watch%20Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch Demo on YouTube">
        </div>
     </div>
  </a>
</div>
  </a>
</div>