# Safety Vest ROI Detection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]
[![OpenCV](https://img.shields.io/badge/OpenCV-enabled-green)]
[![Computer Vision](https://img.shields.io/badge/domain-computer%20vision-orange)]

ROI (Region of Interest) based computer vision project for detecting whether a worker is wearing a safety vest in video footage. The system analyzes only a predefined inspection area, extracts fluorescent yellow and orange colors in HSV space, and visualizes the result with a dedicated side panel.

Author: YIRANG JUNG  
License: Apache License 2.0

## Overview

This repository contains two core scripts:

- `safety_vest_roi_from_image_video.py`: main analysis pipeline for processing a demo video and generating a result video with an information side panel
- `safety_wave_demo_generator.py`: synthetic demo generator that creates a simple worker movement and hand-waving scenario for testing

The main detector:
- defines a Region of Interest (ROI)
- converts the ROI to HSV color space
- extracts yellow and orange pixels that resemble fluorescent safety vest colors
- computes vest pixel ratio inside the ROI
- determines safety vest presence by thresholding the ratio
- renders the result to a side panel to avoid text overlap
- saves the processed result video

## Repository Structure

```text
safety-vest-roi-detection/
├─ demo/
│  └─ safety_wave_demo.mp4
├─ outputs/
│  └─ .gitkeep
├─ docs/
│  └─ .gitkeep
├─ safety_vest_roi_from_image_video.py
├─ safety_wave_demo_generator.py
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ NOTICE
└─ README.md
```

## Features

- ROI-based inspection instead of analyzing the entire frame
- HSV color thresholding for fluorescent vest-like colors
- Morphological noise removal using open and close operations
- Vest ratio calculation inside ROI
- Dedicated side panel visualization for status, threshold, frame index, and mask preview
- Synthetic demo video generation for reproducible testing
- Processed output video export

## Requirements

- Python 3.9 or later recommended
- opencv-python
- numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Included Demo File

The repository includes a demo video here:

```text
demo/safety_wave_demo.mp4
```

This file can be used immediately with the main script.

## How to Run

### 1) Generate a demo video again if needed

```bash
python safety_wave_demo_generator.py
```

This creates:

```text
safety_wave_demo.mp4
```

### 2) Run the safety vest detection pipeline

```bash
python safety_vest_roi_from_image_video.py
```

By default, the script reads:

```text
safety_wave_demo.mp4
```

and writes:

```text
result_safety_wave_demo_panel.mp4
```

If you want to use the included demo file from the repository folder structure, update `VIDEO_PATH` like this:

```python
VIDEO_PATH = "demo/safety_wave_demo.mp4"
```

## Main Workflow

1. Load video input with OpenCV
2. Define ROI based on relative frame coordinates
3. Crop the ROI from each frame
4. Convert ROI from BGR to HSV
5. Extract yellow and orange masks with `cv2.inRange`
6. Remove noise using morphological operations
7. Compute vest pixel ratio inside the ROI
8. Decide `OK` or `NOT DETECTED` using the threshold
9. Render ROI and side panel visualization
10. Save the processed result video

## Core Detection Logic

The detector uses fluorescent color extraction rather than a deep learning object detector. This makes the system lightweight and easy to understand, but also more sensitive to lighting conditions, background colors, and camera exposure.

## Limitations

- Sensitive to illumination changes
- May falsely detect yellow or orange background objects as a vest
- Does not explicitly detect a person
- Accuracy depends on ROI placement and color threshold tuning

## Future Improvements

- Add person detection before vest analysis
- Replace color thresholding with YOLO-based safety vest object detection
- Add webcam mode with `cv2.VideoCapture(0)`
- Add gesture-based start signal for real-time inspection
- Log detection statistics per time segment

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
