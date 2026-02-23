# Bead Tracking & Colour Detection (C++ / OpenCV)

A C++ computer vision project that detects, classifies, and tracks falling beads in video, exporting structured motion data to CSV.

This project demonstrates both:

- Deep Learning detection (YOLO via ONNX)
- Traditional OpenCV contour-based detection

---


## Overview

The system:

- Detects beads entering the frame
- Classifies bead colour (black / white)
- Assigns a unique ID to each bead
- Tracks each bead until it exits the frame
- Records entry/exit frame numbers and x-positions
- Exports results to a structured CSV file

The focus is on clean architecture, tracking stability, and robustness.

---

## Detection Approaches

### 1. YOLO (Deep Learning)

- Custom YOLO model exported to ONNX
- Loaded using OpenCV DNN
- Performs detection + classification
- Uses Non-Maximum Suppression
- CPU-based inference

### 2. Traditional OpenCV Pipeline

- Grayscale conversion
- Thresholding
- Morphological filtering
- Contour detection
- Area filtering
- Centroid extraction

Both approaches integrate with the same tracking pipeline.

---

## Tracking

- CSRT tracker
- Nearest-centroid matching
- Colour consistency validation
- Frame-loss tolerance
- Lifecycle management (enter → track → exit)

---

## Output

Generates:

```
outputData.csv
```

Format:

```
ParticleID, ParticleColour, FrameNumberOnEnter, FrameNumberOnExit, xPosAtEnter, xPosAtExit
```

---

## Project Structure

```
/project-root
│
├── video.avi
├── custom_dataset_YOLO.onnx
├── outputData.csv
├── outputDataTrad.csv
├── main_yolo.cpp
├── main_contour.cpp
└── README.md
```

---

## Requirements

- OpenCV 4.10.0

If using YOLO:
- `opencv_world4100d.dll` must be available at runtime

---


## Key Concepts Demonstrated

- Multi-object detection
- Object tracking
- Data association
- Hybrid CV (Deep Learning + Classical)
- Structured data export (CSV)

---