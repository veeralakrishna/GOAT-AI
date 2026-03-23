# GOAT-AI: Livestock Detection & Biometric Estimation

## Project Overview
**GOAT-AI** is a GenAI-powered computer vision system designed to detect livestock (specifically goats) in video feeds and estimate their physical biometrics in real-time.

The system uses advanced deep learning for object segmentation and geometric analysis to provide measurements such as length, height, and surface area.

## What it Detects
In the processed video, the system identifies:
*   **Goat/Livestock Instances**: Bounded by a green segmentation mask.
*   **Bounding Box**: A red rotated rectangle fitting the animal's posture.
*   **Measurements**:
    *   **L (Length)**: The estimated real-world length of the animal in cm.
    *   **H (Height/Width)**: The estimated real-world width/height of the animal in cm.
    *   **C (Chest Girth)**: The approximated chest circumference ($\pi \times H$) in cm.
    *   **Z (Depth)**: The median physical distance (depth) of the animal from the camera in cm (Orbbec SDK).
    *   (Note: Measurements use true 3D depth from Orbbec `.bag` files, computing real dimensions instead of relative pixel approximations).

## Architecture

The framework consists of the following pipeline:

1.  **Input Layer**:
    *   Ingests PyOrbbecSDK Orbbec Femto recordings (`.bag` files) from `Recordings/` directory with perfectly aligned depth and color streams.

2.  **Detection Module (`src/detector.py`)**:
    *   **Model**: YOLOv8 (Instance Segmentation).
    *   **Logic**: Runs inference to detect objects of class 'sheep' (ID 18) or 'cow' (ID 19) as proxies for goats.
    *   **Output**: Binary segmentation masks for each detected animal.

3.  **Biometric Estimator (`src/measurements.py`)**:
    *   **Input**: Raw usage of segmentation masks.
    *   **Processing**:
        *   Extracts contours from masks.
        *   Computes the Minimum Area Rectangle (Rotated Bounding Box) to handle various orientations.
        *   Calculates Major Axis (Length) and Minor Axis (Width).
        *   Computes Contour Area.
    *   **Calibration**: Converts pixel values to Centimeters using a configurable `PIXELS_PER_CM` factor.

4.  **Visualization (`src/bag_processor.py`)**:
    *   Overlays the segmentation mask (Green).
    *   Draws the rotated bounding box (Red).
    *   Prints the calculated dimensions (L, H, C, Z) near the animal.

5.  **Output**:
    *   Saves the annotated video to `output/`.

## Configuration
All settings can be adjusted in `config.py`:
*   `MODEL_NAME`: Switch between `yolov8n-seg.pt` (Speed) and `yolov8x-seg.pt` (Accuracy).

## Running the Pipeline
```bash
python main.py
```
