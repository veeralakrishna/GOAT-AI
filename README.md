# GOAT-AI: Livestock Detection & Biometric Estimation

## Project Overview
**GOAT-AI** is a GenAI-powered computer vision system designed to detect livestock (specifically goats) in video feeds and estimate their physical biometrics in real-time.

The system uses advanced deep learning for object segmentation and geometric analysis to provide measurements such as length, height, and surface area.

## What it Detects
In the processed video, the system identifies:
*   **Goat/Livestock Instances**: Bounded by a green segmentation mask.
*   **Bounding Box**: A red rotated rectangle fitting the animal's posture.
*   **Measurements**:
    *   **L (Length)**: The estimated length of the animal in cm.
    *   **H (Height/Width)**: The estimated width/girth in cm.
    *   (Note: Measurements are currently relative approximations based on pixel-to-cm calibration).

## Architecture

The framework consists of the following pipeline:

1.  **Input Layer**:
    *   Ingests raw video files (MP4) from the `video/` directory.
    *   Extracts frames using OpenCV.

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

4.  **Visualization (`src/processor.py`)**:
    *   Overlays the segmentation mask (Green).
    *   Draws the rotated bounding box (Red).
    *   Prints the calculated dimensions near the animal.

5.  **Output**:
    *   Saves the annotated video to `output/`.

## Configuration
All settings can be adjusted in `config.py`:
*   `MODEL_NAME`: Switch between `yolov8n-seg.pt` (Speed) and `yolov8x-seg.pt` (Accuracy).
*   `PIXELS_PER_CM`: Calibration scaling factor.

## Running the Pipeline
```bash
python main.py
```
