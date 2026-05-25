# 🐐 GOAT-AI: Livestock Detection & Biometric Estimation

## Project Overview
**GOAT-AI** is a GenAI-powered computer vision framework for detecting livestock (specifically **Sirohi goats**) in Orbbec Femto Bolt depth camera recordings and estimating their physical biometrics — including **live weight** — in real-time.

The system combines cutting-edge deep learning (YOLO-World, SAM2), 3D point cloud analysis (Open3D + PCA), multi-object tracking (BoT-SORT), and veterinary weight formulas to deliver production-grade livestock analytics.

## Architecture

```
.bag Input → YOLO-World Detection → SAM2 Mask Refinement → BoT-SORT Tracking
  → PCA 3D Biometrics → Kalman Smoothing → Weight Estimation → Visualization → Report
```

### Pipeline Stages

| Stage | Engine | Description |
|-------|--------|-------------|
| **Detection** | YOLO-World v2 | Open-vocabulary — detects "goat" natively (no proxy classes) |
| **Refinement** | SAM2 (Base) | Foundation model for pixel-perfect segmentation masks |
| **Tracking** | BoT-SORT | Persistent identity tracking across frames |
| **Biometrics** | Open3D + PCA | 3D point cloud analysis for body length, height, chest girth |
| **Smoothing** | Kalman Filter | Temporal smoothing for stable measurements |
| **Weight** | Dual Formulas | Schaefer Standard + Sirohi Regression (ICAR/AICRP) |
| **Visualization** | Custom Engine | Rich overlays with semi-transparent masks, dashboards |
| **Reporting** | JSON + CSV | Structured data export for downstream analysis |

## Measurements

| Metric | Method | Unit |
|--------|--------|------|
| **Body Length (L)** | PCA primary axis on 3D point cloud | cm |
| **Body Height (H)** | PCA secondary axis | cm |
| **Heart Girth (HG)** | 3D thorax slice + ellipse fitting + Ramanujan's approximation | cm |
| **Weight (W)** — Standard | `HG² × BL / 10,840` (Schaefer) | kg |
| **Weight (W)** — Sirohi | `-28.57 + 0.144×BL + 0.538×HG` (ICAR regression) | kg |
| **Stance (S)** | Front-to-back leg distance | cm |
| **Depth (Z)** | Median depth from Orbbec sensor | cm |
| **Confidence** | Multi-factor quality score | % |

## Project Structure

```
GOAT-AI/
├── config.py                    # Comprehensive pipeline settings
├── main.py                      # CLI entry point
├── app.py                       # Gradio web dashboard
├── requirements.txt             # Python dependencies
├── src/
│   ├── detector.py              # YOLO-World + SAM2 detection engine
│   ├── tracker.py               # BoT-SORT multi-object tracker
│   ├── measurements.py          # PCA-based 3D biometric engine
│   ├── temporal.py              # Kalman temporal smoother
│   ├── weight_estimator.py      # Dual veterinary weight formulas
│   ├── visualizer.py            # Rich overlay renderer
│   ├── reporter.py              # JSON/CSV data export
│   ├── bag_processor.py         # Pipeline orchestrator
│   └── processor.py             # Legacy video processor
├── Recordings/                  # Orbbec .bag input files
└── output/                      # Annotated videos + reports
```

## Running

### CLI Processing
```bash
# Process all .bag files in Recordings/
python main.py

# Process a specific file
python main.py --input path/to/recording.bag

# Disable heavy features for faster processing
python main.py --no-sam2 --no-tracking

# Verbose logging
python main.py -v
```

### Gradio Web Dashboard
```bash
python app.py
# Opens at http://localhost:7860
```

## Hardware
- **Camera**: Orbbec Femto Bolt (RGB-D depth sensor)
- **GPU**: NVIDIA GeForce RTX 3050 Ti (4GB VRAM)
- **RAM**: 16GB

## Breed
- **Sirohi** — Premier goat breed from Rajasthan, India
- Weight estimation formulas calibrated for Indian goat breeds using ICAR/AICRP research data

## Configuration
All pipeline settings are in `config.py` — including model paths, detection thresholds, biometric parameters, weight formula coefficients, visualization styles, and export options.

---
*GOAT-AI © GenZAI — Precision Livestock Farming*
