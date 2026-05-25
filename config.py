"""
GOAT-AI Configuration
=====================
Comprehensive settings for the Livestock Detection & Biometric Estimation framework.
Adjust these settings to tune the pipeline for your specific environment.
"""
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "video")
RECORDINGS_DIR = os.path.join(BASE_DIR, "Recordings")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ─────────────────────────────────────────────
# Detection Engine — YOLO-World + SAM2
# ─────────────────────────────────────────────
# Primary: YOLO-World for open-vocabulary goat detection
YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"
YOLO_WORLD_CLASSES = ["goat", "kid goat", "sheep"]  # Open-vocab prompts
YOLO_WORLD_CONFIDENCE = 0.15  # Lower threshold for open-vocab models

# Primary: Standard YOLO for detection + segmentation + tracking
# Options: yolov8n-seg.pt (fast, ~10 FPS) | yolov8x-seg.pt (quality, ~1 FPS)
YOLO_FALLBACK_MODEL = "yolov8n-seg.pt"
# COCO class IDs (0-indexed): 17=horse, 18=sheep, 19=cow
# NOTE: class 20 = ELEPHANT (NOT horse — common misquote in livestock papers)
# Sirohi goats in side profile are most commonly classified as sheep(18) or horse(17)
YOLO_FALLBACK_CLASSES = [17, 18, 19]  # horse, sheep, cow
YOLO_FALLBACK_CONFIDENCE = 0.15       # Lower = more detections; raise if too many false positives

# SAM2 Mask Refinement (pixel-perfect segmentation)
# NOTE: Disabled by default on 4GB VRAM GPUs. Enable for quality at cost of speed.
ENABLE_SAM2_REFINEMENT = False
SAM2_MODEL = "sam2_b.pt"
SAM2_EVERY_N_FRAMES = 10  # Only run SAM2 every N frames (saves ~80% GPU time)

# Performance
HALF_PRECISION = True      # FP16 inference — ~2x faster on RTX 30xx/40xx
USE_WORLD_MODEL = True     # Use YOLO-World as primary (native "goat" class — better accuracy)

# General detection settings
IOU_THRESHOLD = 0.45

# ─────────────────────────────────────────────
# Detection Filtering
# ─────────────────────────────────────────────
# Reject detections smaller than this pixel area (filters distant/partial animals)
# Tune based on recording distance: ~2000 for close shots, ~500 for distant shots
MIN_DETECTION_AREA_PX = 2000       # pixels² — lowered to catch more valid detections
MIN_MASK_PIXEL_COVERAGE = 0.002    # min fraction of total frame area
WARMUP_FRAMES = 3                  # Skip first N frames while tracker stabilises (reduced)

# Frame skipping for faster draft processing (1 = process every frame)
FRAME_SKIP = 1  # Process every Nth frame; set to 2-4 for faster review

# ─────────────────────────────────────────────
# Multi-Object Tracking — BoT-SORT
# ─────────────────────────────────────────────
ENABLE_TRACKING = True
TRACKER_TYPE = "botsort.yaml"  # "botsort.yaml" or "bytetrack.yaml"

# Reset Kalman smoother if a track disappears for more than this many frames
TRACK_REAPPEAR_GAP_FRAMES = 30

# ─────────────────────────────────────────────
# Depth Sensor — Orbbec Femto Bolt
# ─────────────────────────────────────────────
# Orbbec returns depth in millimetres. Divide by this to convert to cm.
DEPTH_UNIT_SCALE = 10.0   # mm → cm

# ─────────────────────────────────────────────
# 3D Biometric Measurement
# ─────────────────────────────────────────────
# Calibration fallback (used only when depth is unavailable)
PIXELS_PER_CM = 10.0

# Point cloud processing
VOXEL_SIZE = 0.5          # cm — voxel downsampling for consistent density
OUTLIER_NB_NEIGHBORS = 20  # Statistical outlier removal: neighbor count
OUTLIER_STD_RATIO = 2.0    # Statistical outlier removal: std deviation ratio

# Thorax slice location (fraction of body length from head)
THORAX_SLICE_START = 0.20  # 20% from front (behind shoulder)
THORAX_SLICE_END = 0.40    # 40% from front
THORAX_SLICE_THICKNESS = 3.0  # cm — slice half-width for 3D cross-section

# Chest girth ellipse: lateral-to-dorsoventral ratio (calibrated for Sirohi side profile)
# chest_width ≈ LATERAL_BODY_RATIO × chest_height (dorsoventral diameter)
# Typical Sirohi: chest is roughly elliptical, ~60% as wide (lateral) as it is tall (dorsoventral)
# Increase this value if measured girth is consistently lower than tape measure
LATERAL_BODY_RATIO = 0.65

# ─────────────────────────────────────────────
# Measurement Calibration (IMPORTANT for accuracy)
# ─────────────────────────────────────────────
# If the pipeline's measurements are consistently off vs. tape measure:
#   - SCALE_CORRECTION_BODY_LENGTH: multiply raw length_cm by this factor
#   - SCALE_CORRECTION_GIRTH:       multiply raw chest_girth_cm by this factor
# Example: if measured = 72cm but pipeline reports 60cm → set factor = 72/60 = 1.20
SCALE_CORRECTION_BODY_LENGTH = 1.0   # Set to (tape_length / measured_length) if off
SCALE_CORRECTION_GIRTH       = 1.0   # Set to (tape_girth  / measured_girth)  if off

# Minimum valid depth for measurement (ignore pixels closer/farther than this)
DEPTH_MIN_CM = 30.0   # cm — anything closer is likely sensor noise
DEPTH_MAX_CM = 500.0  # cm — anything farther is likely not the target animal



# ─────────────────────────────────────────────
# Temporal Smoothing — Kalman Filter
# ─────────────────────────────────────────────
ENABLE_TEMPORAL_SMOOTHING = True

# Global defaults (used if per-metric values not set)
KALMAN_PROCESS_NOISE = 0.1      # Q — higher = more responsive to changes
KALMAN_MEASUREMENT_NOISE = 5.0  # R — higher = smoother but more lag

# Per-metric Kalman noise tuning (Q, R) — different measurements have different noise
KALMAN_NOISE_PER_METRIC = {
    "length_cm":             (0.05, 3.0),   # Body length: slow change, moderate noise
    "height_cm":             (0.05, 3.0),   # Body height: slow change, moderate noise
    "chest_girth_cm":        (0.05, 2.0),   # Chest girth: very stable, low noise target
    "stance_cm":             (0.20, 8.0),   # Stance: more dynamic, higher noise
    "weight_schaefer_kg":    (0.05, 4.0),   # Weight: derived — moderate smoothing
    "weight_regression_kg":  (0.05, 4.0),   # Weight: derived — moderate smoothing
}

# ─────────────────────────────────────────────
# Weight Estimation — Veterinary Formulas
# ─────────────────────────────────────────────
ENABLE_WEIGHT_ESTIMATION = True

# Breed: Sirohi (Premier goat breed from Rajasthan, India)
BREED = "sirohi"

# Show both formula results on the overlay
SHOW_DUAL_WEIGHT = True

# Maximum acceptable disagreement between formulas before flagging low confidence (%)
WEIGHT_FORMULA_AGREEMENT_THRESHOLD = 40.0

# ── Which formulas to show in the VIDEO OVERLAY ──
# W(Sirohi-Reg) is the most accurate for this breed — show it prominently.
# W(Schaefer) is 2-3x overestimated because our 'body length' is nose-to-tail,
# but Schaefer needs shoulder-to-pin (~65% of that). Set True only for comparison.
SHOW_SCHAEFER_IN_OVERLAY    = False   # True = show Schaefer in overlay
SHOW_BCS_ADJ_IN_OVERLAY     = False   # True = show BCS-adjusted in overlay
SHOW_SIROHI_REG_IN_OVERLAY  = True    # Primary: closest to actual Sirohi weights

# Schaefer body-length correction: our bbox measures nose-to-tail, but Schaefer
# expects shoulder-to-pin-bone length. Approximate ratio ≈ 0.60-0.68 for Sirohi.
SCHAEFER_BL_FRACTION = 0.65   # multiply measured length by this before Schaefer formula

# Formula 1: Schaefer Standard (breed-generic)
# Weight(kg) = HG² × BL / 10840
SCHAEFER_CONSTANT = 10840.0

# Formula 2: Indian Goat Multivariate Regression (Sirohi-adapted from ICAR/AICRP data)
# Weight(kg) = a + b1 × BL + b2 × HG
REGRESSION_INTERCEPT = -28.57
REGRESSION_BL_COEFF = 0.144
REGRESSION_HG_COEFF = 0.538

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
VIS_MASK_OPACITY = 0.35         # Mask overlay transparency (0-1)
VIS_MASK_COLOR = (0, 255, 100)  # Green mask (overridden by confidence color when enabled)
VIS_BOX_COLOR = (50, 50, 255)   # Red rotated bbox
VIS_CHEST_COLOR = (0, 255, 255) # Yellow chest ellipse
VIS_STANCE_COLOR = (255, 0, 255) # Magenta stance line
VIS_TEXT_SCALE = 0.55
VIS_SHOW_TRACK_ID = True
VIS_SHOW_MINI_DASHBOARD = True
VIS_CONFIDENCE_COLOR_CODING = True  # Color-code mask by measurement confidence
VIS_SHOW_SCALE_RULER = True         # Draw real-world scale bar based on depth

# ─────────────────────────────────────────────
# Reporting / Data Export
# ─────────────────────────────────────────────
ENABLE_REPORTING = True
EXPORT_FORMAT = "both"  # "csv", "json", or "both"

# ─────────────────────────────────────────────
# Confidence Scoring
# ─────────────────────────────────────────────
ENABLE_CONFIDENCE_SCORING = True

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False
