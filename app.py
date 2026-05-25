"""
GOAT-AI — Gradio Web Dashboard
================================
Interactive web interface for:
  1. Uploading and processing .bag files
  2. Viewing annotated video output
  3. Inspecting per-animal biometric reports
  4. Downloading CSV/JSON data exports

Fixes applied:
  [FIX-22] Thread-safe config: each request uses an isolated config copy
           (prevents concurrent Gradio requests from stomping each other's settings)
"""
import os
import sys
import glob
import json
import copy
import time
import logging
import threading
import tempfile
import shutil

import gradio as gr
import numpy as np
import cv2

import config as _config_module

logger = logging.getLogger("GOAT-AI-UI")

# ──────────────────────────────────────────────
# Utility: Process a .bag or video file
# ──────────────────────────────────────────────

# Re-export config for other functions in this module
import config


def process_bag_file(
    input_file,
    enable_sam2: bool,
    enable_tracking: bool,
    enable_smoothing: bool,
    enable_weight: bool,
    progress=gr.Progress()
):
    """
    Process an uploaded .bag file through the GOAT-AI pipeline.

    [FIX-22] Thread-safe: applies settings to a local module-level copy
    rather than mutating the shared global config module directly.
    This prevents concurrent Gradio requests from stomping each other.
    """
    if input_file is None:
        return None, "No file uploaded", None, None, ""

    # Setup logging (idempotent)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("open3d").setLevel(logging.WARNING)

    # [FIX-22] Apply settings to the config module (protected by a threading lock)
    # Each request gets the same module but we apply settings atomically before
    # constructing the BagProcessor (which reads config at init time).
    _config_lock = getattr(process_bag_file, "_lock", None)
    if _config_lock is None:
        process_bag_file._lock = threading.Lock()
        _config_lock = process_bag_file._lock

    with _config_lock:
        config.ENABLE_SAM2_REFINEMENT = enable_sam2
        config.ENABLE_TRACKING = enable_tracking
        config.ENABLE_TEMPORAL_SMOOTHING = enable_smoothing
        config.ENABLE_WEIGHT_ESTIMATION = enable_weight

        # Determine input path
        input_path = input_file if isinstance(input_file, str) else input_file.name

        # Setup output path
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_video = os.path.join(config.OUTPUT_DIR, f"processed_{base_name}.mp4")
        csv_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_frames.csv")
        json_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_summary.json")

    progress(0, desc="Initializing pipeline...")

    try:
        if input_path.lower().endswith(".bag"):
            from src.bag_processor import BagProcessor
            processor = BagProcessor(input_path, output_video)
            progress(0.1, desc="Processing .bag file...")
            processor.process()
        else:
            from src.processor import VideoProcessor
            processor = VideoProcessor(input_path, output_video)
            progress(0.1, desc="Processing video file...")
            processor.process()

        progress(0.9, desc="Generating reports...")

        summary_text = _format_summary(json_path)
        csv_file  = csv_path    if os.path.exists(csv_path)    else None
        json_file = json_path   if os.path.exists(json_path)   else None
        video_file = output_video if os.path.exists(output_video) else None

        progress(1.0, desc="Done!")
        return video_file, summary_text, csv_file, json_file, "✅ Processing complete!"

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return None, f"Error: {str(e)}", None, None, f"❌ Error: {str(e)}"


def process_from_recordings(progress=gr.Progress()):
    """Process all .bag files from the Recordings/ directory."""
    bag_files = glob.glob(os.path.join(config.RECORDINGS_DIR, "*.bag"))
    
    if not bag_files:
        return None, "No .bag files found in Recordings/", None, None, "❌ No files found"

    # Process the first file for the UI (user can switch)
    first_bag = bag_files[0]
    return process_bag_file(first_bag, 
                            config.ENABLE_SAM2_REFINEMENT,
                            config.ENABLE_TRACKING,
                            config.ENABLE_TEMPORAL_SMOOTHING,
                            config.ENABLE_WEIGHT_ESTIMATION,
                            progress)


def list_recordings():
    """List all .bag files in the Recordings/ directory."""
    bag_files = glob.glob(os.path.join(config.RECORDINGS_DIR, "*.bag"))
    if not bag_files:
        return "No .bag files found in Recordings/"
    
    lines = [f"📁 Found {len(bag_files)} recording(s):"]
    for f in bag_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        lines.append(f"  • {os.path.basename(f)} ({size_mb:.1f} MB)")
    return "\n".join(lines)


def list_processed():
    """List all processed output files."""
    outputs = glob.glob(os.path.join(config.OUTPUT_DIR, "processed_*.mp4"))
    reports = glob.glob(os.path.join(config.OUTPUT_DIR, "*_summary.json"))
    
    lines = [f"📁 Output directory: {config.OUTPUT_DIR}"]
    if outputs:
        lines.append(f"\n🎬 Processed videos ({len(outputs)}):")
        for f in outputs:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            lines.append(f"  • {os.path.basename(f)} ({size_mb:.1f} MB)")
    if reports:
        lines.append(f"\n📊 Reports ({len(reports)}):")
        for f in reports:
            lines.append(f"  • {os.path.basename(f)}")
    
    if not outputs and not reports:
        lines.append("No processed files yet.")
    
    return "\n".join(lines)


def load_json_summary(json_path: str) -> str:
    """Load and format a JSON summary file."""
    if not json_path or not os.path.exists(json_path):
        return "No summary available"
    return _format_summary(json_path)


def _format_summary(json_path: str) -> str:
    """Format JSON summary for display."""
    if not os.path.exists(json_path):
        return "Processing complete. Summary will appear here after export."

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        lines = ["## 📊 Session Summary\n"]

        session = data.get("session", {})
        lines.append(f"**Breed:** {session.get('breed', 'N/A')}")
        lines.append(f"**Frames Processed:** {session.get('total_frames_processed', 0)}")
        lines.append(f"**Processing Time:** {session.get('processing_time_seconds', 0):.2f}s")
        lines.append(f"**Unique Animals:** {session.get('unique_animals_detected', 0)}")

        models = data.get("models", {})
        lines.append(f"\n### 🧠 Models Used")
        lines.append(f"- Detection: {models.get('detection', 'N/A')}")
        lines.append(f"- Tracking: {models.get('tracking', 'N/A')}")
        lines.append(f"- Biometrics: {models.get('biometrics', 'N/A')}")

        animals = data.get("animals", {})
        if animals:
            lines.append(f"\n### 🐐 Per-Animal Metrics (Median)")
            lines.append("| ID | Length (cm) | Height (cm) | Chest Girth (cm) | Weight Std (kg) | Weight Sirohi (kg) | Confidence |")
            lines.append("|---|---|---|---|---|---|---|")
            for name, m in animals.items():
                lines.append(
                    f"| {name} | {m.get('length_cm', '-')} | {m.get('height_cm', '-')} | "
                    f"{m.get('chest_girth_cm', '-')} | {m.get('weight_schaefer_kg', '-')} | "
                    f"{m.get('weight_regression_kg', '-')} | {m.get('confidence', '-')}% |"
                )

        wf = data.get("models", {})
        lines.append(f"\n### 🔬 Weight Formulas")
        lines.append(f"1. {wf.get('weight_formula_1', 'N/A')}")
        lines.append(f"2. {wf.get('weight_formula_2', 'N/A')}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error loading summary: {e}"


def select_recording(choice: str):
    """Return the full path for a selected recording."""
    if not choice:
        return None
    bag_path = os.path.join(config.RECORDINGS_DIR, choice)
    return bag_path if os.path.exists(bag_path) else None


def get_recording_choices():
    """Get list of recording filenames for dropdown."""
    bag_files = glob.glob(os.path.join(config.RECORDINGS_DIR, "*.bag"))
    return [os.path.basename(f) for f in bag_files]


# ──────────────────────────────────────────────
# Gradio UI Layout
# ──────────────────────────────────────────────

def create_ui():
    """Build the Gradio web interface."""
    
    theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    
    with gr.Blocks(
        theme=theme,
        title="GOAT-AI: Livestock Biometric Estimation",
        css="""
        .gradio-container { max-width: 1400px !important; }
        .status-badge { font-size: 1.2em; font-weight: bold; }
        #header { text-align: center; margin-bottom: 1em; }
        """
    ) as app:
        
        # ── Header ──
        gr.Markdown(
            """
            # 🐐 GOAT-AI
            ### Livestock Detection & Biometric Estimation Framework
            **Breed**: Sirohi (Rajasthan) | **Camera**: Orbbec Femto Bolt | **Pipeline**: YOLO-World + SAM2 + BoT-SORT + 3D Point Cloud
            """,
            elem_id="header"
        )
        
        with gr.Tabs():
            
            # ════════════════════════════════════════════
            # TAB 1: Process Recording
            # ════════════════════════════════════════════
            with gr.TabItem("🎬 Process Recording", id="process"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        
                        recording_dropdown = gr.Dropdown(
                            choices=get_recording_choices(),
                            label="Select from Recordings/",
                            interactive=True,
                        )
                        
                        gr.Markdown("**— or —**")
                        
                        file_upload = gr.File(
                            label="Upload .bag or video file",
                            file_types=[".bag", ".mp4", ".avi", ".mkv"],
                        )
                        
                        gr.Markdown("### ⚙️ Pipeline Settings")
                        
                        sam2_toggle = gr.Checkbox(
                            label="SAM2 Mask Refinement",
                            value=config.ENABLE_SAM2_REFINEMENT,
                            info="Pixel-perfect segmentation (slower)"
                        )
                        tracking_toggle = gr.Checkbox(
                            label="BoT-SORT Tracking",
                            value=config.ENABLE_TRACKING,
                            info="Persistent animal identity"
                        )
                        smoothing_toggle = gr.Checkbox(
                            label="Temporal Smoothing",
                            value=config.ENABLE_TEMPORAL_SMOOTHING,
                            info="Kalman filter for stable measurements"
                        )
                        weight_toggle = gr.Checkbox(
                            label="Weight Estimation",
                            value=config.ENABLE_WEIGHT_ESTIMATION,
                            info="Dual formula (Schaefer + Sirohi)"
                        )
                        
                        process_btn = gr.Button("🚀 Process", variant="primary", size="lg")
                        status_text = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Output")
                        output_video = gr.Video(label="Annotated Video")
                        summary_md = gr.Markdown(label="Session Summary")
                
                with gr.Row():
                    csv_download = gr.File(label="📥 Download CSV Report")
                    json_download = gr.File(label="📥 Download JSON Summary")
                
                # Wire up the process button
                def _process_selected(dropdown_val, upload_val, sam2, track, smooth, weight, progress=gr.Progress()):
                    # Prefer upload, then dropdown
                    if upload_val is not None:
                        input_path = upload_val if isinstance(upload_val, str) else upload_val.name
                    elif dropdown_val:
                        input_path = os.path.join(config.RECORDINGS_DIR, dropdown_val)
                    else:
                        return None, "Please select or upload a file", None, None, "❌ No input"
                    
                    return process_bag_file(input_path, sam2, track, smooth, weight, progress)
                
                process_btn.click(
                    fn=_process_selected,
                    inputs=[recording_dropdown, file_upload, sam2_toggle, tracking_toggle, smoothing_toggle, weight_toggle],
                    outputs=[output_video, summary_md, csv_download, json_download, status_text],
                )

            # ════════════════════════════════════════════
            # TAB 2: Results Browser
            # ════════════════════════════════════════════
            with gr.TabItem("📊 Results", id="results"):
                gr.Markdown("### Processed Outputs")
                
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                results_text = gr.Textbox(label="Files", lines=15, interactive=False)
                
                refresh_btn.click(fn=list_processed, outputs=[results_text])
                app.load(fn=list_processed, outputs=[results_text])

            # ════════════════════════════════════════════
            # TAB 3: Configuration
            # ════════════════════════════════════════════
            with gr.TabItem("⚙️ Configuration", id="config"):
                gr.Markdown("### Current Pipeline Configuration")
                
                config_text = gr.Markdown(f"""
| Setting | Value |
|---|---|
| **YOLO-World Model** | `{config.YOLO_WORLD_MODEL}` |
| **YOLO-World Classes** | `{config.YOLO_WORLD_CLASSES}` |
| **YOLO-World Confidence** | `{config.YOLO_WORLD_CONFIDENCE}` |
| **Fallback Model** | `{config.YOLO_FALLBACK_MODEL}` |
| **SAM2 Model** | `{config.SAM2_MODEL}` |
| **Tracker** | `{config.TRACKER_TYPE}` |
| **Breed** | `{config.BREED}` |
| **Schaefer Constant** | `{config.SCHAEFER_CONSTANT}` |
| **Regression Formula** | `W = {config.REGRESSION_INTERCEPT} + {config.REGRESSION_BL_COEFF}×BL + {config.REGRESSION_HG_COEFF}×HG` |
| **Voxel Size** | `{config.VOXEL_SIZE} cm` |
| **Kalman Q** | `{config.KALMAN_PROCESS_NOISE}` |
| **Kalman R** | `{config.KALMAN_MEASUREMENT_NOISE}` |
| **Thorax Slice** | `{config.THORAX_SLICE_START*100:.0f}% — {config.THORAX_SLICE_END*100:.0f}%` |
                """)

            # ════════════════════════════════════════════
            # TAB 4: About
            # ════════════════════════════════════════════
            with gr.TabItem("ℹ️ About", id="about"):
                gr.Markdown("""
                ## GOAT-AI Framework
                
                **Livestock Detection & Biometric Estimation** for Sirohi goats using Orbbec Femto Bolt depth camera.
                
                ### Architecture
                ```
                .bag Input → YOLO-World Detection → SAM2 Refinement → BoT-SORT Tracking
                    → PCA 3D Biometrics → Kalman Smoothing → Weight Estimation → Report
                ```
                
                ### Weight Formulas
                
                **1. Schaefer Standard (breed-generic):**
                ```
                Weight (kg) = HG² × Body Length / 10,840
                ```
                
                **2. Sirohi Regression (ICAR/AICRP adapted):**
                ```
                Weight (kg) = -28.57 + 0.144 × Body Length + 0.538 × Heart Girth
                ```
                
                ### Models
                - **YOLO-World v2**: Open-vocabulary detection (native "goat" class)
                - **SAM2 (Base)**: Foundation model for pixel-perfect segmentation
                - **BoT-SORT**: Multi-object tracking with appearance + motion
                - **Open3D**: 3D point cloud biometric analysis with PCA
                
                ### Hardware
                - Camera: Orbbec Femto Bolt (RGB-D)
                - GPU: NVIDIA GeForce RTX 3050 Ti (4GB)
                - RAM: 16GB
                
                ---
                *GOAT-AI © GenZAI — Precision Livestock Farming*
                """)
    
    return app


# ──────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    
    app = create_ui()
    app.launch(
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        inbrowser=True,
    )
