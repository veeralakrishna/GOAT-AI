"""
GOAT-AI Calibration Diagnostic Tool
=====================================
Run this on a .bag file to:
  1. See what COCO classes are being detected (horse? sheep? cow? or nothing?)
  2. Check depth values at the goat position
  3. See raw pixel dimensions vs computed real-world cm
  4. Compare against your tape-measure ground truth to compute scale corrections

Usage:
  python calibrate.py --bag Recordings/test.bag
  python calibrate.py --bag Recordings/test.bag --frames 100  # Check first 100 frames
  python calibrate.py --bag Recordings/test.bag --all-classes  # Try all animal classes

After running, the tool prints:
  - Which COCO class IDs the model is actually assigning to your goats
  - The depth sensor readings
  - Raw pixel bbox sizes vs converted cm
  - If you provide ground truth (--gt-length, --gt-girth), it prints the correction factors
"""
import os
import sys
import argparse
import logging
import numpy as np
import cv2

logger = logging.getLogger("calibrate")

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}


def parse_args():
    p = argparse.ArgumentParser(description="GOAT-AI Calibration Diagnostic")
    p.add_argument("--bag", required=True, help="Path to .bag file")
    p.add_argument("--frames", type=int, default=60,
                   help="How many frames to sample (default: 60)")
    p.add_argument("--all-classes", action="store_true",
                   help="Detect ALL animal classes (no class filter) — find what the model sees")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Confidence threshold for calibration (default: 0.05 — very low)")
    p.add_argument("--gt-length", type=float, default=None,
                   help="Ground-truth body length in cm (from tape measure)")
    p.add_argument("--gt-girth", type=float, default=None,
                   help="Ground-truth heart girth in cm (from tape measure)")
    p.add_argument("--output-frames", type=str, default="calibration_frames",
                   help="Directory to save annotated sample frames")
    return p.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def main():
    args = parse_args()
    setup_logging()

    os.makedirs(args.output_frames, exist_ok=True)

    logger.info("=" * 62)
    logger.info("GOAT-AI Calibration Diagnostic")
    logger.info("=" * 62)
    logger.info(f"Bag file: {args.bag}")
    logger.info(f"Frames to sample: {args.frames}")
    logger.info(f"Confidence threshold: {args.conf}")
    if args.all_classes:
        logger.info("Mode: ALL animal classes (finding what model sees)")
    else:
        logger.info("Mode: classes [17=horse, 18=sheep, 19=cow]")

    # ── Load YOLO model ──
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-seg.pt")
        logger.info("YOLO model loaded")
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return

    # ── Open bag file ──
    try:
        from pyorbbecsdk import (
            Pipeline, Config, PlaybackDevice, AlignFilter,
            OBSensorType, OBStreamType, OBFormat,
        )
    except ImportError:
        logger.error("pyorbbecsdk not installed.")
        return

    playback = PlaybackDevice(args.bag)
    pipeline = Pipeline(playback)
    cfg = Config()
    cfg.enable_stream(OBSensorType.COLOR_SENSOR)
    cfg.enable_stream(OBSensorType.DEPTH_SENSOR)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    pipeline.start(cfg)

    # Get intrinsics
    color_profile = pipeline.get_stream_profile_list(
        OBSensorType.COLOR_SENSOR
    ).get_video_stream_profile(0)
    intrinsics = color_profile.get_intrinsic()
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx_i = intrinsics.cx
    cy_i = intrinsics.cy

    logger.info(f"\nCamera Intrinsics:")
    logger.info(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx_i:.2f}  cy={cy_i:.2f}")
    logger.info(f"  (These must be > 0 for real-world cm to be accurate)\n")

    # ── Diagnostic accumulators ──
    class_vote_counts = {}
    depth_samples = []
    length_samples = []
    girth_samples = []
    height_samples = []
    frames_with_detections = 0
    frame_count = 0

    detection_classes = None if args.all_classes else [17, 18, 19]

    try:
        while frame_count < args.frames:
            try:
                frames = pipeline.wait_for_frames(200)
                if not frames:
                    continue
                frames = align_filter.process(frames)
                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not (color_frame and depth_frame):
                    continue

                # Decode color
                color_data = np.asanyarray(color_frame.get_data())
                fmt = color_frame.get_format()
                h = color_frame.get_height()
                w = color_frame.get_width()

                if fmt == OBFormat.MJPG:
                    color_img = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                elif fmt == OBFormat.RGB:
                    color_img = color_data.reshape((h, w, 3))
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                else:
                    color_img = color_data.reshape((h, w, 3))

                if color_img is None:
                    continue

                # Decode depth
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                dh = depth_frame.get_height()
                dw = depth_frame.get_width()
                depth_img = depth_data.reshape((dh, dw))

                frame_count += 1
                annotated = color_img.copy()

                # ── Run YOLO inference ──
                results = model.predict(
                    source=color_img,
                    conf=args.conf,
                    classes=detection_classes,
                    verbose=False,
                    retina_masks=True,
                )

                if not results or results[0].boxes is None:
                    # Label frame with no detections
                    cv2.putText(annotated, f"Frame {frame_count}: NO DETECTIONS",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    _save_frame(annotated, args.output_frames, frame_count)
                    continue

                result = results[0]
                n_det = len(result.boxes)
                if n_det == 0:
                    cv2.putText(annotated, f"Frame {frame_count}: NO DETECTIONS",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    _save_frame(annotated, args.output_frames, frame_count)
                    continue

                frames_with_detections += 1

                # ── Analyse each detection ──
                for i in range(n_det):
                    bbox = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    conf_val = float(result.boxes.conf[i].cpu())
                    cls_id = int(result.boxes.cls[i].cpu())
                    cls_name = COCO_NAMES.get(cls_id, f"cls_{cls_id}")

                    # Vote counter
                    class_vote_counts[f"{cls_id}={cls_name}"] = (
                        class_vote_counts.get(f"{cls_id}={cls_name}", 0) + 1
                    )

                    x1, y1, x2, y2 = bbox
                    bw_px = x2 - x1
                    bh_px = y2 - y1
                    bbox_area = bw_px * bh_px

                    # Depth at bbox centre
                    cxb = (x1 + x2) // 2
                    cyb = (y1 + y2) // 2
                    depth_patch = depth_img[
                        max(0, cyb-15):min(dh, cyb+15),
                        max(0, cxb-15):min(dw, cxb+15),
                    ]
                    valid_d = depth_patch[depth_patch > 0]
                    depth_mm = float(np.median(valid_d)) if len(valid_d) >= 3 else 0
                    depth_cm = depth_mm / 10.0  # Orbbec: mm → cm

                    # Real-world dimensions
                    length_cm_raw = (bw_px * depth_cm / fx) if (depth_cm > 0 and fx > 0) else 0
                    height_cm_raw = (bh_px * depth_cm / fy) if (depth_cm > 0 and fy > 0) else 0

                    # Approx girth from bbox height (ellipse with LATERAL_BODY_RATIO)
                    semi_a = height_cm_raw / 2.0
                    semi_b = semi_a * 0.65
                    girth_cm_raw = 0
                    if semi_a > 0:
                        import math
                        h_g = max(semi_a, semi_b)
                        w_g = min(semi_a, semi_b)
                        hh = ((h_g - w_g) / (h_g + w_g)) ** 2
                        girth_cm_raw = math.pi * (h_g + w_g) * (
                            1 + 3 * hh / (10 + math.sqrt(4 - 3 * hh))
                        )

                    if depth_cm > 0:
                        depth_samples.append(depth_cm)
                        if length_cm_raw > 10:
                            length_samples.append(length_cm_raw)
                        if girth_cm_raw > 10:
                            girth_samples.append(girth_cm_raw)
                        if height_cm_raw > 10:
                            height_samples.append(height_cm_raw)

                    logger.info(
                        f"  Frame {frame_count:3d} | {cls_name}({cls_id}) | conf={conf_val:.2f} | "
                        f"bbox=[{bw_px}×{bh_px}px,area={bbox_area}px²] | "
                        f"depth={depth_cm:.0f}cm | "
                        f"L≈{length_cm_raw:.0f}cm H≈{height_cm_raw:.0f}cm HG≈{girth_cm_raw:.0f}cm"
                    )

                    # Draw on frame
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    label = (
                        f"{cls_name}({cls_id}) {conf_val:.2f} "
                        f"L:{length_cm_raw:.0f} H:{height_cm_raw:.0f} HG:{girth_cm_raw:.0f}cm "
                        f"d:{depth_cm:.0f}cm"
                    )
                    cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                _save_frame(annotated, args.output_frames, frame_count)

            except Exception as e:
                if "timeout" in str(e).lower():
                    logger.info("EOF reached")
                    break
                logger.warning(f"Frame error: {e}")

    finally:
        pipeline.stop()

    # ─── Summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 62)
    logger.info("CALIBRATION DIAGNOSTIC SUMMARY")
    logger.info("=" * 62)
    logger.info(f"Frames sampled      : {frame_count}")
    logger.info(f"Frames with detections: {frames_with_detections} "
                f"({100*frames_with_detections/max(1,frame_count):.0f}%)")

    logger.info("\n── Detection Class Votes (what the model sees your goats as) ──")
    if class_vote_counts:
        for cls_str, count in sorted(class_vote_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {cls_str:25s}: {count:4d} detections")
    else:
        logger.info("  *** NO DETECTIONS AT ALL ***")
        logger.info("  Possible causes:")
        logger.info("    1. Model not finding livestock at all — try --all-classes")
        logger.info("    2. Goats too small in frame — check recording distance")
        logger.info("    3. Very unusual appearance — try a different YOLO model")

    if depth_samples:
        logger.info(f"\n── Depth Readings ──")
        logger.info(f"  Median : {np.median(depth_samples):.0f} cm")
        logger.info(f"  Range  : {np.min(depth_samples):.0f} – {np.max(depth_samples):.0f} cm")
        logger.info(f"  (Typical goat recording: 80–300 cm)")

    if length_samples:
        med_l = np.median(length_samples)
        med_h = np.median(height_samples) if height_samples else 0
        med_g = np.median(girth_samples) if girth_samples else 0

        logger.info(f"\n── Raw Measurements (before calibration) ──")
        logger.info(f"  Body length  : median={med_l:.1f} cm  "
                    f"(range {np.min(length_samples):.0f}–{np.max(length_samples):.0f})")
        logger.info(f"  Body height  : median={med_h:.1f} cm")
        logger.info(f"  Heart girth  : median={med_g:.1f} cm  (estimated from bbox height)")
        logger.info(f"  (Expected adult Sirohi: L=65–85cm, H=60–75cm, HG=70–95cm)")

        if args.gt_length or args.gt_girth:
            logger.info(f"\n── Scale Correction Factors (add these to config.py) ──")
            if args.gt_length and med_l > 0:
                factor_l = args.gt_length / med_l
                logger.info(f"  SCALE_CORRECTION_BODY_LENGTH = {factor_l:.3f}   "
                             f"# tape={args.gt_length}cm, measured={med_l:.1f}cm")
            if args.gt_girth and med_g > 0:
                factor_g = args.gt_girth / med_g
                logger.info(f"  SCALE_CORRECTION_GIRTH       = {factor_g:.3f}   "
                             f"# tape={args.gt_girth}cm, measured={med_g:.1f}cm")
        else:
            logger.info(f"\n  Tip: Re-run with --gt-length <tape_measure_cm> --gt-girth <tape_measure_cm>")
            logger.info(f"  to automatically compute SCALE_CORRECTION factors.")

    logger.info(f"\n── Annotated frames saved to: {args.output_frames}/ ──")
    logger.info("=" * 62)


def _save_frame(img, output_dir, frame_idx):
    path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(path, img)


if __name__ == "__main__":
    main()
