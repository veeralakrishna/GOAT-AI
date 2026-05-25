"""
GOAT-AI — Main Entry Point
============================
Processes all .bag recordings from the Recordings/ directory.

New CLI args:
  --frame-skip N    Process only every Nth frame (faster draft processing)
  --min-area PX     Minimum detection bounding-box area in pixels
  --no-world        Disable YOLO-World (use fallback YOLO only)
"""
import os
import sys
import glob
import logging
import argparse
import time

import config
from src.bag_processor import BagProcessor


def setup_logging(verbose: bool = False):
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO

    log_dir = os.path.join(config.BASE_DIR, "Log")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"goat_ai_{time.strftime('%Y%m%d_%H%M%S')}.log")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    # Suppress noisy third-party libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("open3d").setLevel(logging.WARNING)

    return log_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="GOAT-AI: Livestock Detection & Biometric Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Process all recordings
  python main.py --input rec.bag           # Single file
  python main.py --frame-skip 2            # Every other frame (faster)
  python main.py --min-area 8000           # Stricter detection filter
  python main.py --no-sam2 --no-world -v   # Fast mode with debug logging
        """
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a specific .bag file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--no-sam2", action="store_true",
                        help="Disable SAM2 mask refinement")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable multi-object tracking")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Disable temporal smoothing")
    parser.add_argument("--no-world", action="store_true",
                        help="Disable YOLO-World (use fallback YOLO only)")
    parser.add_argument("--frame-skip", type=int, default=None, metavar="N",
                        help="Process every Nth frame (1=all, 2=half, etc.)")
    parser.add_argument("--min-area", type=int, default=None, metavar="PX",
                        help="Minimum detection bounding-box area in pixels")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


def print_config_summary(logger):
    """Print a formatted configuration summary at startup."""
    logger.info("┌─────────────────────────────────────────────────────────┐")
    logger.info("│            GOAT-AI Configuration Summary                │")
    logger.info("├──────────────────────────────┬──────────────────────────┤")
    logger.info(f"│ Primary Model                │ {'YOLO-World' if config.USE_WORLD_MODEL else 'YOLO Fallback':<24} │")
    logger.info(f"│ Fallback Model               │ {config.YOLO_FALLBACK_MODEL:<24} │")
    logger.info(f"│ Fallback Classes (COCO IDs)  │ {str(config.YOLO_FALLBACK_CLASSES):<24} │")
    logger.info(f"│ Tracking                     │ {'ON (' + config.TRACKER_TYPE + ')' if config.ENABLE_TRACKING else 'OFF':<24} │")
    logger.info(f"│ SAM2 Refinement              │ {'ON (every ' + str(config.SAM2_EVERY_N_FRAMES) + ' frames)' if config.ENABLE_SAM2_REFINEMENT else 'OFF':<24} │")
    logger.info(f"│ Temporal Smoothing           │ {'ON (Kalman)' if config.ENABLE_TEMPORAL_SMOOTHING else 'OFF':<24} │")
    logger.info(f"│ Weight Estimation            │ {'ON (' + config.BREED + ')' if config.ENABLE_WEIGHT_ESTIMATION else 'OFF':<24} │")
    logger.info(f"│ Warmup Frames                │ {str(config.WARMUP_FRAMES):<24} │")
    logger.info(f"│ Frame Skip                   │ {str(config.FRAME_SKIP):<24} │")
    logger.info(f"│ Min Detection Area           │ {str(config.MIN_DETECTION_AREA_PX) + ' px':<24} │")
    logger.info(f"│ Depth Unit Scale             │ {str(config.DEPTH_UNIT_SCALE) + ' (mm→cm)':<24} │")
    logger.info(f"│ Output Directory             │ {config.OUTPUT_DIR:<24} │")
    logger.info("└──────────────────────────────┴──────────────────────────┘")


def main():
    args = parse_args()
    log_file = setup_logging(args.verbose)

    logger = logging.getLogger("GOAT-AI")

    logger.info("=" * 62)
    logger.info("  GOAT-AI: Livestock Detection & Biometric Estimation v2.0")
    logger.info("  Breed: Sirohi | Camera: Orbbec Femto Bolt")
    logger.info("=" * 62)
    logger.info(f"Log file: {log_file}")

    # ── Apply CLI overrides ──
    if args.no_sam2:
        config.ENABLE_SAM2_REFINEMENT = False
        logger.info("SAM2 refinement DISABLED (--no-sam2)")

    if args.no_tracking:
        config.ENABLE_TRACKING = False
        logger.info("Tracking DISABLED (--no-tracking)")

    if args.no_smoothing:
        config.ENABLE_TEMPORAL_SMOOTHING = False
        logger.info("Temporal smoothing DISABLED (--no-smoothing)")

    if args.no_world:
        config.USE_WORLD_MODEL = False
        logger.info("YOLO-World DISABLED (--no-world) — using fallback YOLO only")

    if args.frame_skip is not None:
        config.FRAME_SKIP = max(1, args.frame_skip)
        logger.info(f"Frame skip set to {config.FRAME_SKIP} (--frame-skip)")

    if args.min_area is not None:
        config.MIN_DETECTION_AREA_PX = max(0, args.min_area)
        logger.info(f"Min detection area set to {config.MIN_DETECTION_AREA_PX}px (--min-area)")

    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        logger.info(f"Output directory: {config.OUTPUT_DIR}")

    print_config_summary(logger)

    # ── Find .bag files ──
    if args.input:
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return
        bag_files = [args.input]
    else:
        bag_files = sorted(glob.glob(os.path.join(config.RECORDINGS_DIR, "*.bag")))

    if not bag_files:
        logger.error(
            f"No .bag files found in {config.RECORDINGS_DIR}\n"
            f"  → Use --input to specify a file directly, or\n"
            f"  → Place .bag files in the Recordings/ directory"
        )
        return

    logger.info(f"Found {len(bag_files)} .bag recording(s) to process")

    total_start = time.time()

    for i, bag_path in enumerate(bag_files, 1):
        filename = os.path.basename(bag_path)
        output_path = os.path.join(config.OUTPUT_DIR, f"processed_{filename}.mp4")

        logger.info(f"\n{'─'*62}")
        logger.info(f"Processing [{i}/{len(bag_files)}]: {filename}")
        logger.info(f"{'─'*62}")

        processor = BagProcessor(bag_path, output_path)
        processor.process()

    total_time = time.time() - total_start
    logger.info(f"\n{'='*62}")
    logger.info(f"All {len(bag_files)} recording(s) processed in {total_time:.2f}s")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"{'='*62}")


if __name__ == "__main__":
    main()
