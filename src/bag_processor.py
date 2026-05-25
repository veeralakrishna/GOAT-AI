"""
GOAT-AI Bag Processor (Pipeline Orchestrator)
===============================================
Reads Orbbec .bag files and orchestrates the full pipeline:
  DetectAndTrack (single pass) → Filter → Measure → Smooth → Weight → Visualize → Report

Optimized for RTX 3050 Ti 4GB: single model inference per frame.

Fixes applied:
  [FIX-6]  makedirs guarded against empty dirname (output in root)
  [FIX-9]  pipeline.stop() in finally block — always cleaned up
  [FIX-15] WARMUP_FRAMES: first N frames skip measurement to avoid garbage tracker IDs
  [FIX-18] min_bbox_area filter before BiometricEngine.estimate()
  [NEW]    frame_skip mode: only runs inference on every Nth frame
  [NEW]    per-detection area and depth debug logging
"""
import cv2
import time
import os
import numpy as np
import threading
import queue
import logging

from src.detector import DetectionEngine
from src.tracker import TrackerEngine
from src.measurements import BiometricEngine
from src.temporal import TemporalSmootherEngine
from src.weight_estimator import WeightEstimator
from src.visualizer import Visualizer
from src.reporter import Reporter
import config

from pyorbbecsdk import (
    Pipeline, Config, PlaybackDevice, AlignFilter,
    OBSensorType, OBStreamType, OBFormat, OBPlaybackStatus,
)

logger = logging.getLogger(__name__)


class BagProcessor:
    """
    Full pipeline orchestrator for Orbbec .bag file processing.
    Uses unified detect_and_track() for single-inference-per-frame performance.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.playback_stopped = False

        logger.info("Initializing GOAT-AI pipeline engines...")

        # ── Unified Detection + Tracking (SINGLE inference per frame) ──
        self.detector = DetectionEngine(
            world_model_path=config.YOLO_WORLD_MODEL,
            world_classes=config.YOLO_WORLD_CLASSES,
            world_conf=config.YOLO_WORLD_CONFIDENCE,
            fallback_model_path=config.YOLO_FALLBACK_MODEL,
            fallback_classes=config.YOLO_FALLBACK_CLASSES,
            fallback_conf=config.YOLO_FALLBACK_CONFIDENCE,
            sam2_model_path=config.SAM2_MODEL,
            enable_sam2=config.ENABLE_SAM2_REFINEMENT,
            sam2_every_n=getattr(config, "SAM2_EVERY_N_FRAMES", 10),
            iou_threshold=config.IOU_THRESHOLD,
            enable_tracking=config.ENABLE_TRACKING,
            tracker_type=config.TRACKER_TYPE,
            use_world_model=getattr(config, "USE_WORLD_MODEL", True),
            half_precision=getattr(config, "HALF_PRECISION", True),
            min_area_px=getattr(config, "MIN_DETECTION_AREA_PX", 4000),
            frame_skip=getattr(config, "FRAME_SKIP", 1),
        )

        # Track history manager (lightweight — no model inference)
        self.tracker = TrackerEngine(tracker_type=config.TRACKER_TYPE)

        # Biometric Measurement (intrinsics set after pipeline.start())
        self.estimator = BiometricEngine(
            pixels_per_cm=config.PIXELS_PER_CM,
            voxel_size=config.VOXEL_SIZE,
            outlier_nb_neighbors=config.OUTLIER_NB_NEIGHBORS,
            outlier_std_ratio=config.OUTLIER_STD_RATIO,
            thorax_slice_start=config.THORAX_SLICE_START,
            thorax_slice_end=config.THORAX_SLICE_END,
            thorax_slice_thickness=config.THORAX_SLICE_THICKNESS,
            depth_unit_scale=getattr(config, "DEPTH_UNIT_SCALE", 10.0),
            lateral_body_ratio=getattr(config, "LATERAL_BODY_RATIO", 0.60),
        )

        # Temporal Smoothing — per-metric Kalman noise tuning
        noise_per_metric = getattr(config, "KALMAN_NOISE_PER_METRIC", None)
        self.smoother = TemporalSmootherEngine(
            process_noise=config.KALMAN_PROCESS_NOISE,
            measurement_noise=config.KALMAN_MEASUREMENT_NOISE,
            noise_per_metric=noise_per_metric,
            track_reappear_gap=getattr(config, "TRACK_REAPPEAR_GAP_FRAMES", 30),
        ) if config.ENABLE_TEMPORAL_SMOOTHING else None

        # Weight Estimation
        self.weight_estimator = WeightEstimator(
            breed=config.BREED,
            schaefer_constant=config.SCHAEFER_CONSTANT,
            schaefer_bl_fraction=getattr(config, "SCHAEFER_BL_FRACTION", 0.65),
            regression_intercept=config.REGRESSION_INTERCEPT,
            regression_bl_coeff=config.REGRESSION_BL_COEFF,
            regression_hg_coeff=config.REGRESSION_HG_COEFF,
            agreement_threshold_pct=getattr(config, "WEIGHT_FORMULA_AGREEMENT_THRESHOLD", 40.0),
        ) if config.ENABLE_WEIGHT_ESTIMATION else None

        # Visualization
        self.visualizer = Visualizer(
            mask_opacity=config.VIS_MASK_OPACITY,
            mask_color=config.VIS_MASK_COLOR,
            box_color=config.VIS_BOX_COLOR,
            chest_color=config.VIS_CHEST_COLOR,
            stance_color=config.VIS_STANCE_COLOR,
            text_scale=config.VIS_TEXT_SCALE,
            show_track_id=config.VIS_SHOW_TRACK_ID,
            show_mini_dashboard=config.VIS_SHOW_MINI_DASHBOARD,
            show_dual_weight=config.SHOW_DUAL_WEIGHT,
            confidence_color_coding=getattr(config, "VIS_CONFIDENCE_COLOR_CODING", True),
            show_scale_ruler=getattr(config, "VIS_SHOW_SCALE_RULER", True),
            show_schaefer=getattr(config, "SHOW_SCHAEFER_IN_OVERLAY", False),
            show_bcs_adj=getattr(config, "SHOW_BCS_ADJ_IN_OVERLAY", False),
            show_sirohi_reg=getattr(config, "SHOW_SIROHI_REG_IN_OVERLAY", True),
        )

        # Reporter
        session_name = os.path.splitext(os.path.basename(input_path))[0]
        self.reporter = Reporter(
            output_dir=config.OUTPUT_DIR,
            session_name=session_name,
            export_format=config.EXPORT_FORMAT,
        ) if config.ENABLE_REPORTING else None

        self._warmup_frames = getattr(config, "WARMUP_FRAMES", 5)

        logger.info("All pipeline engines initialized.")

    def on_status_change(self, status):
        if status == OBPlaybackStatus.STOPPED:
            self.playback_stopped = True

    def process(self):
        """Run the full processing pipeline on the .bag file."""
        logger.info(f"Opening bag file: {self.input_path}")

        pipeline = None
        out = None

        try:
            playback = PlaybackDevice(self.input_path)
            playback.set_playback_status_change_callback(self.on_status_change)
            pipeline = Pipeline(playback)
            cfg = Config()
            cfg.enable_stream(OBSensorType.COLOR_SENSOR)
            cfg.enable_stream(OBSensorType.DEPTH_SENSOR)
            align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            pipeline.start(cfg)
        except Exception as e:
            logger.error(f"Failed to open bag file {self.input_path}: {e}")
            return

        try:
            # ── Extract camera intrinsics ──
            color_profile = pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            ).get_video_stream_profile(0)
            intrinsics = color_profile.get_intrinsic()
            self.estimator.fx = intrinsics.fx
            self.estimator.fy = intrinsics.fy
            self.estimator.cx = intrinsics.cx
            self.estimator.cy = intrinsics.cy
            logger.info(
                f"Camera Intrinsics — fx: {intrinsics.fx:.2f}, fy: {intrinsics.fy:.2f}, "
                f"cx: {intrinsics.cx:.2f}, cy: {intrinsics.cy:.2f}"
            )

            width = color_profile.get_width()
            height = color_profile.get_height()
            fps = color_profile.get_fps()

            # ── Output video writer ──
            # [FIX-6]: Use config.OUTPUT_DIR directly instead of dirname(output_path)
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            # ── Threaded Frame Reader ──
            frame_queue: queue.Queue = queue.Queue(maxsize=60)

            def reader_thread():
                last_ts = -1
                consecutive_timeouts = 0

                try:
                    while not self.playback_stopped:
                        try:
                            frames = pipeline.wait_for_frames(100)
                            consecutive_timeouts = 0

                            if not frames:
                                continue

                            frames = align_filter.process(frames)
                            if not frames:
                                continue

                            color_frame = frames.get_color_frame()
                            depth_frame = frames.get_depth_frame()

                            if color_frame and depth_frame:
                                color_data = np.asanyarray(color_frame.get_data()).copy()
                                color_format = color_frame.get_format()
                                c_height = color_frame.get_height()
                                c_width = color_frame.get_width()

                                depth_data = np.frombuffer(
                                    depth_frame.get_data(), dtype=np.uint16
                                ).copy()
                                d_height = depth_frame.get_height()
                                d_width = depth_frame.get_width()

                                ts = color_frame.get_timestamp()

                                if last_ts != -1 and ts < last_ts:
                                    logger.info(
                                        "Bag timestamp restart (loop). Stopping reader."
                                    )
                                    break
                                last_ts = ts

                                frame_queue.put(
                                    (color_data, color_format, c_width, c_height,
                                     depth_data, d_width, d_height, ts)
                                )

                        except Exception as e:
                            err_str = str(e).lower()
                            if "timeout" in err_str:
                                consecutive_timeouts += 1
                                if consecutive_timeouts > 30:
                                    logger.info("EOF (consecutive timeouts). Stopping reader.")
                                    break
                            else:
                                logger.error(f"Reader thread error: {e}", exc_info=True)
                                break
                finally:
                    frame_queue.put(None)  # Always send EOF sentinel

            t = threading.Thread(target=reader_thread, daemon=True)
            t.start()

            # ── Main Processing Loop ──
            frame_count = 0
            written_frames = 0
            warmup_written = 0    # Warmup frames tracked separately from timestamp sync
            start_ts = None
            start_time = time.time()

            while True:
                item = frame_queue.get()
                if item is None:
                    break

                (color_data, color_format, c_width, c_height,
                 depth_data, d_width, d_height, current_ts) = item

                # Decode color frame
                if color_format == OBFormat.MJPG:
                    color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                elif color_format == OBFormat.RGB:
                    color_image = color_data.reshape((c_height, c_width, 3))
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                elif color_format == OBFormat.BGR:
                    color_image = color_data.reshape((c_height, c_width, 3))
                else:
                    logger.warning(f"Unsupported color format {color_format} — skipping frame")
                    continue

                if color_image is None:
                    logger.warning(f"Frame {frame_count}: color decode returned None — skipping")
                    continue

                depth_image = depth_data.reshape((d_height, d_width))
                frame_count += 1

                # ── Capture start_ts from the VERY FIRST frame (including warmup)
                # so timestamp sync covers the full recording duration.
                if start_ts is None:
                    start_ts = current_ts

                # ══════════════════════════════════════════════════════════
                # WARMUP: Skip measurement for first N frames
                # Still run YOLO to warm up tracker state.
                # Write raw (unannotated) frame so video duration is preserved.
                # ══════════════════════════════════════════════════════════
                in_warmup = frame_count <= self._warmup_frames
                if in_warmup:
                    logger.debug(f"Warmup frame {frame_count}/{self._warmup_frames}")
                    # Run detection to warm up tracker (IDs assigned but not recorded)
                    self.detector.detect_and_track(color_image)
                    # Write raw frame using timestamp sync (same as non-warmup path)
                    elapsed_ms = current_ts - start_ts
                    expected_frames = int((elapsed_ms / 1000.0) * fps)
                    frames_to_write = max(1, expected_frames - written_frames)
                    for _ in range(frames_to_write):
                        out.write(color_image)
                        written_frames += 1
                        warmup_written += 1
                    continue

                # ══════════════════════════════════════════════════════════
                # PIPELINE: DetectAndTrack → Measure → Smooth → Weight → Visualize → Report
                # ══════════════════════════════════════════════════════════

                # 1. DETECT + TRACK (SINGLE inference)
                detections = self.detector.detect_and_track(color_image)

                # 2. MEASURE + 3. SMOOTH + 4. WEIGHT
                all_metrics = []
                for det in detections:
                    if len(det.mask_polygon) < 3:
                        all_metrics.append(None)
                        continue

                    # [FIX-18] Minimum bbox area guard before expensive measurement
                    if det.bbox_area < getattr(config, "MIN_DETECTION_AREA_PX", 4000):
                        logger.debug(
                            f"Skipping measurement for small detection "
                            f"(area={det.bbox_area}px, track_id={det.track_id})"
                        )
                        all_metrics.append(None)
                        continue

                    contour = det.mask_polygon
                    metrics = self.estimator.estimate(contour, depth_image=depth_image)

                    # Weight estimation
                    if self.weight_estimator:
                        weight = self.weight_estimator.estimate(
                            body_length_cm=metrics["length_cm"],
                            heart_girth_cm=metrics["chest_girth_cm"],
                            area_cm2=metrics.get("area_cm2", 0.0),
                        )
                        metrics.update(weight)
                        metrics["weight_category"] = self.weight_estimator.get_weight_category(
                            metrics["weight_avg_kg"]
                        )

                    # Temporal smoothing (Kalman per track, EMA for untracked)
                    if self.smoother:
                        metrics = self.smoother.smooth(
                            track_id=det.track_id,
                            metrics=metrics,
                            frame_idx=frame_count,
                        )

                    # Record to tracker history
                    self.tracker.record_metrics(det.track_id, metrics)

                    # Record to reporter
                    if self.reporter:
                        self.reporter.record_frame(
                            frame_id=frame_count,
                            track_id=det.track_id,
                            metrics=metrics,
                            timestamp_ms=current_ts,
                        )

                    all_metrics.append(metrics)

                    # Debug logging
                    logger.debug(
                        f"Frame {frame_count} | Track #{det.track_id} | "
                        f"area={det.bbox_area}px | "
                        f"L={metrics['length_cm']}cm | "
                        f"HG={metrics['chest_girth_cm']}cm | "
                        f"W={metrics.get('weight_avg_kg', 0):.1f}kg | "
                        f"conf={metrics['confidence']}%"
                    )

                # 5. VISUALIZE
                elapsed_time = time.time() - start_time
                fps_proc = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Pass depth_image for scale ruler rendering
                annotated = self.visualizer.draw(
                    color_image, detections, all_metrics,
                    frame_count=frame_count,
                    fps_proc=fps_proc,
                    depth_image=depth_image,
                    fx=self.estimator.fx,
                )

                # 6. WRITE (timestamp-synchronized to preserve original video duration)
                elapsed_ms = current_ts - start_ts
                expected_frames = int((elapsed_ms / 1000.0) * fps)
                frames_to_write = max(1, expected_frames - written_frames)

                for _ in range(frames_to_write):
                    out.write(annotated)
                    written_frames += 1

                if frame_count % 30 == 0:
                    logger.info(
                        f"Frame {frame_count} | Written: {written_frames} | "
                        f"Queue: {frame_queue.qsize()} | FPS: {fps_proc:.1f} | "
                        f"Animals: {len(detections)}"
                    )

            t.join(timeout=5.0)

        finally:
            # [FIX-9]: Always stop pipeline and release writer — even on exception
            if pipeline is not None:
                try:
                    pipeline.stop()
                    logger.debug("Pipeline stopped.")
                except Exception as e:
                    logger.warning(f"Pipeline stop failed: {e}")

            if out is not None:
                out.release()
                logger.debug("Video writer released.")

        # ── Summary & Export ──
        end_time = time.time()
        duration = end_time - start_time
        fps_final = frame_count / duration if duration > 0 else 0

        if self.reporter:
            track_summaries = self.tracker.get_all_track_summaries()
            self.reporter.export(
                track_summaries=track_summaries,
                total_frames=frame_count,
                processing_time=duration,
            )

        logger.info(f"{'='*60}")
        logger.info(f"Processing complete: {self.output_path}")
        logger.info(
            f"Duration: {duration:.2f}s | Frames: {frame_count} "
            f"(+{self._warmup_frames} warmup) | FPS: {fps_final:.2f}"
        )
        summaries = self.tracker.get_all_track_summaries()
        logger.info(f"Unique animals tracked: {len(summaries)}")
        for tid, s in summaries.items():
            logger.info(
                f"  Goat #{tid}: L={s.get('length_cm','?')}cm, "
                f"HG={s.get('chest_girth_cm','?')}cm, "
                f"W={s.get('weight_avg_kg','?')}kg "
                f"({s.get('frame_count',0)} frames, "
                f"conf={s.get('confidence','?')}%)"
            )
        logger.info(f"{'='*60}")

    def get_reporter(self) -> Reporter:
        return self.reporter

    def get_tracker(self) -> TrackerEngine:
        return self.tracker
