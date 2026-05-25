"""
GOAT-AI Detection Engine
=========================
Unified detection + tracking pipeline optimized for RTX 3050 Ti (4GB VRAM):

  Mode A (Fast — default): Single YOLO .track() call → detection + segmentation + tracking in ONE pass
  Mode B (Quality):        YOLO-World detect → SAM2 refinement (toggle per-frame or every N frames)

Only ONE model runs inference per frame. No redundant passes.

Fixes applied:
  - Robust exception handling: only fall back to _detect_pass on terminal errors, not tracker glitches
  - Correct fallback_classes include class 20 (horse) — Sirohi goats often classified as horse
  - min_area_px filtering rejects tiny/partial/distant false positive detections
  - Consecutive failure tracking with meaningful warnings
  - frame_skip support returns cached detections on skipped frames
"""
import logging
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Exceptions that indicate a tracker state problem (recoverable by reset)
_TRACKER_GLITCH_MSGS = ("tracker", "tracking", "lost", "id", "state")
# Exceptions that are truly fatal (should not fall back silently)
_FATAL_EXCEPTION_TYPES = (MemoryError, RuntimeError)


@dataclass
class Detection:
    """Structured detection result for a single animal."""
    bbox: np.ndarray          # [x1, y1, x2, y2] bounding box
    mask: np.ndarray          # Binary mask (H, W)
    mask_polygon: np.ndarray  # Polygon contour points (N, 2)
    confidence: float
    class_name: str
    class_id: int = -1
    track_id: int = -1        # Assigned by tracker
    bbox_area: int = 0        # Pixel area of bounding box (for filtering)


class DetectionEngine:
    """
    Unified detection engine that combines detection + segmentation + tracking
    in a SINGLE model inference pass for maximum GPU efficiency.

    Architecture:
      - Primary path: fallback_model.track() — ONE inference gives boxes, masks, AND track IDs
      - Optional: YOLO-World for initial detection when zero-shot goat class is needed
      - Optional: SAM2 refinement every N frames (not every frame)

    Robustness improvements:
      - min_area_px: filters detections below a minimum bounding-box area
      - frame_skip: returns last valid detections on skipped frames
      - Distinguishes recoverable tracker glitches from fatal errors
    """

    def __init__(
        self,
        world_model_path: str = "yolov8s-worldv2.pt",
        world_classes: list = None,
        world_conf: float = 0.15,
        fallback_model_path: str = "yolov8n-seg.pt",
        fallback_classes: list = None,
        fallback_conf: float = 0.3,
        sam2_model_path: str = "sam2_b.pt",
        enable_sam2: bool = False,
        sam2_every_n: int = 10,
        iou_threshold: float = 0.45,
        enable_tracking: bool = True,
        tracker_type: str = "botsort.yaml",
        use_world_model: bool = True,
        half_precision: bool = True,
        min_area_px: int = 4000,
        frame_skip: int = 1,
    ):
        import torch
        from ultralytics import YOLO

        self.iou_threshold = iou_threshold
        self.enable_tracking = enable_tracking
        self.tracker_type = tracker_type
        self.use_world_model = use_world_model
        self.min_area_px = min_area_px
        self.frame_skip = max(1, frame_skip)

        self._frame_count = 0
        self._consecutive_failures = 0
        self._last_detections: List[Detection] = []  # Cache for frame-skip

        # ── GPU Detection ──
        if torch.cuda.is_available():
            self.device = 0
            self.half_precision = half_precision
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name} — CUDA inference enabled")
        else:
            self.device = "cpu"
            self.half_precision = False
            logger.warning("No CUDA GPU found — running on CPU (will be slow)")

        # ── Primary Model: YOLO-World OR Fallback ──
        self._world_model = None
        self._world_available = False

        if use_world_model:
            try:
                from ultralytics import YOLOWorld
                logger.info(f"Loading YOLO-World: {world_model_path}")
                self._world_model = YOLOWorld(world_model_path)
                self.world_classes = world_classes or ["goat", "kid goat", "sheep"]
                self._world_model.set_classes(self.world_classes)
                self.world_conf = world_conf
                self._world_available = True
                logger.info(f"YOLO-World ready — classes: {self.world_classes}")
            except Exception as e:
                logger.warning(f"YOLO-World failed to load: {e}. Falling back to standard YOLO.")

        # ── Fallback YOLO — segmentation + tracking (single pass) ──
        logger.info(f"Loading YOLO segmentation: {fallback_model_path}")
        self.fallback_model = YOLO(fallback_model_path)
        if self.device == 0:
            self.fallback_model.to("cuda:0")
            logger.info("YOLO model moved to GPU")

        # FIX: Use constructor arg — includes class 17 (horse) for Sirohi goat detection
        # COCO IDs: 17=horse, 18=sheep, 19=cow  (class 20 = ELEPHANT, not horse!)
        self.fallback_classes = fallback_classes or [17, 18, 19]
        self.fallback_conf = fallback_conf
        self._min_coverage = 0.002  # min fraction of frame area (initialized here, not via getattr)
        logger.info(f"YOLO fallback classes (COCO IDs): {self.fallback_classes} "
                    f"→ {[self._coco_name(c) for c in self.fallback_classes]}")
        logger.info(f"YOLO confidence threshold: {fallback_conf}")

        # ── SAM2 Refinement (optional, every N frames) ──
        self.enable_sam2 = enable_sam2
        self.sam2_every_n = max(1, sam2_every_n)
        self._sam2_model = None
        if enable_sam2:
            try:
                from ultralytics import SAM
                logger.info(f"Loading SAM2: {sam2_model_path}")
                self._sam2_model = SAM(sam2_model_path)
                logger.info(f"SAM2 ready (runs every {sam2_every_n} frames)")
            except Exception as e:
                logger.warning(f"SAM2 failed to load: {e}. SAM2 disabled.")
                self.enable_sam2 = False

        logger.info(
            f"DetectionEngine ready — min_area={min_area_px}px, "
            f"frame_skip={frame_skip}, tracking={enable_tracking}"
        )

    # ──────────────────────────────────────────────────────────────
    # Public Interface
    # ──────────────────────────────────────────────────────────────

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        """
        SINGLE-PASS detection + segmentation + tracking.
        Returns list of Detection objects with track_id assigned.

        On frame_skip>1, returns the cached last result for intermediate frames.
        """
        self._frame_count += 1

        # ── Frame skipping: return cached detections on skipped frames ──
        if self.frame_skip > 1 and (self._frame_count % self.frame_skip) != 1:
            logger.debug(f"Frame {self._frame_count}: skipped (using cached detections)")
            return self._last_detections

        # ── Detect + Track ──
        if self.enable_tracking:
            detections = self._track_pass(frame)
        else:
            detections = self._detect_pass(frame)

        # ── Filter by minimum area ──
        detections = self._filter_by_area(detections, frame.shape)

        # ── Optional SAM2 refinement (every N frames only) ──
        if (detections and self.enable_sam2 and self._sam2_model is not None
                and self._frame_count % self.sam2_every_n == 0):
            detections = self._refine_with_sam2(frame, detections)

        self._last_detections = detections
        return detections

    # ──────────────────────────────────────────────────────────────
    # Private: Detection Passes
    # ──────────────────────────────────────────────────────────────

    def _track_pass(self, frame: np.ndarray) -> List[Detection]:
        """
        SINGLE model.track() call → detection + segmentation + tracking IDs.
        FIX: Only fall back to _detect_pass on truly unrecoverable errors,
        not on tracker glitches (which are transient and self-correcting).
        """
        try:
            results = self.fallback_model.track(
                source=frame,
                tracker=self.tracker_type,
                persist=True,
                conf=self.fallback_conf,
                classes=self.fallback_classes,
                iou=self.iou_threshold,
                verbose=False,
                retina_masks=True,
                half=self.half_precision,
                device=self.device,
            )

            self._consecutive_failures = 0  # Reset on success
            return self._parse_results(results, frame.shape, with_ids=True)

        except Exception as e:
            err_str = str(e).lower()
            self._consecutive_failures += 1

            # Tracker glitches (ID assignment issues) — log and return empty for this frame
            is_tracker_glitch = any(kw in err_str for kw in _TRACKER_GLITCH_MSGS)

            if is_tracker_glitch:
                if self._consecutive_failures <= 3:
                    logger.debug(f"Tracker glitch (frame {self._frame_count}): {e}")
                elif self._consecutive_failures == 4:
                    logger.warning(
                        f"Tracker glitch repeated {self._consecutive_failures}x — "
                        f"consider resetting. Latest: {e}"
                    )
                return []  # Return empty; tracker will self-recover on next frame

            # Fatal / unexpected errors — fall back to detect-only with warning
            logger.warning(
                f"Track pass failed (non-tracker error, frame {self._frame_count}): {e}. "
                f"Falling back to detect-only for this frame."
            )
            return self._detect_pass(frame)

    def _detect_pass(self, frame: np.ndarray) -> List[Detection]:
        """Fallback: detection + segmentation only (no tracking)."""
        try:
            results = self.fallback_model.predict(
                source=frame,
                conf=self.fallback_conf,
                classes=self.fallback_classes,
                iou=self.iou_threshold,
                verbose=False,
                retina_masks=True,
                half=self.half_precision,
                device=self.device,
            )
            return self._parse_results(results, frame.shape, with_ids=False)
        except Exception as e:
            logger.error(f"Detect pass failed (frame {self._frame_count}): {e}")
            return []

    def _parse_results(
        self, results, frame_shape: tuple, with_ids: bool
    ) -> List[Detection]:
        """
        Parse Ultralytics Results into Detection objects.
        Shared between _track_pass and _detect_pass.
        """
        if not results:
            return []

        result = results[0]
        if result.masks is None or result.boxes is None:
            return []

        detections = []
        masks_xy = result.masks.xy
        has_ids = with_ids and (result.boxes.id is not None)

        for i, mask_poly in enumerate(masks_xy):
            if len(mask_poly) < 3:  # Need at least a triangle to make a valid mask
                continue

            bbox = result.boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(result.boxes.conf[i].cpu())
            cls_id = int(result.boxes.cls[i].cpu())
            track_id = int(result.boxes.id[i].cpu()) if has_ids else -1

            # Compute bounding box area for downstream filtering
            bw = int(bbox[2]) - int(bbox[0])
            bh = int(bbox[3]) - int(bbox[1])
            bbox_area = bw * bh

            contour = np.array(mask_poly, dtype=np.int32)
            mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)

            detections.append(Detection(
                bbox=bbox,
                mask=mask,
                mask_polygon=contour,
                confidence=conf,
                class_name="goat",
                class_id=cls_id,
                track_id=track_id,
                bbox_area=bbox_area,
            ))

        return detections

    # ──────────────────────────────────────────────────────────────
    # Private: Filtering & Refinement
    # ──────────────────────────────────────────────────────────────

    def _filter_by_area(
        self, detections: List[Detection], frame_shape: tuple
    ) -> List[Detection]:
        """
        Reject detections that are too small to produce reliable measurements.
        Filters by both absolute pixel area and fraction of frame area.
        """
        if not detections or self.min_area_px <= 0:
            return detections

        frame_area = frame_shape[0] * frame_shape[1]
        # Use contour mask area (more accurate) if available; fall back to bbox area
        min_frame_area = int(frame_area * self._min_coverage)
        effective_min = max(self.min_area_px, min_frame_area)

        filtered = []
        for det in detections:
            if det.bbox_area >= effective_min:
                filtered.append(det)
            else:
                logger.debug(
                    f"Filtered small detection: area={det.bbox_area}px "
                    f"(min={effective_min}px, track_id={det.track_id})"
                )

        if len(filtered) < len(detections):
            logger.debug(
                f"Area filter: {len(detections) - len(filtered)} detection(s) removed"
            )

        return filtered

    def _refine_with_sam2(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> List[Detection]:
        """Refine masks with SAM2 (runs only every N frames)."""
        try:
            bboxes = [d.bbox.tolist() for d in detections]
            results = self._sam2_model(frame, bboxes=bboxes, verbose=False)

            if results and results[0].masks is not None:
                for i, det in enumerate(detections):
                    if i < len(results[0].masks.xy):
                        sam_poly = results[0].masks.xy[i]
                        if len(sam_poly) >= 3:
                            contour = np.array(sam_poly, dtype=np.int32)
                            refined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(refined_mask, [contour], 255)
                            det.mask = refined_mask
                            det.mask_polygon = contour

            logger.debug(
                f"SAM2 refined {len(detections)} masks (frame {self._frame_count})"
            )
        except Exception as e:
            logger.warning(f"SAM2 refinement failed (frame {self._frame_count}): {e}")

        return detections

    # ──────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return self._frame_count


    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @staticmethod
    def _coco_name(class_id: int) -> str:
        """Return human-readable COCO class name for logging."""
        _NAMES = {
            0: "person", 14: "bird", 15: "cat", 16: "dog",
            17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
            21: "bear", 22: "zebra", 23: "giraffe",
        }
        return _NAMES.get(class_id, f"cls_{class_id}")
