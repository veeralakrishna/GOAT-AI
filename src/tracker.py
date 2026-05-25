"""
GOAT-AI Multi-Object Tracker
==============================
Persistent identity tracking using Ultralytics' built-in BoT-SORT/ByteTrack
(assigned directly inside DetectionEngine.detect_and_track).

This module manages per-track HISTORY only — no second inference call.

Note: The TrackerEngine.update() method previously existed but caused a
second .track() inference call which corrupted tracking state.
It has been removed. Track IDs are now assigned inside DetectionEngine._track_pass().
"""
import logging
import numpy as np
from typing import List, Dict, Optional
from src.detector import Detection

logger = logging.getLogger(__name__)


class TrackerEngine:
    """
    Manages per-animal measurement history and generates track summaries.
    Does NOT run model inference — that is handled by DetectionEngine.
    """

    def __init__(self, tracker_type: str = "botsort.yaml"):
        self.tracker_type = tracker_type
        self._track_history: Dict[int, List[dict]] = {}
        self._track_first_frame: Dict[int, int] = {}
        self._track_last_frame: Dict[int, int] = {}
        logger.info(f"Tracker history manager initialized: {tracker_type}")

    def record_metrics(self, track_id: int, metrics: dict, frame_id: int = -1):
        """Record a frame's metrics for a tracked animal."""
        if track_id < 0:
            return  # Untracked detections not recorded in per-animal history

        if track_id not in self._track_history:
            self._track_history[track_id] = []
            self._track_first_frame[track_id] = frame_id

        self._track_history[track_id].append(dict(metrics))
        self._track_last_frame[track_id] = frame_id

    def get_track_history(self, track_id: int) -> List[dict]:
        """Get the full measurement history for a specific track."""
        return self._track_history.get(track_id, [])

    def get_all_track_ids(self) -> List[int]:
        """Return all known track IDs."""
        return list(self._track_history.keys())

    def get_all_track_summaries(self) -> Dict[int, dict]:
        """
        Generate per-track summary statistics (median + std of each metric).
        Used for the final report.
        """
        summaries = {}
        for track_id, history in self._track_history.items():
            if not history:
                continue

            summary = {
                "track_id": track_id,
                "frame_count": len(history),
                "first_frame": self._track_first_frame.get(track_id, -1),
                "last_frame": self._track_last_frame.get(track_id, -1),
            }

            # Median estimates for stable per-animal biometrics
            numeric_keys = [
                "length_cm", "height_cm", "chest_girth_cm",
                "weight_schaefer_kg", "weight_regression_kg",
                "weight_bcs_kg", "weight_avg_kg",
                "median_depth_cm", "confidence",
            ]
            for key in numeric_keys:
                values = [
                    h[key] for h in history
                    if key in h and h[key] is not None and float(h[key]) > 0
                ]
                if values:
                    arr = np.array(values, dtype=float)
                    summary[key] = round(float(np.median(arr)), 2)
                    summary[f"{key}_std"] = round(float(np.std(arr)), 2)

            # Most common weight category
            categories = [h.get("weight_category") for h in history if h.get("weight_category")]
            if categories:
                summary["weight_category"] = max(set(categories), key=categories.count)

            summaries[track_id] = summary

        return summaries

    def get_stable_measurements(
        self,
        track_id: int,
        min_frames: int = 10,
    ) -> Optional[dict]:
        """
        Return median measurements for a track only if it has enough observations.
        Useful for final reporting — avoids single-frame noisy estimates.
        """
        history = self._track_history.get(track_id, [])
        if len(history) < min_frames:
            return None

        summaries = self.get_all_track_summaries()
        return summaries.get(track_id)
