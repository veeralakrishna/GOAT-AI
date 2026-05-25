"""
GOAT-AI Reporter
==================
Structured data export for downstream analysis.
Exports per-frame CSV, per-animal JSON summary, and session metadata.

Improvements vs baseline:
  [FIX-20] Fixed CSV fieldnames — no longer derived from first record's keys
            (avoids silent column drops when weight estimation is disabled)
  [NEW]    Per-session statistics: std dev, min, max per metric in JSON
  [NEW]    weight_formula_agreement_pct in JSON summary
"""
import csv
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

import config

logger = logging.getLogger(__name__)


# [FIX-20]: Fixed schema — always present regardless of pipeline settings
_CSV_FIELDNAMES = [
    "frame_id",
    "track_id",
    "timestamp_ms",
    "length_cm",
    "height_cm",
    "chest_girth_cm",
    "stance_cm",
    "median_depth_cm",
    "weight_schaefer_kg",
    "weight_regression_kg",
    "weight_bcs_kg",
    "weight_avg_kg",
    "formula_agreement_pct",
    "formula_disagreement_flag",
    "confidence",
    "faces_left",
]

# Numeric fields for statistics computation
_STAT_FIELDS = [
    "length_cm", "height_cm", "chest_girth_cm", "stance_cm",
    "weight_schaefer_kg", "weight_regression_kg", "weight_bcs_kg",
    "weight_avg_kg", "confidence",
]


class Reporter:
    """
    Exports biometric data in CSV and JSON formats with statistics.
    """

    def __init__(
        self,
        output_dir: str,
        session_name: str,
        export_format: str = "both",
    ):
        self.output_dir = output_dir
        self.session_name = session_name
        self.export_format = export_format

        self._frame_records: List[dict] = []
        self._session_start = datetime.now()

        os.makedirs(output_dir, exist_ok=True)

    def record_frame(
        self,
        frame_id: int,
        track_id: int,
        metrics: dict,
        timestamp_ms: float = 0,
    ):
        """Record a single frame's metrics for a tracked animal."""
        record = {
            "frame_id": frame_id,
            "track_id": track_id,
            "timestamp_ms": round(float(timestamp_ms), 2),
            "length_cm":             metrics.get("length_cm", 0) or 0,
            "height_cm":             metrics.get("height_cm", 0) or 0,
            "chest_girth_cm":        metrics.get("chest_girth_cm", 0) or 0,
            "stance_cm":             metrics.get("stance_cm", 0) or 0,
            "median_depth_cm":       metrics.get("median_depth_cm") or 0,
            "weight_schaefer_kg":    metrics.get("weight_schaefer_kg", 0) or 0,
            "weight_regression_kg":  metrics.get("weight_regression_kg", 0) or 0,
            "weight_bcs_kg":         metrics.get("weight_bcs_kg", 0) or 0,
            "weight_avg_kg":         metrics.get("weight_avg_kg", 0) or 0,
            "formula_agreement_pct": metrics.get("formula_agreement_pct", 100.0),
            "formula_disagreement_flag": int(metrics.get("formula_disagreement_flag", False)),
            "confidence":            metrics.get("confidence", 0) or 0,
            "faces_left":            int(metrics.get("faces_left", True)),
        }
        self._frame_records.append(record)

    def export(
        self,
        track_summaries: Optional[Dict[int, dict]] = None,
        total_frames: int = 0,
        processing_time: float = 0,
    ):
        """Export all recorded data to files."""
        if self.export_format in ("csv", "both"):
            self._export_csv()
        if self.export_format in ("json", "both"):
            self._export_json(track_summaries, total_frames, processing_time)
        logger.info(f"Reports exported to {self.output_dir}")

    def _export_csv(self):
        """Export per-frame data as CSV with fixed schema."""
        csv_path = os.path.join(self.output_dir, f"{self.session_name}_frames.csv")

        if not self._frame_records:
            logger.warning("No frame records to export")
            return

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            # [FIX-20]: Use pre-defined fieldnames, not dynamic keys from records
            writer = csv.DictWriter(
                f,
                fieldnames=_CSV_FIELDNAMES,
                extrasaction="ignore",  # Ignore any extra keys in records
            )
            writer.writeheader()
            for record in self._frame_records:
                # Fill any missing keys with 0 so CSV never has empty cells
                row = {k: record.get(k, 0) for k in _CSV_FIELDNAMES}
                writer.writerow(row)

        logger.info(f"CSV exported: {csv_path} ({len(self._frame_records)} rows)")

    def _export_json(
        self,
        track_summaries: Optional[Dict[int, dict]] = None,
        total_frames: int = 0,
        processing_time: float = 0,
    ):
        """Export session summary as JSON with per-track statistics."""
        json_path = os.path.join(self.output_dir, f"{self.session_name}_summary.json")

        session = {
            "session": {
                "name": self.session_name,
                "breed": config.BREED,
                "start_time": self._session_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_frames_processed": total_frames,
                "processing_time_seconds": round(processing_time, 2),
                "unique_animals_detected": len(track_summaries) if track_summaries else 0,
            },
            "models": {
                "detection": f"YOLO-World v2 ({config.YOLO_WORLD_MODEL}) + Fallback ({config.YOLO_FALLBACK_MODEL})",
                "tracking": config.TRACKER_TYPE,
                "segmentation": f"SAM2 ({config.SAM2_MODEL})" if config.ENABLE_SAM2_REFINEMENT else "YOLO-masks",
                "biometrics": "2D-projection + Open3D 3D cross-section",
                "weight_formula_1": f"Schaefer Standard (HG²×BL/{config.SCHAEFER_CONSTANT:.0f})",
                "weight_formula_2": (
                    f"Sirohi Regression "
                    f"({config.REGRESSION_INTERCEPT} + {config.REGRESSION_BL_COEFF}×BL "
                    f"+ {config.REGRESSION_HG_COEFF}×HG)"
                ),
                "weight_formula_3": "BCS-Adjusted Schaefer",
            },
            "animals": {},
        }

        if track_summaries:
            for track_id, summary in track_summaries.items():
                # [NEW]: Add per-track per-metric statistics from frame records
                track_records = [
                    r for r in self._frame_records if r["track_id"] == track_id
                ]
                stats = self._compute_statistics(track_records)
                entry = dict(summary)
                entry["statistics"] = stats
                entry["weight_formula_agreement_pct"] = (
                    round(float(np.mean([
                        r["formula_agreement_pct"]
                        for r in track_records
                        if r.get("formula_agreement_pct") is not None
                    ])), 1) if track_records else None
                )
                session["animals"][f"goat_{track_id}"] = entry

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, default=str)

        logger.info(f"JSON exported: {json_path}")

    def _compute_statistics(self, records: List[dict]) -> dict:
        """
        Compute per-metric statistics (mean, median, std, min, max) from a list of records.
        """
        stats = {}
        for field in _STAT_FIELDS:
            values = [
                float(r[field]) for r in records
                if field in r and r[field] is not None and float(r[field]) > 0
            ]
            if values:
                arr = np.array(values)
                stats[field] = {
                    "mean":   round(float(np.mean(arr)), 2),
                    "median": round(float(np.median(arr)), 2),
                    "std":    round(float(np.std(arr)), 2),
                    "min":    round(float(np.min(arr)), 2),
                    "max":    round(float(np.max(arr)), 2),
                    "n":      len(values),
                }
        return stats

    def get_records(self) -> List[dict]:
        """Get all frame records (for Gradio UI display)."""
        return self._frame_records
