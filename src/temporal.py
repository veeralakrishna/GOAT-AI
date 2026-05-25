"""
GOAT-AI Temporal Smoother
==========================
Kalman filter-based temporal smoothing per tracked animal.
Smooths jittery frame-to-frame measurement noise into stable biometrics.

Improvements vs baseline:
  [FIX-13] Per-metric Kalman noise tuning — different measurements have different noise characteristics
  [NEW]    EMA (Exponential Moving Average) fallback for untracked detections (track_id == -1)
  [NEW]    reset_on_gap — Kalman state is reset if a track reappears after a long absence
           (prevents stale lag from old state being applied to a re-acquired animal)
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Default per-metric (Q, R) tuning — overridden by config.KALMAN_NOISE_PER_METRIC
_DEFAULT_NOISE_PER_METRIC: Dict[str, Tuple[float, float]] = {
    "length_cm":             (0.05, 3.0),
    "height_cm":             (0.05, 3.0),
    "chest_girth_cm":        (0.05, 2.0),
    "stance_cm":             (0.20, 8.0),
    "weight_schaefer_kg":    (0.05, 4.0),
    "weight_regression_kg":  (0.05, 4.0),
    "weight_bcs_kg":         (0.05, 4.0),
    "weight_avg_kg":         (0.05, 3.0),
}

# EMA alpha for untracked detections (higher = more responsive, less smooth)
_EMA_ALPHA = 0.3


class KalmanSmoother:
    """
    Simple 1D Kalman filter for smoothing a single measurement.
    Tracks the frame index of last update to support gap detection.
    """

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 5.0):
        self.Q = process_noise
        self.R = measurement_noise
        self.x: Optional[float] = None   # State estimate
        self.P: float = 1.0              # Error covariance
        self._initialized: bool = False
        self._last_update_frame: int = -1

    def update(self, measurement: float, frame_idx: int = -1) -> float:
        """Process a new measurement and return the smoothed value."""
        self._last_update_frame = frame_idx

        if not self._initialized or self.x is None:
            self.x = measurement
            self.P = self.R
            self._initialized = True
            return self.x

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update (Kalman gain)
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        return float(self.x)

    def reset(self):
        """Reset filter state (called when a track reappears after a gap)."""
        self.x = None
        self.P = 1.0
        self._initialized = False
        self._last_update_frame = -1

    def get_state(self) -> Optional[float]:
        return self.x

    @property
    def last_update_frame(self) -> int:
        return self._last_update_frame


class AnimalSmoother:
    """
    Per-animal smoother that manages Kalman filters for each biometric measurement.
    Supports per-metric noise tuning and EMA fallback.
    """

    # All metrics we smooth
    METRICS = [
        "length_cm", "height_cm", "chest_girth_cm", "stance_cm",
        "weight_schaefer_kg", "weight_regression_kg", "weight_bcs_kg", "weight_avg_kg",
    ]

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 5.0,
        noise_per_metric: Optional[Dict[str, Tuple[float, float]]] = None,
        track_reappear_gap: int = 30,
    ):
        self.Q_default = process_noise
        self.R_default = measurement_noise
        self.track_reappear_gap = track_reappear_gap

        # Merge user-supplied per-metric noise with defaults
        self._noise_map = dict(_DEFAULT_NOISE_PER_METRIC)
        if noise_per_metric:
            self._noise_map.update(noise_per_metric)

        # [FIX-13] Create separate Kalman filter per metric with tuned Q, R
        self.filters: Dict[str, KalmanSmoother] = {}
        for metric in self.METRICS:
            Q, R = self._noise_map.get(metric, (process_noise, measurement_noise))
            self.filters[metric] = KalmanSmoother(Q, R)

        # EMA states (for untracked fallback)
        self._ema: Dict[str, Optional[float]] = {m: None for m in self.METRICS}

    def smooth(self, metrics: dict, frame_idx: int = -1) -> dict:
        """
        Apply temporal smoothing to a metrics dict.
        Returns updated metrics dict with smoothed values.
        """
        smoothed = dict(metrics)

        for key in self.METRICS:
            if key not in metrics or metrics[key] is None or metrics[key] <= 0:
                continue
            raw = float(metrics[key])
            smoothed[key] = round(self.filters[key].update(raw, frame_idx), 2)

        return smoothed

    def smooth_ema(self, metrics: dict) -> dict:
        """
        Apply EMA smoothing (for untracked detections — no persistent ID).
        Uses a global EMA state rather than per-track Kalman.
        """
        smoothed = dict(metrics)

        for key in self.METRICS:
            if key not in metrics or metrics[key] is None or metrics[key] <= 0:
                continue
            raw = float(metrics[key])
            if self._ema[key] is None:
                self._ema[key] = raw
            else:
                self._ema[key] = _EMA_ALPHA * raw + (1 - _EMA_ALPHA) * self._ema[key]
            smoothed[key] = round(self._ema[key], 2)

        return smoothed

    def check_and_reset_on_gap(self, current_frame: int):
        """
        Reset Kalman filters if the animal hasn't been seen for `track_reappear_gap` frames.
        Prevents stale pre-gap state being applied to a re-acquired track.
        """
        for key, filt in self.filters.items():
            if (filt.last_update_frame >= 0 and
                    current_frame - filt.last_update_frame > self.track_reappear_gap):
                logger.debug(
                    f"Track gap detected ({current_frame - filt.last_update_frame} frames) "
                    f"for metric '{key}' — resetting Kalman filter"
                )
                filt.reset()


class TemporalSmootherEngine:
    """
    Manages per-track AnimalSmoothers for all tracked animals.
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 5.0,
        noise_per_metric: Optional[Dict[str, Tuple[float, float]]] = None,
        track_reappear_gap: int = 30,
    ):
        self._smoothers: Dict[int, AnimalSmoother] = {}
        self._global_ema = AnimalSmoother(process_noise, measurement_noise, noise_per_metric)
        self.Q = process_noise
        self.R = measurement_noise
        self.noise_per_metric = noise_per_metric
        self.track_reappear_gap = track_reappear_gap

        logger.info(
            f"Temporal smoother initialized — Q={process_noise}, R={measurement_noise}, "
            f"per-metric tuning={'yes' if noise_per_metric else 'default'}"
        )

    def smooth(self, track_id: int, metrics: dict, frame_idx: int = -1) -> dict:
        """
        Apply temporal smoothing for a specific tracked animal.

        Args:
            track_id:  Unique track ID (use -1 for untracked detections → EMA)
            metrics:   Raw metrics dict from BiometricEngine
            frame_idx: Current frame index (for gap detection)

        Returns:
            Smoothed metrics dict
        """
        # [NEW] Untracked detections use global EMA instead of being returned raw
        if track_id < 0:
            return self._global_ema.smooth_ema(metrics)

        # Per-track Kalman smoother
        if track_id not in self._smoothers:
            self._smoothers[track_id] = AnimalSmoother(
                self.Q, self.R,
                noise_per_metric=self.noise_per_metric,
                track_reappear_gap=self.track_reappear_gap,
            )
        else:
            # [NEW] Check for track gap and reset if needed
            self._smoothers[track_id].check_and_reset_on_gap(frame_idx)

        return self._smoothers[track_id].smooth(metrics, frame_idx)

    def get_smoother(self, track_id: int) -> Optional[AnimalSmoother]:
        return self._smoothers.get(track_id)

    def reset(self, track_id: int = None):
        """Reset smoother(s). If track_id is None, reset all."""
        if track_id is not None:
            if track_id in self._smoothers:
                del self._smoothers[track_id]
        else:
            self._smoothers.clear()
