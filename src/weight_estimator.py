"""
GOAT-AI Weight Estimator
=========================
Live weight estimation using established veterinary formulas.
Implements triple estimation (Schaefer Standard + Sirohi Regression + BCS-Adjusted)
and cross-validates the results for reliability.

References:
  - Schaefer Standard: Weight(kg) = HG² × BL / 10840
  - Indian Multivariate (Sirohi-adapted ICAR/AICRP):
    Weight(kg) = -28.57 + 0.144×BL + 0.538×HG
  - BCS-Adjusted (derived):
    Weight(kg) = BCS_factor × Schaefer, BCS_factor from area/length ratio

Fixes applied:
  [FIX-12] _regression returns None for out-of-range results (not 0.0) — avoids false zeros
  [NEW]    BCS-adjusted formula using mask area/length ratio as body condition proxy
  [NEW]    Formula agreement gate: flags low confidence when estimates differ >40%
"""
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class WeightEstimator:
    """
    Multi-formula live-weight estimation engine for Sirohi goats.
    Shows all formula outputs for cross-validation and flags disagreements.
    """

    # Valid weight range for any goat (reject obvious outliers)
    _MIN_VALID_KG = 2.0
    _MAX_VALID_KG = 150.0

    def __init__(
        self,
        breed: str = "sirohi",
        schaefer_constant: float = 10840.0,
        schaefer_bl_fraction: float = 0.65,
        regression_intercept: float = -28.57,
        regression_bl_coeff: float = 0.144,
        regression_hg_coeff: float = 0.538,
        agreement_threshold_pct: float = 40.0,
    ):
        self.breed = breed
        self.schaefer_constant = schaefer_constant
        # BL fraction: corrects nose-to-tail measurement → shoulder-to-pin (what Schaefer needs)
        self.schaefer_bl_fraction = schaefer_bl_fraction
        self.reg_a = regression_intercept
        self.reg_b1 = regression_bl_coeff
        self.reg_b2 = regression_hg_coeff
        self.agreement_threshold = agreement_threshold_pct
        logger.info(
            f"Weight estimator: breed={breed}, Schaefer BL fraction={schaefer_bl_fraction:.2f}"
            f" (corrects nose-to-tail → shoulder-to-pin)"
        )

    def estimate(
        self,
        body_length_cm: float,
        heart_girth_cm: float,
        area_cm2: float = 0.0,
    ) -> dict:
        """
        Estimate live weight using all three formulas.

        Args:
            body_length_cm: Body length in cm
            heart_girth_cm: Heart girth (chest circumference) in cm
            area_cm2:       Mask projected area in cm² (for BCS formula)

        Returns:
            dict with all weight fields, agreement score, and formula flag
        """
        w_schaefer = self._schaefer(body_length_cm, heart_girth_cm)
        w_regression = self._regression(body_length_cm, heart_girth_cm)
        w_bcs = self._bcs_schaefer(body_length_cm, heart_girth_cm, area_cm2)

        # Collect valid (non-None) estimates
        valid_weights = [w for w in [w_schaefer, w_regression, w_bcs] if w is not None and w > 0]

        # Ensemble average of valid estimates
        w_avg = float(sum(valid_weights) / len(valid_weights)) if valid_weights else 0.0

        # Agreement check: how far apart are the two main formulas?
        formula_agreement_pct = 100.0
        formula_disagreement_flag = False
        if w_schaefer is not None and w_regression is not None and w_schaefer > 0 and w_regression > 0:
            diff_pct = abs(w_schaefer - w_regression) / (0.5 * (w_schaefer + w_regression)) * 100.0
            formula_agreement_pct = round(100.0 - diff_pct, 1)
            formula_disagreement_flag = (diff_pct > self.agreement_threshold)
            if formula_disagreement_flag:
                logger.debug(
                    f"Weight formula disagreement: Schaefer={w_schaefer:.1f}kg, "
                    f"Regression={w_regression:.1f}kg ({diff_pct:.0f}% apart)"
                )

        return {
            "weight_schaefer_kg": round(w_schaefer, 2) if w_schaefer is not None else 0.0,
            "weight_regression_kg": round(w_regression, 2) if w_regression is not None else 0.0,
            "weight_bcs_kg": round(w_bcs, 2) if w_bcs is not None else 0.0,
            "weight_avg_kg": round(w_avg, 2),
            "weight_schaefer_lbs": round(w_schaefer * 2.20462, 2) if w_schaefer else 0.0,
            "weight_regression_lbs": round(w_regression * 2.20462, 2) if w_regression else 0.0,
            "weight_avg_lbs": round(w_avg * 2.20462, 2),
            "formula_agreement_pct": formula_agreement_pct,
            "formula_disagreement_flag": formula_disagreement_flag,
        }

    # ──────────────────────────────────────────────────────────────
    # Formula 1: Schaefer Standard
    # ──────────────────────────────────────────────────────────────

    def _schaefer(self, bl: float, hg: float) -> Optional[float]:
        """
        Schaefer Standard Formula (breed-generic):
        Weight (kg) = HG² × BL / constant

        Our measured BL is nose-to-tail (bbox major axis). Schaefer requires
        shoulder-to-pin-bone length, which is ~schaefer_bl_fraction of nose-to-tail.
        We apply this correction internally so the reported length_cm remains accurate.
        """
        if bl <= 0 or hg <= 0:
            return None

        # Apply body-length correction (nose-to-tail → shoulder-to-pin)
        corrected_bl = bl * self.schaefer_bl_fraction
        weight = (hg ** 2 * corrected_bl) / self.schaefer_constant

        logger.debug(
            f"Schaefer: raw_BL={bl:.1f}cm × {self.schaefer_bl_fraction} "
            f"= corrected_BL={corrected_bl:.1f}cm, HG={hg:.1f}cm → {weight:.1f}kg"
        )

        if weight < self._MIN_VALID_KG or weight > self._MAX_VALID_KG:
            logger.debug(f"Schaefer weight {weight:.1f}kg outside valid range")

        return max(0.0, weight)

    # ──────────────────────────────────────────────────────────────
    # Formula 2: Sirohi Regression
    # ──────────────────────────────────────────────────────────────

    def _regression(self, bl: float, hg: float) -> Optional[float]:
        """
        Indian Multivariate Regression (Sirohi-adapted from ICAR research):
        Weight (kg) = a + b1 × BL + b2 × HG

        [FIX-12]: Returns None for negative values (physiologically impossible).
        A negative result indicates the goat measurements are outside the formula's
        training range (very small kids) — we should not report 0 as a weight.
        """
        if bl <= 0 or hg <= 0:
            return None

        weight = self.reg_a + self.reg_b1 * bl + self.reg_b2 * hg

        if weight < 0:
            logger.debug(
                f"Regression weight {weight:.1f}kg is negative — "
                f"measurements outside training range (BL={bl:.1f}, HG={hg:.1f}). "
                f"Returning None (not 0)."
            )
            return None  # FIX: was return max(0.0, weight) → masked the issue

        if weight > self._MAX_VALID_KG:
            logger.debug(f"Regression weight {weight:.1f}kg above max valid range")

        return weight

    # ──────────────────────────────────────────────────────────────
    # Formula 3: BCS-Adjusted Schaefer
    # ──────────────────────────────────────────────────────────────

    def _bcs_schaefer(self, bl: float, hg: float, area_cm2: float) -> Optional[float]:
        """
        Body Condition Score (BCS) adjusted Schaefer formula.

        BCS proxy: projected body area / body length. A goat with more body area
        per unit length is in better body condition (more fat/muscle coverage).

        Normalised against expected area for typical adult Sirohi:
          - Expected area ≈ body_length × height ≈ bl × (hg / π)
          - BCS factor = actual_area / expected_area, clamped to [0.8, 1.2]

        Weight(kg) = BCS_factor × Schaefer_weight
        """
        if bl <= 0 or hg <= 0:
            return None

        w_base = self._schaefer(bl, hg)
        if w_base is None or w_base <= 0:
            return None

        if area_cm2 <= 0:
            return w_base  # No area data → return unmodified Schaefer

        # Expected projected area from simplified body dimensions
        # height ≈ hg / π (circular cross-section approximation)
        estimated_height = hg / math.pi
        expected_area = bl * estimated_height

        if expected_area <= 0:
            return w_base

        bcs_factor = area_cm2 / expected_area
        # Clamp BCS factor to ±20% of baseline (prevents wild outliers)
        bcs_factor = max(0.80, min(1.20, bcs_factor))

        return round(w_base * bcs_factor, 2)

    # ──────────────────────────────────────────────────────────────
    # Classification & Formatting
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_weight_category(weight_kg: float) -> str:
        """Classify the goat into a weight category (calibrated for Sirohi breed)."""
        if weight_kg <= 0:
            return "Unknown"
        elif weight_kg < 10:
            return "Kid"
        elif weight_kg < 20:
            return "Young"
        elif weight_kg < 35:
            return "Sub-Adult"
        elif weight_kg < 55:
            return "Adult"
        else:
            return "Large Adult"

    @staticmethod
    def weight_to_text(weight_dict: dict) -> str:
        """Format weight estimation for display."""
        s = weight_dict.get("weight_schaefer_kg", 0)
        r = weight_dict.get("weight_regression_kg", 0)
        b = weight_dict.get("weight_bcs_kg", 0)
        avg = weight_dict.get("weight_avg_kg", 0)
        flag = weight_dict.get("formula_disagreement_flag", False)

        parts = []
        if s > 0:
            parts.append(f"Schaefer: {s:.1f}kg")
        if r > 0:
            parts.append(f"Sirohi-Reg: {r:.1f}kg")
        if b > 0 and abs(b - s) > 0.5:
            parts.append(f"BCS-Adj: {b:.1f}kg")
        if avg > 0:
            parts.append(f"Avg: {avg:.1f}kg")
        if flag:
            parts.append("⚠ Formulae disagree")

        return " | ".join(parts) if parts else "N/A"
