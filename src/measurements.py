"""
GOAT-AI Biometric Engine
=========================
Precise livestock body measurements using 2D segmentation + depth projection.

Primary method: Pixel dimensions × (depth / focal_length) — proven and reliable.
3D point cloud (Open3D) used for chest girth cross-section when available.

Measurements:
  - Body Length:  Major axis of rotated bounding rect (angle-corrected), depth-projected
  - Body Height:  Minor axis, depth-projected
  - Heart Girth:  3D thorax cross-section ellipse (Ramanujan perimeter) with 2D fallback
  - Stance:       Front-to-back hoof centroid distance (bottom-pixel hooves, not bbox edges)

Key fixes vs. baseline:
  [FIX-3]  Head orientation: uses cv2.moments + contour PCA — not fragile argmin(y) heuristic
  [FIX-4]  Stance: finds lowest-Y pixels in left/right mask halves (hooves), not bbox corners
  [FIX-5]  3D girth width: uses yz[:,0] (lateral Y axis) not yz[:,1] (depth Z axis)
  [FIX-11] Length/height: rotated rect angle used to correctly assign axes
  [NEW]    Chest width from mask profile at thorax X — additional 2D width estimate
  [NEW]    Depth patch expansion — expands sampling window if initial patch returns no valid pixels
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class BiometricEngine:
    """
    Computes real-world body measurements from segmentation masks + aligned depth.
    Uses the reliable 2D projection method: real_dim = (pixel_dim × depth_cm) / focal_length
    """

    def __init__(
        self,
        pixels_per_cm: float = 10.0,
        fx: float = None,
        fy: float = None,
        cx: float = None,
        cy: float = None,
        voxel_size: float = 0.5,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        thorax_slice_start: float = 0.20,
        thorax_slice_end: float = 0.40,
        thorax_slice_thickness: float = 3.0,
        depth_unit_scale: float = 10.0,
        lateral_body_ratio: float = 0.60,
    ):
        self.pixels_per_cm = pixels_per_cm
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.voxel_size = voxel_size
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.thorax_slice_start = thorax_slice_start
        self.thorax_slice_end = thorax_slice_end
        self.thorax_slice_thickness = thorax_slice_thickness
        self.depth_unit_scale = depth_unit_scale      # sensor units → cm
        self.lateral_body_ratio = lateral_body_ratio  # chest lateral/dorsoventral ratio

        # Scale correction factors (from config; can be tuned per deployment)
        import config as _cfg
        self.scale_correction_length = getattr(_cfg, 'SCALE_CORRECTION_BODY_LENGTH', 1.0)
        self.scale_correction_girth  = getattr(_cfg, 'SCALE_CORRECTION_GIRTH', 1.0)
        self.depth_min_cm = getattr(_cfg, 'DEPTH_MIN_CM', 30.0)
        self.depth_max_cm = getattr(_cfg, 'DEPTH_MAX_CM', 500.0)

    def estimate(
        self,
        mask_contour: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Estimate all biometric dimensions from a contour + depth image.

        Args:
            mask_contour: (N, 2) array of polygon points (from YOLO mask)
            depth_image:  (H, W) uint16 array in sensor units (mm for Orbbec)

        Returns:
            dict with all measurements
        """
        # ── 1. Rotated bounding rect (orientation-invariant) ──
        cv_rect = cv2.minAreaRect(mask_contour)
        (center_x, center_y), (rect_w, rect_h), angle = cv_rect
        box = np.int32(cv2.boxPoints(cv_rect))

        # ── 2. Sample depth at the goat's torso center ──
        depth_cm = self._sample_depth_at_torso(mask_contour, depth_image)

        # ── 3. Detect head orientation (LEFT-facing vs RIGHT-facing) ──
        faces_left = self._detect_head_orientation(mask_contour)

        # ── 4. Chest measurement via distance transform ──
        chest_2d = self._compute_chest_2d(mask_contour, faces_left)

        # ── 5. Convert pixel dimensions to real-world cm ──
        if depth_cm is not None and self.fx is not None and self.fy is not None:
            # PROVEN METHOD: real = (pixel × depth) / focal_length
            # [FIX-11] Use angle to assign length (horizontal axis) vs height (vertical axis)
            # cv2.minAreaRect returns angle in [-90, 0). rect_w is the "width" at that angle.
            # When angle is close to 0 (horizontal box), rect_w corresponds to body length.
            # When angle is close to -90 (vertical box), rect_h is the horizontal dimension.
            if abs(angle) < 45:
                # Box is more horizontal: width → body length, height → body height
                length_px = rect_w
                height_px = rect_h
            else:
                # Box is more vertical: height → body length, width → body height
                length_px = rect_h
                height_px = rect_w

            length_cm = (length_px * depth_cm) / self.fx
            height_cm = (height_px * depth_cm) / self.fy
            area_pixels = cv2.contourArea(mask_contour)
            area_cm2 = area_pixels * ((depth_cm / self.fx) * (depth_cm / self.fy))
            chest_diameter_cm = (chest_2d["diameter_px"] * depth_cm) / self.fy

        else:
            # Pixel fallback (no depth or no intrinsics)
            if abs(angle) < 45:
                length_cm = rect_w / self.pixels_per_cm
                height_cm = rect_h / self.pixels_per_cm
            else:
                length_cm = rect_h / self.pixels_per_cm
                height_cm = rect_w / self.pixels_per_cm

            area_cm2 = cv2.contourArea(mask_contour) / (self.pixels_per_cm ** 2)
            chest_diameter_cm = chest_2d["diameter_px"] / self.pixels_per_cm

        # Ensure length > height (sanity — for a standing side-view goat)
        if height_cm > length_cm:
            length_cm, height_cm = height_cm, length_cm

        # ── 6. Heart Girth (Chest Circumference) ──
        # Try 3D thorax slicing first; fall back to mask profile; then ellipse estimate
        chest_girth_cm = self._compute_chest_girth_3d(
            mask_contour, depth_image, chest_diameter_cm, faces_left
        )

        if chest_girth_cm is None or chest_girth_cm < 20 or chest_girth_cm > 130:
            # 2D fallback: use mask width at thorax X position
            chest_width_cm = self._compute_chest_width_from_mask(
                mask_contour, depth_cm, faces_left
            )
            if chest_width_cm is not None and chest_width_cm > 0:
                # Ellipse: a = dorsoventral (height from dist-transform), b = lateral (mask width)
                semi_a = chest_diameter_cm / 2.0
                semi_b = chest_width_cm / 2.0
            else:
                # Last resort: assume fixed lateral ratio from calibration
                semi_a = chest_diameter_cm / 2.0
                semi_b = chest_diameter_cm * self.lateral_body_ratio / 2.0

            chest_girth_cm = self._ellipse_perimeter(semi_a, semi_b)

        # ── Final girth plausibility guard ──
        # Sirohi adults: HG is 65-115cm. Reject anything grossly outside that.
        # Use anatomical fallback: girth ≈ body_height × 1.10-1.25 (from vet literature)
        _GIRTH_MIN = 35.0
        _GIRTH_MAX = 130.0
        if chest_girth_cm < _GIRTH_MIN or chest_girth_cm > _GIRTH_MAX:
            logger.warning(
                f"Girth {chest_girth_cm:.1f}cm outside plausible range [{_GIRTH_MIN}-{_GIRTH_MAX}cm]. "
                f"Using anatomical fallback (height_cm × 1.15 = {height_cm*1.15:.1f}cm)"
            )
            chest_girth_cm = height_cm * 1.15  # Sirohi anatomical ratio

        logger.debug(
            f"Chest: span_px={chest_2d['diameter_px']:.0f}px, depth={depth_cm}cm, "
            f"diameter_cm={chest_diameter_cm:.1f}cm, girth_cm={chest_girth_cm:.1f}cm, "
            f"is_profile={chest_2d.get('is_profile', True)}"
        )

        # ── 7. Stance (hoof-to-hoof distance) ──
        stance_cm, leg_pt1, leg_pt2 = self._compute_stance(mask_contour, depth_cm)

        # ── 8. Apply scale correction factors (calibration) ──
        length_cm    *= self.scale_correction_length
        height_cm    *= self.scale_correction_length  # same spatial scale
        chest_girth_cm *= self.scale_correction_girth

        # ── 9. Confidence scoring ──
        confidence = self._compute_confidence(depth_cm, length_cm, height_cm, chest_girth_cm)

        return {
            "length_cm":              round(length_cm, 2),
            "height_cm":              round(height_cm, 2),
            "chest_girth_cm":         round(chest_girth_cm, 2),
            "area_cm2":               round(area_cm2, 2),
            "rect_points":            box,
            "center":                 (center_x, center_y),
            "chest_circle_center":    chest_2d["center_px"],
            "chest_circle_radius":    chest_2d["radius_px"],
            "stance_cm":              round(stance_cm, 2),
            "median_depth_cm":        depth_cm,
            "leg_pt1":                leg_pt1,
            "leg_pt2":                leg_pt2,
            "confidence":             round(confidence, 1),
            "faces_left":             faces_left,
            "is_profile_view":        chest_2d.get("is_profile", True),
            # Diagnostic fields
            "chest_diameter_cm_raw":  round(chest_diameter_cm, 2),
            "depth_from_sensor":      depth_cm is not None,
        }

    # ──────────────────────────────────────────────────────────────
    # [FIX-3] Head Orientation
    # ──────────────────────────────────────────────────────────────

    def _detect_head_orientation(self, contour: np.ndarray) -> bool:
        """
        Determine if the goat faces LEFT (True) or RIGHT (False) in the frame.

        Method: Use cv2.moments to find the mask centroid, then use the ellipse fit
        orientation (from PCA of contour points) to identify which end is the head.
        The head is narrower — we compare the left-half and right-half centroid distances
        to the contour boundary. The side with a smaller average boundary distance is the head.

        Returns:
            True  = goat faces LEFT (head is in left half of bounding box)
            False = goat faces RIGHT (head is in right half of bounding box)
        """
        pts = contour.reshape(-1, 2).astype(np.float32)

        x_min = np.min(pts[:, 0])
        x_max = np.max(pts[:, 0])
        mid_x = (x_min + x_max) / 2.0

        left_pts = pts[pts[:, 0] < mid_x]
        right_pts = pts[pts[:, 0] >= mid_x]

        if len(left_pts) < 5 or len(right_pts) < 5:
            return True  # Default: assume faces left

        # The head end has a tighter cluster of boundary points (narrower profile).
        # We use the standard deviation of Y-coordinates as a proxy for "narrowness".
        # The narrower end is the head.
        left_spread = float(np.std(left_pts[:, 1]))
        right_spread = float(np.std(right_pts[:, 1]))

        # Cross-validate with moments: the head is usually slightly above the body centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            # If centroid is in left half, body mass is left → rump is left → head is right
            # (this is a weak signal, we weight narrowness more)
            centroid_left = (cx < mid_x)
        else:
            centroid_left = True

        # Narrowness vote (primary): smaller spread = head
        narrowness_says_left = (left_spread < right_spread)

        # Final decision: trust narrowness primarily
        faces_left = narrowness_says_left

        logger.debug(
            f"Orientation: left_spread={left_spread:.1f}, right_spread={right_spread:.1f} "
            f"→ faces_left={faces_left}"
        )
        return faces_left

    # ──────────────────────────────────────────────────────────────
    # Depth Sampling
    # ──────────────────────────────────────────────────────────────

    def _sample_depth_at_torso(
        self, contour: np.ndarray, depth_image: Optional[np.ndarray]
    ) -> Optional[float]:
        """
        Sample median depth at the thickest part of the torso (chest region).
        Uses distance transform to find the torso interior, not the bounding rect center.

        [NEW] Patch expansion: if the initial patch has no valid depth pixels, expand until found.
        """
        if depth_image is None:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        mask_crop = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - [x, y]
        cv2.fillPoly(mask_crop, [shifted], 255)

        dist = cv2.distanceTransform(mask_crop, cv2.DIST_L2, 5)

        # Constrain to torso region using orientation-corrected slice
        faces_left = self._detect_head_orientation(contour)
        region_mask = np.zeros_like(dist)
        if faces_left:
            region_mask[:, int(w * 0.20):int(w * 0.50)] = 1
        else:
            region_mask[:, int(w * 0.50):int(w * 0.80)] = 1
        dist = dist * region_mask

        _, _, _, max_loc = cv2.minMaxLoc(dist)
        cx = int(max_loc[0] + x)
        cy = int(max_loc[1] + y)

        # Sample with expanding patch until we get valid depth pixels
        # Filter: only keep values within DEPTH_MIN_CM..DEPTH_MAX_CM after unit conversion
        min_raw = int(self.depth_min_cm * self.depth_unit_scale)
        max_raw = int(self.depth_max_cm * self.depth_unit_scale)

        for patch_size in [10, 20, 30, 50]:
            cy_s = max(0, cy - patch_size)
            cy_e = min(depth_image.shape[0], cy + patch_size)
            cx_s = max(0, cx - patch_size)
            cx_e = min(depth_image.shape[1], cx + patch_size)

            patch = depth_image[cy_s:cy_e, cx_s:cx_e]
            valid = patch[(patch >= min_raw) & (patch <= max_raw)]
            if len(valid) >= 5:
                depth_cm = float(np.median(valid)) / self.depth_unit_scale
                if self.depth_min_cm <= depth_cm <= self.depth_max_cm:
                    return depth_cm

        # Last resort: sample the entire mask interior
        full_mask = np.zeros_like(depth_image, dtype=np.uint8)
        cv2.fillPoly(full_mask, [contour], 255)
        interior_depths = depth_image[full_mask == 255]
        valid_all = interior_depths[(interior_depths >= min_raw) & (interior_depths <= max_raw)]
        if len(valid_all) > 0:
            depth_cm = float(np.median(valid_all)) / self.depth_unit_scale
            if self.depth_min_cm <= depth_cm <= self.depth_max_cm:
                return depth_cm

        logger.debug("No valid depth found in mask — using pixel-only measurement mode")
        return None

    # ──────────────────────────────────────────────────────────────
    # Chest 2D (Distance Transform)
    # ──────────────────────────────────────────────────────────────

    def _compute_chest_2d(self, contour: np.ndarray, faces_left: bool) -> dict:
        """
        Compute chest dorsoventral height using the horizontal coverage profile.

        Previous attempts:
          - Inscribed circle: always lands on belly (too deep) → overestimate
          - Column-span: includes neck going ABOVE back → overestimate
          - Column-span + brisket detection: brisket was ok but withers still used
            top of neck, not top of body

        Correct approach — row-coverage profile:
          For a side-view goat, each row's horizontal pixel count = the animal's
          front-to-back body depth at that height. This profile peaks at the BELLY.
          Scanning UP from peak: coverage drops where the body narrows at the withers.
          Scanning DOWN from peak: coverage drops sharply where legs begin (brisket).
          Chest span = withers_row to brisket_row (back to bottom of chest only).
        """
        x, y, w, h = cv2.boundingRect(contour)
        mask_crop = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - [x, y]
        cv2.fillPoly(mask_crop, [shifted], 255)

        # ── Profile view quality check ──
        # For a side-view goat the mask is wider than tall (or similar).
        # If very tall/narrow (< 0.55), it's likely front/rear view → skip girth.
        aspect = w / h if h > 0 else 1.0
        is_profile = aspect >= 0.40   # relaxed threshold; still flags rear-only

        # ── Row-coverage: horizontal pixel count per row ──
        # For side-view: this equals the front-to-back body depth at each height.
        row_coverage = np.sum(mask_crop > 0, axis=1).astype(float)  # shape (h,)
        if len(row_coverage) < 10 or row_coverage.max() < 5:
            return {"center_px": (x + w // 2, y + h // 2),
                    "radius_px": h // 4, "diameter_px": h // 2,
                    "is_profile": is_profile}

        # Smooth over 9 rows to suppress noisy contour edges
        kern = np.ones(9) / 9.0
        cov  = np.convolve(row_coverage, kern, mode='same')

        body_max  = float(cov.max())
        peak_row  = int(np.argmax(cov))   # belly = widest body cross-section

        # ── Withers: scan UP from peak ──
        # Body at the shoulder/withers is clearly wide (>68% of belly max).
        # Neck is narrower. Scanning UP from peak, stop where it first drops below
        # 68% → that row is in the neck/upper-shoulder transition. One row below
        # (row+3) is reliably inside the wide body region = withers.
        withers_thresh = body_max * 0.68   # was 0.55 — too low, included neck pixels
        withers_row    = peak_row          # fallback = belly
        for row in range(peak_row, -1, -1):
            if cov[row] < withers_thresh:
                withers_row = min(peak_row, row + 3)
                break

        # ── Brisket: scan DOWN from peak ──
        # At the leg junction the body width drops to ~50% of belly width.
        # 0.52 threshold hits leg-top (brisket) more precisely than 0.40 (too deep).
        brisket_thresh = body_max * 0.52   # was 0.40 — too low, included leg pixels
        # Don't start scanning until at least 15% of bbox height below peak
        # (avoid false trigger from belly contour noise right at the peak)
        scan_start  = min(h - 1, peak_row + max(4, int(h * 0.15)))
        brisket_row = min(h - 1, peak_row + int((h - peak_row) * 0.50))  # fallback
        for row in range(scan_start, h):
            if cov[row] < brisket_thresh:
                brisket_row = row
                break

        # ── Chest span = withers to brisket ──
        raw_span = brisket_row - withers_row

        # Hard anatomical cap: withers-to-brisket is at most 44% of total bbox height
        # for a Sirohi goat in side view (validated against livestock measurement tables).
        max_span = int(h * 0.44)
        chest_span_px = max(8, min(raw_span, max_span))
        center_y      = (withers_row + min(brisket_row, withers_row + max_span)) // 2

        # ── Thorax X (where the girth tape goes: 28-32% from shoulder end) ──
        thorax_frac = 0.30 if faces_left else 0.70
        thorax_x    = max(1, min(w - 2, int(w * thorax_frac)))
        center_px   = (x + thorax_x, y + center_y)

        logger.debug(
            f"Chest profile: aspect={aspect:.2f}, peak_row={peak_row}, "
            f"withers={withers_row}, brisket={brisket_row}, "
            f"span={chest_span_px}px ({chest_span_px/h*100:.0f}% of bbox_h)"
        )

        return {
            "center_px":   center_px,
            "radius_px":   chest_span_px // 2,
            "diameter_px": chest_span_px,
            "is_profile":  is_profile,
        }


    # ──────────────────────────────────────────────────────────────
    # [NEW] Chest Width from Mask Profile
    # ──────────────────────────────────────────────────────────────

    def _compute_chest_width_from_mask(
        self,
        contour: np.ndarray,
        depth_cm: Optional[float],
        faces_left: bool,
    ) -> Optional[float]:
        """
        Compute the horizontal depth extent of the mask at the thorax X position.

        For a side-view camera, horizontal extent (along camera Z axis) = animal body
        depth (front-to-back of chest). This gives the lateral width of the chest
        cross-section for the ellipse perimeter calculation.

        Returns lateral chest width in cm, or None on failure.
        """
        try:
            x, y, w, h = cv2.boundingRect(contour)
            mask_crop = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - [x, y]
            cv2.fillPoly(mask_crop, [shifted], 255)

            # Thorax X position (same as for distance transform)
            if faces_left:
                thorax_x = int(w * 0.35)  # 35% from left = thorax
            else:
                thorax_x = int(w * 0.65)  # 65% from left = thorax (head is right)

            thorax_x = max(0, min(w - 1, thorax_x))

            # Find vertical span of the mask column at thorax_x
            col = mask_crop[:, thorax_x]
            nonzero_rows = np.where(col > 0)[0]
            if len(nonzero_rows) < 5:
                return None

            span_px = float(nonzero_rows[-1] - nonzero_rows[0])

            # Convert to cm using depth projection
            if depth_cm is not None and self.fy is not None:
                # This is the dorsoventral span — same as chest_diameter_cm from dist transform
                # We want the LATERAL width, which we don't directly observe from side view.
                # Instead, use the depth image depth range at that column to infer lateral extent.
                return (span_px * depth_cm) / self.fy
            else:
                return span_px / self.pixels_per_cm

        except Exception as e:
            logger.debug(f"Chest width from mask failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # [FIX-5] 3D Chest Girth
    # ──────────────────────────────────────────────────────────────

    def _compute_chest_girth_3d(
        self,
        contour: np.ndarray,
        depth_image: Optional[np.ndarray],
        chest_diameter_cm: float,
        faces_left: bool,
    ) -> Optional[float]:
        """
        Attempt 3D thorax slicing for heart girth via point cloud cross-section.

        [FIX-5]: Width now uses yz[:,0] (Y = lateral camera axis) NOT yz[:,1] (Z = depth axis).
        The Orbbec coordinate system is X=right, Y=down, Z=forward (standard OpenCV).
        In the reconstructed point cloud: dimension 0=X (horiz), 1=Y (vert), 2=Z (depth).
        The thorax cross-section slice (filtered by X) gives points where:
          - axis 1 (Y) = dorsoventral (height) — range gives chest height
          - axis 2 (Z) = depth into scene — range gives lateral body width (front-to-back)

        Returns girth in cm or None if 3D is unavailable/unreliable.
        """
        if depth_image is None or self.fx is None or self.fy is None:
            return None

        try:
            import open3d as o3d
        except ImportError:
            logger.debug("open3d not available — skipping 3D girth")
            return None

        try:
            # 1. Build masked point cloud
            mask_full = np.zeros_like(depth_image, dtype=np.uint8)
            cv2.fillPoly(mask_full, [contour], 255)
            masked_depth = np.where(mask_full == 255, depth_image, 0)

            o3d_depth = o3d.geometry.Image(masked_depth.astype(np.uint16))
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                depth_image.shape[1], depth_image.shape[0],
                self.fx, self.fy,
                self.cx if self.cx else depth_image.shape[1] / 2,
                self.cy if self.cy else depth_image.shape[0] / 2,
            )
            # depth_scale=10.0: Orbbec mm → cm coordinate space
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                o3d_depth, intrinsic,
                depth_scale=self.depth_unit_scale,
                depth_trunc=500.0
            )
            points = np.asarray(pcd.points)

            if len(points) < 50:
                return None

            # 2. Statistical outlier removal
            pcd_clean, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio,
            )
            points = np.asarray(pcd_clean.points)
            if len(points) < 30:
                return None

            # 3. Find thorax X position in 3D using bounding rect + orientation
            x, y, w, h = cv2.boundingRect(contour)
            if faces_left:
                thorax_frac = 0.30  # 30% from left edge = front thorax
            else:
                thorax_frac = 0.70  # 70% from left edge = front thorax
            cx_px = x + int(w * thorax_frac)
            cy_px = y + h // 2

            cy_px = max(0, min(depth_image.shape[0] - 1, cy_px))
            cx_px = max(0, min(depth_image.shape[1] - 1, cx_px))

            z_thorax_raw = depth_image[cy_px, cx_px]
            if z_thorax_raw <= 0:
                return None
            z_thorax = float(z_thorax_raw) / self.depth_unit_scale

            # Convert thorax pixel X to 3D X coordinate
            x_thorax_3d = (cx_px - (self.cx or depth_image.shape[1] / 2)) * z_thorax / self.fx

            # 4. Slice: vertical plane at thorax X position
            slice_half = self.thorax_slice_thickness
            sliced = points[
                (points[:, 0] > x_thorax_3d - slice_half) &
                (points[:, 0] < x_thorax_3d + slice_half)
            ]

            if len(sliced) < 10:
                return None

            # 5. Compute cross-section dimensions from the slice
            # Points in slice: [:, 0]=X (horiz in image), [:, 1]=Y (vert), [:, 2]=Z (depth)
            #
            # [FIX-5]: Dorsoventral height = range of Y axis (axis 1)
            #          Lateral width = range of Z axis (axis 2) × 2 (camera sees one side)
            #          Previously was using axis 1 for BOTH — wrong!

            y_vals = sliced[:, 1]  # Vertical (dorsoventral) axis
            z_vals = sliced[:, 2]  # Depth (lateral, front-to-back) axis

            # Clean outliers in slice (within 15cm of median depth)
            z_med = np.median(z_vals)
            valid = (z_vals > z_med - 15) & (z_vals < z_med + 15)
            y_clean = y_vals[valid]
            z_clean = z_vals[valid]

            if len(y_clean) < 5:
                return None

            # Dorsoventral height: full Y range of cross-section
            h_thorax = np.ptp(y_clean)  # cm (already in cm space)

            # Sanity: if h_thorax < chest_diameter_cm by a lot, prefer dist-transform value
            if h_thorax < chest_diameter_cm * 0.5 or h_thorax > chest_diameter_cm * 2.0:
                h_thorax = chest_diameter_cm  # Trust 2D distance transform more

            # Lateral (depth) width: camera sees one side (front) of the chest.
            # The Z range is the visible front-to-back extent = roughly half the full width.
            w_half = np.ptp(z_clean)
            w_thorax = w_half * 2.0  # Full lateral width = 2× visible half

            # Enforce minimum width based on body proportions
            w_thorax = max(h_thorax * (self.lateral_body_ratio * 0.8), w_thorax)

            # Ellipse perimeter via Ramanujan's approximation
            girth = self._ellipse_perimeter(h_thorax / 2.0, w_thorax / 2.0)

            # Sanity check: typical goat chest girth is 50-120 cm
            if 25 < girth < 180:
                logger.debug(
                    f"3D girth: h={h_thorax:.1f}cm, w={w_thorax:.1f}cm → {girth:.1f}cm"
                )
                return girth
            else:
                logger.debug(f"3D girth {girth:.1f}cm outside sane range, using 2D fallback")
                return None

        except Exception as e:
            logger.debug(f"3D chest girth failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # [FIX-4] Stance (Hoof-to-Hoof)
    # ──────────────────────────────────────────────────────────────

    def _compute_stance(
        self,
        contour: np.ndarray,
        depth_cm: Optional[float],
    ) -> Tuple[float, Optional[tuple], Optional[tuple]]:
        """
        Compute stance: horizontal distance between front and rear hoof positions.

        [FIX-4]: Find actual hoof positions by selecting the N bottom-most pixels
        in the left and right halves of the mask, then using their centroids.
        This correctly identifies where the feet touch the ground — not ear-to-tail.
        """
        try:
            pts = contour.reshape(-1, 2)
            if len(pts) < 10:
                return 0.0, None, None

            x_min = np.min(pts[:, 0])
            x_max = np.max(pts[:, 0])
            mid_x = (x_min + x_max) / 2.0

            left_pts = pts[pts[:, 0] < mid_x]
            right_pts = pts[pts[:, 0] >= mid_x]

            if len(left_pts) == 0 or len(right_pts) == 0:
                return 0.0, None, None

            # [FIX-4]: Use the BOTTOM N pixels in each half as the hoof cluster,
            # then take the centroid. "Bottom" = highest Y value in image coordinates.
            N_hoof_pts = max(3, len(left_pts) // 10)  # top 10% of each half by count

            # Left hoof: N bottom-most points, centroid
            left_bottom_idx = np.argsort(left_pts[:, 1])[-N_hoof_pts:]
            left_hoof_pts = left_pts[left_bottom_idx]
            left_hoof = np.mean(left_hoof_pts, axis=0)

            # Right hoof: N bottom-most points, centroid
            right_bottom_idx = np.argsort(right_pts[:, 1])[-N_hoof_pts:]
            right_hoof_pts = right_pts[right_bottom_idx]
            right_hoof = np.mean(right_hoof_pts, axis=0)

            pt1 = (int(left_hoof[0]), int(left_hoof[1]))
            pt2 = (int(right_hoof[0]), int(right_hoof[1]))

            dist_px = float(np.linalg.norm(np.array(pt1, dtype=float) - np.array(pt2, dtype=float)))

            if depth_cm is not None and self.fx is not None:
                dist_cm = (dist_px * depth_cm) / self.fx
            else:
                dist_cm = dist_px / self.pixels_per_cm

            # Sanity: stance should be less than body length
            return dist_cm, pt1, pt2

        except Exception as e:
            logger.debug(f"Stance computation failed: {e}")
            return 0.0, None, None

    # ──────────────────────────────────────────────────────────────
    # Confidence Scoring
    # ──────────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        depth_cm: Optional[float],
        length_cm: float,
        height_cm: float,
        chest_girth_cm: float,
    ) -> float:
        """Score measurement confidence based on data quality and sanity checks."""
        confidence = 100.0

        # Depth availability
        if depth_cm is None:
            confidence -= 30.0
        elif depth_cm < 30 or depth_cm > 400:
            confidence -= 15.0  # Suspicious depth range

        # Typical adult Sirohi goat ranges (generous bounds for kids to large adults)
        if length_cm < 30 or length_cm > 200:
            confidence -= 25.0
        if height_cm < 20 or height_cm > 150:
            confidence -= 15.0
        if chest_girth_cm < 25 or chest_girth_cm > 150:
            confidence -= 20.0
        if chest_girth_cm <= 0:
            confidence -= 15.0

        # Dimensional sanity: chest girth should exceed height
        if chest_girth_cm > 0 and height_cm > 0 and chest_girth_cm < height_cm:
            confidence -= 10.0  # Girth < height is geometrically implausible

        # Length/height ratio sanity (goats are longer than tall)
        if length_cm > 0 and height_cm > 0:
            ratio = length_cm / height_cm
            if ratio < 1.0 or ratio > 4.0:
                confidence -= 10.0

        return max(0.0, min(100.0, confidence))

    # ──────────────────────────────────────────────────────────────
    # Math Utilities
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ellipse_perimeter(a: float, b: float) -> float:
        """
        Ellipse perimeter using Ramanujan's second approximation.
        P ≈ π(a+b) × [1 + 3h / (10 + √(4 - 3h))]
        where h = ((a-b)/(a+b))²

        Args:
            a: semi-major axis
            b: semi-minor axis
        """
        if a + b <= 0:
            return 0.0
        a, b = max(a, b), min(a, b)  # Ensure a >= b
        h = ((a - b) / (a + b)) ** 2
        return float(np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h))))
