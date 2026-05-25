"""
GOAT-AI Visualizer
===================
Rich overlay renderer for annotated video output.
Draws segmentation masks, bounding boxes, chest ellipse, stance lines,
biometric labels, tracking IDs, weight estimates, scale ruler, and mini dashboard.

Design principles:
  - Semi-transparent panel behind text (no per-char shadows → no blur)
  - Clean HUD card with colored metric rows
  - Confidence-coded mask fill (green/amber/red)
  - Anatomical girth indicator (vertical line at thorax X)
  - Scale ruler with depth-aware tick spacing
"""
import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


class Visualizer:
    """Renders rich visual overlays on video frames."""

    # Palette — BGR
    COLOR_GREEN   = (50,  210,  80)
    COLOR_AMBER   = (30,  170, 255)
    COLOR_RED     = (50,   50, 220)
    COLOR_YELLOW  = (30,  230, 230)
    COLOR_CYAN    = (220, 220,  30)
    COLOR_WHITE   = (245, 245, 245)
    COLOR_BLACK   = (  0,   0,   0)
    COLOR_ACCENT  = (255, 200,  40)   # vivid blue-white for header
    COLOR_DIM     = (160, 160, 160)

    # Panel styling
    _PANEL_BG       = (18, 18, 22)
    _PANEL_ALPHA    = 0.82            # panel opacity
    _PANEL_BORDER   = (0, 200, 255)   # cyan accent border
    _PANEL_RADIUS   = 8               # corner radius (approximated)
    _LINE_H         = 22              # pixels per text row
    _PAD_X          = 10
    _PAD_Y          = 8

    # Confidence thresholds
    _CONF_HIGH = 70.0
    _CONF_MED  = 40.0

    def __init__(
        self,
        mask_opacity: float = 0.38,
        mask_color: tuple = (50, 210, 80),
        box_color: tuple = (40, 40, 200),
        chest_color: tuple = (30, 230, 230),
        stance_color: tuple = (200, 40, 200),
        text_scale: float = 0.50,
        show_track_id: bool = True,
        show_mini_dashboard: bool = True,
        show_dual_weight: bool = True,
        confidence_color_coding: bool = True,
        show_scale_ruler: bool = True,
        show_schaefer: bool = False,
        show_bcs_adj: bool = False,
        show_sirohi_reg: bool = True,
    ):
        self.mask_opacity          = mask_opacity
        self.mask_color            = mask_color
        self.box_color             = box_color
        self.chest_color           = chest_color
        self.stance_color          = stance_color
        self.text_scale            = text_scale
        self.show_track_id         = show_track_id
        self.show_mini_dashboard   = show_mini_dashboard
        self.show_dual_weight      = show_dual_weight
        self.confidence_color_coding = confidence_color_coding
        self.show_scale_ruler      = show_scale_ruler
        self.show_schaefer         = show_schaefer
        self.show_bcs_adj          = show_bcs_adj
        self.show_sirohi_reg       = show_sirohi_reg

    # ──────────────────────────────────────────────────────────────
    # Public Entry Point
    # ──────────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        detections: list,
        all_metrics: list,
        frame_count: int = 0,
        fps_proc: float = 0.0,
        depth_image: Optional[np.ndarray] = None,
        fx: Optional[float] = None,
    ) -> np.ndarray:
        """Draw all overlays on a frame. Returns annotated copy."""
        if frame is None:
            return frame

        out = frame.copy()

        for det, metrics in zip(detections, all_metrics):
            if metrics is None:
                # Draw minimal bbox even without measurements
                if hasattr(det, 'bbox') and det.bbox is not None:
                    x1, y1, x2, y2 = det.bbox.astype(int)
                    cv2.rectangle(out, (x1, y1), (x2, y2), self.COLOR_DIM, 1)
                continue
            self._draw_detection(out, det, metrics)

        if self.show_scale_ruler and depth_image is not None and fx:
            self._draw_scale_ruler(out, depth_image, fx)

        if self.show_mini_dashboard:
            self._draw_dashboard(out, len(detections), frame_count, fps_proc)

        return out

    # ──────────────────────────────────────────────────────────────
    # Per-Detection Overlay
    # ──────────────────────────────────────────────────────────────

    def _draw_detection(self, frame: np.ndarray, detection, metrics: dict):
        h_frame, w_frame = frame.shape[:2]
        contour = detection.mask_polygon

        if contour is None or len(contour) < 3:
            return

        # ── 1. Segmentation mask (confidence-coded fill) ──
        conf = metrics.get("confidence", 50)
        if self.confidence_color_coding:
            if conf >= self._CONF_HIGH:
                fill_color = self.COLOR_GREEN
            elif conf >= self._CONF_MED:
                fill_color = self.COLOR_AMBER
            else:
                fill_color = self.COLOR_RED
        else:
            fill_color = self.mask_color

        # Soft fill with blended overlay
        mask_layer = frame.copy()
        cv2.fillPoly(mask_layer, [contour], fill_color)
        cv2.addWeighted(mask_layer, self.mask_opacity, frame, 1 - self.mask_opacity, 0, frame)

        # Clean contour edge (antialiased, slightly brighter than fill)
        edge_color = tuple(min(255, int(c * 1.3)) for c in fill_color)
        cv2.polylines(frame, [contour], isClosed=True, color=edge_color, thickness=2,
                      lineType=cv2.LINE_AA)

        # ── 2. Rotated bounding box ──
        box = metrics.get("rect_points")
        if box is not None:
            cv2.drawContours(frame, [box], 0, self.box_color, 1, cv2.LINE_AA)

        # ── 3. Chest girth: ellipse at thorax position ──
        if "chest_circle_center" in metrics:
            cc = metrics["chest_circle_center"]
            cr = int(metrics.get("chest_circle_radius", 0))
            if cr > 4:
                semi_v = cr
                semi_h = max(3, int(cr * 0.65))
                try:
                    cv2.ellipse(frame, cc, (semi_h, semi_v), 0, 0, 360,
                                self.chest_color, 2, cv2.LINE_AA)
                except Exception:
                    pass
                # Vertical marker at the thorax position
                cx_t, cy_t = cc
                cv2.line(frame,
                         (cx_t, cy_t - semi_v - 6), (cx_t, cy_t + semi_v + 6),
                         self.chest_color, 1, cv2.LINE_AA)

        # ── 4. Stance line (hoof-to-hoof) ──
        lp1 = metrics.get("leg_pt1")
        lp2 = metrics.get("leg_pt2")
        if lp1 is not None and lp2 is not None:
            p1 = (int(lp1[0]), int(lp1[1]))
            p2 = (int(lp2[0]), int(lp2[1]))
            cv2.line(frame, p1, p2, self.stance_color, 2, cv2.LINE_AA)
            cv2.circle(frame, p1, 4, self.stance_color, -1, cv2.LINE_AA)
            cv2.circle(frame, p2, 4, self.stance_color, -1, cv2.LINE_AA)
            stance = metrics.get("stance_cm", 0)
            if stance > 0:
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 + 14)
                self._text_pill(frame, f"{stance:.0f}cm", mid, self.stance_color)

        # ── 5. Info panel ──
        rows = self._build_rows(detection, metrics)
        if rows:
            anchor = self._panel_anchor(box, contour, h_frame, w_frame, len(rows))
            self._draw_info_panel(frame, rows, anchor, h_frame, w_frame)

    # ──────────────────────────────────────────────────────────────
    # Info Panel (clean card — no per-char shadows)
    # ──────────────────────────────────────────────────────────────

    def _build_rows(self, detection, metrics: dict) -> list:
        """
        Build list of (label, value, value_color) tuples for the panel.
        One row per metric — two-column layout (label left, value right).
        """
        rows = []
        conf = metrics.get("confidence", 50)

        # ── Header row ──
        tid = getattr(detection, "track_id", -1)
        if self.show_track_id and tid >= 0:
            conf_color = (
                self.COLOR_GREEN if conf >= self._CONF_HIGH
                else self.COLOR_AMBER if conf >= self._CONF_MED
                else self.COLOR_RED
            )
            rows.append(("HEADER", f"Goat  #{tid}", conf_color))
        else:
            rows.append(("HEADER", "Goat", self.COLOR_ACCENT))

        # ── Biometrics ──
        rows.append(("Length",     f"{metrics['length_cm']:.1f} cm",      self.COLOR_GREEN))
        rows.append(("Height",     f"{metrics['height_cm']:.1f} cm",      self.COLOR_AMBER))

        cg = metrics.get("chest_girth_cm", 0)
        if cg > 0:
            rows.append(("Heart Girth", f"{cg:.1f} cm  ({cg*0.3937:.1f}\")", self.COLOR_YELLOW))

        dep = metrics.get("median_depth_cm")
        if dep is not None:
            rows.append(("Depth",    f"{dep:.0f} cm",                     self.COLOR_DIM))

        # ── Weight ──
        if self.show_sirohi_reg:
            wr = metrics.get("weight_regression_kg")
            if wr and wr > 0:
                rows.append(("Weight", f"{wr:.1f} kg  (Sirohi-Reg)",      self.COLOR_WHITE))

        if self.show_schaefer:
            ws = metrics.get("weight_schaefer_kg")
            if ws and ws > 0:
                rows.append(("Schaefer", f"{ws:.1f} kg",                  (160, 160, 255)))

        if self.show_bcs_adj:
            wb = metrics.get("weight_bcs_kg")
            ws = metrics.get("weight_schaefer_kg", 0) or 0
            if wb and wb > 0 and abs(wb - ws) > 0.5:
                rows.append(("BCS-adj", f"{wb:.1f} kg",                   (160, 255, 210)))

        wavg = metrics.get("weight_avg_kg")
        if wavg and wavg > 0:
            cat = metrics.get("weight_category", "")
            lbl = f"Avg [{cat}]" if cat else "Avg"
            rows.append((lbl,        f"{wavg:.1f} kg",                    self.COLOR_ACCENT))

        # ── Flags ──
        if metrics.get("formula_disagreement_flag"):
            rows.append(("!", "Formulas disagree",                         self.COLOR_RED))

        # ── Confidence badge ──
        conf_label = (
            "High" if conf >= self._CONF_HIGH
            else "Medium" if conf >= self._CONF_MED
            else "Low"
        )
        conf_color = (
            self.COLOR_GREEN if conf >= self._CONF_HIGH
            else self.COLOR_AMBER if conf >= self._CONF_MED
            else self.COLOR_RED
        )
        rows.append(("Conf", f"{conf:.0f}%  {conf_label}",                conf_color))

        return rows

    def _panel_anchor(self, box, contour, h_frame, w_frame, n_rows):
        """Choose best anchor point for the panel — prefers top-left of detection."""
        panel_h = n_rows * self._LINE_H + self._PAD_Y * 2 + 24  # +24 for header
        panel_w = 210

        if box is not None:
            raw_x = int(np.min(box[:, 0]))
            raw_y = int(np.min(box[:, 1])) - panel_h - 6
        else:
            pts = np.array(contour)
            raw_x = int(pts[:, 0].min())
            raw_y = int(pts[:, 1].min()) - panel_h - 6

        x = max(4, min(raw_x, w_frame - panel_w - 4))
        y = max(4, min(raw_y, h_frame - panel_h - 4))
        return (x, y)

    def _draw_info_panel(self, frame, rows, anchor, h_frame, w_frame):
        """
        Draw a clean translucent card with two-column text layout.
        No per-character shadows — the dark background panel provides contrast.
        """
        ax, ay = anchor
        line_h = self._LINE_H
        pad_x  = self._PAD_X
        pad_y  = self._PAD_Y

        # ── Measure panel width from longest text ──
        max_label_w = 0
        max_value_w = 0
        for row in rows:
            kind, val, _ = row
            if kind == "HEADER":
                lw, _ = cv2.getTextSize(val, _FONT_BOLD, self.text_scale + 0.05, 1)[0]
                max_label_w = max(max_label_w, lw + pad_x)
            else:
                lw, _ = cv2.getTextSize(kind + ":", _FONT, self.text_scale - 0.03, 1)[0]
                vw, _ = cv2.getTextSize(val, _FONT, self.text_scale, 1)[0]
                max_label_w = max(max_label_w, lw)
                max_value_w = max(max_value_w, vw)

        col_gap = 8
        header_h = line_h + 6
        body_h   = (len(rows) - 1) * line_h
        panel_w  = max_label_w + col_gap + max_value_w + pad_x * 2
        panel_w  = max(panel_w, 200)
        panel_h  = header_h + body_h + pad_y * 2

        # Clamp to frame
        ax = max(4, min(ax, w_frame - panel_w - 4))
        ay = max(4, min(ay, h_frame - panel_h - 4))

        x1, y1 = ax, ay
        x2, y2 = ax + panel_w, ay + panel_h

        # ── Background panel (semi-transparent dark) ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self._PANEL_BG, -1)
        cv2.addWeighted(overlay, self._PANEL_ALPHA, frame, 1.0 - self._PANEL_ALPHA, 0, frame)

        # ── Accent border ──
        cv2.rectangle(frame, (x1, y1), (x2, y2), self._PANEL_BORDER, 1, cv2.LINE_AA)

        # ── Top accent bar ──
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 3), self._PANEL_BORDER, -1)

        # ── Draw rows ──
        text_x = ax + pad_x
        val_x  = ax + pad_x + max_label_w + col_gap
        cur_y  = ay + pad_y

        for idx, (kind, val, color) in enumerate(rows):
            row_y = cur_y + line_h

            if kind == "HEADER":
                # Full-width header with slightly larger font
                row_y = cur_y + line_h + 2
                cv2.putText(frame, val, (text_x, row_y),
                            _FONT_BOLD, self.text_scale + 0.05,
                            color, 1, cv2.LINE_AA)
                # Thin separator below header
                sep_y = row_y + 5
                cv2.line(frame, (x1 + 2, sep_y), (x2 - 2, sep_y),
                         (60, 60, 70), 1)
                cur_y += line_h + 8
            else:
                # Label (dim) + value (colored)
                label_str = kind + ":"
                cv2.putText(frame, label_str, (text_x, row_y),
                            _FONT, self.text_scale - 0.03,
                            self.COLOR_DIM, 1, cv2.LINE_AA)
                cv2.putText(frame, val, (val_x, row_y),
                            _FONT, self.text_scale,
                            color, 1, cv2.LINE_AA)
                cur_y += line_h

    # ──────────────────────────────────────────────────────────────
    # Small pill label (used for stance midpoint)
    # ──────────────────────────────────────────────────────────────

    def _text_pill(self, frame, text, center, color):
        """Draw a small pill-shaped label at center point."""
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.38, 1)
        px, py = int(center[0] - tw // 2), int(center[1])
        pad = 3
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (px - pad, py - th - pad),
                      (px + tw + pad, py + pad),
                      self._PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, text, (px, py), _FONT, 0.38, color, 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────
    # Scale Ruler
    # ──────────────────────────────────────────────────────────────

    def _draw_scale_ruler(
        self, frame: np.ndarray, depth_image: np.ndarray, fx: float
    ):
        """Draw a depth-calibrated scale bar in the bottom-left corner."""
        h, w = frame.shape[:2]

        # Sample median depth from frame centre
        strip = depth_image[h // 3: 2 * h // 3, w // 4: 3 * w // 4]
        valid = strip[strip > 0]
        if len(valid) < 20 or fx <= 0:
            return

        depth_cm = float(np.median(valid)) / 10.0   # Orbbec mm → cm
        if depth_cm <= 0:
            return

        px_per_10cm = int((10.0 * fx) / depth_cm)
        if px_per_10cm <= 5 or px_per_10cm > w // 3:
            return

        ticks = 5
        ruler_len = min(ticks * px_per_10cm, w - 50)
        rx, ry = 16, h - 28

        # Background pill
        ovl = frame.copy()
        cv2.rectangle(ovl, (rx - 6, ry - 16), (rx + ruler_len + 6, ry + 14),
                      self._PANEL_BG, -1)
        cv2.addWeighted(ovl, 0.72, frame, 0.28, 0, frame)

        # Main line
        cv2.line(frame, (rx, ry), (rx + ruler_len, ry), self.COLOR_WHITE, 2, cv2.LINE_AA)

        # Ticks
        for t in range(ticks + 1):
            tx = min(rx + t * px_per_10cm, rx + ruler_len)
            th = 7 if t % 5 == 0 else 4
            cv2.line(frame, (tx, ry - th), (tx, ry + th), self.COLOR_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{t*10}", (tx - 5, ry + 13),
                        _FONT, 0.32, self.COLOR_DIM, 1, cv2.LINE_AA)

        # Unit label
        cv2.putText(frame, f"cm   d={depth_cm:.0f}cm",
                    (rx, ry - 18), _FONT, 0.36, (160, 210, 255), 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────
    # Dashboard Bar
    # ──────────────────────────────────────────────────────────────

    def _draw_dashboard(
        self, frame: np.ndarray, num_animals: int, frame_count: int, fps: float
    ):
        """Top bar: model name, animal count, frame, FPS."""
        h, w = frame.shape[:2]
        bar_h = 32

        ovl = frame.copy()
        cv2.rectangle(ovl, (0, 0), (w, bar_h), (12, 12, 16), -1)
        cv2.addWeighted(ovl, 0.80, frame, 0.20, 0, frame)

        # Bottom accent
        cv2.line(frame, (0, bar_h), (w, bar_h), self._PANEL_BORDER, 1)

        items = [
            ("GOAT-AI",               (0, 200, 255)),
            (f"| Animals: {num_animals}", self.COLOR_GREEN if num_animals > 0 else self.COLOR_DIM),
            (f"| Frame {frame_count}", self.COLOR_DIM),
            (f"| {fps:.1f} FPS",      self.COLOR_YELLOW),
        ]

        x = 10
        for text, color in items:
            (tw, _), _ = cv2.getTextSize(text, _FONT_BOLD, 0.48, 1)
            cv2.putText(frame, text, (x, 22), _FONT_BOLD, 0.48, color, 1, cv2.LINE_AA)
            x += tw + 10
