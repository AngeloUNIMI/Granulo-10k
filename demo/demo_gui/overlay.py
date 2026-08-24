from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class StrandGeometry:
    center: np.ndarray
    major_start: np.ndarray
    major_end: np.ndarray
    minor_start: np.ndarray
    minor_end: np.ndarray
    contour: np.ndarray


# -------------------------------------------------------------------------
# Foreground / strand localization
# -------------------------------------------------------------------------

def _estimate_background_color(rgb: np.ndarray) -> np.ndarray:
    """Estimate the background RGB color from the image borders."""
    height, width, _ = rgb.shape

    border = max(
        4,
        min(height, width) // 40,
    )

    border_pixels = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    return np.median(
        border_pixels,
        axis=0,
    )


def _foreground_mask(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Estimate a rough foreground mask by measuring color distance from
    the border background.

    This mask is used only for visualization and arrow placement.
    """
    background = _estimate_background_color(
        rgb
    )

    distance = np.linalg.norm(
        rgb.astype(np.float32)
        - background.astype(np.float32),
        axis=2,
    )

    distance_u8 = cv2.normalize(
        distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    _, mask = cv2.threshold(
        distance_u8,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    kernel_size = max(
        3,
        min(rgb.shape[:2]) // 150,
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return mask


def _prepare_mask(mask_image: Image.Image) -> np.ndarray:
    """Convert a dataset segmentation mask to a clean binary mask."""
    mask = np.asarray(mask_image.convert("L"))

    _, binary = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Ensure the strand is foreground.
    foreground_fraction = np.count_nonzero(binary) / binary.size
    if foreground_fraction > 0.5:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    return binary


def find_strand_geometry(
    image: Image.Image,
    mask_image: Image.Image | None = None,
) -> StrandGeometry | None:
    """
    Estimate the strand major and minor axes.

    A provided segmentation mask is used directly. If no mask is available,
    RGB foreground extraction is used as a fallback.
    """
    rgb = np.asarray(image.convert("RGB"))

    if mask_image is not None:
        mask = _prepare_mask(mask_image)

        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(
                mask,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    else:
        mask = _foreground_mask(rgb)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    image_area = rgb.shape[0] * rgb.shape[1]

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > image_area * 0.0002
    ]

    if not valid_contours:
        return None

    contour = max(
        valid_contours,
        key=cv2.contourArea,
    )

    points = contour[:, 0, :].astype(np.float32)

    if len(points) < 5:
        return None

    center = points.mean(axis=0)
    centered = points - center

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]

    major_axis = eigenvectors[:, order[0]]
    minor_axis = eigenvectors[:, order[1]]

    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis

    major_start = center + major_axis * major_projection.min()
    major_end = center + major_axis * major_projection.max()

    minor_start = center + minor_axis * minor_projection.min()
    minor_end = center + minor_axis * minor_projection.max()

    return StrandGeometry(
        center=center,
        major_start=major_start,
        major_end=major_end,
        minor_start=minor_start,
        minor_end=minor_end,
        contour=contour,
    )


# -------------------------------------------------------------------------
# Drawing utilities
# -------------------------------------------------------------------------

def _font(
    size: int,
):
    try:
        return ImageFont.truetype(
            "arial.ttf",
            size,
        )
    except OSError:
        return ImageFont.load_default()


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start,
    end,
    fill,
    width: int,
):
    """Draw a double-headed arrow."""
    start = tuple(
        float(v)
        for v in start
    )

    end = tuple(
        float(v)
        for v in end
    )

    draw.line(
        [start, end],
        fill=fill,
        width=width,
    )

    angle = math.atan2(
        end[1] - start[1],
        end[0] - start[0],
    )

    arrow_length = max(
        10,
        width * 4,
    )

    arrow_angle = math.radians(
        28
    )

    for point, direction in (
        (end, angle + math.pi),
        (start, angle),
    ):
        p1 = (
            point[0]
            + arrow_length
            * math.cos(
                direction
                - arrow_angle
            ),
            point[1]
            + arrow_length
            * math.sin(
                direction
                - arrow_angle
            ),
        )

        p2 = (
            point[0]
            + arrow_length
            * math.cos(
                direction
                + arrow_angle
            ),
            point[1]
            + arrow_length
            * math.sin(
                direction
                + arrow_angle
            ),
        )

        draw.polygon(
            [
                point,
                p1,
                p2,
            ],
            fill=fill,
        )


def _label_box(
    draw: ImageDraw.ImageDraw,
    xy,
    text: str,
    font,
    fill,
):
    """Draw a compact label with dark background."""
    x, y = xy

    bbox = draw.textbbox(
        (x, y),
        text,
        font=font,
    )

    padding = 5

    rectangle = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )

    draw.rounded_rectangle(
        rectangle,
        radius=5,
        fill=(
            0,
            0,
            0,
            180,
        ),
    )

    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
    )


def _draw_results_panel(
    draw: ImageDraw.ImageDraw,
    image_size,
    prediction: dict,
    font,
):
    """Draw numerical predictions in a dedicated top-right panel."""
    image_width, _ = image_size

    padding = 14
    line_gap = 8

    if prediction["orientation"] == "frontal":
        lines = [
            f"Height (H) = {prediction['height_mm']:.2f} mm",
            f"Width (W) = {prediction['width_mm']:.2f} mm",
        ]
    else:
        lines = [
            f"Height (H) = {prediction['height_mm']:.2f} mm",
            f"Thickness (T) = {prediction['thickness_mm']:.3f} mm",
        ]

    text_boxes = [
        draw.textbbox(
            (0, 0),
            line,
            font=font,
        )
        for line in lines
    ]

    text_width = max(
        box[2] - box[0]
        for box in text_boxes
    )

    text_heights = [
        box[3] - box[1]
        for box in text_boxes
    ]

    panel_width = text_width + 2 * padding

    panel_height = (
        sum(text_heights)
        + (len(lines) - 1) * line_gap
        + 2 * padding
    )

    margin = 20

    x0 = image_width - panel_width - margin
    y0 = margin

    x1 = x0 + panel_width
    y1 = y0 + panel_height

    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=10,
        fill=(0, 0, 0, 190),
        outline=(255, 255, 255, 170),
        width=2,
    )

    y = y0 + padding

    for line, text_height in zip(
        lines,
        text_heights,
    ):
        draw.text(
            (
                x0 + padding,
                y,
            ),
            line,
            font=font,
            fill=(255, 255, 255, 255),
        )

        y += text_height + line_gap

# -------------------------------------------------------------------------
# Main annotation function
# -------------------------------------------------------------------------

def annotate_measurements(
    image: Image.Image,
    prediction: dict,
    mask_image: Image.Image | None = None,
) -> Image.Image:
    """
    Overlay predicted strand dimensions.

    The CNN provides the numerical measurements.
    Image processing is used only to place the arrows visually.
    """
    base = image.convert(
        "RGBA"
    )

    overlay = Image.new(
        "RGBA",
        base.size,
        (
            0,
            0,
            0,
            0,
        ),
    )

    draw = ImageDraw.Draw(
        overlay
    )

    geometry = find_strand_geometry(
        image,
        mask_image,
    )

    image_width, image_height = (
        base.size
    )

    line_width = max(
        3,
        min(
            image_width,
            image_height,
        ) // 180,
    )

    font = _font(
        max(
            16,
            min(
                image_width,
                image_height,
            ) // 28,
        )
    )

    small_font = _font(
        max(
            14,
            min(
                image_width,
                image_height,
            ) // 34,
        )
    )

    # ------------------------------------------------------------------
    # Fallback if automatic geometry extraction fails
    # ------------------------------------------------------------------

    if geometry is None:
        center = np.array(
            [
                image_width / 2.0,
                image_height / 2.0,
            ],
            dtype=np.float32,
        )

        geometry = StrandGeometry(
            center=center,
            major_start=np.array(
                [
                    image_width * 0.50,
                    image_height * 0.20,
                ],
                dtype=np.float32,
            ),
            major_end=np.array(
                [
                    image_width * 0.50,
                    image_height * 0.80,
                ],
                dtype=np.float32,
            ),
            minor_start=np.array(
                [
                    image_width * 0.40,
                    image_height * 0.50,
                ],
                dtype=np.float32,
            ),
            minor_end=np.array(
                [
                    image_width * 0.60,
                    image_height * 0.50,
                ],
                dtype=np.float32,
            ),
            contour=np.empty(
                (
                    0,
                    1,
                    2,
                ),
                dtype=np.int32,
            ),
        )

    height_color = (
        255,
        80,
        80,
        255,
    )

    secondary_color = (
        80,
        200,
        255,
        255,
    )

    # Extend the minor-axis annotation slightly so W/T remains visible
    # even when the strand is only a few pixels thick.
    minor_vector = geometry.minor_end - geometry.minor_start
    minor_length = float(np.linalg.norm(minor_vector))

    if minor_length > 1e-6:
        minor_unit = minor_vector / minor_length
        visual_padding = max(6.0, line_width * 2.0)

        geometry.minor_start = (
            geometry.minor_start - minor_unit * visual_padding
        )
        geometry.minor_end = (
            geometry.minor_end + minor_unit * visual_padding
        )

    # ------------------------------------------------------------------
    # Height arrow
    # ------------------------------------------------------------------

    _draw_arrow(
        draw,
        geometry.major_start,
        geometry.major_end,
        fill=height_color,
        width=line_width,
    )

    # Place H near the visually upper endpoint of the height arrow.
    if geometry.major_start[1] < geometry.major_end[1]:
        height_top = geometry.major_start
    else:
        height_top = geometry.major_end

    h_x = float(height_top[0] + 15)
    h_y = float(height_top[1] + 15)

    # Keep the label inside the image bounds.
    h_x = max(10, min(h_x, image_width - 50))
    h_y = max(10, min(h_y, image_height - 50))

    _label_box(
        draw,
        (h_x, h_y),
        "H",
        small_font,
        height_color,
    )

    # ------------------------------------------------------------------
    # Width / thickness arrow
    # ------------------------------------------------------------------

    _draw_arrow(
        draw,
        geometry.minor_start,
        geometry.minor_end,
        fill=secondary_color,
        width=line_width,
    )

    secondary_mid = (
        geometry.minor_start
        + geometry.minor_end
    ) / 2.0

    secondary_letter = (
        "W"
        if prediction["orientation"]
        == "frontal"
        else "T"
    )

    _label_box(
        draw,
        (
            float(
                secondary_mid[0]
                + 12
            ),
            float(
                secondary_mid[1]
                - 35
            ),
        ),
        secondary_letter,
        small_font,
        secondary_color,
    )

    # ------------------------------------------------------------------
    # Numerical result panel
    # ------------------------------------------------------------------

    _draw_results_panel(
        draw,
        base.size,
        prediction,
        font,
    )

    return Image.alpha_composite(
        base,
        overlay,
    ).convert(
        "RGB"
    )