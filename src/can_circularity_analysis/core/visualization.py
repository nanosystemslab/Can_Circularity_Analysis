"""Visualization utilities for creating analysis overlays and debug images."""

import os
from typing import Any

import cv2
import numpy as np


def create_analysis_overlay(
    image: np.ndarray,
    contour: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    points: np.ndarray,
    ellipse_result: dict[str, Any] | None,
    best_contour_result: dict[str, Any] | None,
    scale_metadata: dict[str, Any] | None,
    offset: tuple[int, int],
    output_dir: str,
    stem: str,
    **kwargs,
) -> str:
    """Create analysis overlay image with detected features.

    Args:
        image: Input image
        contour: Detected contour
        center_x, center_y, radius: Circle parameters
        points: Rim points
        ellipse_result: Ellipse fitting results
        best_contour_result: Best contour fitting results
        scale_metadata: Scale/calibration information
        offset: Crop offset
        output_dir: Output directory
        stem: Output file stem
        **kwargs: Additional parameters

    Returns:
        Path to saved overlay image
    """
    overlay = image.copy()

    # Draw contour
    contour_color = (255, 0, 0) if kwargs.get("method", "binary") == "binary" else (0, 165, 255)
    cv2.drawContours(overlay, [contour], -1, contour_color, 1)

    # Draw rim points
    for point in points.astype(int):
        cv2.circle(overlay, tuple(point), 1, (0, 0, 255), -1)  # Red dots

    # Draw fitted circle
    circle_center = (int(round(center_x)), int(round(center_y)))
    cv2.circle(overlay, circle_center, int(round(radius)), (0, 255, 0), 2)  # Green circle
    cv2.circle(overlay, circle_center, 3, (0, 255, 255), -1)  # Yellow center

    # Draw ellipse if available
    if ellipse_result and "major_px" in ellipse_result:
        ellipse_center = ellipse_result["center_px"]
        ellipse_axes = (
            int(round(ellipse_result["major_px"] / 2.0)),
            int(round(ellipse_result["minor_px"] / 2.0)),
        )
        ellipse_angle = ellipse_result["angle_deg"]

        cv2.ellipse(
            overlay,
            (int(round(ellipse_center[0])), int(round(ellipse_center[1]))),
            ellipse_axes,
            ellipse_angle,
            0,
            360,
            (200, 0, 200),  # Purple
            2,
        )

    # Draw best contour if available
    if best_contour_result and "contour" in best_contour_result:
        best_contour = best_contour_result["contour"]
        cv2.drawContours(overlay, [best_contour], -1, (0, 255, 255), 2)  # Cyan color

    # Draw scale line if available
    draw_scale_reference(overlay, scale_metadata, offset)

    # Add text label
    diameter_px = 2.0 * radius
    mm_per_pixel = kwargs.get("mm_per_pixel")

    label_lines = []
    if mm_per_pixel:
        diameter_mm = diameter_px * mm_per_pixel
        label_lines.append(f"Circle: {diameter_mm:.2f} mm")
    else:
        label_lines.append(f"Circle: {diameter_px:.0f} px")

    if ellipse_result and ellipse_result.get("ellipticity"):
        label_lines.append(f"Ellipse: e={ellipse_result['ellipticity']:.3f}")

    if best_contour_result and best_contour_result.get("method"):
        method = best_contour_result["method"]
        if "radius_variation" in best_contour_result:
            variation = best_contour_result["radius_variation"]
            label_lines.append(f"Best ({method}): var={variation:.3f}")
        else:
            label_lines.append(f"Best: {method}")

    # Draw text with outline for visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    for i, label in enumerate(label_lines):
        text_pos = (12, 28 + i * 25)
        cv2.putText(overlay, label, text_pos, font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(overlay, label, text_pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Save overlay
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    return os.path.abspath(overlay_path)


def draw_scale_reference(
    image: np.ndarray,
    scale_metadata: dict[str, Any] | None,
    offset: tuple[int, int] = (0, 0),
    line_color: tuple[int, int, int] = (255, 255, 0),
    point_color: tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
    radius: int = 5,
) -> None:
    """Draw scale reference line on image if scale points are available.

    Args:
        image: Image to draw on (modified in-place)
        scale_metadata: Dictionary containing scale information
        offset: Crop offset to apply to coordinates
        line_color: RGB color for reference line
        point_color: RGB color for reference points
        thickness: Line thickness
        radius: Point circle radius
    """
    if not scale_metadata:
        return

    scale_points = scale_metadata.get("scale_points_full_px")
    if not scale_points:
        return

    x1, y1, x2, y2 = scale_points
    x_offset, y_offset = offset

    # Adjust coordinates for crop offset
    p1 = (int(round(x1 - x_offset)), int(round(y1 - y_offset)))
    p2 = (int(round(x2 - x_offset)), int(round(y2 - y_offset)))

    # Check if points are within image bounds
    height, width = image.shape[:2]

    def point_in_bounds(point):
        return 0 <= point[0] < width and 0 <= point[1] < height

    # Draw line and points if they're visible
    if point_in_bounds(p1) or point_in_bounds(p2):
        cv2.line(image, p1, p2, line_color, thickness)
        cv2.circle(image, p1, radius, point_color, -1)
        cv2.circle(image, p2, radius, point_color, -1)


def save_debug_images(binary_image: np.ndarray | None, edges: np.ndarray | None, output_dir: str, stem: str) -> None:
    """Save debug images for troubleshooting segmentation.

    Args:
        binary_image: Binary segmentation result
        edges: Edge detection result
        output_dir: Output directory
        stem: Output file stem
    """
    if binary_image is not None:
        debug_path = os.path.join(output_dir, f"{stem}_binary.png")
        cv2.imwrite(debug_path, binary_image)

    if edges is not None:
        debug_path = os.path.join(output_dir, f"{stem}_edges.png")
        cv2.imwrite(debug_path, edges)


def create_comparison_overlay(images: list, titles: list, output_path: str, max_height: int = 600) -> str:
    """Create side-by-side comparison of multiple images.

    Args:
        images: List of images to compare
        titles: List of titles for each image
        output_path: Output file path
        max_height: Maximum height for output image

    Returns:
        Path to saved comparison image
    """
    if not images or len(images) != len(titles):
        raise ValueError("Images and titles lists must have same length")

    # Resize images to same height
    resized_images = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, max_height))

        resized_images.append(img)

    # Add titles to images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    for _i, (img, title) in enumerate(zip(resized_images, titles, strict=False)):
        cv2.putText(img, title, (10, 30), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(img, title, (10, 30), font, font_scale, (255, 255, 255), thickness)

    # Concatenate horizontally
    comparison = np.hstack(resized_images)

    # Save result
    cv2.imwrite(output_path, comparison)
    return os.path.abspath(output_path)
