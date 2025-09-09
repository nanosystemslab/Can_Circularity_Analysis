"""Calibration utilities for pixel-to-millimeter conversion."""

import json
import math
import os

import cv2
import numpy as np


def calibrate_pixels_per_mm(
    img: np.ndarray, known_mm: float = 50.0, window_name: str = "Calibration"
) -> tuple[float, tuple[float, float, float, float]]:
    """Interactive calibration: click two points on ruler that are `known_mm` apart.

    Args:
        img: Input image for calibration
        known_mm: Known distance in millimeters between the two points
        window_name: OpenCV window name for display

    Returns:
        Tuple of (pixels_per_mm, (x1, y1, x2, y2)) in full-image coordinates

    Raises:
        RuntimeError: If calibration is canceled or OpenCV GUI not available
    """
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 6, (255, 0, 255), -1)
            if len(points) == 2:
                cv2.line(display_img, points[0], points[1], (255, 255, 0), 2)
            cv2.imshow(window_name, display_img)

    display_img = img.copy()

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_img)
        cv2.setMouseCallback(window_name, on_mouse)

        print(f"[calibrate] Click two points on the ruler that are exactly {known_mm} mm apart.")
        print("Press ESC to cancel.")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                raise RuntimeError("Calibration canceled by user.")
            if len(points) == 2:
                break

        cv2.destroyWindow(window_name)

    except cv2.error as e:
        raise RuntimeError(f"OpenCV GUI not available: {e}")

    p1, p2 = np.array(points[0], dtype=float), np.array(points[1], dtype=float)
    distance_pixels = float(np.linalg.norm(p1 - p2))
    pixels_per_mm = distance_pixels / known_mm

    return pixels_per_mm, (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))


def calibrate_from_points(
    scale_points: str, known_mm: float
) -> tuple[float, tuple[float, float, float, float]]:
    """Non-GUI calibration from coordinate string.

    Args:
        scale_points: String in format 'x1,y1,x2,y2'
        known_mm: Known distance in millimeters

    Returns:
        Tuple of (pixels_per_mm, (x1, y1, x2, y2))
    """
    x1, y1, x2, y2 = map(float, scale_points.split(","))
    distance_pixels = math.hypot(x2 - x1, y2 - y1)
    pixels_per_mm = distance_pixels / known_mm
    return pixels_per_mm, (x1, y1, x2, y2)


def save_calibration(
    file_path: str,
    pixels_per_mm: float,
    known_mm: float | None = None,
    source_image: str | None = None,
    scale_points: tuple[float, float, float, float] | None = None,
) -> str:
    """Save calibration data to JSON file.

    Args:
        file_path: Output file path
        pixels_per_mm: Calibration value
        known_mm: Known distance used for calibration
        source_image: Source image file path
        scale_points: Calibration points (x1, y1, x2, y2)

    Returns:
        Absolute path to saved file
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    data = {
        "pixels_per_mm": float(pixels_per_mm),
        "known_mm": float(known_mm) if known_mm is not None else None,
        "source_image": source_image,
        "scale_points_full_px": scale_points,
    }

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    return os.path.abspath(file_path)


def load_calibration(file_path: str) -> tuple[float, dict]:
    """Load calibration data from JSON file.

    Args:
        file_path: Input file path

    Returns:
        Tuple of (pixels_per_mm, metadata_dict)

    Raises:
        ValueError: If file doesn't contain required calibration data
        FileNotFoundError: If file doesn't exist
    """
    with open(file_path) as f:
        data = json.load(f)

    if "pixels_per_mm" not in data:
        raise ValueError(f"{file_path} does not contain 'pixels_per_mm' field.")

    return float(data["pixels_per_mm"]), data


def parse_crop_string(crop_str: str | None) -> tuple[int, int, int, int] | None:
    """Parse crop string into coordinates.

    Args:
        crop_str: String in format 'x,y,w,h' or None

    Returns:
        Tuple (x, y, w, h) or None if input is None/empty
    """
    if not crop_str:
        return None
    x, y, w, h = map(int, crop_str.split(","))
    return (x, y, w, h)
