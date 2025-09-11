"""Core detection algorithms for circular contours and rim analysis."""

import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.measure import CircleModel, ransac

from can_circularity_analysis.utils.file_io import save_results
from can_circularity_analysis.utils.image_processing import preprocess_image, segment_image

from .calibration import calibrate_from_points, calibrate_pixels_per_mm, load_calibration
from .metrics import (
    calculate_circularity_metrics,
    calculate_deviation_metrics,
)


def score_contour(
    contour: np.ndarray,
    image_width: int,
    image_height: int,
    prefer_center: bool = False,
    center_radius_fraction: float = 0.4,
) -> float:
    """Score a contour based on area, circularity, and optionally center preference.

    Args:
        contour: OpenCV contour
        image_width: Image width in pixels
        image_height: Image height in pixels
        prefer_center: Whether to prefer contours near image center
        center_radius_fraction: Radius fraction for center preference

    Returns:
        Numerical score (higher is better)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Circularity score (4π×area / perimeter²)
    circularity = (4.0 * math.pi * area) / (perimeter * perimeter + 1e-12) if perimeter > 0 else 0.0
    score = area * circularity  # Prefer large and round

    if prefer_center and area > 0:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]

            img_center_x, img_center_y = image_width / 2.0, image_height / 2.0
            distance_from_center = math.hypot(center_x - img_center_x, center_y - img_center_y)
            preference_radius = center_radius_fraction * min(image_width, image_height)

            if distance_from_center < preference_radius:
                score *= 2.0

    return score


def fit_circle_to_contour(
    contour: np.ndarray,
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    """Fit a circle to contour points using RANSAC.

    Args:
        contour: OpenCV contour array

    Returns:
        Tuple of ((center_x, center_y, radius), points, inlier_mask)
    """
    points = contour[:, 0, :].astype(float)

    try:
        model_robust, inliers = ransac(points, CircleModel, min_samples=3, residual_threshold=2.0, max_trials=1000)

        if model_robust is None:
            # Fallback to least squares if RANSAC fails
            model = CircleModel()
            model.estimate(points)
            center_x, center_y, radius = model.params
            inliers = np.ones(len(points), dtype=bool)
        else:
            center_x, center_y, radius = model_robust.params

    except Exception:
        # Final fallback: use minimum enclosing circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        inliers = np.ones(len(points), dtype=bool)

    return (float(center_x), float(center_y), float(radius)), points, inliers


def fit_ellipse_to_contour(contour: np.ndarray) -> dict[str, Any]:
    """Fit an ellipse to contour points.

    Args:
        contour: OpenCV contour array (needs >=5 points)

    Returns:
        Dictionary with ellipse parameters or error information
    """
    if len(contour) < 5:
        return {"error": "Not enough points to fit ellipse (need >=5)"}

    try:
        contour_float = contour.astype(np.float32)
        (center_x, center_y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour_float)

        # Ensure major_axis >= minor_axis
        if minor_axis > major_axis:
            major_axis, minor_axis = minor_axis, major_axis

        return {
            "center_px": (float(center_x), float(center_y)),
            "major_px": float(major_axis),
            "minor_px": float(minor_axis),
            "angle_deg": float(angle),
            "ellipticity": float(major_axis / minor_axis) if minor_axis > 0 else None,
            "eccentricity": math.sqrt(1.0 - (minor_axis / major_axis) ** 2) if major_axis > 0 else None,
            "area_px2": math.pi * (major_axis / 2.0) * (minor_axis / 2.0),
            "ovalization": (major_axis - minor_axis) / ((major_axis + minor_axis) / 2.0)
            if (major_axis + minor_axis) > 0
            else None,
        }

    except Exception as e:
        return {"error": str(e)}


def select_best_contour(
    binary_image: np.ndarray,
    prefer_center: bool = False,
    center_radius_fraction: float = 0.4,
    mm_per_pixel: float | None = None,
    min_diameter_mm: float = 0.0,
    max_diameter_mm: float = 1e9,
) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray, np.ndarray]:
    """Select the best circular contour from binary image.

    Args:
        binary_image: Binary image with white contours
        prefer_center: Whether to prefer contours near image center
        center_radius_fraction: Radius fraction for center preference
        mm_per_pixel: Conversion factor for diameter filtering
        min_diameter_mm: Minimum allowed diameter in mm
        max_diameter_mm: Maximum allowed diameter in mm

    Returns:
        Tuple of (best_contour, (center_x, center_y, radius), points, inliers)

    Raises:
        RuntimeError: If no suitable contours found
    """
    height, width = binary_image.shape[:2]
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("No contours found in binary image.")

    # Score and sort contours
    scored_contours = sorted(
        contours,
        key=lambda c: score_contour(c, width, height, prefer_center, center_radius_fraction),
        reverse=True,
    )

    # Apply diameter filter if calibration available
    if mm_per_pixel and (min_diameter_mm > 0 or max_diameter_mm < 1e9):
        for contour in scored_contours:
            (center_x, center_y, radius), points, inliers = fit_circle_to_contour(contour)
            diameter_mm = 2.0 * radius * mm_per_pixel

            if min_diameter_mm <= diameter_mm <= max_diameter_mm:
                return contour, (center_x, center_y, radius), points, inliers

        # If no contour passes filter, return best one anyway
        best_contour = scored_contours[0]
        return best_contour, *fit_circle_to_contour(best_contour)

    # No diameter filtering - return best scored contour
    best_contour = scored_contours[0]
    return best_contour, *fit_circle_to_contour(best_contour)


def detect_circular_contour(image: np.ndarray, method: str = "binary", **kwargs) -> dict[str, Any]:
    """Detect circular contour in image using specified method.

    Args:
        image: Input image (BGR or grayscale)
        method: Detection method ("binary" or "canny")
        **kwargs: Additional parameters for detection

    Returns:
        Dictionary with detection results
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Preprocess image
    gray = preprocess_image(gray)

    # Segment image
    binary_mask, edges = segment_image(gray, method=method, **kwargs)

    # Detect contour
    if method == "binary":
        contour, circle_params, points, inliers = select_best_contour(
            binary_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "prefer_center",
                    "center_radius_fraction",
                    "mm_per_pixel",
                    "min_diameter_mm",
                    "max_diameter_mm",
                ]
            },
        )
        debug_image = binary_mask
    else:
        try:
            contour, circle_params, points, inliers = select_best_contour(
                edges,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in [
                        "prefer_center",
                        "center_radius_fraction",
                        "mm_per_pixel",
                        "min_diameter_mm",
                        "max_diameter_mm",
                    ]
                },
            )
            debug_image = edges
        except RuntimeError:
            # Hough circle fallback
            result = _hough_circle_fallback(gray, edges, **kwargs)
            if result:
                return result
            raise RuntimeError("No circular contours detected with any method.")

    return {
        "contour": contour,
        "circle_params": circle_params,
        "points": points,
        "inliers": inliers,
        "debug_image": debug_image,
    }


def _hough_circle_fallback(gray: np.ndarray, edges: np.ndarray | None, **kwargs) -> dict[str, Any] | None:
    """Fallback Hough circle detection when contour methods fail."""
    # Apply CLAHE for better circle detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    circles = cv2.HoughCircles(
        gray_enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(gray.shape) // 3,
        param1=max(50, kwargs.get("canny_high", 200)),
        param2=30,
        minRadius=int(0.15 * min(gray.shape)),
        maxRadius=int(0.6 * min(gray.shape)),
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))
    best_circle = max(circles[0, :], key=lambda c: c[2])  # Select largest radius
    center_x, center_y, radius = map(float, best_circle)

    # Check diameter filter if available
    mm_per_pixel = kwargs.get("mm_per_pixel")
    if mm_per_pixel:
        diameter_mm = 2.0 * radius * mm_per_pixel
        min_diam = kwargs.get("min_diameter_mm", 0.0)
        max_diam = kwargs.get("max_diameter_mm", 1e9)

        if not (min_diam <= diameter_mm <= max_diam):
            return None

    # Create synthetic contour from edge points near the detected circle
    if edges is None:
        edges = cv2.Canny(
            gray_enhanced,
            max(10, kwargs.get("canny_low", 75) // 2),
            max(40, kwargs.get("canny_high", 200) // 2),
            L2gradient=True,
        )

    y_coords, x_coords = np.where(edges > 0)
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) - radius
    mask = np.abs(distances) < 3.0

    rim_points = np.stack([x_coords[mask], y_coords[mask]], axis=1).astype(np.float32)

    if len(rim_points) < 20:
        return None

    contour = rim_points.reshape(-1, 1, 2).astype(np.int32)
    inliers = np.ones(len(rim_points), dtype=bool)

    return {
        "contour": contour,
        "circle_params": (center_x, center_y, radius),
        "points": rim_points,
        "inliers": inliers,
        "debug_image": edges,
        "method": "hough_fallback",
    }


def analyze_image_file(image_path: str, output_dir: str = "out", **kwargs) -> dict[str, Any]:
    """Analyze circularity of a can in an image file.

    Args:
        image_path: Path to input image
        output_dir: Output directory for results
        **kwargs: Analysis parameters

    Returns:
        Dictionary with complete analysis results
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Set up output paths
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem

    # Handle cropping
    crop = kwargs.get("crop")
    offset = (0, 0)
    if crop:
        x, y, w, h = crop
        image = image[y : y + h, x : x + w].copy()
        offset = (x, y)

    # Handle calibration
    pixels_per_mm, scale_metadata = _setup_calibration(image_path, **kwargs)
    mm_per_pixel = (1.0 / pixels_per_mm) if pixels_per_mm else None

    # Detect circular contour
    detection_result = detect_circular_contour(image, mm_per_pixel=mm_per_pixel, **kwargs)

    contour = detection_result["contour"]
    center_x, center_y, radius = detection_result["circle_params"]
    points = detection_result["points"]
    inliers = detection_result["inliers"]

    # Calculate metrics
    metrics = calculate_circularity_metrics(contour, center_x, center_y, radius, mm_per_pixel)
    deviation_metrics = calculate_deviation_metrics(points, center_x, center_y, radius, mm_per_pixel)

    # Optional ellipse fitting
    ellipse_result = None
    if kwargs.get("fit_ellipse", False):
        ellipse_result = fit_ellipse_to_contour(contour)
        if mm_per_pixel and ellipse_result and "major_px" in ellipse_result:
            ellipse_result.update(
                {
                    "major_mm": ellipse_result["major_px"] * mm_per_pixel,
                    "minor_mm": ellipse_result["minor_px"] * mm_per_pixel,
                    "area_mm2": ellipse_result["area_px2"] * (mm_per_pixel**2),
                    "ovalization_mm": (ellipse_result["major_px"] - ellipse_result["minor_px"]) * mm_per_pixel,
                }
            )

    # Compile final results
    results = {
        "image_path": os.path.abspath(image_path),
        "pixels_per_mm": pixels_per_mm,
        "center_px": [center_x + offset[0], center_y + offset[1]],
        "radius_px": radius,
        "diameter_px": 2.0 * radius,
        "offset": offset,
        "inlier_ratio": float(np.mean(inliers)),
        **metrics,
        **deviation_metrics,
        "ellipse": ellipse_result,
        "detection_method": detection_result.get("method", kwargs.get("method", "binary")),
    }

    # Add mm measurements if calibration available
    if mm_per_pixel:
        results.update(
            {
                "center_mm": [
                    (center_x + offset[0]) * mm_per_pixel,
                    (center_y + offset[1]) * mm_per_pixel,
                ],
                "radius_mm": radius * mm_per_pixel,
                "diameter_mm": 2.0 * radius * mm_per_pixel,
            }
        )

    # Save results
    save_results(results, points, center_x, center_y, offset, mm_per_pixel, output_dir, stem)

    # Generate overlay visualization if requested
    if not kwargs.get("no_overlay", False):
        from .visualization import create_analysis_overlay

        overlay_path = create_analysis_overlay(
            image,
            contour,
            center_x,
            center_y,
            radius,
            points,
            ellipse_result,
            scale_metadata,
            offset,
            output_dir,
            stem,
            **kwargs,
        )
        results["overlay_image"] = overlay_path

    return results


def _setup_calibration(image_path: str, **kwargs) -> tuple[float | None, dict | None]:
    """Set up pixel-to-mm calibration from various sources."""
    scale_metadata = None

    # Priority order: scale-from -> scale-pts -> click-scale -> pixels-per-mm
    if kwargs.get("scale_from"):
        pixels_per_mm, scale_metadata = load_calibration(kwargs["scale_from"])
    elif kwargs.get("scale_pts"):
        pixels_per_mm, scale_points = calibrate_from_points(kwargs["scale_pts"], kwargs.get("known_mm", 50.0))
        scale_metadata = {"scale_points_full_px": scale_points}
    elif kwargs.get("click_scale"):
        try:
            full_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            pixels_per_mm, scale_points = calibrate_pixels_per_mm(
                full_image.copy(), known_mm=kwargs.get("known_mm", 50.0)
            )
            scale_metadata = {"scale_points_full_px": scale_points}
        except (cv2.error, RuntimeError):
            raise SystemExit("OpenCV GUI not available. Use --scale-pts or --pixels-per-mm instead.")
    else:
        pixels_per_mm = kwargs.get("pixels_per_mm")

    return pixels_per_mm, scale_metadata
