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


def score_circle_by_edge_support(edges: np.ndarray, x: int, y: int, r: int) -> float:
    """Score a circle hypothesis by counting nearby edge pixels."""
    # Create a mask of pixels that should be on the circle
    h, w = edges.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

    # Count edge pixels within 2 pixels of the circle circumference
    circle_mask = np.abs(dist_from_center - r) <= 2.0
    edge_on_circle = np.sum(edges[circle_mask] > 0)

    # Normalize by circle circumference
    circumference = 2 * np.pi * r
    return edge_on_circle / circumference if circumference > 0 else 0.0


def create_synthetic_contour(edges: np.ndarray, x: int, y: int, r: int) -> tuple:
    """Create a synthetic contour from edge points near a detected circle."""
    # Find edge pixels near the circle
    h, w = edges.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

    # Get pixels within 3 pixels of the circumference
    circle_mask = np.abs(dist_from_center - r) <= 3.0
    edge_pixels = edges > 0
    rim_mask = circle_mask & edge_pixels

    y_coords, x_coords = np.where(rim_mask)
    if len(x_coords) < 10:
        # Fallback: create perfect circle points
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        x_coords = x + r * np.cos(angles)
        y_coords = y + r * np.sin(angles)
        x_coords = np.clip(x_coords, 0, w - 1).astype(int)
        y_coords = np.clip(y_coords, 0, h - 1).astype(int)

    points = np.column_stack([x_coords, y_coords]).astype(np.float32)
    contour = points.reshape(-1, 1, 2).astype(np.int32)

    return contour, points


def detect_broken_circle(gray_image: np.ndarray, **kwargs) -> dict:
    """Enhanced detection for broken/incomplete circles like can rims.

    This method specifically handles cases where the rim appears as
    incomplete circle segments rather than closed contours.
    """
    # Apply preprocessing
    processed = preprocess_image(gray_image)

    # Get edge map
    canny_low = kwargs.get("canny_low", 50)
    canny_high = kwargs.get("canny_high", 150)

    # Use adaptive thresholds based on image statistics
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(processed)

    edges = cv2.Canny(enhanced, canny_low, canny_high, L2gradient=True)

    # Close gaps in edges using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Filter kwargs for select_best_contour
    contour_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ["prefer_center", "center_radius_fraction", "mm_per_pixel", "min_diameter_mm", "max_diameter_mm"]
    }

    # Try contour detection on closed edges first
    try:
        contour, circle_params, points, inliers = select_best_contour(edges_closed, **contour_kwargs)
        return {
            "contour": contour,
            "circle_params": circle_params,
            "points": points,
            "inliers": inliers,
            "debug_image": edges_closed,
            "method": "closed_edges",
        }
    except RuntimeError:
        pass

    # Fallback 1: Hough circles on original edges
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min(gray_image.shape) // 4,
        param1=max(50, canny_high),
        param2=30,
        minRadius=int(0.1 * min(gray_image.shape)),
        maxRadius=int(0.7 * min(gray_image.shape)),
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Score circles and pick the best one
        best_circle = None
        best_score = 0

        # Check diameter filter during scoring
        mm_per_pixel = kwargs.get("mm_per_pixel")
        min_diameter_mm = kwargs.get("min_diameter_mm", 0.0)
        max_diameter_mm = kwargs.get("max_diameter_mm", 1e9)

        for x, y, r in circles:
            # Apply diameter filter if calibration available
            if mm_per_pixel:
                diameter_mm = 2.0 * r * mm_per_pixel
                if not (min_diameter_mm <= diameter_mm <= max_diameter_mm):
                    continue  # Skip this circle

            # Score based on how many edge pixels are near this circle
            score = score_circle_by_edge_support(edges, x, y, r)
            if score > best_score:
                best_score = score
                best_circle = (x, y, r)

        if best_circle:
            x, y, r = best_circle

            # Create synthetic contour from edge points near the circle
            contour, points = create_synthetic_contour(edges, x, y, r)
            inliers = np.ones(len(points), dtype=bool)

            return {
                "contour": contour,
                "circle_params": (float(x), float(y), float(r)),
                "points": points,
                "inliers": inliers,
                "debug_image": edges,
                "method": "hough_enhanced",
            }

    # Fallback 2: RANSAC on all edge points
    y_coords, x_coords = np.where(edges > 0)
    if len(x_coords) > 100:  # Need sufficient points
        edge_points = np.column_stack([x_coords, y_coords]).astype(float)

        try:
            model_robust, inliers = ransac(
                edge_points, CircleModel, min_samples=3, residual_threshold=3.0, max_trials=2000
            )

            if model_robust is not None:
                center_x, center_y, radius = model_robust.params

                # Apply diameter filter if calibration available
                mm_per_pixel = kwargs.get("mm_per_pixel")
                if mm_per_pixel:
                    diameter_mm = 2.0 * radius * mm_per_pixel
                    min_diameter_mm = kwargs.get("min_diameter_mm", 0.0)
                    max_diameter_mm = kwargs.get("max_diameter_mm", 1e9)

                    if not (min_diameter_mm <= diameter_mm <= max_diameter_mm):
                        raise RuntimeError(
                            f"RANSAC circle diameter {diameter_mm:.2f}mm outside range [{min_diameter_mm}, {max_diameter_mm}]"
                        )

                # Create contour from inlier points
                inlier_points = edge_points[inliers]
                contour = inlier_points.reshape(-1, 1, 2).astype(np.int32)

                return {
                    "contour": contour,
                    "circle_params": (center_x, center_y, radius),
                    "points": inlier_points,
                    "inliers": inliers,
                    "debug_image": edges,
                    "method": "ransac_edges",
                }
        except ImportError:
            pass

    raise RuntimeError("No circular features detected with any method")


def fit_best_contour(
    contour: np.ndarray, center_x: float, center_y: float, method: str = "adaptive", **kwargs
) -> dict[str, Any]:
    """Fit the best representation of the actual contour shape.

    Args:
        contour: Detected contour points
        center_x, center_y: Circle center coordinates
        method: "smooth", "fourier", or "adaptive"
        **kwargs: Method-specific parameters
    """
    if len(contour) < 5:
        return {"error": "Not enough points for best contour fitting"}

    try:
        points = contour[:, 0, :].astype(float)

        # Convert to polar coordinates relative to circle center
        dx = points[:, 0] - center_x
        dy = points[:, 1] - center_y
        angles = np.arctan2(dy, dx)
        radii = np.sqrt(dx**2 + dy**2)

        # Remove duplicate angles and very close points
        angles_clean, radii_clean = _remove_duplicate_angles(angles, radii)

        if len(angles_clean) < 5:
            return {"error": "Not enough unique points after cleaning duplicates"}

        # Sort by angle to ensure proper ordering
        sorted_indices = np.argsort(angles_clean)
        angles_sorted = angles_clean[sorted_indices]
        radii_sorted = radii_clean[sorted_indices]

        if method == "adaptive":
            return _fit_adaptive_contour(
                angles_sorted, radii_sorted, center_x, center_y, kwargs.get("smoothing_factor", 0.1)
            )
        elif method == "smooth":
            return _fit_smooth_contour(angles_sorted, radii_sorted, center_x, center_y, kwargs.get("resolution", 360))
        elif method == "fourier":
            return _fit_fourier_contour(
                angles_sorted,
                radii_sorted,
                center_x,
                center_y,
                kwargs.get("num_harmonics", 10),
                kwargs.get("resolution", 360),
            )
        else:
            return {"error": f"Unknown best contour method: {method}"}

    except Exception as e:
        return {"error": f"Best contour fitting failed: {str(e)}"}


def _remove_duplicate_angles(
    angles: np.ndarray, radii: np.ndarray, min_angle_diff: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate or very close angles from polar coordinate data.

    Args:
        angles: Angular coordinates in radians
        radii: Radial coordinates
        min_angle_diff: Minimum angle difference to keep points separate

    Returns:
        Tuple of (cleaned_angles, cleaned_radii)
    """
    if len(angles) == 0:
        return angles, radii

    # Sort by angle first
    sorted_indices = np.argsort(angles)
    angles_sorted = angles[sorted_indices]
    radii_sorted = radii[sorted_indices]

    # Find points that are far enough apart
    keep_mask = np.ones(len(angles_sorted), dtype=bool)

    for i in range(1, len(angles_sorted)):
        # Check if this angle is too close to the previous kept angle
        prev_kept_idx = i - 1
        while prev_kept_idx >= 0 and not keep_mask[prev_kept_idx]:
            prev_kept_idx -= 1

        if prev_kept_idx >= 0:
            angle_diff = angles_sorted[i] - angles_sorted[prev_kept_idx]
            if angle_diff < min_angle_diff:
                # Too close - average the radii and keep only one point
                avg_radius = (radii_sorted[i] + radii_sorted[prev_kept_idx]) / 2
                radii_sorted[prev_kept_idx] = avg_radius
                keep_mask[i] = False

    # Handle wrap-around: check if first and last points are too close
    if len(angles_sorted) > 1 and keep_mask[0] and keep_mask[-1]:
        angle_diff = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
        if angle_diff < min_angle_diff:
            # Average the radii
            avg_radius = (radii_sorted[0] + radii_sorted[-1]) / 2
            radii_sorted[0] = avg_radius
            keep_mask[-1] = False

    return angles_sorted[keep_mask], radii_sorted[keep_mask]


def _fit_adaptive_contour(
    angles: np.ndarray, radii: np.ndarray, center_x: float, center_y: float, smoothing_factor: float = 0.1
) -> dict[str, Any]:
    """Adaptive contour that balances following actual points vs. smoothness."""
    try:
        from scipy.interpolate import interp1d
        from scipy.ndimage import uniform_filter1d

        # Ensure we have enough points
        if len(angles) < 5:
            return {"error": "Not enough points for adaptive fitting after cleaning"}

        # Apply adaptive smoothing with circular boundary conditions
        # Extend data for proper circular smoothing
        extended_angles = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])
        extended_radii = np.tile(radii, 3)

        # Apply smoothing
        window_size = max(3, int(len(radii) * smoothing_factor))
        if window_size % 2 == 0:
            window_size += 1

        smoothed_extended = uniform_filter1d(extended_radii, size=window_size, mode="constant")
        smoothed_radii = smoothed_extended[len(radii) : 2 * len(radii)]

        # Create interpolation function with more robust settings
        # Ensure angles are strictly increasing for interpolation
        angles_for_interp = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])
        radii_for_interp = np.tile(smoothed_radii, 3)

        # Use linear interpolation to avoid oscillations
        interp_func = interp1d(
            angles_for_interp, radii_for_interp, kind="linear", bounds_error=False, fill_value="extrapolate"
        )

        # Generate high-resolution contour
        angles_dense = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        radii_dense = interp_func(angles_dense)

        # Ensure positive radii
        radii_dense = np.maximum(radii_dense, 1.0)

        # Convert back to Cartesian
        x_dense = center_x + radii_dense * np.cos(angles_dense)
        y_dense = center_y + radii_dense * np.sin(angles_dense)

        dense_points = np.column_stack([x_dense, y_dense])
        dense_contour = dense_points.reshape(-1, 1, 2).astype(np.int32)

        # Calculate metrics
        area = cv2.contourArea(dense_contour)
        perimeter = cv2.arcLength(dense_contour, True)
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

        mean_radius = np.mean(radii_dense)
        radius_std = np.std(radii_dense)
        max_deviation = np.max(np.abs(radii_dense - mean_radius))
        radius_variation = (np.max(radii_dense) - np.min(radii_dense)) / mean_radius if mean_radius > 0 else 0

        return {
            "contour": dense_contour,
            "points": dense_points,
            "area_px2": float(area),
            "perimeter_px": float(perimeter),
            "circularity": float(circularity),
            "mean_radius_px": float(mean_radius),
            "radius_std_px": float(radius_std),
            "max_deviation_px": float(max_deviation),
            "radius_variation": float(radius_variation),
            "smoothing_factor": smoothing_factor,
            "radii": radii_dense,
            "angles": angles_dense,
            "method": "adaptive_smooth",
        }

    except ImportError:
        return {"error": "scipy required for adaptive contour fitting"}
    except Exception as e:
        return {"error": f"Adaptive contour fitting failed: {str(e)}"}


def _fit_smooth_contour(
    angles: np.ndarray, radii: np.ndarray, center_x: float, center_y: float, resolution: int = 360
) -> dict[str, Any]:
    """Fit a smooth spline curve to the actual detected contour points."""
    try:
        from scipy.interpolate import UnivariateSpline

        # Handle wrap-around: add points at beginning and end for continuity
        angles_extended = np.concatenate([angles[-3:] - 2 * np.pi, angles, angles[:3] + 2 * np.pi])
        radii_extended = np.concatenate([radii[-3:], radii, radii[:3]])

        # Fit smooth spline in polar coordinates
        spline = UnivariateSpline(angles_extended, radii_extended, s=len(radii) * 0.1, k=3)

        # Generate smooth contour at target resolution
        angles_smooth = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        radii_smooth = spline(angles_smooth)

        # Convert back to Cartesian coordinates
        x_smooth = center_x + radii_smooth * np.cos(angles_smooth)
        y_smooth = center_y + radii_smooth * np.sin(angles_smooth)

        # Create smooth contour in OpenCV format
        smooth_points = np.column_stack([x_smooth, y_smooth])
        smooth_contour = smooth_points.reshape(-1, 1, 2).astype(np.int32)

        # Calculate metrics
        area = cv2.contourArea(smooth_contour)
        perimeter = cv2.arcLength(smooth_contour, True)
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

        mean_radius = np.mean(radii_smooth)
        radius_std = np.std(radii_smooth)
        max_deviation = np.max(np.abs(radii_smooth - mean_radius))
        radius_variation = (np.max(radii_smooth) - np.min(radii_smooth)) / mean_radius if mean_radius > 0 else 0

        return {
            "contour": smooth_contour,
            "points": smooth_points,
            "area_px2": float(area),
            "perimeter_px": float(perimeter),
            "circularity": float(circularity),
            "mean_radius_px": float(mean_radius),
            "radius_std_px": float(radius_std),
            "max_deviation_px": float(max_deviation),
            "radius_variation": float(radius_variation),
            "radii": radii_smooth,
            "angles": angles_smooth,
            "method": "smooth_spline",
        }

    except ImportError:
        return {"error": "scipy required for smooth contour fitting"}


def _fit_fourier_contour(
    angles: np.ndarray,
    radii: np.ndarray,
    center_x: float,
    center_y: float,
    num_harmonics: int = 10,
    resolution: int = 360,
) -> dict[str, Any]:
    """Fit a Fourier series to represent the contour shape."""
    if len(radii) < num_harmonics * 2:
        return {"error": f"Need at least {num_harmonics * 2} points for Fourier fitting"}

    # Fit Fourier series: r(θ) = a0 + Σ[an*cos(nθ) + bn*sin(nθ)]
    mean_radius = np.mean(radii)

    # Calculate Fourier coefficients
    coeffs = {"a0": mean_radius}

    for n in range(1, num_harmonics + 1):
        an = np.mean(radii * np.cos(n * angles)) * 2
        bn = np.mean(radii * np.sin(n * angles)) * 2
        coeffs[f"a{n}"] = an
        coeffs[f"b{n}"] = bn

    # Generate smooth contour using Fourier series
    angles_smooth = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    radii_smooth = np.full(resolution, coeffs["a0"])

    for n in range(1, num_harmonics + 1):
        radii_smooth += coeffs[f"a{n}"] * np.cos(n * angles_smooth)
        radii_smooth += coeffs[f"b{n}"] * np.sin(n * angles_smooth)

    # Convert back to Cartesian
    x_smooth = center_x + radii_smooth * np.cos(angles_smooth)
    y_smooth = center_y + radii_smooth * np.sin(angles_smooth)

    smooth_points = np.column_stack([x_smooth, y_smooth])
    smooth_contour = smooth_points.reshape(-1, 1, 2).astype(np.int32)

    # Calculate metrics
    area = cv2.contourArea(smooth_contour)
    perimeter = cv2.arcLength(smooth_contour, True)
    circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

    # Fourier-specific metrics
    fundamental_amplitude = np.sqrt(coeffs["a1"] ** 2 + coeffs["b1"] ** 2)
    total_harmonic_power = sum(coeffs[f"a{n}"] ** 2 + coeffs[f"b{n}"] ** 2 for n in range(1, num_harmonics + 1))

    return {
        "contour": smooth_contour,
        "points": smooth_points,
        "area_px2": float(area),
        "perimeter_px": float(perimeter),
        "circularity": float(circularity),
        "mean_radius_px": float(coeffs["a0"]),
        "fundamental_amplitude": float(fundamental_amplitude),
        "total_harmonic_power": float(total_harmonic_power),
        "fourier_coefficients": coeffs,
        "radii": radii_smooth,
        "angles": angles_smooth,
        "method": "fourier_series",
    }


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
        method: Detection method ("binary", "canny", or "broken_circle")
        **kwargs: Additional parameters for detection

    Returns:
        Dictionary with detection results
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Try the new broken circle method for edge-based detection
    if method == "broken_circle" or (method == "canny" and kwargs.get("use_broken_circle", True)):
        try:
            return detect_broken_circle(gray, **kwargs)
        except RuntimeError:
            # Fall back to original canny method if broken circle fails
            if method == "broken_circle":
                print("Warning: Broken circle detection failed, falling back to standard canny")
            pass

    # Original method logic
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

    # Save debug images if requested
    if kwargs.get("save_debug", False):
        from .visualization import save_debug_images

        # Extract debug images from detection result
        debug_image = detection_result.get("debug_image")
        method = kwargs.get("method", "binary")

        if method == "binary" and debug_image is not None:
            # For binary method, debug_image is the binary mask
            save_debug_images(debug_image, None, output_dir, stem)
        elif method in ["canny", "broken_circle"] and debug_image is not None:
            # For canny/broken_circle method, debug_image is the edge image
            save_debug_images(None, debug_image, output_dir, stem)

        # Also save both if we can reconstruct them from the segmentation
        # This provides more complete debug info
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        gray = preprocess_image(gray)

        # Create a copy of kwargs without conflicting parameters for segment_image
        segment_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["binary_block_size", "binary_C", "binary_invert", "erode_iterations", "canny_low", "canny_high"]
        }

        binary_mask, edges = segment_image(gray, method=method, **segment_kwargs)
        save_debug_images(binary_mask, edges, output_dir, stem)

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

    # Optional best contour fitting
    best_contour_result = None
    if kwargs.get("fit_best_contour", False):
        best_contour_method = kwargs.get("best_contour_method", "adaptive")
        best_contour_result = fit_best_contour(
            contour,
            center_x,
            center_y,
            method=best_contour_method,
            smoothing_factor=kwargs.get("smoothing_factor", 0.1),
            num_harmonics=kwargs.get("num_harmonics", 10),
        )

        # Add mm measurements if calibration available
        if mm_per_pixel and best_contour_result and "mean_radius_px" in best_contour_result:
            best_contour_result.update(
                {
                    "mean_radius_mm": best_contour_result["mean_radius_px"] * mm_per_pixel,
                    "area_mm2": best_contour_result["area_px2"] * (mm_per_pixel**2),
                    "perimeter_mm": best_contour_result["perimeter_px"] * mm_per_pixel,
                    "radius_std_mm": best_contour_result.get("radius_std_px", 0) * mm_per_pixel,
                    "max_deviation_mm": best_contour_result.get("max_deviation_px", 0) * mm_per_pixel,
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
        "best_contour": best_contour_result,
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
            best_contour_result,
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
