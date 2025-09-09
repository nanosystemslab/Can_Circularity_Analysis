"""Metrics calculation for circularity and deviation analysis."""

import math
from typing import Optional, Dict, Any

import cv2
import numpy as np


def calculate_circularity_metrics(
    contour: np.ndarray,
    center_x: float,
    center_y: float, 
    radius: float,
    mm_per_pixel: Optional[float] = None
) -> Dict[str, Any]:
    """Calculate basic circularity metrics from contour.
    
    Args:
        contour: OpenCV contour array
        center_x: Circle center x-coordinate
        center_y: Circle center y-coordinate
        radius: Circle radius in pixels
        mm_per_pixel: Conversion factor for mm measurements
        
    Returns:
        Dictionary with circularity metrics
    """
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    
    # Circularity: 4π×Area / Perimeter²
    circularity = (4.0 * math.pi * area) / (perimeter * perimeter + 1e-12) if perimeter > 0 else 0.0
    
    metrics = {
        "area_px2": area,
        "perimeter_px": perimeter,
        "circularity_4piA_P2": float(circularity),
    }
    
    # Add mm measurements if calibration available
    if mm_per_pixel:
        metrics.update({
            "area_mm2": area * (mm_per_pixel ** 2),
            "perimeter_mm": perimeter * mm_per_pixel,
        })
    
    return metrics


def calculate_deviation_metrics(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    mm_per_pixel: Optional[float] = None
) -> Dict[str, Any]:
    """Calculate radial deviation metrics from rim points.
    
    Args:
        points: Array of rim points [(x, y), ...]
        center_x: Circle center x-coordinate
        center_y: Circle center y-coordinate
        radius: Circle radius in pixels
        mm_per_pixel: Conversion factor for mm measurements
        
    Returns:
        Dictionary with deviation metrics
    """
    # Calculate radial distances
    radii = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
    deviations = radii - radius  # Signed deviation from fitted circle
    
    # Calculate statistics in pixels
    rms_px = float(np.sqrt(np.mean(deviations**2)))
    max_abs_px = float(np.max(np.abs(deviations)))
    std_px = float(np.std(deviations))
    range_px = float(np.max(deviations) - np.min(deviations))
    
    metrics = {
        "rms_out_of_round_px": rms_px,
        "max_out_of_round_px": max_abs_px,
        "std_out_of_round_px": std_px,
        "range_out_of_round_px": range_px,
    }
    
    # Add mm measurements if calibration available
    if mm_per_pixel:
        metrics.update({
            "rms_out_of_round_mm": rms_px * mm_per_pixel,
            "max_out_of_round_mm": max_abs_px * mm_per_pixel,
            "std_out_of_round_mm": std_px * mm_per_pixel,
            "range_out_of_round_mm": range_px * mm_per_pixel,
        })
    
    return metrics


def calculate_cylindrical_coordinates(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    offset: tuple = (0, 0),
    mm_per_pixel: Optional[float] = None
) -> list:
    """Convert rim points to cylindrical coordinates for export.
    
    Args:
        points: Array of rim points [(x, y), ...]
        center_x: Circle center x-coordinate
        center_y: Circle center y-coordinate
        offset: Crop offset (x_offset, y_offset)
        mm_per_pixel: Conversion factor for mm measurements
        
    Returns:
        List of dictionaries with point coordinates
    """
    # Calculate cylindrical coordinates
    dx = points[:, 0] - center_x
    dy = points[:, 1] - center_y
    radii = np.sqrt(dx**2 + dy**2)
    theta_rad = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta_rad)
    
    rows = []
    for i in range(len(points)):
        x_px = float(points[i, 0])
        y_px = float(points[i, 1])
        
        row = {
            "x_px": x_px + offset[0],
            "y_px": y_px + offset[1],
            "theta_rad": float(theta_rad[i]),
            "theta_deg": float(theta_deg[i]),
            "r_px": float(radii[i]),
            "x_center_px": center_x + offset[0],
            "y_center_px": center_y + offset[1],
        }
        
        # Add mm measurements if calibration available
        if mm_per_pixel:
            row.update({
                "x_mm": (x_px + offset[0]) * mm_per_pixel,
                "y_mm": (y_px + offset[1]) * mm_per_pixel,
                "r_mm": float(radii[i] * mm_per_pixel),
            })
            
        rows.append(row)
    
    return rows


def validate_diameter_range(
    radius: float,
    mm_per_pixel: Optional[float],
    min_diameter_mm: float = 0.0,
    max_diameter_mm: float = 1e9
) -> None:
    """Validate that detected diameter falls within specified range.
    
    Args:
        radius: Detected radius in pixels
        mm_per_pixel: Conversion factor
        min_diameter_mm: Minimum allowed diameter in mm
        max_diameter_mm: Maximum allowed diameter in mm
        
    Raises:
        RuntimeError: If diameter is outside allowed range
    """
    if mm_per_pixel is not None:
        diameter_mm = 2.0 * radius * mm_per_pixel
        if not (min_diameter_mm <= diameter_mm <= max_diameter_mm):
            raise RuntimeError(
                f"Detected rim diameter {diameter_mm:.2f} mm is outside "
                f"allowed range [{min_diameter_mm}, {max_diameter_mm}] mm"
            )
