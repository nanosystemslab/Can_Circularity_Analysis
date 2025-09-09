"""Can Circularity Analysis

A computer vision tool for analyzing the circularity and rim quality of cylindrical objects (cans).
Provides both CLI tools and a Python API for programmatic use.
"""

__version__ = "0.1.0"
__author__ = "Can Circularity Analysis Team"

# Import main API functions for convenience
from .core.detection import analyze_image_file, detect_circular_contour
from .core.calibration import calibrate_pixels_per_mm, load_calibration, save_calibration
from .core.metrics import calculate_circularity_metrics, calculate_deviation_metrics

__all__ = [
    "analyze_image_file",
    "detect_circular_contour", 
    "calibrate_pixels_per_mm",
    "load_calibration",
    "save_calibration",
    "calculate_circularity_metrics",
    "calculate_deviation_metrics",
]
