"""Can Circularity Analysis

A computer vision tool for analyzing the circularity and rim quality of cylindrical objects (cans).
Provides both CLI tools and a Python API for programmatic use, including 3D reconstruction capabilities.
"""

__version__ = "0.2.0"
__author__ = "Can Circularity Analysis Team"

# Import main API functions for convenience
from .core.calibration import calibrate_pixels_per_mm, load_calibration, save_calibration
from .core.detection import analyze_image_file, detect_circular_contour
from .core.metrics import calculate_circularity_metrics, calculate_deviation_metrics

# Conditionally import reconstruction functionality
try:
    from .core.reconstruction import create_can_step_file

    RECONSTRUCTION_AVAILABLE = True
    __all_reconstruction__ = ["create_can_step_file"]
except ImportError:
    RECONSTRUCTION_AVAILABLE = False
    __all_reconstruction__ = []

__all__ = [
    "analyze_image_file",
    "detect_circular_contour",
    "calibrate_pixels_per_mm",
    "load_calibration",
    "save_calibration",
    "calculate_circularity_metrics",
    "calculate_deviation_metrics",
    *__all_reconstruction__,
]
