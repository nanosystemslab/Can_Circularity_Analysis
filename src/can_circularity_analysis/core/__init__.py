"""Core analysis modules for can circularity analysis."""

from .calibration import (
    calibrate_from_points,
    calibrate_pixels_per_mm,
    load_calibration,
    parse_crop_string,
    save_calibration,
)
from .detection import (
    analyze_image_file,
    detect_circular_contour,
    fit_circle_to_contour,
    fit_ellipse_to_contour,
    select_best_contour,
)
from .metrics import (
    calculate_circularity_metrics,
    calculate_cylindrical_coordinates,
    calculate_deviation_metrics,
    validate_diameter_range,
)
from .visualization import create_analysis_overlay, draw_scale_reference, save_debug_images

# Conditionally import reconstruction if dependencies available
try:
    from .reconstruction import CanReconstructionError, CanSTEPGenerator, create_can_step_file

    RECONSTRUCTION_AVAILABLE = True
    __all_reconstruction__ = [
        "create_can_step_file",
        "CanSTEPGenerator",
        "CanReconstructionError",
    ]
except ImportError:
    RECONSTRUCTION_AVAILABLE = False
    __all_reconstruction__ = []

__all__ = [
    # Calibration
    "calibrate_pixels_per_mm",
    "calibrate_from_points",
    "load_calibration",
    "save_calibration",
    "parse_crop_string",
    # Detection
    "analyze_image_file",
    "detect_circular_contour",
    "fit_circle_to_contour",
    "fit_ellipse_to_contour",
    "select_best_contour",
    # Metrics
    "calculate_circularity_metrics",
    "calculate_deviation_metrics",
    "calculate_cylindrical_coordinates",
    "validate_diameter_range",
    # Visualization
    "create_analysis_overlay",
    "draw_scale_reference",
    "save_debug_images",
    # Reconstruction (if available)
    *__all_reconstruction__,
]
