"""Core analysis modules for can circularity analysis."""

from .calibration import (
    calibrate_pixels_per_mm,
    calibrate_from_points, 
    load_calibration,
    save_calibration,
    parse_crop_string
)

from .detection import (
    analyze_image_file,
    detect_circular_contour,
    fit_circle_to_contour,
    fit_ellipse_to_contour,
    select_best_contour
)

from .metrics import (
    calculate_circularity_metrics,
    calculate_deviation_metrics,
    calculate_cylindrical_coordinates,
    validate_diameter_range
)

from .visualization import (
    create_analysis_overlay,
    draw_scale_reference,
    save_debug_images
)

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
]
