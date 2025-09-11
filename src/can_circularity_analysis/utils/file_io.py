"""File I/O utilities for saving analysis results and data."""

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def save_results(
    metrics: dict[str, Any],
    points: np.ndarray,
    center_x: float,
    center_y: float,
    offset: tuple,
    mm_per_pixel: float | None,
    output_dir: str,
    stem: str,
) -> dict[str, str]:
    """Save complete analysis results to files.

    Args:
        metrics: Analysis metrics dictionary
        points: Rim points array
        center_x, center_y: Circle center coordinates
        offset: Crop offset tuple
        mm_per_pixel: Pixel-to-mm conversion factor
        output_dir: Output directory path
        stem: Output file stem

    Returns:
        Dictionary with paths to saved files
    """
    output_paths = {}

    # Save rim points as CSV
    csv_path = save_rim_points_csv(points, center_x, center_y, offset, mm_per_pixel, output_dir, stem)
    output_paths["csv_points"] = csv_path
    metrics["csv_points"] = csv_path

    # Save metrics as JSON
    json_path = save_metrics_json(metrics, output_dir, stem)
    output_paths["metrics_json"] = json_path

    return output_paths


def save_rim_points_csv(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    offset: tuple,
    mm_per_pixel: float | None,
    output_dir: str,
    stem: str,
) -> str:
    """Save rim points to CSV file with Cartesian and cylindrical coordinates.

    Args:
        points: Array of rim points
        center_x, center_y: Circle center coordinates
        offset: Crop offset tuple
        mm_per_pixel: Pixel-to-mm conversion factor
        output_dir: Output directory path
        stem: Output file stem

    Returns:
        Absolute path to saved CSV file
    """
    # Import here to avoid circular imports - FIXED IMPORT PATH
    from can_circularity_analysis.core.metrics import calculate_cylindrical_coordinates

    rows = calculate_cylindrical_coordinates(points, center_x, center_y, offset, mm_per_pixel)

    csv_path = os.path.join(output_dir, f"{stem}_points.csv")

    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return os.path.abspath(csv_path)


def save_metrics_json(metrics: dict[str, Any], output_dir: str, stem: str) -> str:
    """Save analysis metrics to JSON file.

    Args:
        metrics: Dictionary of analysis metrics
        output_dir: Output directory path
        stem: Output file stem

    Returns:
        Absolute path to saved JSON file
    """
    json_path = os.path.join(output_dir, f"{stem}_metrics.json")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_serializer)

    return os.path.abspath(json_path)


def load_metrics_json(file_path: str) -> dict[str, Any]:
    """Load analysis metrics from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary of analysis metrics

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(file_path) as f:
        return json.load(f)


def save_batch_summary(
    results_list: list[dict[str, Any]], output_dir: str, filename: str = "batch_summary.json"
) -> str:
    """Save summary of batch analysis results.

    Args:
        results_list: List of analysis result dictionaries
        output_dir: Output directory path
        filename: Output filename

    Returns:
        Absolute path to saved summary file
    """
    summary = {
        "total_images": len(results_list),
        "successful_analyses": len([r for r in results_list if not r.get("error")]),
        "failed_analyses": len([r for r in results_list if r.get("error")]),
        "results": results_list,
    }

    summary_path = os.path.join(output_dir, filename)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_serializer)

    return os.path.abspath(summary_path)


def find_metrics_files(directory: str, pattern: str = "*_metrics.json") -> list[Path]:
    """Find all metrics JSON files in a directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern for metrics files

    Returns:
        List of Path objects for found metrics files
    """
    search_dir = Path(directory)
    return sorted(search_dir.glob(pattern))


def create_output_directory(output_dir: str, exist_ok: bool = True) -> str:
    """Create output directory if it doesn't exist.

    Args:
        output_dir: Path to output directory
        exist_ok: Whether to ignore if directory already exists

    Returns:
        Absolute path to output directory

    Raises:
        OSError: If directory creation fails
    """
    os.makedirs(output_dir, exist_ok=exist_ok)
    return os.path.abspath(output_dir)


def validate_image_file(file_path: str) -> bool:
    """Validate that file exists and is a supported image format.

    Args:
        file_path: Path to image file

    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.isfile(file_path):
        return False

    # Check file extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    file_ext = Path(file_path).suffix.lower()

    return file_ext in valid_extensions


def get_image_files(directory: str, recursive: bool = False) -> list[str]:
    """Get list of image files in directory.

    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories

    Returns:
        List of image file paths
    """
    search_dir = Path(directory)

    if not search_dir.exists():
        return []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []

    if recursive:
        for ext in valid_extensions:
            image_files.extend(search_dir.rglob(f"*{ext}"))
            image_files.extend(search_dir.rglob(f"*{ext.upper()}"))
    else:
        for ext in valid_extensions:
            image_files.extend(search_dir.glob(f"*{ext}"))
            image_files.extend(search_dir.glob(f"*{ext.upper()}"))

    return sorted([str(f) for f in image_files])


def _json_serializer(obj):
    """Custom JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def ensure_file_extension(filename: str, extension: str) -> str:
    """Ensure filename has the specified extension.

    Args:
        filename: Input filename
        extension: Required extension (with or without leading dot)

    Returns:
        Filename with correct extension
    """
    if not extension.startswith("."):
        extension = "." + extension

    if not filename.lower().endswith(extension.lower()):
        filename += extension

    return filename


def backup_existing_file(file_path: str) -> str | None:
    """Create backup of existing file if it exists.

    Args:
        file_path: Path to file that might be overwritten

    Returns:
        Path to backup file if created, None if original didn't exist
    """
    if not os.path.exists(file_path):
        return None

    path = Path(file_path)
    backup_path = path.with_suffix(f"{path.suffix}.backup")

    counter = 1
    while backup_path.exists():
        backup_path = path.with_suffix(f"{path.suffix}.backup{counter}")
        counter += 1

    os.rename(file_path, backup_path)
    return str(backup_path)
