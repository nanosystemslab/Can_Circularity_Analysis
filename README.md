# Can Circularity Analysis

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/nanosystemslab/Can_Circularity_Analysis)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/nanosystemslab/Can_Circularity_Analysis)
[![License](https://img.shields.io/badge/license-GPL--3.0-green)](LICENSE)
[![Poetry](https://img.shields.io/badge/dependency--management-poetry-blue)](https://python-poetry.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/badge/linter-ruff-red)](https://github.com/astral-sh/ruff)

![Analysis Types](https://img.shields.io/badge/Analysis-Circularity%20%7C%20Diameter%20%7C%20Ovality-blue)
![Detection Methods](https://img.shields.io/badge/Detection-Binary%20%7C%20Canny%20%7C%20Hough-orange)
![Measurements](https://img.shields.io/badge/Measures-RMS%20%7C%20Ellipticity%20%7C%20Deviation-green)
![3D Reconstruction](https://img.shields.io/badge/3D-Reconstruction%20%7C%20STL%20%7C%20OBJ%20%7C%20PLY-purple)
![Interfaces](https://img.shields.io/badge/Interface-CLI%20%7C%20Python%20API-red)
![Output Types](https://img.shields.io/badge/Outputs-JSON%20%7C%20CSV%20%7C%20Plots%20%7C%203D%20Models-purple)

A computer vision tool for automated analysis of can rim circularity and quality control in manufacturing. This package provides both command-line utilities and a Python API for measuring circular deviation, diameter accuracy, and out-of-round characteristics of cylindrical objects, with advanced 3D reconstruction capabilities for generating mesh models from top and bottom rim profiles.

## Features

- **Automated rim detection** using binary segmentation or Canny edge detection
- **Pixel-to-millimeter calibration** with interactive or programmatic setup
- **Diameter filtering** to focus analysis on specific size ranges
- **Comprehensive circularity metrics** including area, perimeter, and deviation analysis
- **Out-of-round quantification** with RMS, standard deviation, and range statistics
- **Ellipse fitting** for ovalization and eccentricity analysis
- **3D reconstruction** from top and bottom rim profiles with configurable profile types
- **Multiple mesh formats** (STL, OBJ, PLY) with proper wall thickness implementation
- **Batch processing** for analyzing multiple images efficiently
- **Statistical summaries** with visualization plots
- **Export capabilities** for rim point coordinates and analysis results

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Calibration](#calibration)
- [Analysis Workflow](#analysis-workflow)
- [3D Reconstruction](#3d-reconstruction)
- [Batch Processing](#batch-processing)
- [Output Files](#output-files)
- [Command Reference](#command-reference)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Installation

### Requirements

- Python 3.11 or higher
- Poetry (recommended) or pip

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/nanosystemslab/Can_Circularity_Analysis.git
cd Can_Circularity_Analysis

# Install dependencies
poetry install

# For 3D reconstruction capabilities
poetry install --extras "reconstruction"

# Verify installation
poetry run can-analyze --help
poetry run can-reconstruct --help
```

### Using pip

```bash
git clone https://github.com/nanosystemslab/Can_Circularity_Analysis.git
cd Can_Circularity_Analysis
pip install -e .

# For 3D reconstruction
pip install -e ".[reconstruction]"
```

### Dependencies

Core dependencies are automatically installed:
- `opencv-python` - Computer vision and image processing
- `scikit-image` - Circle fitting algorithms
- `numpy` - Numerical computations
- `pandas` - Data analysis (for summarization)
- `matplotlib` - Plotting and visualization

Additional dependencies for 3D reconstruction:
- `trimesh` - 3D mesh processing and export
- `scipy` - Advanced interpolation and scientific computing

## Quick Start

### 1. Basic Analysis

```bash
# Analyze a single image
poetry run can-analyze image.jpg --out-dir results

# With known pixel-to-mm ratio
poetry run can-analyze image.jpg --pixels-per-mm 10.5 --out-dir results
```

### 2. Interactive Calibration

```bash
# Click two points on a ruler to calibrate
poetry run can-analyze image.jpg --click-scale --known-mm 50.0 --out-dir results
```

### 3. 3D Reconstruction

```bash
# Create 3D model from top and bottom rim analysis
poetry run can-reconstruct \
  results/can_top_metrics.json \
  results/can_bottom_metrics.json \
  models/can_model.stl \
  --top-profile ellipse \
  --bottom-profile ellipse \
  --wall-thickness 0.1 \
  --height 100.0

# Preview profiles before reconstruction
poetry run can-reconstruct \
  results/can_top_metrics.json \
  results/can_bottom_metrics.json \
  models/can_model.obj \
  --plot-profiles
```

### 4. Batch Processing

```bash
# Process multiple images
poetry run can-analyze data/*.jpg --pixels-per-mm 10.5 --out-dir batch_results

# Generate summary report
poetry run can-summarize --in-dir batch_results --make-plots --out-dir reports
```

## Calibration

Accurate pixel-to-millimeter calibration is essential for meaningful measurements. The tool supports several calibration methods:

### Method 1: Interactive Calibration

```bash
poetry run can-analyze image.jpg --click-scale --known-mm 50.0
```

This opens a GUI where you click two points on a ruler that are exactly 50mm apart.

### Method 2: Manual Coordinate Entry

```bash
poetry run can-analyze image.jpg --scale-pts "100,200,150,200" --known-mm 50.0
```

Provide pixel coordinates (x1,y1,x2,y2) of two points with known distance.

### Method 3: Saved Calibration

```bash
# Save calibration for reuse
poetry run can-analyze image1.jpg --click-scale --save-scale calibration.json

# Reuse calibration
poetry run can-analyze image2.jpg --scale-from calibration.json
```

### Method 4: Known Ratio

```bash
poetry run can-analyze image.jpg --pixels-per-mm 10.5
```

Use a pre-determined pixels-per-millimeter ratio.

## Analysis Workflow

The recommended workflow follows a systematic approach: **calibrate once, process in batches, fix individual issues, then summarize results**.

### Step 1: Create Calibration for Each Image Set

Before processing images, establish accurate pixel-to-millimeter calibration for each imaging setup or can size.

#### Interactive Calibration (Recommended)

```bash
# Create calibration for each can size dataset
poetry run can-analyze data/60mm_topDown_ovality/sample_image.jpg \
  --click-scale \
  --known-mm 50.0 \
  --save-scale scales/ppmm_60mm.json

poetry run can-analyze data/120mm_topDown_ovality/sample_image.jpg \
  --click-scale \
  --known-mm 50.0 \
  --save-scale scales/ppmm_120mm.json
```

#### Manual Calibration (Headless Systems)

```bash
# If you know the pixel coordinates of two points 50mm apart
poetry run can-analyze data/60mm_topDown_ovality/sample_image.jpg \
  --scale-pts "100,200,150,200" \
  --known-mm 50.0 \
  --save-scale scales/ppmm_60mm.json
```

#### Verify Calibration

```bash
# Test calibration on a few images to ensure accuracy
poetry run can-analyze data/60mm_topDown_ovality/test_image.jpg \
  --scale-from scales/ppmm_60mm.json \
  --min-diameter-mm 58 \
  --max-diameter-mm 62
```

### Step 2: Batch Processing

Once calibration is established, process entire directories efficiently.

#### Process by Can Size

```bash
# 60mm cans with ellipse fitting for reconstruction
poetry run can-analyze data/60mm_topDown_ovality/*.jpg \
  --scale-from scales/ppmm_60mm.json \
  --method canny \
  --canny-low 50 --canny-high 150 \
  --prefer-center \
  --min-diameter-mm 58 --max-diameter-mm 62 \
  --fit-ellipse \
  --crop 0,750,99999,99999 \
  --out-dir ./results/60mm/

# 120mm cans
poetry run can-analyze data/120mm_topDown_ovality/*.jpg \
  --scale-from scales/ppmm_120mm.json \
  --method canny \
  --canny-low 50 --canny-high 150 \
  --prefer-center \
  --min-diameter-mm 115 --max-diameter-mm 125 \
  --fit-ellipse \
  --crop 0,750,99999,99999 \
  --out-dir ./results/120mm/
```

### Step 3: 3D Reconstruction (Optional)

Generate 3D models from top and bottom rim analyses for visualization, simulation, or manufacturing applications.

#### Single Can Reconstruction

```bash
# Reconstruct with elliptical profiles (recommended for oval cans)
poetry run can-reconstruct \
  results/60mm/can_sample_top_metrics.json \
  results/60mm/can_sample_bottom_metrics.json \
  models/can_sample_ellipse.stl \
  --top-profile ellipse \
  --bottom-profile ellipse \
  --wall-thickness 0.1 \
  --height 100.0

# Reconstruct with circular profiles
poetry run can-reconstruct \
  results/60mm/can_sample_top_metrics.json \
  results/60mm/can_sample_bottom_metrics.json \
  models/can_sample_circle.obj \
  --top-profile circle \
  --bottom-profile circle \
  --wall-thickness 0.15 \
  --height 120.0

# Use actual measured points for maximum accuracy
poetry run can-reconstruct \
  results/60mm/can_sample_top_metrics.json \
  results/60mm/can_sample_bottom_metrics.json \
  models/can_sample_measured.ply \
  --top-profile measured \
  --bottom-profile measured \
  --wall-thickness 0.1 \
  --height 100.0
```

#### Preview Before Reconstruction

```bash
# Show 3D plot of profiles before creating mesh
poetry run can-reconstruct \
  results/60mm/can_sample_top_metrics.json \
  results/60mm/can_sample_bottom_metrics.json \
  models/can_sample.stl \
  --plot-profiles \
  --wall-thickness 0.1 \
  --height 100.0
```

#### Batch Reconstruction

```bash
# Reconstruct all matching pairs in directories
poetry run can-reconstruct \
  --batch results/60mm_tops/ results/60mm_bottoms/ \
  --out-dir models/60mm/ \
  --top-profile ellipse \
  --bottom-profile ellipse \
  --wall-thickness 0.1 \
  --height 100.0
```

### Step 4: Handle Failed Detections

After batch processing, identify and fix individual problematic images.

#### Identify Failed Analyses

```bash
# Check batch summary for failures
cat results/60mm/batch_summary.json | grep -A 5 "failed_analyses"

# Find specific failed files
grep -l "error" results/60mm/*_metrics.json
```

#### Interactive Problem Solving

For images that failed or produced questionable results:

```bash
# Re-analyze problematic images with different parameters
poetry run can-analyze data/60mm_topDown_ovality/problematic_image.jpg \
  --scale-from scales/ppmm_60mm.json \
  --method binary \
  --binary-block-size 71 \
  --save-debug \
  --out-dir ./results/60mm_fixed/

# Try alternative detection method
poetry run can-analyze data/60mm_topDown_ovality/problematic_image.jpg \
  --scale-from scales/ppmm_60mm.json \
  --method canny \
  --canny-low 30 --canny-high 100 \
  --save-debug \
  --out-dir ./results/60mm_fixed/

# Manual region selection if needed
poetry run can-analyze data/60mm_topDown_ovality/problematic_image.jpg \
  --scale-from scales/ppmm_60mm.json \
  --crop 200,200,600,600 \
  --prefer-center \
  --out-dir ./results/60mm_fixed/
```

### Step 5: Generate Comprehensive Summaries

Once all images are successfully processed, create summary reports and visualizations.

#### Individual Size Summaries

```bash
# Generate summary for each can size
poetry run can-summarize \
  --in-dir ./results/60mm/ \
  --out-dir ./reports/60mm/ \
  --make-plots

poetry run can-summarize \
  --in-dir ./results/120mm/ \
  --out-dir ./reports/120mm/ \
  --make-plots
```

## 3D Reconstruction

The 3D reconstruction feature allows you to create detailed mesh models of cans from top and bottom rim analyses. This is useful for visualization, manufacturing simulation, quality control documentation, and digital twin applications.

### Profile Types

Choose the appropriate profile type based on your analysis needs:

#### Ellipse Profile (Recommended for Oval Cans)
```bash
poetry run can-reconstruct top.json bottom.json model.stl \
  --top-profile ellipse \
  --bottom-profile ellipse
```
- Uses fitted ellipse data from analysis
- Captures ovality and eccentricity
- Best for deformed or intentionally oval cans

#### Circle Profile
```bash
poetry run can-reconstruct top.json bottom.json model.stl \
  --top-profile circle \
  --bottom-profile circle
```
- Uses fitted circle data from analysis
- Creates perfectly circular cross-sections
- Good for comparing against ideal geometry

#### Measured Profile (Highest Accuracy)
```bash
poetry run can-reconstruct top.json bottom.json model.stl \
  --top-profile measured \
  --bottom-profile measured
```
- Uses actual measured rim points from CSV files
- Captures all irregularities and asymmetries
- Most accurate representation of actual can shape

### Wall Thickness Implementation

The wall thickness parameter controls how the 3D model represents the material:

```bash
# Thin wall (0.1mm) - typical for aluminum cans
poetry run can-reconstruct top.json bottom.json thin_can.stl \
  --wall-thickness 0.1

# Thick wall (0.5mm) - for steel cans or simulation
poetry run can-reconstruct top.json bottom.json thick_can.stl \
  --wall-thickness 0.5
```

**How it works:**
- Your analyzed profile becomes the **centerline** of the wall
- Inner surface: centerline - (thickness/2)
- Outer surface: centerline + (thickness/2)
- Creates proper hollow geometry with realistic wall structure

### Output Formats

Multiple 3D formats are supported:

```bash
# STL format (most common for 3D printing)
poetry run can-reconstruct top.json bottom.json model.stl

# OBJ format (good for CAD software)
poetry run can-reconstruct top.json bottom.json model.obj

# PLY format (preserves more metadata)
poetry run can-reconstruct top.json bottom.json model.ply

# Auto-detect from extension
poetry run can-reconstruct top.json bottom.json model.stl \
  --output-format auto
```

### Advanced Options

#### Profile Alignment
```bash
# Align ellipse orientations for straight walls (default)
poetry run can-reconstruct top.json bottom.json aligned.stl \
  --align-profiles

# Preserve individual orientations (may create twisted walls)
poetry run can-reconstruct top.json bottom.json twisted.stl \
  --no-align-profiles
```

#### Mesh Resolution
```bash
# High resolution (128 points around profile)
poetry run can-reconstruct top.json bottom.json hires.stl \
  --resolution 128

# Low resolution (32 points) for faster processing
poetry run can-reconstruct top.json bottom.json lowres.stl \
  --resolution 32
```

#### Visualization and Debugging
```bash
# Preview profiles in 3D before reconstruction
poetry run can-reconstruct top.json bottom.json model.stl \
  --plot-profiles

# Check profile alignment and wall thickness visually
poetry run can-reconstruct top.json bottom.json debug.stl \
  --plot-profiles \
  --wall-thickness 0.2 \
  --height 150.0
```

### Batch Reconstruction

Process multiple can pairs automatically:

```bash
# Reconstruct all matching pairs
poetry run can-reconstruct \
  --batch analysis/tops/ analysis/bottoms/ \
  --out-dir models/ \
  --top-profile ellipse \
  --bottom-profile ellipse \
  --wall-thickness 0.1 \
  --height 120.0

# Mixed profile types
poetry run can-reconstruct \
  --batch analysis/tops/ analysis/bottoms/ \
  --out-dir models/mixed/ \
  --top-profile ellipse \
  --bottom-profile circle \
  --wall-thickness 0.15
```

### Applications

#### Quality Control Documentation
```bash
# Create models showing actual vs. ideal geometry
poetry run can-reconstruct top.json bottom.json actual.stl \
  --top-profile measured --bottom-profile measured

poetry run can-reconstruct top.json bottom.json ideal.stl \
  --top-profile circle --bottom-profile circle
```

#### Manufacturing Simulation
```bash
# Generate models for FEA or CFD analysis
poetry run can-reconstruct top.json bottom.json simulation.obj \
  --top-profile ellipse \
  --bottom-profile ellipse \
  --wall-thickness 0.12 \
  --resolution 128
```

#### 3D Printing and Prototyping
```bash
# Create physical prototypes
poetry run can-reconstruct top.json bottom.json prototype.stl \
  --wall-thickness 2.0 \
  --resolution 64
```

### Metadata Files

Each reconstruction generates a metadata file with complete parameters:

```json
{
  "output_file": "can_model.stl",
  "reconstruction_params": {
    "wall_thickness_mm": 0.1,
    "height_mm": 100.0,
    "top_profile_type": "ellipse",
    "bottom_profile_type": "ellipse",
    "reconstruction_method": "enhanced_walled_loft"
  },
  "top_rim_data": {
    "diameter_mm": 59.84,
    "ellipse_data": {...}
  },
  "bottom_rim_data": {
    "diameter_mm": 59.78,
    "ellipse_data": {...}
  }
}
```

## Batch Processing

### Processing Multiple Can Sizes

```bash
# Process different can sizes with appropriate settings
for size in 40mm 60mm 80mm 100mm 120mm; do
  poetry run can-analyze data/${size}_*.jpg \
    --scale-from scales/ppmm_${size}.json \
    --min-diameter-mm $((${size%mm} - 5)) \
    --max-diameter-mm $((${size%mm} + 5)) \
    --method canny \
    --canny-low 50 --canny-high 150 \
    --prefer-center \
    --fit-ellipse \
    --out-dir results/${size}/
done
```

### Automated Quality Control Pipeline

```bash
#!/bin/bash
# quality_control.sh

# Process images
poetry run can-analyze production_images/*.jpg \
  --scale-from calibration/line1.json \
  --method canny \
  --min-diameter-mm 58 \
  --max-diameter-mm 62 \
  --fit-ellipse \
  --out-dir qc_results/

# Generate summary report
poetry run can-summarize \
  --in-dir qc_results/ \
  --out-dir qc_reports/ \
  --make-plots

# Create 3D models for documentation
poetry run can-reconstruct \
  --batch qc_results/tops/ qc_results/bottoms/ \
  --out-dir qc_models/ \
  --top-profile ellipse \
  --bottom-profile ellipse

echo "Quality control analysis complete. Check qc_reports/ for results and qc_models/ for 3D models."
```

## Output Files

### Per-Image Results

Each analyzed image generates:

#### `*_metrics.json`
Complete analysis results including:
```json
{
  "diameter_mm": 59.84,
  "circularity_4piA_P2": 0.987,
  "rms_out_of_round_mm": 0.156,
  "center_mm": [125.4, 98.7],
  "ellipse": {
    "ellipticity": 1.023,
    "ovalization_mm": 0.089,
    "major_mm": 60.12,
    "minor_mm": 59.87,
    "angle_deg": 45.2
  }
}
```

#### `*_points.csv`
Rim point coordinates in both Cartesian and cylindrical formats:
```csv
x_px,y_px,x_mm,y_mm,theta_deg,r_mm
245.2,178.9,23.35,17.04,36.8,29.94
```

#### `*_overlay.png`
Visualization showing:
- Original image with detected rim
- Fitted circle (green)
- Individual rim points (red dots)
- Fitted ellipse (purple, if enabled)
- Measurement annotations

### 3D Reconstruction Results

#### Mesh Files
- `*.stl` - STL format for 3D printing and CAD
- `*.obj` - OBJ format for graphics and simulation
- `*.ply` - PLY format with extended metadata

#### Metadata Files
- `*_metadata.json` - Complete reconstruction parameters and source data

### Batch Results

#### `batch_summary.json`
Summary of all analyses:
```json
{
  "total_images": 32,
  "successful_analyses": 31,
  "failed_analyses": 1,
  "results": [...]
}
```

### Summary Reports

Generated by `can-summarize --make-plots`:

#### Statistical Plots
- `diameter_histogram.png` - Diameter distribution
- `rms_histogram.png` - Out-of-round distribution
- `circularity_histogram.png` - Circularity distribution
- `rim_overlay.png` - Normalized rim shape overlay

#### Data Tables
- `metrics_all.csv` - All results in spreadsheet format
- `stats_overall.csv` - Statistical summaries

## Command Reference

### can-analyze

```bash
poetry run can-analyze [OPTIONS] IMAGE [IMAGE...]

Main options:
  --out-dir TEXT              Output directory (default: out)
  --method [binary|canny]     Detection method (default: binary)
  --pixels-per-mm FLOAT       Known pixels-per-mm conversion
  --scale-from TEXT           Load calibration from JSON file
  --click-scale               Interactive calibration (requires GUI)
  --scale-pts TEXT            Manual calibration points 'x1,y1,x2,y2'
  --known-mm FLOAT            Distance for calibration (default: 50.0)
  --min-diameter-mm FLOAT     Minimum allowed diameter (default: 0.0)
  --max-diameter-mm FLOAT     Maximum allowed diameter (default: inf)
  --crop TEXT                 Crop region 'x,y,w,h'
  --fit-ellipse               Include ellipse analysis
  --save-debug                Save intermediate processing images
  --prefer-center             Prefer contours near image center
  --no-overlay                Skip overlay image generation

Binary segmentation:
  --binary-block-size INT     Adaptive threshold block size (default: 51)
  --binary-C INT              Adaptive threshold constant (default: 2)
  --binary-invert             Invert binary threshold

Canny edge detection:
  --canny-low INT             Lower threshold (default: 75)
  --canny-high INT            Upper threshold (default: 200)
```

### can-reconstruct

```bash
poetry run can-reconstruct [OPTIONS] TOP_JSON BOTTOM_JSON OUTPUT_FILE

Profile Selection:
  --top-profile [circle|ellipse|measured]     Top rim profile type (default: ellipse)
  --bottom-profile [circle|ellipse|measured] Bottom rim profile type (default: ellipse)

Reconstruction Parameters:
  --wall-thickness FLOAT      Wall thickness in mm (default: 0.1)
  --height FLOAT              Height between profiles in mm (default: 100.0)
  --resolution INT            Points around each profile (default: 64)

Output Options:
  --output-format [stl|obj|ply|auto]  Output format (default: auto)
  --plot-profiles             Show 3D plot before meshing
  --align-profiles            Align orientations for straight walls (default: True)
  --no-align-profiles         Preserve individual orientations

Batch Processing:
  --batch TOP_DIR BOTTOM_DIR  Batch process matching files
  --out-dir TEXT              Output directory for batch (default: models)
```

### can-summarize

```bash
poetry run can-summarize [OPTIONS]

Options:
  --in-dir TEXT               Input directory with *_metrics.json files
  --out-dir TEXT              Output directory for reports
  --make-plots                Generate visualization plots
```

## Python API

### Basic Usage

```python
from can_circularity_analysis import analyze_image_file

# Analyze single image
results = analyze_image_file(
    "can_image.jpg",
    output_dir="results",
    pixels_per_mm=10.5,
    min_diameter_mm=85,
    max_diameter_mm=95,
    fit_ellipse=True
)

print(f"Diameter: {results['diameter_mm']:.2f} mm")
print(f"Circularity: {results['circularity_4piA_P2']:.3f}")
print(f"Ellipticity: {results['ellipse']['ellipticity']:.3f}")
```

### 3D Reconstruction API

```python
from can_circularity_analysis.core.reconstruction import create_can_mesh_file

# Create 3D model from analysis results
mesh_path = create_can_mesh_file(
    top_json="results/can_top_metrics.json",
    bottom_json="results/can_bottom_metrics.json",
    output_path="models/can_model.stl",
    wall_thickness=0.1,
    height=100.0,
    top_profile_type="ellipse",
    bottom_profile_type="ellipse",
    plot_profiles=True
)

print(f"3D model created: {mesh_path}")
```

### Advanced Usage

```python
import cv2
from can_circularity_analysis.core import (
    detect_circular_contour,
    calculate_circularity_metrics,
    calibrate_pixels_per_mm
)

# Load and preprocess image
image = cv2.imread("image.jpg")

# Detect rim
result = detect_circular_contour(
    image,
    method="canny",
    canny_low=50,
    canny_high=150,
    prefer_center=True
)

# Extract results
contour = result["contour"]
center_x, center_y, radius = result["circle_params"]
points = result["points"]

# Calculate metrics
metrics = calculate_circularity_metrics(contour, center_x, center_y, radius)
print(f"Circularity: {metrics['circularity_4piA_P2']:.3f}")
```

### Batch Processing with 3D Reconstruction

```python
from pathlib import Path
from can_circularity_analysis import analyze_image_file
from can_circularity_analysis.core.reconstruction import create_can_mesh_file
from can_circularity_analysis.utils.file_io import save_batch_summary

# Process images
results = []
image_dir = Path("production_images")

for image_path in image_dir.glob("*.jpg"):
    try:
        result = analyze_image_file(
            str(image_path),
            output_dir="batch_results",
            pixels_per_mm=10.5,
            method="canny",
            fit_ellipse=True
        )
        results.append(result)
    except Exception as e:
        results.append({
            "image_path": str(image_path),
            "error": str(e)
        })

# Save batch summary
save_batch_summary(results, "batch_results")

# Create 3D models for successful analyses
for result in results:
    if not result.get("error"):
        # Example: create model if both top and bottom exist
        top_file = result["metrics_json"]
        bottom_file = top_file.replace("_top_", "_bottom_")

        if Path(bottom_file).exists():
            model_path = top_file.replace("_metrics.json", "_model.stl")
            create_can_mesh_file(
                top_json=top_file,
                bottom_json=bottom_file,
                output_path=model_path,
                top_profile_type="ellipse",
                bottom_profile_type="ellipse"
            )
```

## Configuration

### Default Detection Parameters

The tool uses optimized defaults for typical can imaging:

```python
# Binary segmentation
BINARY_BLOCK_SIZE = 51      # Adaptive threshold neighborhood
BINARY_C = 2                # Threshold adjustment constant

# Canny edge detection
CANNY_LOW = 75              # Lower threshold
CANNY_HIGH = 200            # Upper threshold

# Contour selection
PREFER_CENTER = False       # Prefer center contours
CENTER_RADIUS_FRAC = 0.4    # Center preference radius

# Quality filters
MIN_DIAMETER_MM = 0.0       # No minimum by default
MAX_DIAMETER_MM = inf       # No maximum by default

# 3D Reconstruction
WALL_THICKNESS = 0.1        # Default wall thickness (mm)
MESH_RESOLUTION = 64        # Points around profile
ALIGN_PROFILES = True       # Align ellipse orientations
```

### Calibration Files

Calibration data is stored in JSON format:

```json
{
  "pixels_per_mm": 10.567,
  "known_mm": 50.0,
  "source_image": "/path/to/calibration_image.jpg",
  "scale_points_full_px": [100.0, 200.0, 150.0, 200.0]
}
```

## Troubleshooting

### Common Issues

#### No contours detected
- **Problem**: "No contours found in binary image"
- **Solutions**:
  - Try different detection method (`--method canny`)
  - Adjust threshold parameters
  - Check image contrast and lighting
