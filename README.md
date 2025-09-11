# Can Circularity Analysis

[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/nanosystemslab/Can_Circularity_Analysis)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/nanosystemslab/Can_Circularity_Analysis)
[![License](https://img.shields.io/badge/license-GPL--3.0-green)](LICENSE)
[![Poetry](https://img.shields.io/badge/dependency--management-poetry-blue)](https://python-poetry.org/)

![Analysis Types](https://img.shields.io/badge/Analysis-Circularity%20%7C%20Diameter%20%7C%20Ovality-blue)
![Detection Methods](https://img.shields.io/badge/Detection-Binary%20%7C%20Canny%20%7C%20Hough-orange)
![Measurements](https://img.shields.io/badge/Measures-RMS%20%7C%20Ellipticity%20%7C%20Deviation-green)
![3D Reconstruction](https://img.shields.io/badge/3D-Reconstruction%20%7C%20STL%20%7C%20OBJ%20%7C%20PLY-purple)
![Interfaces](https://img.shields.io/badge/Interface-CLI%20%7C%20Python%20API-red)
![Output Types](https://img.shields.io/badge/Outputs-JSON%20%7C%20CSV%20%7C%20Plots%20%7C%203D%20Models-purple)

A computer vision tool for automated analysis of can rim circularity and quality control in manufacturing. This package provides both command-line utilities and a Python API for measuring circular deviation, diameter accuracy, and out-of-round characteristics of cylindrical objects, with advanced 3D reconstruction capabilities for generating mesh models from top and bottom rim profiles.

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
poetry run can-summarize --help
poetry run can-reconstruct --help
```

### Dependencies

Core dependencies:
- `opencv-python` - Computer vision and image processing
- `scikit-image` - Circle fitting algorithms
- `numpy` - Numerical computations
- `pandas` - Data analysis (for summarization)
- `matplotlib` - Plotting and visualization

Additional dependencies for 3D reconstruction:
- `trimesh` - 3D mesh processing and export
- `scipy` - Advanced interpolation and scientific computing

---

# Section 1: Can Analysis

The analysis component provides automated rim detection and circularity measurement for individual images or batch processing.

## Features

- **Automated rim detection** using binary segmentation or Canny edge detection
- **Pixel-to-millimeter calibration** with interactive or programmatic setup
- **Diameter filtering** to focus analysis on specific size ranges
- **Comprehensive circularity metrics** including area, perimeter, and deviation analysis
- **Out-of-round quantification** with RMS, standard deviation, and range statistics
- **Ellipse fitting** for ovalization and eccentricity analysis
- **Batch processing** for analyzing multiple images efficiently

## Quick Start

### Basic Analysis
```bash
# Analyze a single image
poetry run can-analyze image.jpg --out-dir results

# With known pixel-to-mm ratio
poetry run can-analyze image.jpg --pixels-per-mm 10.5 --out-dir results
```

### Interactive Calibration
```bash
# Click two points on a ruler to calibrate
poetry run can-analyze image.jpg --click-scale --known-mm 50.0 --out-dir results
```

### Batch Processing
```bash
# Process multiple images
poetry run can-analyze data/*.jpg --pixels-per-mm 10.5 --out-dir batch_results
```

## Calibration Methods

### Method 1: Interactive Calibration
```bash
poetry run can-analyze image.jpg --click-scale --known-mm 50.0
```
Opens a GUI where you click two points on a ruler that are exactly 50mm apart.

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

## Command Reference

```bash
poetry run can-analyze [OPTIONS] IMAGE [IMAGE...]
```

### Main Options
- `--out-dir TEXT` - Output directory (default: out)
- `--method [binary|canny]` - Detection method (default: binary)
- `--pixels-per-mm FLOAT` - Known pixels-per-mm conversion
- `--scale-from TEXT` - Load calibration from JSON file
- `--click-scale` - Interactive calibration (requires GUI)
- `--scale-pts TEXT` - Manual calibration points 'x1,y1,x2,y2'
- `--known-mm FLOAT` - Distance for calibration (default: 50.0)
- `--min-diameter-mm FLOAT` - Minimum allowed diameter (default: 0.0)
- `--max-diameter-mm FLOAT` - Maximum allowed diameter (default: inf)
- `--crop TEXT` - Crop region 'x,y,w,h'
- `--fit-ellipse` - Include ellipse analysis
- `--save-debug` - Save intermediate processing images
- `--prefer-center` - Prefer contours near image center
- `--no-overlay` - Skip overlay image generation

### Binary Segmentation Parameters
- `--binary-block-size INT` - Adaptive threshold block size (default: 51)
- `--binary-C INT` - Adaptive threshold constant (default: 2)
- `--binary-invert` - Invert binary threshold

### Canny Edge Detection Parameters
- `--canny-low INT` - Lower threshold (default: 75)
- `--canny-high INT` - Upper threshold (default: 200)

## Workflow Examples

### Create Calibration for Each Image Set
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

### Batch Processing by Can Size
```bash
# 60mm cans with ellipse fitting
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

### Handle Failed Detections
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
```

## Output Files

### Per-Image Results

#### `*_metrics.json`
Complete analysis results including:
- `diameter_mm` - Diameter in millimeters
- `circularity_4piA_P2` - Circularity metric (4πA/P²)
- `rms_out_of_round_mm` - RMS out-of-round measurement
- `center_mm` - Center coordinates in millimeters
- `ellipse` - Ellipse fitting results (if enabled)

#### `*_points.csv`
Rim point coordinates in both Cartesian and cylindrical formats

#### `*_overlay.png`
Visualization showing:
- Original image with detected rim
- Fitted circle (green)
- Individual rim points (red dots)
- Fitted ellipse (purple, if enabled)
- Measurement annotations

#### `batch_summary.json`
Summary of all analyses when processing multiple images

---

# Section 2: Can Summary

The summary component generates statistical reports and visualizations from analysis results.

## Features

- **Statistical summaries** with comprehensive metrics
- **Visualization plots** including histograms and overlays
- **Data aggregation** from multiple analysis results
- **Export capabilities** for spreadsheet analysis

## Quick Start

```bash
# Generate summary report with plots
poetry run can-summarize --in-dir batch_results --make-plots --out-dir reports
```

## Command Reference

```bash
poetry run can-summarize [OPTIONS]
```

### Options
- `--in-dir TEXT` - Input directory with *_metrics.json files (default: out)
- `--out-dir TEXT` - Output directory for reports (default: out)
- `--make-plots` - Generate visualization plots

## Workflow Examples

### Individual Size Summaries
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

### Cross-Size Comparison
```bash
# Compare across all sizes
poetry run can-summarize \
  --in-dir ./results/ \
  --out-dir ./reports/comparison/ \
  --make-plots
```

### Quality Control Pipeline
```bash
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
```

## Output Files

### Statistical Plots
- `diameter_histogram.png` - Diameter distribution
- `rms_histogram.png` - Out-of-round distribution
- `circularity_histogram.png` - Circularity distribution
- `rim_overlay.png` - Normalized rim shape overlay

### Data Tables
- `metrics_all.csv` - All results in spreadsheet format
- `stats_overall.csv` - Statistical summaries

---

# Section 3: Can Reconstruction

The reconstruction component creates 3D mesh models from top and bottom rim analyses for visualization, simulation, or manufacturing applications.

## Features

- **3D reconstruction** from top and bottom rim profiles with configurable profile types
- **Multiple profile types** (circle, ellipse, measured points)
- **Proper wall thickness** implementation with inner/outer surfaces
- **Profile alignment** for straight walls
- **Multiple mesh formats** (STL, OBJ, PLY) with auto-detection
- **Batch reconstruction** processing
- **3D visualization** and debugging tools

## Quick Start

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

## Profile Types

### Ellipse Profile (Recommended for Oval Cans)
```bash
poetry run can-reconstruct top.json bottom.json model.stl \
  --top-profile ellipse \
  --bottom-profile ellipse
```
- Uses fitted ellipse data from analysis
- Captures ovality and eccentricity
- Best for deformed or intentionally oval cans

### Circle Profile
```bash
poetry run can-reconstruct top.json bottom.json model.stl \
  --top-profile circle \
  --bottom-profile circle
```
- Uses fitted circle data from analysis
- Creates perfectly circular cross-sections
- Good for comparing against ideal geometry

### Measured Profile (Highest Accuracy)
```bash
poetry run can-reconstruct top.json bottom.json model.stl \
  --top-profile measured \
  --bottom-profile measured
```
- Uses actual measured rim points from CSV files
- Captures all irregularities and asymmetries
- Most accurate representation of actual can shape

## Wall Thickness Implementation

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

## Command Reference

```bash
poetry run can-reconstruct [OPTIONS] TOP_JSON BOTTOM_JSON OUTPUT_FILE
```

### Profile Selection
- `--top-profile [circle|ellipse|measured]` - Top rim profile type (default: ellipse)
- `--bottom-profile [circle|ellipse|measured]` - Bottom rim profile type (default: ellipse)

### Reconstruction Parameters
- `--wall-thickness FLOAT` - Wall thickness in mm (default: 0.1)
- `--height FLOAT` - Height between profiles in mm (default: 100.0)
- `--resolution INT` - Points around each profile (default: 64)

### Output Options
- `--output-format [stl|obj|ply|auto]` - Output format (default: auto)
- `--plot-profiles` - Show 3D plot before meshing
- `--align-profiles` - Align orientations for straight walls (default: True)
- `--no-align-profiles` - Preserve individual orientations

### Batch Processing
- `--batch TOP_DIR BOTTOM_DIR` - Batch process matching files
- `--out-dir TEXT` - Output directory for batch (default: models)

## Workflow Examples

### Single Can Reconstruction
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
```

### Preview Before Reconstruction
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

### Batch Reconstruction
```bash
# Reconstruct all matching pairs in directories
poetry run can-reconstruct \
  --batch results/60mm_tops/ results/60mm_bottoms/ \
  --out-dir models/60mm/ \
  --top-profile ellipse \
  --bottom-profile ellipse \
  --wall-thickness 0.1 \
  --height 100.0

# Mixed profile types
poetry run can-reconstruct \
  --batch analysis/tops/ analysis/bottoms/ \
  --out-dir models/mixed/ \
  --top-profile ellipse \
  --bottom-profile circle \
  --wall-thickness 0.15
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

## Output Formats

### Supported Formats
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

## Applications

### Quality Control Documentation
```bash
# Create models showing actual vs. ideal geometry
poetry run can-reconstruct top.json bottom.json actual.stl \
  --top-profile measured --bottom-profile measured

poetry run can-reconstruct top.json bottom.json ideal.stl \
  --top-profile circle --bottom-profile circle
```

### Manufacturing Simulation
```bash
# Generate models for FEA or CFD analysis
poetry run can-reconstruct top.json bottom.json simulation.obj \
  --top-profile ellipse \
  --bottom-profile ellipse \
  --wall-thickness 0.12 \
  --resolution 128
```

### 3D Printing and Prototyping
```bash
# Create physical prototypes
poetry run can-reconstruct top.json bottom.json prototype.stl \
  --wall-thickness 2.0 \
  --resolution 64
```

## Output Files

### Mesh Files
- `*.stl` - STL format for 3D printing and CAD
- `*.obj` - OBJ format for graphics and simulation
- `*.ply` - PLY format with extended metadata

### Metadata Files
- `*_metadata.json` - Complete reconstruction parameters and source data

---

## Troubleshooting

### Can Analysis Issues
- **No contours detected**: Try different detection method, adjust threshold parameters, check image contrast
- **Poor circle fitting**: Improve image quality, use `--prefer-center`, adjust edge detection parameters
- **Calibration errors**: Use `--scale-pts` on headless systems, provide known `--pixels-per-mm` value
- **Diameter filtering issues**: Verify calibration accuracy, adjust diameter range parameters

### Can Summary Issues
- **No metrics files found**: Check input directory path, verify analysis completed successfully
- **Plot generation fails**: Install matplotlib, check output directory permissions

### Can Reconstruction Issues
- **Missing dependencies**: Install with `poetry install --extras "reconstruction"`
- **Twisted walls**: Use `--align-profiles` (default) to align ellipse orientations
- **Profile visualization issues**: Install matplotlib, use `--no-plot-profiles` to skip
- **Missing ellipse data**: Re-run analysis with `--fit-ellipse` flag
- **File not found errors**: Verify CSV files exist alongside JSON files

## License

This project is licensed under the GPL-3.0-or-later License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/nanosystemslab/Can_Circularity_Analysis/issues)
- **Documentation**: Additional examples and tutorials in the `docs/` directory
- **Discussions**: Join discussions on [GitHub Discussions](https://github.com/nanosystemslab/Can_Circularity_Analysis/discussions)

## Acknowledgments

- Built with OpenCV for computer vision capabilities
- Uses scikit-image for robust circle fitting algorithms
- Trimesh library for 3D mesh processing and export
- SciPy for advanced mathematical interpolation
- Developed at Nanosystems Laboratory
