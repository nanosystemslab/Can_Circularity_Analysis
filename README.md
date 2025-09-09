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
![Interfaces](https://img.shields.io/badge/Interface-CLI%20%7C%20Python%20API-red)
![Output Types](https://img.shields.io/badge/Outputs-JSON%20%7C%20CSV%20%7C%20Plots-purple)

A computer vision tool for automated analysis of can rim circularity and quality control in manufacturing. This package provides both command-line utilities and a Python API for measuring circular deviation, diameter accuracy, and out-of-round characteristics of cylindrical objects.

## Features

- **Automated rim detection** using binary segmentation or Canny edge detection
- **Pixel-to-millimeter calibration** with interactive or programmatic setup
- **Diameter filtering** to focus analysis on specific size ranges
- **Comprehensive circularity metrics** including area, perimeter, and deviation analysis
- **Out-of-round quantification** with RMS, standard deviation, and range statistics
- **Ellipse fitting** for ovalization and eccentricity analysis
- **Batch processing** for analyzing multiple images efficiently
- **Statistical summaries** with visualization plots
- **Export capabilities** for rim point coordinates and analysis results

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Calibration](#calibration)
- [Analysis Workflow](#analysis-workflow)
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

# Verify installation
poetry run can-analyze --help
```

### Using pip

```bash
git clone https://github.com/nanosystemslab/Can_Circularity_Analysis.git
cd Can_Circularity_Analysis
pip install -e .
```

### Dependencies

Core dependencies are automatically installed:
- `opencv-python` - Computer vision and image processing
- `scikit-image` - Circle fitting algorithms
- `numpy` - Numerical computations
- `pandas` - Data analysis (for summarization)
- `matplotlib` - Plotting and visualization

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

### 3. Batch Processing

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
# 60mm cans
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

#### Automated Processing Script

Create `process_all_sizes.sh`:

```bash
#!/bin/bash
set -e

echo "Starting batch processing of all can sizes..."

# Array of can sizes and their parameters
declare -A can_configs=(
    ["40mm"]="38:42"
    ["60mm"]="58:62"
    ["80mm"]="78:82"
    ["100mm"]="98:102"
    ["120mm"]="118:122"
)

for size in "${!can_configs[@]}"; do
    IFS=':' read -r min_diam max_diam <<< "${can_configs[$size]}"

    echo "Processing ${size} cans (${min_diam}-${max_diam}mm)..."

    poetry run can-analyze data/${size}_topDown_ovality/*.jpg \
      --scale-from scales/ppmm_${size}.json \
      --method canny \
      --canny-low 50 --canny-high 150 \
      --prefer-center \
      --min-diameter-mm ${min_diam} \
      --max-diameter-mm ${max_diam} \
      --fit-ellipse \
      --crop 0,750,99999,99999 \
      --out-dir ./results/${size}/

    echo "✓ Completed ${size}"
done

echo "Batch processing complete!"
```

### Step 3: Handle Failed Detections

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

#### Debug Analysis

Use debug output to understand detection issues:

```bash
# Generate debug images to see what went wrong
poetry run can-analyze problematic_image.jpg \
  --scale-from scales/ppmm_60mm.json \
  --save-debug \
  --out-dir debug/

# Examine intermediate results:
# debug/problematic_image_binary.png - thresholding result
# debug/problematic_image_edges.png - edge detection result
# debug/problematic_image_overlay.png - final detection
```

### Step 4: Generate Comprehensive Summaries

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

#### Cross-Size Comparison

```bash
# Compare across all sizes
poetry run can-summarize \
  --in-dir ./results/ \
  --out-dir ./reports/comparison/ \
  --make-plots
```

#### Production Report Generation

Create `generate_reports.sh`:

```bash
#!/bin/bash

echo "Generating comprehensive analysis reports..."

# Individual size reports
for size in 40mm 60mm 80mm 100mm 120mm; do
    if [ -d "results/${size}" ]; then
        echo "Generating ${size} report..."
        poetry run can-summarize \
          --in-dir ./results/${size}/ \
          --out-dir ./reports/${size}/ \
          --make-plots
    fi
done

# Overall summary
echo "Generating overall comparison report..."
poetry run can-summarize \
  --in-dir ./results/ \
  --out-dir ./reports/overall/ \
  --make-plots

echo "Reports complete! Check ./reports/ directory"
echo ""
echo "Key files generated:"
echo "  - reports/*/plots/rim_overlay.png (shape comparisons)"
echo "  - reports/*/plots/*_histogram.png (quality distributions)"
echo "  - reports/*/metrics_all.csv (complete data)"
echo ""
echo "Summary statistics:"
for size in 40mm 60mm 80mm 100mm 120mm; do
    if [ -f "results/${size}/batch_summary.json" ]; then
        total=$(jq '.total_images' results/${size}/batch_summary.json)
        success=$(jq '.successful_analyses' results/${size}/batch_summary.json)
        echo "  ${size}: ${success}/${total} successful"
    fi
done
```

### Quality Control Workflow

#### Real-time Production Monitoring

```bash
# Daily production analysis
DATE=$(date +%Y%m%d)
mkdir -p production_analysis/${DATE}

# Process today's images
poetry run can-analyze production_images/${DATE}/*.jpg \
  --scale-from scales/production_line_cal.json \
  --method canny \
  --min-diameter-mm 58 --max-diameter-mm 62 \
  --out-dir production_analysis/${DATE}/

# Generate quality report
poetry run can-summarize \
  --in-dir production_analysis/${DATE}/ \
  --out-dir quality_reports/${DATE}/ \
  --make-plots

# Alert if quality issues detected
python check_quality_thresholds.py quality_reports/${DATE}/metrics_all.csv
```

#### Detection Methods and Parameters

**Binary Segmentation** - Best for high-contrast images:
```bash
poetry run can-analyze image.jpg --method binary --binary-block-size 51 --binary-C 2
```

**Canny Edge Detection** - Better for noisy or low-contrast images:
```bash
poetry run can-analyze image.jpg --method canny --canny-low 50 --canny-high 150
```

#### Troubleshooting Common Issues

**Low success rate in batch processing:**
1. Check calibration accuracy with known samples
2. Verify image quality (focus, lighting, contrast)
3. Adjust detection parameters for your specific setup
4. Consider different crop regions

**Inconsistent measurements:**
1. Ensure consistent imaging conditions
2. Verify stable camera positioning
3. Check for vibration or movement during imaging
4. Validate calibration across the full image area

**Processing speed optimization:**
```bash
# Parallel processing for large batches
find data/ -name "*.jpg" | \
xargs -n 1 -P 4 -I {} poetry run can-analyze {} \
  --scale-from scales/ppmm_60mm.json \
  --out-dir results/
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

echo "Quality control analysis complete. Check qc_reports/ for results."
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
    "ovalization_mm": 0.089
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
    max_diameter_mm=95
)

print(f"Diameter: {results['diameter_mm']:.2f} mm")
print(f"Circularity: {results['circularity_4piA_P2']:.3f}")
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

### Batch Processing

```python
from pathlib import Path
from can_circularity_analysis import analyze_image_file
from can_circularity_analysis.utils.file_io import save_batch_summary

results = []
image_dir = Path("production_images")

for image_path in image_dir.glob("*.jpg"):
    try:
        result = analyze_image_file(
            str(image_path),
            output_dir="batch_results",
            pixels_per_mm=10.5,
            method="canny"
        )
        results.append(result)
    except Exception as e:
        results.append({
            "image_path": str(image_path),
            "error": str(e)
        })

# Save batch summary
save_batch_summary(results, "batch_results")
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
  - Use `--save-debug` to inspect intermediate images

#### Poor circle fitting
- **Problem**: Inconsistent or poor quality measurements
- **Solutions**:
  - Improve image quality (lighting, focus, contrast)
  - Use `--prefer-center` for centered objects
  - Adjust edge detection parameters
  - Consider `--crop` to focus on region of interest

#### Calibration errors
- **Problem**: "OpenCV GUI not available"
- **Solutions**:
  - Use `--scale-pts` instead of `--click-scale` on headless systems
  - Provide known `--pixels-per-mm` value
  - Run on system with display capability

#### Diameter filtering issues
- **Problem**: "Detected rim X mm is outside range"
- **Solutions**:
  - Verify calibration accuracy
  - Adjust `--min-diameter-mm` and `--max-diameter-mm`
  - Check for incorrect scale or units

### Performance Optimization

#### Large batch processing
```bash
# Process images in parallel (if system supports)
find data/ -name "*.jpg" | xargs -n 1 -P 4 poetry run can-analyze
```

#### Memory usage
- Use `--crop` to reduce processing area
- Process images individually for very large datasets
- Consider resizing images if extremely high resolution

### Debug Mode

```bash
# Enable all debug outputs
poetry run can-analyze image.jpg \
  --save-debug \
  --method binary \
  --out-dir debug_output/

# Check intermediate files:
# - debug_output/*_binary.png (segmentation result)
# - debug_output/*_overlay.png (detection visualization)
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/nanosystemslab/Can_Circularity_Analysis.git
cd Can_Circularity_Analysis

# Install development dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Run linting
poetry run ruff check
poetry run ruff format
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=can_circularity_analysis

# Test specific functionality
poetry run pytest tests/test_detection.py
```

### Code Quality

```bash
# Format code
poetry run ruff format

# Lint code
poetry run ruff check --fix

# Type checking
poetry run mypy src/can_circularity_analysis
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`poetry run pytest`)
6. Run code quality checks (`poetry run ruff check`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Contribution Guidelines

- Follow existing code style (enforced by ruff)
- Add docstrings for all public functions
- Include unit tests for new features
- Update documentation for user-facing changes
- Ensure backward compatibility when possible

## Citation

If you use this software in your research or industrial applications, please cite:

```bibtex
@software{can_circularity_analysis,
  title={Can Circularity Analysis: Computer Vision for Manufacturing Quality Control},
  author={Nanosystems Lab},
  year={2025},
  url={https://github.com/nanosystemslab/Can_Circularity_Analysis},
  version={0.1.0}
}
```

## License

This project is licensed under the GPL-3.0-or-later License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/nanosystemslab/Can_Circularity_Analysis/issues)
- **Documentation**: Additional examples and tutorials in the `docs/` directory
- **Discussions**: Join discussions on [GitHub Discussions](https://github.com/nanosystemslab/Can_Circularity_Analysis/discussions)

## Acknowledgments

- Built with OpenCV for computer vision capabilities
- Uses scikit-image for robust circle fitting algorithms
- Inspired by manufacturing quality control requirements
- Developed at Nanosystems Laboratory
