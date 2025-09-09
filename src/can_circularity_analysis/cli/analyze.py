"""CLI for analyzing can circularity in images."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from ..core.detection import analyze_image_file
from ..core.calibration import parse_crop_string, save_calibration
from ..utils.file_io import validate_image_file, get_image_files, create_output_directory, save_batch_summary


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for the analyze command."""
    parser = argparse.ArgumentParser(
        description="Analyze can circularity in images with optional diameter filtering and calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image with interactive calibration
  can-analyze image.jpg --click-scale --known-mm 50

  # Analyze with pre-defined scale points
  can-analyze image.jpg --scale-pts "100,200,150,200" --known-mm 50

  # Batch analysis with diameter filtering
  can-analyze *.jpg --min-diameter-mm 85 --max-diameter-mm 95 --pixels-per-mm 10

  # Use saved calibration for multiple images
  can-analyze image1.jpg --save-scale calibration.json --click-scale
  can-analyze image2.jpg image3.jpg --scale-from calibration.json
        """
    )
    
    # Input/Output
    parser.add_argument(
        "images", 
        nargs="+", 
        help="Path(s) to input images (JPEG/PNG). Supports wildcards."
    )
    parser.add_argument(
        "--out-dir", 
        default="out", 
        help="Output directory for results (default: ./out)"
    )
    
    # Calibration options
    calib_group = parser.add_argument_group("Calibration")
    calib_group.add_argument(
        "--pixels-per-mm", 
        type=float, 
        help="Known pixels-per-mm conversion factor"
    )
    calib_group.add_argument(
        "--click-scale", 
        action="store_true",
        help="Interactive calibration: click two ruler points (requires GUI)"
    )
    calib_group.add_argument(
        "--known-mm", 
        type=float, 
        default=50.0,
        help="Distance in mm between calibration points (default: 50.0)"
    )
    calib_group.add_argument(
        "--scale-pts", 
        help="Non-GUI calibration points as 'x1,y1,x2,y2' in full image coordinates"
    )
    calib_group.add_argument(
        "--save-scale", 
        help="Save calibration to JSON file for reuse"
    )
    calib_group.add_argument(
        "--scale-from", 
        help="Load calibration from existing JSON file"
    )
    
    # Diameter filtering
    filter_group = parser.add_argument_group("Diameter Filtering")
    filter_group.add_argument(
        "--min-diameter-mm", 
        type=float, 
        default=0.0,
        help="Minimum allowed rim diameter in mm (requires calibration)"
    )
    filter_group.add_argument(
        "--max-diameter-mm", 
        type=float, 
        default=1e9,
        help="Maximum allowed rim diameter in mm (requires calibration)"
    )
    
    # Detection method
    detect_group = parser.add_argument_group("Detection Method")
    detect_group.add_argument(
        "--method", 
        choices=["binary", "canny"], 
        default="binary",
        help="Rim detection method (default: binary)"
    )
    
    # Binary segmentation parameters
    binary_group = parser.add_argument_group("Binary Segmentation")
    binary_group.add_argument(
        "--binary-block-size", 
        type=int, 
        default=51,
        help="Adaptive threshold block size, must be odd (default: 51)"
    )
    binary_group.add_argument(
        "--binary-C", 
        type=int, 
        default=2,
        help="Adaptive threshold constant C (default: 2)"
    )
    binary_group.add_argument(
        "--binary-invert", 
        action="store_true",
        help="Invert binary threshold (use if rim is darker than background)"
    )
    binary_group.add_argument(
        "--erode", 
        type=int, 
        default=0,
        help="Number of erosion iterations to shrink detected regions (default: 0)"
    )
    
    # Canny edge detection parameters  
    canny_group = parser.add_argument_group("Canny Edge Detection")
    canny_group.add_argument(
        "--canny-low", 
        type=int, 
        default=75,
        help="Canny lower threshold (default: 75)"
    )
    canny_group.add_argument(
        "--canny-high", 
        type=int, 
        default=200,
        help="Canny upper threshold (default: 200)"
    )
    
    # Image processing
    process_group = parser.add_argument_group("Image Processing")
    process_group.add_argument(
        "--crop", 
        help="Crop region as 'x,y,width,height' before analysis"
    )
    process_group.add_argument(
        "--prefer-center", 
        action="store_true",
        help="Prefer contours near image center when multiple candidates exist"
    )
    process_group.add_argument(
        "--center-radius-frac", 
        type=float, 
        default=0.4,
        help="Radius fraction for center preference (default: 0.4)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--no-overlay", 
        action="store_true",
        help="Skip generating overlay visualization images"
    )
    output_group.add_argument(
        "--save-debug", 
        action="store_true",
        help="Save debug images (binary masks, edge maps)"
    )
    output_group.add_argument(
        "--fit-ellipse", 
        action="store_true",
        help="Also fit ellipse and calculate ellipticity metrics"
    )
    output_group.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress output, only show final summary"
    )
    
    return parser


def validate_arguments(args) -> None:
    """Validate command line arguments."""
    # Ensure binary block size is odd
    if args.binary_block_size % 2 == 0:
        args.binary_block_size += 1
        if not args.quiet:
            print(f"Warning: Adjusted binary-block-size to {args.binary_block_size} (must be odd)")
    
    # Parse crop string
    args.crop = parse_crop_string(args.crop)
    
    # Validate images exist
    missing_images = [img for img in args.images if not validate_image_file(img)]
    if missing_images:
        print(f"Error: The following image files were not found or are invalid:")
        for img in missing_images:
            print(f"  {img}")
        sys.exit(1)


def analyze_single_image(image_path: str, args) -> dict:
    """Analyze a single image and return results."""
    try:
        if not args.quiet:
            print(f"Analyzing: {image_path}")
        
        # Convert args to kwargs
        kwargs = {
            'output_dir': args.out_dir,
            'method': args.method,
            'crop': args.crop,
            'pixels_per_mm': args.pixels_per_mm,
            'click_scale': args.click_scale,
            'known_mm': args.known_mm,
            'scale_pts': args.scale_pts,
            'scale_from': args.scale_from,
            'min_diameter_mm': args.min_diameter_mm,
            'max_diameter_mm': args.max_diameter_mm,
            'binary_block_size': args.binary_block_size,
            'binary_C': args.binary_C,
            'binary_invert': args.binary_invert,
            'erode_iterations': args.erode,
            'canny_low': args.canny_low,
            'canny_high': args.canny_high,
            'prefer_center': args.prefer_center,
            'center_radius_frac': args.center_radius_frac,
            'no_overlay': args.no_overlay,
            'save_debug': args.save_debug,
            'fit_ellipse': args.fit_ellipse,
        }
        
        # Analyze image
        results = analyze_image_file(image_path, **kwargs)
        
        # Save calibration if requested
        if args.save_scale and results.get('pixels_per_mm'):
            save_calibration(
                args.save_scale,
                results['pixels_per_mm'],
                known_mm=results.get('known_mm_for_scale'),
                source_image=image_path
            )
            if not args.quiet:
                print(f"Saved calibration to: {args.save_scale}")
        
        return results
        
    except Exception as e:
        error_result = {
            'image_path': image_path,
            'error': str(e),
            'success': False
        }
        if not args.quiet:
            print(f"Error analyzing {image_path}: {e}")
        return error_result


def print_summary(all_results: List[dict], args) -> None:
    """Print analysis summary."""
    total = len(all_results)
    successful = len([r for r in all_results if not r.get('error')])
    failed = total - successful
    
    print(f"\nAnalysis Summary:")
    print(f"  Total images: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if successful > 0:
        # Calculate statistics for successful analyses
        diameters_mm = [r.get('diameter_mm') for r in all_results 
                       if not r.get('error') and r.get('diameter_mm')]
        
        if diameters_mm:
            import numpy as np
            mean_diam = np.mean(diameters_mm)
            std_diam = np.std(diameters_mm)
            print(f"  Mean diameter: {mean_diam:.2f} ± {std_diam:.2f} mm")
            print(f"  Range: {min(diameters_mm):.2f} - {max(diameters_mm):.2f} mm")


def main():
    """Main entry point for the analyze command."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Expand image paths (handle wildcards)
    image_paths = []
    for pattern in args.images:
        if '*' in pattern or '?' in pattern:
            from glob import glob
            matches = glob(pattern)
            if matches:
                image_paths.extend(matches)
            else:
                if not args.quiet:
                    print(f"Warning: No files match pattern '{pattern}'")
        else:
            image_paths.append(pattern)
    
    if not image_paths:
        print("Error: No valid image files specified")
        sys.exit(1)
    
    # Create output directory
    create_output_directory(args.out_dir)
    
    # Process images
    all_results = []
    
    for image_path in image_paths:
        result = analyze_single_image(image_path, args)
        all_results.append(result)
    
    # Save batch summary if multiple images
    if len(all_results) > 1:
        summary_path = save_batch_summary(all_results, args.out_dir)
        if not args.quiet:
            print(f"Batch summary saved to: {summary_path}")
    
    # Print summary
    if not args.quiet:
        print_summary(all_results, args)
    
    # Print JSON output for single image (for scripting)
    if len(all_results) == 1 and not all_results[0].get('error'):
        if args.quiet:
            print(json.dumps(all_results[0], indent=2))
    
    # Exit with error code if any analyses failed
    failed_count = len([r for r in all_results if r.get('error')])
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
