"""CLI for reconstructing 3D STEP files from can analysis data."""

import argparse
import os
import sys
from pathlib import Path


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for the reconstruct command."""
    parser = argparse.ArgumentParser(
        description="Create 3D mesh files from can circularity analysis data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction with best contour data (most accurate)
  can-reconstruct top.json bottom.json output.stl --top-profile best_contour --bottom-profile best_contour

  # Plot profiles first to check geometry
  can-reconstruct top.json bottom.json can.stl --top-profile best_contour --bottom-profile best_contour --plot-profiles

  # Fallback to ellipse if best contour not available
  can-reconstruct top.json bottom.json output.obj --top-profile ellipse --bottom-profile ellipse

  # Use measured points for both profiles
  can-reconstruct top.json bottom.json can.stl --top-profile measured --bottom-profile measured

  # Custom wall thickness and height with best contour
  can-reconstruct top.json bottom.json can.obj --top-profile best_contour --bottom-profile best_contour --wall-thickness 0.15 --height 120

  # Batch reconstruction from analysis directories using best contour
  can-reconstruct --batch out/60mm_top out/60mm_bottom --out-dir models/ --top-profile best_contour --bottom-profile best_contour
        """,
    )

    # Input files
    parser.add_argument("top_json", nargs="?", help="Top rim analysis JSON file (*_metrics.json)")
    parser.add_argument("bottom_json", nargs="?", help="Bottom rim analysis JSON file (*_metrics.json)")
    parser.add_argument("output_file", nargs="?", help="Output mesh file path (.stl, .obj, .ply)")

    # Profile type selection
    profile_group = parser.add_argument_group("Profile Selection")
    profile_group.add_argument(
        "--top-profile",
        choices=["circle", "ellipse", "measured", "best_contour"],
        default="best_contour",
        help="Profile type for top rim (default: best_contour)",
    )
    profile_group.add_argument(
        "--bottom-profile",
        choices=["circle", "ellipse", "measured", "best_contour"],
        default="best_contour",
        help="Profile type for bottom rim (default: best_contour)",
    )

    # Batch processing
    batch_group = parser.add_argument_group("Batch Processing")
    batch_group.add_argument(
        "--batch",
        nargs=2,
        metavar=("TOP_DIR", "BOTTOM_DIR"),
        help="Batch process: directories containing top and bottom analysis results",
    )
    batch_group.add_argument(
        "--out-dir", default="models", help="Output directory for batch processing (default: models)"
    )

    # Reconstruction parameters
    recon_group = parser.add_argument_group("Reconstruction Parameters")
    recon_group.add_argument("--wall-thickness", type=float, default=0.1, help="Wall thickness in mm (default: 0.1)")
    recon_group.add_argument(
        "--height", type=float, default=100.0, help="Height between top and bottom profiles in mm (default: 100.0)"
    )
    recon_group.add_argument(
        "--resolution", type=int, default=64, help="Number of points around each profile (default: 64)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-format",
        choices=["stl", "obj", "ply", "auto"],
        default="auto",
        help="Output format (default: auto-detect from extension)",
    )
    output_group.add_argument(
        "--plot-profiles",
        action="store_true",
        help="Show 3D plot of profiles before creating mesh (requires matplotlib)",
    )
    output_group.add_argument("--quiet", action="store_true", help="Suppress progress output")
    output_group.add_argument(
        "--align-profiles",
        action="store_true",
        default=True,
        help="Align ellipse orientations for straight walls (default: True)",
    )
    output_group.add_argument(
        "--no-align-profiles",
        dest="align_profiles",
        action="store_false",
        help="Preserve individual ellipse orientations (may create twisted walls)",
    )

    return parser


def validate_arguments(args) -> None:
    """Validate command line arguments."""
    if args.batch:
        # Batch mode validation
        top_dir, bottom_dir = args.batch
        if not os.path.isdir(top_dir):
            print(f"Error: Top directory '{top_dir}' does not exist")
            sys.exit(1)
        if not os.path.isdir(bottom_dir):
            print(f"Error: Bottom directory '{bottom_dir}' does not exist")
            sys.exit(1)
    else:
        # Single file mode validation
        if not all([args.top_json, args.bottom_json, args.output_file]):
            print("Error: top_json, bottom_json, and output_file are required for single file mode")
            print("Use --batch for batch processing or provide all three arguments")
            sys.exit(1)

        if not os.path.isfile(args.top_json):
            print(f"Error: Top JSON file '{args.top_json}' does not exist")
            sys.exit(1)
        if not os.path.isfile(args.bottom_json):
            print(f"Error: Bottom JSON file '{args.bottom_json}' does not exist")
            sys.exit(1)

    # Validate parameters
    if args.wall_thickness <= 0:
        print("Error: Wall thickness must be positive")
        sys.exit(1)
    if args.height <= 0:
        print("Error: Height must be positive")
        sys.exit(1)
    if args.resolution < 8:
        print("Error: Resolution must be at least 8 points")
        sys.exit(1)


def find_matching_files(top_dir: str, bottom_dir: str) -> list[tuple[str, str, str]]:
    """Find matching analysis files in top and bottom directories.

    Returns:
        List of tuples (top_file, bottom_file, output_name)
    """
    top_files = {}
    bottom_files = {}

    # Find all metrics files
    for file_path in Path(top_dir).glob("*_metrics.json"):
        # Extract base name (remove _metrics.json)
        base_name = file_path.stem.replace("_metrics", "")
        top_files[base_name] = str(file_path)

    for file_path in Path(bottom_dir).glob("*_metrics.json"):
        base_name = file_path.stem.replace("_metrics", "")
        bottom_files[base_name] = str(file_path)

    # Find matches
    matches = []
    for base_name in top_files:
        if base_name in bottom_files:
            output_name = f"{base_name}.stl"  # Default to STL for batch
            matches.append((top_files[base_name], bottom_files[base_name], output_name))
        else:
            print(f"Warning: No matching bottom file for '{base_name}'")

    return matches


def reconstruct_single(top_json: str, bottom_json: str, output_file: str, args) -> bool:
    """Reconstruct single mesh file from analysis data.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Import here to provide better error message if dependency missing
        try:
            from can_circularity_analysis.core.reconstruction import create_can_mesh_file
        except ImportError:
            print("Error: Trimesh and scipy are required for 3D reconstruction.")
            print("Install with: pip install trimesh scipy")
            return False

        if not args.quiet:
            print(f"Reconstructing: {os.path.basename(output_file)}")
            print(f"  Top: {os.path.basename(top_json)} (profile: {args.top_profile})")
            print(f"  Bottom: {os.path.basename(bottom_json)} (profile: {args.bottom_profile})")
            print(f"  Wall thickness: {args.wall_thickness} mm")
            print(f"  Height: {args.height} mm")
            print(f"  Resolution: {args.resolution} points")
            print(
                f"  Align profiles: {'Yes (straight walls)' if args.align_profiles else 'No (preserve orientations)'}"
            )
            if args.plot_profiles:
                print("  Will show profile plot before meshing...")

        # Create mesh file with enhanced parameters
        mesh_path = create_can_mesh_file(
            top_json,
            bottom_json,
            output_file,
            wall_thickness=args.wall_thickness,
            height=args.height,
            top_profile_type=args.top_profile,
            bottom_profile_type=args.bottom_profile,
            resolution=args.resolution,
            output_format=args.output_format,
            plot_profiles=args.plot_profiles,
            align_profiles=args.align_profiles,
        )

        if not args.quiet:
            print(f"  Created: {mesh_path}")

        return True

    except Exception as e:
        print(f"Error reconstructing {output_file}: {e}")
        import traceback

        traceback.print_exc()
        return False


def reconstruct_batch(args) -> None:
    """Perform batch reconstruction."""
    top_dir, bottom_dir = args.batch

    if not args.quiet:
        print("Batch reconstruction:")
        print(f"  Top directory: {top_dir}")
        print(f"  Bottom directory: {bottom_dir}")
        print(f"  Output directory: {args.out_dir}")
        print(f"  Top profile: {args.top_profile}")
        print(f"  Bottom profile: {args.bottom_profile}")

    # Find matching files
    matches = find_matching_files(top_dir, bottom_dir)

    if not matches:
        print("No matching analysis files found")
        return

    if not args.quiet:
        print(f"Found {len(matches)} matching pairs")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Process each match
    success_count = 0
    for top_file, bottom_file, output_name in matches:
        output_path = os.path.join(args.out_dir, output_name)

        if reconstruct_single(top_file, bottom_file, output_path, args):
            success_count += 1

    if not args.quiet:
        print("\nBatch reconstruction complete:")
        print(f"  Successful: {success_count}/{len(matches)}")
        print(f"  Output directory: {os.path.abspath(args.out_dir)}")


def main():
    """Main entry point for the reconstruct command."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_arguments(args)

    if args.batch:
        # Batch processing
        reconstruct_batch(args)
    else:
        # Single file processing
        success = reconstruct_single(args.top_json, args.bottom_json, args.output_file, args)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
