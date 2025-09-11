"""CLI for summarizing can analysis results and generating plots."""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def create_argument_parser():
    """Create argument parser for the summarize command."""
    parser = argparse.ArgumentParser(description="Summarize can analysis results and generate plots.")
    parser.add_argument("--in-dir", default="out", help="Directory containing *_metrics.json files (default: ./out)")
    parser.add_argument("--out-dir", default="out", help="Output directory for summaries and plots (default: ./out)")
    parser.add_argument("--make-plots", action="store_true", help="Generate overlay plots and histograms")
    return parser


def load_case_data(metrics_files):
    """Load actual rim data from metrics and CSV files."""
    cases = []

    for metrics_file in metrics_files:
        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Find corresponding CSV file
            csv_path = metrics.get("csv_points")
            if not csv_path or not Path(csv_path).exists():
                # Try to construct CSV path from metrics file name
                csv_path = str(metrics_file).replace("_metrics.json", "_points.csv")

            if not Path(csv_path).exists():
                print(f"Warning: CSV file not found for {metrics_file}")
                continue

            # Load rim points
            df = pd.read_csv(csv_path)

            if df.empty:
                print(f"Warning: Empty CSV file {csv_path}")
                continue

            # Extract center and radius from metrics
            center_px = metrics.get("center_px", [0, 0])
            radius_px = metrics.get("radius_px", 1)

            # Get rim points relative to center
            x_pts = df["x_px"].values - center_px[0]
            y_pts = df["y_px"].values - center_px[1]

            # Normalize by radius
            x_norm = x_pts / radius_px
            y_norm = y_pts / radius_px

            case = {
                "name": Path(metrics_file).stem.replace("_metrics", ""),
                "x_norm": x_norm,
                "y_norm": y_norm,
                "diameter_mm": metrics.get("diameter_mm"),
                "diameter_px": metrics.get("diameter_px"),
                "circularity": metrics.get("circularity_4piA_P2"),
                "rms_mm": metrics.get("rms_out_of_round_mm"),
                "rms_px": metrics.get("rms_out_of_round_px"),
                "std_mm": metrics.get("std_out_of_round_mm"),
                "range_mm": metrics.get("range_out_of_round_mm"),
            }
            cases.append(case)

        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
            continue

    return cases


def create_rim_overlay_plot(cases, output_path):
    """Create overlay plot with actual rim data."""
    plt.figure(figsize=(14, 10))  # Made slightly wider to accommodate legend

    # Plot perfect circle reference
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.8, linewidth=2, label="Perfect Circle")

    # Plot actual rim data
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(cases), 10)))

    # Always label up to 12 cases for legend
    max_legend_items = 32

    for i, case in enumerate(cases):
        color = colors[i % len(colors)]
        alpha = 0.7 if len(cases) <= 5 else 0.5
        size = 2 if len(cases) <= 5 else 1

        # Show label for first max_legend_items cases
        label = case["name"] if i < max_legend_items else ""

        plt.scatter(case["x_norm"], case["y_norm"], c=[color], alpha=alpha, s=size, label=label)

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x / r_fit")
    plt.ylabel("y / r_fit")
    plt.title(f"Rim Overlay (Normalized) - {len(cases)} samples")

    # Always show legend, but adjust based on number of cases
    if len(cases) > max_legend_items:
        # Add a note about additional unlabeled cases
        plt.figtext(
            0.02,
            0.02,
            (
                f"Note: Showing first {max_legend_items} labels. "
                f"{len(cases) - max_legend_items} additional cases unlabeled."
            ),
            fontsize=8,
            style="italic",
        )

    # Position legend outside plot area
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Created: {output_path}")


def create_histogram_plots(cases, plots_dir):
    """Create histogram plots for key metrics."""
    # Diameter histogram
    diameters = [c["diameter_mm"] for c in cases if c["diameter_mm"] is not None]
    if diameters:
        plt.figure(figsize=(8, 6))
        plt.hist(diameters, bins=min(15, len(diameters)), alpha=0.7, edgecolor="black")
        plt.xlabel("Diameter (mm)")
        plt.ylabel("Count")
        plt.title(f"Diameter Distribution (n={len(diameters)})")
        plt.grid(True, alpha=0.3)

        # Add statistics text
        mean_d = np.mean(diameters)
        std_d = np.std(diameters)
        plt.axvline(mean_d, color="red", linestyle="--", alpha=0.8)
        plt.text(
            0.02,
            0.98,
            (f"Mean: {mean_d:.2f}±{std_d:.2f} mm\n" f"Range: {min(diameters):.2f}-{max(diameters):.2f} mm"),
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.savefig(plots_dir / "diameter_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Created: diameter_histogram.png")

    # RMS out-of-round histogram
    rms_values = [c["rms_mm"] for c in cases if c["rms_mm"] is not None]
    if rms_values:
        plt.figure(figsize=(8, 6))
        plt.hist(rms_values, bins=min(15, len(rms_values)), alpha=0.7, edgecolor="black", color="orange")
        plt.xlabel("RMS Out-of-Round (mm)")
        plt.ylabel("Count")
        plt.title(f"Out-of-Round Distribution (n={len(rms_values)})")
        plt.grid(True, alpha=0.3)

        # Add statistics text
        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)
        plt.axvline(mean_rms, color="red", linestyle="--", alpha=0.8)
        plt.text(
            0.02,
            0.98,
            f"Mean: {mean_rms:.3f}±{std_rms:.3f} mm\nRange: {min(rms_values):.3f}-{max(rms_values):.3f} mm",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.savefig(plots_dir / "rms_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Created: rms_histogram.png")

    # Circularity histogram
    circularities = [c["circularity"] for c in cases if c["circularity"] is not None]
    if circularities:
        plt.figure(figsize=(8, 6))
        plt.hist(
            circularities,
            bins=min(15, len(circularities)),
            alpha=0.7,
            edgecolor="black",
            color="green",
        )
        plt.xlabel("Circularity (4πA/P²)")
        plt.ylabel("Count")
        plt.title(f"Circularity Distribution (n={len(circularities)})")
        plt.grid(True, alpha=0.3)

        # Add statistics and ideal line
        mean_circ = np.mean(circularities)
        std_circ = np.std(circularities)
        plt.axvline(1.0, color="blue", linestyle="--", alpha=0.8, label="Perfect Circle")
        plt.axvline(mean_circ, color="red", linestyle="--", alpha=0.8, label="Mean")
        plt.text(
            0.02,
            0.98,
            f"Mean: {mean_circ:.3f}±{std_circ:.3f}\nRange: {min(circularities):.3f}-{max(circularities):.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig(plots_dir / "circularity_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Created: circularity_histogram.png")


def print_summary_stats(cases):
    """Print summary statistics to console."""
    if not cases:
        print("No data to summarize")
        return

    print(f"\nSummary Statistics (n={len(cases)} samples):")
    print("=" * 50)

    # Diameter stats
    diameters = [c["diameter_mm"] for c in cases if c["diameter_mm"] is not None]
    if diameters:
        print("Diameter (mm):")
        print(f"  Mean: {np.mean(diameters):.2f} ± {np.std(diameters):.2f}")
        print(f"  Range: {np.min(diameters):.2f} - {np.max(diameters):.2f}")
        print(f"  Median: {np.median(diameters):.2f}")

    # Circularity stats
    circularities = [c["circularity"] for c in cases if c["circularity"] is not None]
    if circularities:
        print("\nCircularity (4πA/P²):")
        print(f"  Mean: {np.mean(circularities):.3f} ± {np.std(circularities):.3f}")
        print(f"  Range: {np.min(circularities):.3f} - {np.max(circularities):.3f}")
        print(f"  Median: {np.median(circularities):.3f}")

    # Out-of-round stats
    rms_values = [c["rms_mm"] for c in cases if c["rms_mm"] is not None]
    if rms_values:
        print("\nRMS Out-of-Round (mm):")
        print(f"  Mean: {np.mean(rms_values):.3f} ± {np.std(rms_values):.3f}")
        print(f"  Range: {np.min(rms_values):.3f} - {np.max(rms_values):.3f}")
        print(f"  Median: {np.median(rms_values):.3f}")


def main():
    """Main entry point for the summarize command."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Check if input directory exists
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"Error: Input directory '{args.in_dir}' does not exist")
        sys.exit(1)

    # Find metrics files
    metrics_files = list(in_dir.glob("*_metrics.json"))
    if not metrics_files:
        print(f"No *_metrics.json files found in '{args.in_dir}'")
        return

    print(f"Found {len(metrics_files)} metrics files in {in_dir}")

    if not PLOTTING_AVAILABLE:
        print("Warning: pandas/matplotlib not available. Install with: pip install pandas matplotlib")
        return

    # Load actual data
    print("Loading rim data from CSV files...")
    cases = load_case_data(metrics_files)

    if not cases:
        print("Error: No valid data could be loaded from CSV files")
        return

    print(f"Successfully loaded {len(cases)} cases with rim data")

    # Print summary statistics
    print_summary_stats(cases)

    # Generate plots if requested
    if args.make_plots:
        print("\nGenerating plots...")
        plots_dir = Path(args.out_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create rim overlay plot
        create_rim_overlay_plot(cases, plots_dir / "rim_overlay.png")

        # Create histogram plots
        create_histogram_plots(cases, plots_dir)

        print(f"\nAll plots saved to: {plots_dir}")
        print("\nGenerated files:")
        print("  - rim_overlay.png (normalized rim shapes)")
        print("  - diameter_histogram.png")
        print("  - rms_histogram.png")
        print("  - circularity_histogram.png")
    else:
        print("\nUse --make-plots to generate visualization plots")


if __name__ == "__main__":
    main()
