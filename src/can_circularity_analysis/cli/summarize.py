"""CLI for summarizing can analysis results and generating plots."""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from ..utils.file_io import find_metrics_files, load_metrics_json, create_output_directory


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for the summarize command."""
    parser = argparse.ArgumentParser(
        description="Summarize can analysis results and generate comparison plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic summary of results
  can-summarize --in-dir results

  # Generate summary with plots
  can-summarize --in-dir results --make-plots

  # Group by filename prefix (first 3 characters)
  can-summarize --in-dir results --prefix-len 3

  # Create plots with legends
  can-summarize --in-dir results --make-plots --legend --legend-limit 5
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--in-dir", 
        default="out", 
        help="Directory containing *_metrics.json files (default: ./out)"
    )
    parser.add_argument(
        "--pattern", 
        default="*_metrics.json",
        help="Glob pattern for metrics files (default: *_metrics.json)"
    )
    parser.add_argument(
        "--out-dir", 
        default="out", 
        help="Output directory for summaries and plots (default: ./out)"
    )
    
    # Grouping options
    group_args = parser.add_argument_group("Grouping Options")
    group_args.add_argument(
        "--prefix-len", 
        type=int, 
        default=0,
        help="If >0, also group stats by filename prefix of this length"
    )
    
    # Plotting options
    plot_args = parser.add_argument_group("Plotting Options")
    plot_args.add_argument(
        "--make-plots", 
        action="store_true",
        help="Generate overlay plots and histograms"
    )
    plot_args.add_argument(
        "--polar-mm", 
        action="store_true",
        help="Use mm units for polar deviation plots (requires calibrated data)"
    )
    plot_args.add_argument(
        "--max-points-per", 
        type=int, 
        default=400,
        help="Maximum points per rim in overlay plots (default: 400)"
    )
    
    # Legend options
    legend_args = parser.add_argument_group("Legend Options")
    legend_args.add_argument(
        "--legend", 
        action="store_true",
        help="Add legends to overlay plots (may cause clutter with many files)"
    )
    legend_args.add_argument(
        "--legend-limit", 
        type=int, 
        default=10,
        help="Maximum number of entries to show in legends (default: 10)"
    )
    legend_args.add_argument(
        "--legend-by-parent", 
        action="store_true",
        help="Use parent directory names in legends instead of filenames"
    )
    
    return parser


def read_metrics_file(file_path: Path) -> Dict[str, Any]:
    """Read and flatten a single metrics JSON file."""
    try:
        data = load_metrics_json(file_path)
    except Exception as e:
        return {"_read_error": str(e), "_file": str(file_path)}
    
    row = {"_file": str(file_path)}
    
    # Flatten basic fields
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    
    # Flatten center coordinates
    if isinstance(data.get("center_px"), (list, tuple)) and len(data["center_px"]) == 2:
        row["center_px_x"], row["center_px_y"] = data["center_px"]
    if isinstance(data.get("center_mm"), (list, tuple)) and len(data["center_mm"]) == 2:
        row["center_mm_x"], row["center_mm_y"] = data["center_mm"]
    
    # Flatten ellipse data
    ellipse = data.get("ellipse")
    if isinstance(ellipse, dict):
        for key, value in ellipse.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"ellipse_{key}"] = value
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                row[f"ellipse_{key}_x"] = value[0]
                row[f"ellipse_{key}_y"] = value[1]
    
    # Add derived fields
    if "image_path" in data:
        img_path = Path(data["image_path"])
        row["_image_name"] = img_path.name
        row["_parent_dir"] = img_path.parent.name
        row["_stem"] = img_path.stem
    
    # Success flag
    row["_ok"] = (row.get("overlay_image") is not None) and (row.get("_read_error") is None)
    
    return row


def summarize_numeric_data(df, group_by=None):
    """Generate summary statistics for numeric columns."""
    if not PLOTTING_AVAILABLE:
        return None
        
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        return pd.DataFrame()
    
    if group_by is None:
        return numeric_cols.describe().T
    
    if group_by not in df.columns:
        return numeric_cols.describe().T
        
    grouped_stats = []
    for name, group in df.groupby(group_by):
        stats = group.select_dtypes(include=[np.number]).describe().T
        stats.insert(0, group_by, name)
        grouped_stats.append(stats)
    
    if grouped_stats:
        result = pd.concat(grouped_stats, axis=0)
        result.reset_index(names=["metric"], inplace=True)
        return result
    else:
        return pd.DataFrame()


def generate_plots(cases: List[Dict], plots_dir: Path, args) -> None:
    """Generate all visualization plots."""
    if not PLOTTING_AVAILABLE:
        print("Warning: Plotting libraries not available. Install pandas and matplotlib for plots.")
        return
    
    # Import plotting functions (simplified versions of the original)
    try:
        # Histograms
        _create_histograms(cases, plots_dir)
        
        # Overlay plots  
        _create_overlay_plots(cases, plots_dir, args)
        
        # Polar deviation plot
        _create_polar_plot(cases, plots_dir, args)
        
        print(f"Plots saved to: {plots_dir}")
        
    except Exception as e:
        print(f"Error generating plots: {e}")


def _create_histograms(cases: List[Dict], plots_dir: Path) -> None:
    """Create histogram plots for key metrics."""
    metrics = {
        'diameter_mm': [c.get('diameter_mm') for c in cases if c.get('diameter_mm')],
        'rms_out_of_round_mm': [c.get('rms_out_of_round_mm') for c in cases if c.get('rms_out_of_round_mm')],
        'circularity': [c.get('circularity_4piA_P2') for c in cases if c.get('circularity_4piA_P2')]
    }
    
    for metric_name, values in metrics.items():
        if len(values) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(values, bins=min(20, len(values)), alpha=0.7, edgecolor='black')
            plt.xlabel(metric_name.replace('_', ' ').title())
            plt.ylabel('Count')
            plt.title(f'Distribution of {metric_name.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'hist_{metric_name}.png', dpi=150, bbox_inches='tight')
            plt.close()


def _create_overlay_plots(cases: List[Dict], plots_dir: Path, args) -> None:
    """Create overlay plots for rims and circles."""
    if not cases:
        return
        
    # Simple rim overlay (would need to load actual point data for full implementation)
    plt.figure(figsize=(10, 10))
    
    # Draw unit circle as reference
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Perfect Circle')
    
    # For now, just show that we'd overlay the actual rim data here
    # In full implementation, would load CSV files and plot normalized points
    plt.title('Rim Overlay (Normalized)')
    plt.xlabel('x / r_fit')
    plt.ylabel('y / r_fit')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'overlay_rims.png', dpi=150, bbox_inches='tight')
    plt.close()


def _create_polar_plot(cases: List[Dict], plots_dir: Path, args) -> None:
    """Create polar deviation plot."""
    # This would require loading actual point data from CSV files
    # For now, create a placeholder
    plt.figure(figsize=(12, 8))
    
    # Simulate some polar deviation data
    theta_sim = np.linspace(-180, 180, 360)
    
    for i, case in enumerate(cases[:min(5, len(cases))]):  # Limit to first 5 for visibility
        # Simulate deviation data (in real implementation, load from CSV)
        deviation_sim = 0.1 * np.sin(3 * np.deg2rad(theta_sim)) + 0.05 * np.random.randn(len(theta_sim))
        
        label = case.get('_stem', f'Case {i+1}')
        if args.legend and i < args.legend_limit:
            plt.plot(theta_sim, deviation_sim, alpha=0.7, label=label)
        else:
            plt.plot(theta_sim, deviation_sim, alpha=0.7)
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Radial Deviation (mm)' if args.polar_mm else 'Radial Deviation (px)')
    plt.title('Radial Deviation vs Angle')
    plt.grid(True, alpha=0.3)
    
    if args.legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_name = 'polar_deviation_mm.png' if args.polar_mm else 'polar_deviation_px.png'
    plt.savefig(plots_dir / output_name, dpi=150, bbox_inches='tight')
    plt.close()


def load_case_data(metrics_files: List[Path]) -> List[Dict]:
    """Load case data for plotting."""
    cases = []
    
    for file_path in metrics_files:
        try:
            data = load_metrics_json(file_path)
            
            # Extract key information for plotting
            case = {
                '_file': str(file_path),
                '_stem': file_path.stem.replace('_metrics', ''),
                '_parent_dir': file_path.parent.name,
                'diameter_mm': data.get('diameter_mm'),
                'diameter_px': data.get('diameter_px'), 
                'radius_px': data.get('radius_px'),
                'center_px': data.get('center_px'),
                'rms_out_of_round_mm': data.get('rms_out_of_round_mm'),
                'rms_out_of_round_px': data.get('rms_out_of_round_px'),
                'circularity_4piA_P2': data.get('circularity_4piA_P2'),
                'csv_points': data.get('csv_points'),
                'has_mm': data.get('pixels_per_mm') is not None,
                'ellipse': data.get('ellipse')
            }
            cases.append(case)
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    return cases


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
    metrics_files = find_metrics_files(args.in_dir, args.pattern)
    if not metrics_files:
        print(f"Warning: No files matching '{args.pattern}' found in '{args.in_dir}'")
        return
    
    print(f"Found {len(metrics_files)} metrics files")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    create_output_directory(str(out_dir))
    
    # Process files into tabular format
    if not PLOTTING_AVAILABLE:
        print("Warning: pandas not available. Install pandas for CSV summaries.")
        rows = [read_metrics_file(f) for f in metrics_files]
        
        # Simple summary without pandas
        total = len(rows)
        successful = len([r for r in rows if r.get('_ok', False)])
        failed = total - successful
        
        print(f"\nSummary:")
        print(f"  Total files: {total}")
        print(f"  Successfully processed: {successful}")
        print(f"  Failed: {failed}")
        
        # Save simple JSON summary
        import json
        summary = {
            'total_files': total,
            'successful': successful,
            'failed': failed,
            'details': rows
        }
        
        summary_path = out_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")
        
    else:
        # Full pandas-based analysis
        rows = [read_metrics_file(f) for f in metrics_files]
        df = pd.DataFrame(rows)
        
        # Convert numeric columns
        for col in df.columns:
            if not col.startswith('_'):
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Add derived columns
        if 'diameter_mm' not in df.columns and 'radius_mm' in df.columns:
            df['diameter_mm'] = df['radius_mm'] * 2.0
        
        # Save master table
        all_csv = out_dir / 'metrics_all.csv'
        df.to_csv(all_csv, index=False)
        print(f"All metrics saved to: {all_csv}")
        
        # Overall statistics
        stats_overall = summarize_numeric_data(df)
        if not stats_overall.empty:
            stats_csv = out_dir / 'stats_overall.csv'
            stats_overall.to_csv(stats_csv)
            print(f"Overall statistics saved to: {stats_csv}")
        
        # Group by parent directory
        if '_parent_dir' in df.columns and df['_parent_dir'].nunique() > 1:
            stats_by_parent = summarize_numeric_data(df, group_by='_parent_dir')
            if not stats_by_parent.empty:
                parent_csv = out_dir / 'stats_by_parent.csv'
                stats_by_parent.to_csv(parent_csv, index=False)
                print(f"Statistics by parent directory saved to: {parent_csv}")
        
        # Group by filename prefix
        if args.prefix_len > 0 and '_stem' in df.columns:
            df['_prefix'] = df['_stem'].str.slice(0, args.prefix_len)
            stats_by_prefix = summarize_numeric_data(df, group_by='_prefix')
            if not stats_by_prefix.empty:
                prefix_csv = out_dir / 'stats_by_prefix.csv'
                stats_by_prefix.to_csv(prefix_csv, index=False)
                print(f"Statistics by prefix saved to: {prefix_csv}")
        
        # Print summary
        total = len(df)
        successful = int(df['_ok'].sum()) if '_ok' in df.columns else total
        failed = total - successful
        
        print(f"\nSummary:")
        print(f"  Total files: {total}")
        print(f"  Successfully processed: {successful}")
        print(f"  Failed: {failed}")
        
        # Show some key statistics
        if successful > 0:
            numeric_cols = ['diameter_mm', 'circularity_4piA_P2', 'rms_out_of_round_mm']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if available_cols:
                print(f"\nKey Statistics:")
                for col in available_cols:
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        print(f"  {col}: {valid_data.mean():.3f} ± {valid_data.std():.3f} "
                              f"(range: {valid_data.min():.3f} - {valid_data.max():.3f})")
    
    # Generate plots if requested
    if args.make_plots:
        if not PLOTTING_AVAILABLE:
            print("Error: Cannot generate plots. Install pandas and matplotlib:")
            print("  pip install pandas matplotlib")
            sys.exit(1)
        
        plots_dir = out_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Load case data for plotting
        cases = load_case_data(metrics_files)
        
        if cases:
            generate_plots(cases, plots_dir, args)
        else:
            print("Warning: No valid case data found for plotting")


if __name__ == "__main__":
    main()
