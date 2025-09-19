"""Enhanced 3D reconstruction module using Trimesh - compatible with v4.8+

This module creates 3D models by lofting between two circular/oval profiles
with enhanced profile control and proper wall thickness implementation.
Now supports best contour data for maximum accuracy.
"""

import json
import math
import os
import warnings
from pathlib import Path

try:
    import numpy as np
    import pandas as pd

    NUMPY_PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_PANDAS_AVAILABLE = False

try:
    import trimesh
    from scipy.interpolate import interp1d

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class CanReconstructionError(Exception):
    """Custom exception for reconstruction errors."""

    pass


class EnhancedCanMeshGenerator:
    """Enhanced generator for 3D mesh files with profile control and proper wall thickness."""

    def __init__(self):
        if not TRIMESH_AVAILABLE:
            raise ImportError(
                "Trimesh and scipy are required for 3D reconstruction. " "Install with: pip install trimesh scipy"
            )
        if not NUMPY_PANDAS_AVAILABLE:
            raise ImportError(
                "NumPy and pandas are required for data processing. " "Install with: pip install numpy pandas"
            )

    def create_mesh_file(
        self,
        top_json_path: str,
        bottom_json_path: str,
        output_path: str,
        wall_thickness_mm: float = 0.1,
        height_mm: float = 100.0,
        top_profile_type: str = "best_contour",
        bottom_profile_type: str = "best_contour",
        resolution: int = 64,
        output_format: str = "auto",
        plot_profiles: bool = False,
        align_profiles: bool = True,
    ) -> str:
        """Create 3D mesh file from two can analysis JSON files with enhanced control.

        Args:
            top_json_path: Path to top rim analysis JSON
            bottom_json_path: Path to bottom rim analysis JSON
            output_path: Output file path
            wall_thickness_mm: Wall thickness in mm
            height_mm: Height between top and bottom profiles
            top_profile_type: Profile type for top ("circle", "ellipse", "best_contour", "measured")
            bottom_profile_type: Profile type for bottom ("circle", "ellipse", "best_contour", "measured")
            resolution: Number of points around each profile
            output_format: Output format ("stl", "ply", "obj", "auto")
            plot_profiles: Whether to show 3D plot of profiles before meshing
            align_profiles: Whether to align profile orientations for straight walls

        Returns:
            Absolute path to created file
        """
        # Load and validate analysis data
        top_data = self._load_analysis_data(top_json_path)
        bottom_data = self._load_analysis_data(bottom_json_path)

        # Validate compatibility
        self._validate_compatibility(top_data, bottom_data)

        # Determine output format
        if output_format == "auto":
            output_format = self._detect_format(output_path)

        # Determine reference angle for alignment if needed
        reference_angle = 0.0  # Default to 0 degrees
        if align_profiles:
            # Use top ellipse angle as reference, or 0 if not available
            top_ellipse = top_data.get("ellipse", {})
            reference_angle = top_ellipse.get("angle_deg", 0.0)

        # Create profiles with specified types
        top_midline = self._create_profile_by_type(
            top_data, height_mm, top_profile_type, resolution, reference_angle if align_profiles else None
        )
        bottom_midline = self._create_profile_by_type(
            bottom_data, 0.0, bottom_profile_type, resolution, reference_angle if align_profiles else None
        )

        # Generate wall profiles (inner and outer) from midlines
        top_inner, top_outer = self._generate_wall_profiles(top_midline, wall_thickness_mm)
        bottom_inner, bottom_outer = self._generate_wall_profiles(bottom_midline, wall_thickness_mm)

        # Plot profiles if requested
        if plot_profiles:
            self._plot_profiles_3d(
                top_midline,
                bottom_midline,
                top_inner,
                top_outer,
                bottom_inner,
                bottom_outer,
                top_profile_type,
                bottom_profile_type,
                wall_thickness_mm,
                height_mm,
            )

        # Create complete mesh with proper wall geometry
        mesh = self._create_walled_mesh(bottom_inner, bottom_outer, top_inner, top_outer)

        # Export mesh
        output_path = self._export_mesh(
            mesh,
            output_path,
            output_format,
            top_data,
            bottom_data,
            wall_thickness_mm,
            height_mm,
            top_profile_type,
            bottom_profile_type,
        )

        return output_path

    def _create_profile_by_type(
        self, analysis_data: dict, z_height: float, profile_type: str, resolution: int, override_angle: float = None
    ) -> np.ndarray:
        """Create profile curve based on specified type.

        Args:
            override_angle: If provided, override the ellipse rotation angle for alignment

        Returns:
            Array of 3D points [[x, y, z], ...] centered at origin
        """
        if profile_type == "best_contour":
            return self._create_best_contour_profile(analysis_data, z_height, resolution)
        elif profile_type == "measured":
            return self._create_measured_profile(analysis_data, z_height, resolution)
        elif profile_type == "ellipse":
            return self._create_ellipse_profile(analysis_data, z_height, resolution, override_angle)
        elif profile_type == "circle":
            return self._create_circle_profile(analysis_data, z_height, resolution)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

    def _create_best_contour_profile(self, analysis_data: dict, z_height: float, resolution: int) -> np.ndarray:
        """Create profile from best contour fitting data."""
        best_contour_data = analysis_data.get("best_contour")

        if not best_contour_data or "error" in best_contour_data:
            warnings.warn(f"Best contour data not available or failed: {best_contour_data}, falling back to ellipse")
            return self._create_ellipse_profile(analysis_data, z_height, resolution)

        try:
            # Check if we have the radii and angles arrays
            if "radii" in best_contour_data and "angles" in best_contour_data:
                radii = np.array(best_contour_data["radii"])
                angles = np.array(best_contour_data["angles"])

                # Convert from pixels to mm if needed
                mm_per_pixel = self._get_mm_per_pixel(analysis_data)
                if mm_per_pixel:
                    radii = radii * mm_per_pixel
                else:
                    warnings.warn("No calibration data found, using pixel coordinates as mm")

                # Resample to target resolution if needed
                if len(radii) != resolution:
                    radii = self._resample_polar_data(angles, radii, resolution)

                # Create 3D points centered at origin
                angles_target = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
                x_coords = radii * np.cos(angles_target)
                y_coords = radii * np.sin(angles_target)
                z_coords = np.full(resolution, z_height)

                return np.column_stack([x_coords, y_coords, z_coords])

            elif "points" in best_contour_data:
                # Use the contour points directly
                points = np.array(best_contour_data["points"])

                if len(points) < 3:
                    raise CanReconstructionError("Insufficient points in best contour data")

                # Convert to mm and center at origin
                mm_per_pixel = self._get_mm_per_pixel(analysis_data)
                if mm_per_pixel:
                    points_mm = points * mm_per_pixel
                else:
                    points_mm = points
                    warnings.warn("No calibration data found, using pixel coordinates as mm")

                # Center at origin
                center_x = np.mean(points_mm[:, 0])
                center_y = np.mean(points_mm[:, 1])
                x_centered = points_mm[:, 0] - center_x
                y_centered = points_mm[:, 1] - center_y

                # Convert to polar and resample
                angles = np.arctan2(y_centered, x_centered)
                radii = np.sqrt(x_centered**2 + y_centered**2)

                # Sort by angle
                sorted_indices = np.argsort(angles)
                angles_sorted = angles[sorted_indices]
                radii_sorted = radii[sorted_indices]

                # Resample to target resolution
                radii_resampled = self._resample_polar_data(angles_sorted, radii_sorted, resolution)

                # Create 3D points
                angles_target = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
                x_coords = radii_resampled * np.cos(angles_target)
                y_coords = radii_resampled * np.sin(angles_target)
                z_coords = np.full(resolution, z_height)

                return np.column_stack([x_coords, y_coords, z_coords])

            else:
                raise CanReconstructionError("Best contour data missing required fields (radii/angles or points)")

        except Exception as e:
            warnings.warn(f"Failed to use best contour data ({e}), falling back to ellipse")
            return self._create_ellipse_profile(analysis_data, z_height, resolution)

    def _resample_polar_data(self, angles: np.ndarray, radii: np.ndarray, target_points: int) -> np.ndarray:
        """Resample polar coordinate data to target number of points."""
        if len(angles) == target_points:
            return radii

        # Extend for circular interpolation
        angles_extended = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])
        radii_extended = np.tile(radii, 3)

        # Create interpolation function
        interp_func = interp1d(
            angles_extended, radii_extended, kind="linear", bounds_error=False, fill_value="extrapolate"
        )

        # Generate target angles and interpolate
        angles_target = np.linspace(0, 2 * np.pi, target_points, endpoint=False)
        radii_resampled = interp_func(angles_target)

        # Ensure positive radii
        radii_resampled = np.maximum(radii_resampled, 0.1)

        return radii_resampled

    def _get_mm_per_pixel(self, analysis_data: dict) -> float | None:
        """Get mm per pixel conversion factor from analysis data."""
        pixels_per_mm = analysis_data.get("pixels_per_mm")
        if pixels_per_mm and pixels_per_mm > 0:
            return 1.0 / pixels_per_mm
        return None

    def _create_measured_profile(self, analysis_data: dict, z_height: float, resolution: int) -> np.ndarray:
        """Create profile from actual measured rim points."""
        csv_path = analysis_data.get("csv_points")
        if not csv_path or not os.path.exists(csv_path):
            warnings.warn(f"CSV file not found: {csv_path}, falling back to ellipse")
            return self._create_ellipse_profile(analysis_data, z_height, resolution)

        try:
            # Load points from CSV
            df = pd.read_csv(csv_path)

            if df.empty or len(df) < 3:
                raise CanReconstructionError("Insufficient points in CSV file")

            # Get coordinates in mm
            x_coords, y_coords = self._get_coordinates_mm(df, analysis_data)

            # Center the coordinates at origin
            x_coords = x_coords - np.mean(x_coords)
            y_coords = y_coords - np.mean(y_coords)

            # Sort points by angle to ensure proper ordering
            angles = np.arctan2(y_coords, x_coords)
            sorted_indices = np.argsort(angles)
            x_coords = x_coords[sorted_indices]
            y_coords = y_coords[sorted_indices]

            # Resample to target resolution
            if len(x_coords) != resolution:
                x_coords, y_coords = self._resample_curve(x_coords, y_coords, resolution)

            # Create 3D points
            points_3d = np.column_stack([x_coords, y_coords, np.full(resolution, z_height)])

            return points_3d

        except Exception as e:
            warnings.warn(f"Failed to use measured points ({e}), falling back to ellipse")
            return self._create_ellipse_profile(analysis_data, z_height, resolution)

    def _create_ellipse_profile(
        self, analysis_data: dict, z_height: float, resolution: int, override_angle: float = None
    ) -> np.ndarray:
        """Create elliptical profile from fitted ellipse data."""
        ellipse_data = analysis_data.get("ellipse")

        if not ellipse_data or "major_px" not in ellipse_data:
            warnings.warn("No ellipse data found, falling back to circle")
            return self._create_circle_profile(analysis_data, z_height, resolution)

        # Get ellipse parameters in mm
        if "major_mm" in ellipse_data and "minor_mm" in ellipse_data:
            major_radius = ellipse_data["major_mm"] / 2.0
            minor_radius = ellipse_data["minor_mm"] / 2.0
        else:
            # Convert from pixels
            mm_per_pixel = self._get_mm_per_pixel(analysis_data)
            if mm_per_pixel:
                major_radius = ellipse_data["major_px"] * mm_per_pixel / 2.0
                minor_radius = ellipse_data["minor_px"] * mm_per_pixel / 2.0
            else:
                warnings.warn("No calibration data found, using pixel radius as mm")
                major_radius = ellipse_data["major_px"] / 2.0
                minor_radius = ellipse_data["minor_px"] / 2.0

        # Get rotation angle - use override if provided
        if override_angle is not None:
            angle_deg = override_angle
        else:
            angle_deg = ellipse_data.get("angle_deg", 0.0)

        angle_rad = math.radians(angle_deg)

        # Create ellipse points centered at origin
        t = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        x_ellipse = major_radius * np.cos(t)
        y_ellipse = minor_radius * np.sin(t)

        # Apply rotation
        x_coords = x_ellipse * np.cos(angle_rad) - y_ellipse * np.sin(angle_rad)
        y_coords = x_ellipse * np.sin(angle_rad) + y_ellipse * np.cos(angle_rad)
        z_coords = np.full(resolution, z_height)

        return np.column_stack([x_coords, y_coords, z_coords])

    def _create_circle_profile(self, analysis_data: dict, z_height: float, resolution: int) -> np.ndarray:
        """Create circular profile from fitted circle data."""
        # Get radius in mm
        radius_mm = self._get_radius_mm(analysis_data)

        # Create circle points centered at origin
        angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        x_coords = radius_mm * np.cos(angles)
        y_coords = radius_mm * np.sin(angles)
        z_coords = np.full(resolution, z_height)

        return np.column_stack([x_coords, y_coords, z_coords])

    def _get_coordinates_mm(self, df: pd.DataFrame, analysis_data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract coordinates in mm from DataFrame."""
        if "x_mm" in df.columns and "y_mm" in df.columns:
            return df["x_mm"].values, df["y_mm"].values
        elif "x_px" in df.columns and "y_px" in df.columns:
            # Convert pixels to mm if calibration available
            mm_per_pixel = self._get_mm_per_pixel(analysis_data)
            if mm_per_pixel:
                return df["x_px"].values * mm_per_pixel, df["y_px"].values * mm_per_pixel
            else:
                # No calibration - use pixels as mm (warn user)
                warnings.warn("No calibration data found, using pixel coordinates as mm")
                return df["x_px"].values, df["y_px"].values
        else:
            raise CanReconstructionError("No valid coordinate columns found in CSV")

    def _get_radius_mm(self, analysis_data: dict) -> float:
        """Get radius in mm from analysis data."""
        if "radius_mm" in analysis_data:
            return analysis_data["radius_mm"]
        else:
            # Convert from pixels
            radius_px = analysis_data["radius_px"]
            mm_per_pixel = self._get_mm_per_pixel(analysis_data)
            if mm_per_pixel:
                return radius_px * mm_per_pixel
            else:
                warnings.warn("No calibration data found, using pixel radius as mm")
                return radius_px

    def _resample_curve(
        self, x_coords: np.ndarray, y_coords: np.ndarray, target_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample curve to specified number of points using interpolation."""
        # Calculate angles and radii for polar interpolation
        angles_orig = np.arctan2(y_coords, x_coords)
        radii_orig = np.sqrt(x_coords**2 + y_coords**2)

        # Ensure angles are increasing
        angles_sorted = np.sort(angles_orig)
        sorted_indices = np.argsort(angles_orig)
        radii_sorted = radii_orig[sorted_indices]

        # Create target angles
        angles_new = np.linspace(0, 2 * np.pi, target_points, endpoint=False)

        # Interpolate radii with periodic boundary conditions
        angles_extended = np.concatenate([angles_sorted - 2 * np.pi, angles_sorted, angles_sorted + 2 * np.pi])
        radii_extended = np.tile(radii_sorted, 3)

        interp_func = interp1d(angles_extended, radii_extended, kind="linear")
        radii_new = interp_func(angles_new)

        # Convert back to Cartesian
        x_new = radii_new * np.cos(angles_new)
        y_new = radii_new * np.sin(angles_new)

        return x_new, y_new

    # ... (rest of the methods remain the same: _generate_wall_profiles, _create_walled_mesh, etc.)
    # I'll include the key remaining methods below

    def _generate_wall_profiles(self, midline: np.ndarray, wall_thickness: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate inner and outer wall profiles from midline profile."""
        half_thickness = wall_thickness / 2.0

        # Calculate inward and outward normals for each point
        inner_profile = []
        outer_profile = []

        n_points = len(midline)

        for i in range(n_points):
            # Get current point and neighbors (with wraparound)
            prev_i = (i - 1) % n_points
            next_i = (i + 1) % n_points

            curr_pt = midline[i]
            prev_pt = midline[prev_i]
            next_pt = midline[next_i]

            # Calculate tangent vector (average of incoming and outgoing)
            tangent1 = curr_pt[:2] - prev_pt[:2]  # Only x,y components
            tangent2 = next_pt[:2] - curr_pt[:2]
            tangent = (tangent1 + tangent2) / 2.0

            # Normalize tangent
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > 1e-10:
                tangent = tangent / tangent_len
            else:
                tangent = np.array([1.0, 0.0])  # Fallback

            # Calculate normal (90 degrees CCW from tangent)
            normal = np.array([-tangent[1], tangent[0]])

            # Create inner and outer points
            inner_pt = curr_pt.copy()
            outer_pt = curr_pt.copy()

            inner_pt[:2] = curr_pt[:2] - normal * half_thickness
            outer_pt[:2] = curr_pt[:2] + normal * half_thickness

            inner_profile.append(inner_pt)
            outer_profile.append(outer_pt)

        return np.array(inner_profile), np.array(outer_profile)

    def _create_walled_mesh(
        self, bottom_inner: np.ndarray, bottom_outer: np.ndarray, top_inner: np.ndarray, top_outer: np.ndarray
    ) -> trimesh.Trimesh:
        """Create complete mesh with proper wall geometry."""
        vertices = []
        faces = []

        # Add all vertices
        all_profiles = [bottom_inner, bottom_outer, top_inner, top_outer]
        for profile in all_profiles:
            vertices.extend(profile)

        vertices = np.array(vertices)
        n_points = len(bottom_inner)

        # Helper function to add quad faces (split into two triangles)
        def add_quad_faces(v1_start, v2_start, reverse=False):
            nonlocal faces
            for i in range(n_points):
                next_i = (i + 1) % n_points

                # Quad vertices
                v1 = v1_start + i
                v2 = v1_start + next_i
                v3 = v2_start + next_i
                v4 = v2_start + i

                if reverse:
                    # Reverse winding for inward-facing surfaces
                    faces.append([v1, v4, v2])
                    faces.append([v2, v4, v3])
                else:
                    # Normal winding for outward-facing surfaces
                    faces.append([v1, v2, v4])
                    faces.append([v2, v3, v4])

        # Vertex layout:
        # 0 to n_points-1: bottom_inner
        # n_points to 2*n_points-1: bottom_outer
        # 2*n_points to 3*n_points-1: top_inner
        # 3*n_points to 4*n_points-1: top_outer

        # Bottom ring surface (between bottom_inner and bottom_outer)
        add_quad_faces(0, n_points, reverse=True)

        # Top ring surface (between top_inner and top_outer)
        add_quad_faces(2 * n_points, 3 * n_points, reverse=False)

        # Inner wall (connecting bottom_inner to top_inner)
        add_quad_faces(0, 2 * n_points, reverse=True)

        # Outer wall (connecting bottom_outer to top_outer)
        add_quad_faces(n_points, 3 * n_points, reverse=False)

        # Create the mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Clean up the mesh
        try:
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.fill_holes()
        except:
            pass  # Continue even if cleanup fails

        return mesh

    # ... (include other helper methods like _detect_format, _load_analysis_data, etc.)
    def _detect_format(self, output_path: str) -> str:
        """Detect output format from file extension."""
        ext = Path(output_path).suffix.lower()
        format_map = {
            ".stl": "stl",
            ".ply": "ply",
            ".obj": "obj",
        }
        return format_map.get(ext, "stl")

    def _load_analysis_data(self, json_path: str) -> dict:
        """Load and validate analysis JSON data."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Analysis file not found: {json_path}")

        try:
            with open(json_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}")

        # Validate required fields
        required_fields = ["center_px", "radius_px"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(f"Missing required fields in {json_path}: {missing_fields}")

        return data

    def _validate_compatibility(self, top_data: dict, bottom_data: dict) -> None:
        """Validate that top and bottom data are compatible for reconstruction."""
        # Check if both have calibration or neither
        top_has_calib = "pixels_per_mm" in top_data and top_data["pixels_per_mm"] is not None
        bottom_has_calib = "pixels_per_mm" in bottom_data and bottom_data["pixels_per_mm"] is not None

        if top_has_calib != bottom_has_calib:
            warnings.warn("One rim has calibration data and the other doesn't. " "Results may not be accurate.")

    def _export_mesh(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        output_format: str,
        top_data: dict,
        bottom_data: dict,
        wall_thickness: float,
        height: float,
        top_profile_type: str,
        bottom_profile_type: str,
    ) -> str:
        """Export mesh to specified format."""

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Add extension if missing
        if not any(output_path.lower().endswith(ext) for ext in [".stl", ".ply", ".obj"]):
            output_path += f".{output_format}"

        try:
            # Export using trimesh
            mesh.export(output_path)

            # Create metadata file
            self._create_metadata_file(
                output_path,
                top_data,
                bottom_data,
                wall_thickness,
                height,
                output_format,
                top_profile_type,
                bottom_profile_type,
            )

            return os.path.abspath(output_path)

        except Exception as e:
            raise CanReconstructionError(f"Export failed: {e}")

    def _create_metadata_file(
        self,
        output_path: str,
        top_data: dict,
        bottom_data: dict,
        wall_thickness: float,
        height: float,
        output_format: str,
        top_profile_type: str,
        bottom_profile_type: str,
    ) -> None:
        """Create metadata file with reconstruction parameters."""
        metadata = {
            "output_file": os.path.basename(output_path),
            "output_format": output_format,
            "reconstruction_params": {
                "wall_thickness_mm": wall_thickness,
                "height_mm": height,
                "top_profile_type": top_profile_type,
                "bottom_profile_type": bottom_profile_type,
                "top_analysis": os.path.basename(top_data.get("image_path", "unknown")),
                "bottom_analysis": os.path.basename(bottom_data.get("image_path", "unknown")),
                "creation_date": pd.Timestamp.now().isoformat(),
                "reconstruction_method": "enhanced_walled_loft_with_best_contour",
            },
            "top_rim_data": {
                "diameter_mm": top_data.get("diameter_mm"),
                "diameter_px": top_data.get("diameter_px"),
                "circularity": top_data.get("circularity_4piA_P2"),
                "rms_out_of_round_mm": top_data.get("rms_out_of_round_mm"),
                "pixels_per_mm": top_data.get("pixels_per_mm"),
                "ellipse_data": top_data.get("ellipse"),
                "best_contour_data": top_data.get("best_contour"),
            },
            "bottom_rim_data": {
                "diameter_mm": bottom_data.get("diameter_mm"),
                "diameter_px": bottom_data.get("diameter_px"),
                "circularity": bottom_data.get("circularity_4piA_P2"),
                "rms_out_of_round_mm": bottom_data.get("rms_out_of_round_mm"),
                "pixels_per_mm": bottom_data.get("pixels_per_mm"),
                "ellipse_data": bottom_data.get("ellipse"),
                "best_contour_data": bottom_data.get("best_contour"),
            },
        }

        metadata_path = output_path.rsplit(".", 1)[0] + "_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _plot_profiles_3d(
        self,
        top_midline: np.ndarray,
        bottom_midline: np.ndarray,
        top_inner: np.ndarray,
        top_outer: np.ndarray,
        bottom_inner: np.ndarray,
        bottom_outer: np.ndarray,
        top_profile_type: str,
        bottom_profile_type: str,
        wall_thickness: float,
        height: float,
    ) -> None:
        """Plot all profiles in 3D space for visualization."""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available for plotting. Install with: pip install matplotlib")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot midlines
        ax.plot(
            top_midline[:, 0],
            top_midline[:, 1],
            top_midline[:, 2],
            "r-",
            linewidth=3,
            label=f"Top Midline ({top_profile_type})",
        )
        ax.plot(
            bottom_midline[:, 0],
            bottom_midline[:, 1],
            bottom_midline[:, 2],
            "b-",
            linewidth=3,
            label=f"Bottom Midline ({bottom_profile_type})",
        )

        # Plot wall profiles
        ax.plot(
            top_inner[:, 0], top_inner[:, 1], top_inner[:, 2], "r--", linewidth=2, alpha=0.7, label="Top Inner Wall"
        )
        ax.plot(top_outer[:, 0], top_outer[:, 1], top_outer[:, 2], "r:", linewidth=2, alpha=0.7, label="Top Outer Wall")
        ax.plot(
            bottom_inner[:, 0],
            bottom_inner[:, 1],
            bottom_inner[:, 2],
            "b--",
            linewidth=2,
            alpha=0.7,
            label="Bottom Inner Wall",
        )
        ax.plot(
            bottom_outer[:, 0],
            bottom_outer[:, 1],
            bottom_outer[:, 2],
            "b:",
            linewidth=2,
            alpha=0.7,
            label="Bottom Outer Wall",
        )

        # Mark centers
        ax.scatter([0], [0], [height], c="red", s=100, marker="x", label="Top Center")
        ax.scatter([0], [0], [0], c="blue", s=100, marker="x", label="Bottom Center")

        # Add some connection lines to show the structure
        n_lines = 8  # Show a few connecting lines
        step = len(top_midline) // n_lines
        for i in range(0, len(top_midline), step):
            # Connect midlines
            ax.plot(
                [bottom_midline[i, 0], top_midline[i, 0]],
                [bottom_midline[i, 1], top_midline[i, 1]],
                [bottom_midline[i, 2], top_midline[i, 2]],
                "k-",
                alpha=0.3,
                linewidth=1,
            )

        # Set labels and title
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"Can Profiles - Wall Thickness: {wall_thickness} mm, Height: {height} mm")

        # Equal aspect ratio
        max_range = (
            max(
                np.ptp(np.vstack([top_midline, bottom_midline])[:, 0]),
                np.ptp(np.vstack([top_midline, bottom_midline])[:, 1]),
                height,
            )
            / 2.0
        )

        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, height])

        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

        # Print some debug info
        print("\nProfile Debug Information:")
        print(
            f"Top midline radius range: {np.min(np.sqrt(top_midline[:, 0]**2 + top_midline[:, 1]**2)):.3f} - {np.max(np.sqrt(top_midline[:, 0]**2 + top_midline[:, 1]**2)):.3f} mm"
        )
        print(
            f"Bottom midline radius range: {np.min(np.sqrt(bottom_midline[:, 0]**2 + bottom_midline[:, 1]**2)):.3f} - {np.max(np.sqrt(bottom_midline[:, 0]**2 + bottom_midline[:, 1]**2)):.3f} mm"
        )
        print(f"Wall thickness: {wall_thickness} mm")
        print(f"Number of points per profile: {len(top_midline)}")


# Legacy compatibility classes and functions
class CanMeshGenerator(EnhancedCanMeshGenerator):
    """Legacy compatibility class."""

    pass


class CanSTEPGenerator(EnhancedCanMeshGenerator):
    """Legacy compatibility class."""

    pass


def create_can_mesh_file(
    top_json: str,
    bottom_json: str,
    output_path: str,
    wall_thickness: float = 0.1,
    height: float = 100.0,
    top_profile_type: str = "best_contour",
    bottom_profile_type: str = "best_contour",
    resolution: int = 64,
    output_format: str = "auto",
    plot_profiles: bool = False,
    align_profiles: bool = True,
) -> str:
    """Convenience function to create 3D mesh file from can analysis data.

    Args:
        top_json: Path to top rim analysis JSON file
        bottom_json: Path to bottom rim analysis JSON file
        output_path: Output file path
        wall_thickness: Wall thickness in mm (default: 0.1)
        height: Height between profiles in mm (default: 100.0)
        top_profile_type: Profile type for top ("circle", "ellipse", "best_contour", "measured")
        bottom_profile_type: Profile type for bottom ("circle", "ellipse", "best_contour", "measured")
        resolution: Number of points around each profile (default: 64)
        output_format: Output format ("stl", "ply", "obj", "auto")
        plot_profiles: Whether to show 3D plot of profiles before meshing
        align_profiles: Whether to align profile orientations for straight walls (default: True)

    Returns:
        Absolute path to created file
    """
    generator = EnhancedCanMeshGenerator()
    return generator.create_mesh_file(
        top_json,
        bottom_json,
        output_path,
        wall_thickness,
        height,
        top_profile_type,
        bottom_profile_type,
        resolution,
        output_format,
        plot_profiles,
        align_profiles,
    )


# Maintain compatibility for old function names
def create_can_step_file(*args, **kwargs):
    """Compatibility function - creates mesh file with STL default."""
    # Handle legacy arguments
    if len(args) >= 3:
        args = list(args)
        output_path = args[2]
        if output_path.lower().endswith((".step", ".stp")):
            print("⚠️  Note: STEP export not available. Creating STL file instead.")
            args[2] = output_path.rsplit(".", 1)[0] + ".stl"
        args = tuple(args)

    # Map old parameters to new ones
    old_to_new = {"use_measured_points": lambda x: ("measured" if x else "ellipse", "measured" if x else "ellipse")}

    if "use_measured_points" in kwargs:
        use_measured = kwargs.pop("use_measured_points")
        top_type, bottom_type = old_to_new["use_measured_points"](use_measured)
        kwargs["top_profile_type"] = top_type
        kwargs["bottom_profile_type"] = bottom_type

    if "output_format" not in kwargs:
        kwargs["output_format"] = "stl"

    return create_can_mesh_file(*args, **kwargs)
