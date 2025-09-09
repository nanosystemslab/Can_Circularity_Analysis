"""Image processing utilities for preprocessing and segmentation."""

import cv2
import numpy as np


def preprocess_image(gray_image: np.ndarray) -> np.ndarray:
    """Apply preprocessing to grayscale image.

    Args:
        gray_image: Input grayscale image

    Returns:
        Preprocessed grayscale image
    """
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray_image, 9, 50, 50)
    return filtered


def segment_image(
    gray_image: np.ndarray,
    method: str = "binary",
    binary_block_size: int = 51,
    binary_C: int = 2,
    binary_invert: bool = False,
    erode_iterations: int = 0,
    canny_low: int = 75,
    canny_high: int = 200,
    **kwargs,  # Accept additional keyword arguments and ignore them
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Segment image using specified method.

    Args:
        gray_image: Input grayscale image
        method: Segmentation method ("binary" or "canny")
        binary_block_size: Block size for adaptive threshold (must be odd)
        binary_C: Constant subtracted from weighted mean
        binary_invert: Whether to invert binary threshold
        erode_iterations: Number of erosion iterations to apply
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Tuple of (binary_mask, edges) - only one will be non-None
    """
    if method == "binary":
        return _segment_binary(
            gray_image, binary_block_size, binary_C, binary_invert, erode_iterations
        ), None
    else:
        return None, _segment_canny(gray_image, canny_low, canny_high)


def _segment_binary(
    gray_image: np.ndarray, block_size: int, C: int, invert: bool, erode_iterations: int
) -> np.ndarray:
    """Segment image using adaptive thresholding.

    Args:
        gray_image: Input grayscale image
        block_size: Size of the neighborhood area for threshold calculation
        C: Constant subtracted from the mean
        invert: Whether to invert the threshold
        erode_iterations: Number of erosion iterations

    Returns:
        Binary segmented image
    """
    # Ensure block size is odd
    if block_size % 2 == 0:
        block_size += 1

    # Apply adaptive threshold
    threshold_type = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    binary = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block_size, C
    )

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Opening to remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Closing to fill gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional erosion
    if erode_iterations > 0:
        binary = cv2.erode(binary, kernel, iterations=erode_iterations)

    return binary


def _segment_canny(gray_image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Segment image using Canny edge detection.

    Args:
        gray_image: Input grayscale image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection

    Returns:
        Edge-detected binary image
    """
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)

    # Apply Canny edge detection
    edges = cv2.Canny(enhanced, low_threshold, high_threshold, L2gradient=True)

    return edges


def enhance_contrast(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Enhance image contrast using CLAHE.

    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of the grid for histogram equalization

    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """Apply Gaussian blur to image.

    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation for Gaussian kernel (0 = auto-calculate)

    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def crop_image(
    image: np.ndarray, crop_region: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop image to specified region.

    Args:
        image: Input image
        crop_region: Tuple of (x, y, width, height)

    Returns:
        Tuple of (cropped_image, (x_offset, y_offset))
    """
    x, y, w, h = crop_region

    # Ensure crop region is within image bounds
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    cropped = image[y : y + h, x : x + w].copy()
    return cropped, (x, y)


def resize_image(image: np.ndarray, max_dimension: int = 1024) -> tuple[np.ndarray, float]:
    """Resize image while maintaining aspect ratio.

    Args:
        image: Input image
        max_dimension: Maximum allowed dimension (width or height)

    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim <= max_dimension:
        return image.copy(), 1.0

    scale = max_dimension / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale
