import cv2
import numpy as np
from skimage import filters


def compute_brightness(gray):
    return float(np.mean(gray))


def compute_contrast(gray):
    return float(np.std(gray))


def compute_blur_score(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_overexposure(gray, threshold=240):
    overexposed = int(np.sum(gray >= threshold))
    return float(overexposed / gray.size * 100)


def compute_underexposure(gray, threshold=15):
    underexposed = int(np.sum(gray <= threshold))
    return float(underexposed / gray.size * 100)


def compute_edge_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / edges.size * 100)


def compute_coverage(gray):
    try:
        thresh_val = filters.threshold_otsu(gray)
        binary = gray > thresh_val
        return float(np.sum(binary) / binary.size * 100)
    except Exception:
        return 0.0


def compute_intensity_histogram(gray, bins=32):
    counts, edges = np.histogram(gray.astype(np.float32), bins=bins, range=(0, 256))
    bin_centers = ((edges[:-1] + edges[1:]) / 2).astype(int)
    return bin_centers.tolist(), counts.tolist()


def analyze_image(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy().astype(np.uint8)

    metrics = {
        "brightness": round(compute_brightness(gray), 2),
        "contrast": round(compute_contrast(gray), 2),
        "blur_score": round(compute_blur_score(gray), 2),
        "overexposure_pct": round(compute_overexposure(gray, threshold=240), 2),
        "underexposure_pct": round(compute_underexposure(gray, threshold=15), 2),
        "edge_density_pct": round(compute_edge_density(gray), 2),
        "coverage_pct": round(compute_coverage(gray), 2),
    }

    bin_centers, hist_counts = compute_intensity_histogram(gray)
    return metrics, bin_centers, hist_counts
