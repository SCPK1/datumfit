"""Numerical point-cloud diagnostics and geometry auto-classification."""

from dataclasses import dataclass

import numpy as np

from gdt_engine.fitting.cylinder import fit_cylinder
from gdt_engine.fitting.sphere import fit_sphere


@dataclass(frozen=True)
class PointCloudDiagnostics:
    """Descriptive statistics and PCA classification for a point cloud."""

    bounding_box_min: np.ndarray
    bounding_box_max: np.ndarray
    extents: np.ndarray
    centroid: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    geometry: str


def detect_feature_type(points: np.ndarray) -> tuple[str, float]:
    """Suggest a geometric feature from PCA shape ratios and fit residual RMS."""
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if len(data) < 3:
        raise ValueError("feature detection requires at least 3 points")
    if not np.isfinite(data).all():
        raise ValueError("points must be finite")

    centered = data - data.mean(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(data))[::-1], 0.0)
    total = float(eigenvalues.sum())
    if total <= np.finfo(np.float64).eps:
        return "sphere", 0.0

    ratios = eigenvalues / total
    if ratios[0] > 0.90:
        return "line", float(np.clip((ratios[0] - 0.90) / 0.10 * 100.0, 0.0, 100.0))
    if ratios[2] < 0.005:
        return "plane", float(np.clip((0.005 - ratios[2]) / 0.005 * 100.0, 0.0, 100.0))

    rms_by_type: dict[str, float] = {}
    try:
        sphere = fit_sphere(data)
        rms_by_type["sphere"] = float(np.sqrt(np.mean(np.square(sphere.radial_residuals))))
    except (ValueError, np.linalg.LinAlgError):
        rms_by_type["sphere"] = float("inf")
    try:
        _, _, _, residuals = fit_cylinder(data)
        rms_by_type["cylinder"] = float(np.sqrt(np.mean(np.square(residuals))))
    except (ValueError, np.linalg.LinAlgError):
        rms_by_type["cylinder"] = float("inf")

    best = min(rms_by_type, key=rms_by_type.get)
    other = "cylinder" if best == "sphere" else "sphere"
    best_rms, other_rms = rms_by_type[best], rms_by_type[other]
    if not np.isfinite(best_rms):
        return "sphere", 0.0
    confidence = 100.0 if not np.isfinite(other_rms) else float(
        np.clip((other_rms - best_rms) / max(other_rms, np.finfo(float).eps) * 100.0, 0.0, 100.0)
    )
    return best, confidence


def analyze_point_cloud(points: np.ndarray) -> PointCloudDiagnostics:
    """Compute bounding-box and PCA diagnostics, classifying plane or sphere.

    A planar cloud has negligible variance along its third principal axis. A
    sphere-like cloud has comparable variance along all three axes. The
    classification threshold is relative to the largest eigenvalue, making it
    independent of the coordinate units and overall scale.
    """
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if data.shape[0] < 3:
        raise ValueError("diagnostics requires at least 3 points")
    if not np.isfinite(data).all():
        raise ValueError("points must be finite")

    centroid = data.mean(axis=0)
    centered = data - centroid
    covariance = centered.T @ centered / data.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    largest = eigenvalues[0]
    is_plane = largest == 0.0 or eigenvalues[2] / largest <= 1e-3

    return PointCloudDiagnostics(
        bounding_box_min=data.min(axis=0),
        bounding_box_max=data.max(axis=0),
        extents=data.max(axis=0) - data.min(axis=0),
        centroid=centroid,
        covariance=covariance,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        geometry="plane" if is_plane else "sphere",
    )
