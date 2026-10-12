"""Algebraic least-squares sphere fitting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SphereFitResult:
    center: np.ndarray
    radius: float
    radial_residuals: np.ndarray


def fit_sphere(points: np.ndarray) -> SphereFitResult:
    """Fit ``|p-c|² = R²`` using a linear algebraic least-squares system."""
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if data.shape[0] < 4:
        raise ValueError("sphere fitting requires at least 4 points")
    if not np.isfinite(data).all():
        raise ValueError("points must be finite")

    design = np.column_stack((2.0 * data, np.ones(data.shape[0])))
    if np.linalg.matrix_rank(design) < 4:
        raise ValueError("sphere fitting requires four non-coplanar points")
    solution, _, _, _ = np.linalg.lstsq(design, np.sum(data**2, axis=1), rcond=None)
    center = solution[:3]
    radius_squared = solution[3] + np.dot(center, center)
    if radius_squared <= 0:
        raise ValueError("fitted radius must be positive")
    radius = float(np.sqrt(radius_squared))
    radial_residuals = np.linalg.norm(data - center, axis=1) - radius
    return SphereFitResult(
        center=center,
        radius=radius,
        radial_residuals=radial_residuals,
    )
