"""Orthogonal least-squares plane fitting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneFitResult:
    centroid: np.ndarray
    normal: np.ndarray
    signed_distances: np.ndarray
    ptv: float


def fit_plane(points: np.ndarray) -> PlaneFitResult:
    """Fit a plane by SVD and return signed orthogonal residuals."""
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if data.shape[0] < 3:
        raise ValueError("plane fitting requires at least 3 points")
    if not np.isfinite(data).all():
        raise ValueError("points must be finite")

    centroid = data.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(data - centroid, full_matrices=False)
    if singular_values[1] <= np.finfo(np.float64).eps * singular_values[0]:
        raise ValueError("points must not be collinear")
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    signed_distances = (data - centroid) @ normal
    return PlaneFitResult(
        centroid=centroid,
        normal=normal,
        signed_distances=signed_distances,
        ptv=float(np.ptp(signed_distances)),
    )
