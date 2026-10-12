"""Orthogonal least-squares fitting of a 3D line."""

import numpy as np


def fit_line_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid, unit principal direction and point-to-line distances."""
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if len(data) < 2:
        raise ValueError("line fitting requires at least 2 points")
    if not np.isfinite(data).all():
        raise ValueError("points must be finite")
    centroid = data.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(data - centroid, full_matrices=False)
    if singular_values[0] <= np.finfo(float).eps:
        raise ValueError("points must not be coincident")
    direction = vh[0]
    direction /= np.linalg.norm(direction)
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    delta = data - centroid
    perpendicular = delta - np.outer(delta @ direction, direction)
    residuals = np.linalg.norm(perpendicular, axis=1)
    return centroid, direction, residuals
