"""Non-linear orthogonal radial fitting of a cylinder."""

import numpy as np
from scipy.optimize import least_squares


def _validate_points(points: np.ndarray) -> np.ndarray:
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if data.shape[0] < 5:
        raise ValueError("cylinder fitting requires at least 5 points")
    if not np.isfinite(data).all():
        raise ValueError("points must be finite")
    return data


def fit_cylinder(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit ``P0``, unit axis ``V`` and radius using radial least squares.

    The residual for every point is its shortest distance to the fitted axis,
    minus the fitted radius.  The returned point on the axis is the centroid
    projected onto the fitted axis coordinate system (its axial origin is
    therefore stable for a given point cloud).
    """
    data = _validate_points(points)
    centroid = data.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(data - centroid, full_matrices=False)
    if singular_values[1] <= np.finfo(float).eps * singular_values[0]:
        raise ValueError("points must span a cylindrical surface")
    axis0 = vh[0]
    radial0 = data - centroid - np.outer((data - centroid) @ axis0, axis0)
    radius0 = float(np.mean(np.linalg.norm(radial0, axis=1)))
    if radius0 <= np.finfo(float).eps:
        raise ValueError("fitted radius must be positive")

    def residuals(parameters: np.ndarray) -> np.ndarray:
        p0 = parameters[:3]
        vector = parameters[3:6]
        norm = np.linalg.norm(vector)
        if norm <= np.finfo(float).eps:
            return np.full(data.shape[0], 1e6)
        axis = vector / norm
        delta = data - p0
        distances = np.linalg.norm(delta - np.outer(delta @ axis, axis), axis=1)
        return distances - parameters[6]

    candidates = [axis0, *vh[1:]]
    candidates.extend(np.eye(3))
    fits = []
    for candidate_axis in candidates:
        candidate_radial = (
            data
            - centroid
            - np.outer((data - centroid) @ candidate_axis, candidate_axis)
        )
        candidate_radius = float(np.mean(np.linalg.norm(candidate_radial, axis=1)))
        initial = np.concatenate((centroid, candidate_axis, [candidate_radius]))
        fits.append(least_squares(residuals, initial, method="trf", x_scale="jac"))
    result = min(fits, key=lambda item: float(np.sum(item.fun**2)))
    if not result.success or result.x[6] <= 0:
        raise ValueError("cylinder fitting did not converge")
    p0 = result.x[:3]
    axis = result.x[3:6] / np.linalg.norm(result.x[3:6])
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    radial_residuals = residuals(np.r_[p0, axis, result.x[6]])
    return p0, axis, float(result.x[6]), radial_residuals
