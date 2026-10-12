"""Numerical evaluations of ISO 1101 form tolerances."""

import numpy as np

from gdt_engine.fitting.cylinder import fit_cylinder


def _points(points: np.ndarray, dimensions: int, minimum: int = 3) -> np.ndarray:
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != dimensions:
        raise ValueError(f"points must be an Nx{dimensions} array")
    if len(data) < minimum or not np.isfinite(data).all():
        raise ValueError("points are insufficient or non-finite")
    return data


def evaluate_flatness(points: np.ndarray) -> float:
    data = _points(points, 3)
    centered = data - data.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    distances = centered @ vh[-1]
    return float(np.ptp(distances))


def evaluate_straightness(points: np.ndarray) -> float:
    data = _points(points, 3, 2)
    centered = data - data.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= np.finfo(float).eps:
        return 0.0
    axis = vh[0]
    distances = np.linalg.norm(centered - np.outer(centered @ axis, axis), axis=1)
    return float(np.max(distances))


def evaluate_circularity(points_2d: np.ndarray) -> float:
    data = _points(points_2d, 2, 3)
    x, y = data.T
    design = np.column_stack((x, y, np.ones(len(data))))
    rhs = x * x + y * y
    solution, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
    center = solution[:2] / 2.0
    radius = float(np.sqrt(solution[2] + np.dot(center, center)))
    if radius <= 0:
        raise ValueError("fitted radius must be positive")
    return float(np.ptp(np.linalg.norm(data - center, axis=1)))


def evaluate_cylindricity(points: np.ndarray) -> float:
    _, _, _, residuals = fit_cylinder(points)
    return float(np.ptp(residuals))
