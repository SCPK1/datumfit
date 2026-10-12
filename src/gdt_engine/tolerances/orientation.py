"""Numerical orientation and datum relationship evaluations."""

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(value)
    if (
        value.shape != (3,)
        or not np.isfinite(value).all()
        or norm <= np.finfo(float).eps
    ):
        raise ValueError("vectors must be finite, non-zero 3D vectors")
    return value / norm


def _acute_angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(abs(np.dot(_unit(first), _unit(second))), 0.0, 1.0)))


def evaluate_parallelism(
    n_datum: np.ndarray, n_feature: np.ndarray, length: float
) -> float:
    theta = _acute_angle(n_datum, n_feature)
    return float(length) * np.sin(theta)


def evaluate_perpendicularity(
    n_datum: np.ndarray, n_feature: np.ndarray, length: float
) -> float:
    phi = abs(np.pi / 2.0 - _acute_angle(n_datum, n_feature))
    return float(length) * np.sin(phi)


def evaluate_angularity(
    n_datum: np.ndarray, n_feature: np.ndarray, nominal_angle_deg: float, length: float
) -> float:
    deviation = abs(_acute_angle(n_datum, n_feature) - np.deg2rad(nominal_angle_deg))
    return float(length) * np.sin(deviation)


def evaluate_coaxiality(
    axis_datum_p0: np.ndarray,
    axis_datum_v: np.ndarray,
    axis_feat_p0: np.ndarray,
    axis_feat_v: np.ndarray,
    length: float,
) -> float:
    datum_point = np.asarray(axis_datum_p0, dtype=np.float64)
    datum_axis = _unit(axis_datum_v)
    feature_point = np.asarray(axis_feat_p0, dtype=np.float64)
    feature_axis = _unit(axis_feat_v)
    if datum_point.shape != (3,) or feature_point.shape != (3,) or length < 0:
        raise ValueError("axis points must be 3D and length must be non-negative")
    samples = np.array([feature_point, feature_point + float(length) * feature_axis])
    delta = samples - datum_point
    distances = np.linalg.norm(delta - np.outer(delta @ datum_axis, datum_axis), axis=1)
    return float(np.max(distances))
