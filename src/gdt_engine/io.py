"""Input/output helpers for point clouds."""

import re
from pathlib import Path

import numpy as np


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load a finite Nx3 point cloud from an XYZ or CSV-like text file."""
    file_path = Path(path)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read point-cloud file: {file_path}") from exc

    data_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        raise ValueError("point cloud must contain at least 3 points")

    sample = data_lines[0]
    delimiter = "," if "," in sample else ";" if ";" in sample else None
    rows: list[list[float]] = []
    for line in data_lines:
        fields = line.split(delimiter) if delimiter else re.split(r"\s+", line)
        if len(fields) != 3:
            raise ValueError("point cloud must have exactly three columns")
        try:
            rows.append([float(field.strip()) for field in fields])
        except ValueError as exc:
            raise ValueError("point cloud values must be numeric") from exc

    points = np.asarray(rows, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("point cloud must contain at least 3 points")
    if not np.isfinite(points).all():
        raise ValueError("point cloud values must be finite")
    return points
