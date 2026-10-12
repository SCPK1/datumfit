from pathlib import Path

import numpy as np
import pytest

from gdt_engine.fitting.line import fit_line_3d


def test_fit_line_3d_returns_centroid_axis_and_point_to_line_residuals() -> None:
    points = np.loadtxt(Path("data/nist/line_sample.xyz"))

    p0, direction, residuals = fit_line_3d(points)

    assert p0.shape == (3,)
    assert direction.shape == (3,)
    assert residuals.shape == (len(points),)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert np.all(residuals >= 0.0)
    assert np.max(residuals) == pytest.approx(0.3, abs=0.08)
    assert np.dot(direction, np.array([1.0, 0.0, 0.0])) > 0.99


def test_fit_line_3d_rejects_invalid_clouds() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_line_3d(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="finite"):
        fit_line_3d(np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]))
