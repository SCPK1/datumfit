import json
from pathlib import Path

import numpy as np
import pytest

from gdt_engine.fitting.sphere import fit_sphere
from gdt_engine.io import load_point_cloud


def test_fit_sphere_matches_nist_reference() -> None:
    data_dir = Path("data/nist")
    result = fit_sphere(load_point_cloud(data_dir / "sphere_sample.xyz"))
    reference = json.loads((data_dir / "sphere_sample_ref.json").read_text())

    np.testing.assert_allclose(result.center, reference["center"], atol=1e-8)
    assert result.radius == pytest.approx(reference["radius"], abs=1e-8)
    assert result.radial_residuals.shape == (6,)
    assert np.max(np.abs(result.radial_residuals)) < 1e-8


def test_fit_sphere_returns_residuals_for_noisy_data() -> None:
    center = np.array([2.0, -1.0, 4.0])
    points = center + np.array(
        [[3, 0, 0], [-3, 0, 0], [0, 3, 0], [0, -3, 0], [0, 0, 3], [0, 0, -3]],
        dtype=float,
    )
    points[0, 0] += 0.1
    result = fit_sphere(points)
    assert result.radial_residuals[0] > 0
    assert result.radius == pytest.approx(3.0, abs=0.1)


def test_fit_sphere_requires_four_non_coplanar_points() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        fit_sphere(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
