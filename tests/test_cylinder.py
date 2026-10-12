import json
from pathlib import Path

import numpy as np
import pytest

from gdt_engine.fitting.cylinder import fit_cylinder
from gdt_engine.io import load_point_cloud


def test_fit_cylinder_matches_nist_reference() -> None:
    data_dir = Path("data/nist")
    points = load_point_cloud(data_dir / "cylinder_sample.xyz")
    reference = json.loads((data_dir / "cylinder_sample_ref.json").read_text())

    p0, axis, radius, residuals = fit_cylinder(points)

    assert radius == pytest.approx(reference["radius"], abs=0.02)
    assert abs(float(np.dot(axis, reference["axis"]))) > 1 - 2e-4
    assert np.linalg.norm(np.cross(p0 - reference["center"], reference["axis"])) < 0.05
    assert np.ptp(residuals) < 0.02


def test_fit_cylinder_returns_radial_residuals_and_rejects_invalid_input() -> None:
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points = np.column_stack(
        (5 * np.cos(theta), 5 * np.sin(theta), np.linspace(0, 10, 12))
    )
    p0, axis, radius, residuals = fit_cylinder(points)

    assert p0.shape == (3,)
    assert axis.shape == (3,)
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert radius == pytest.approx(5.0, abs=1e-8)
    assert residuals.shape == (12,)
    with pytest.raises(ValueError, match="at least 5"):
        fit_cylinder(points[:4])
