import json
from pathlib import Path

import numpy as np
import pytest

from gdt_engine.fitting.plane import fit_plane
from gdt_engine.io import load_point_cloud


def test_fit_plane_matches_nist_reference() -> None:
    data_dir = Path("data/nist")
    points = load_point_cloud(data_dir / "plane_sample.xyz")
    reference = json.loads((data_dir / "plane_sample_ref.json").read_text())

    result = fit_plane(points)

    np.testing.assert_allclose(result.centroid, reference["centroid"], atol=1e-12)
    assert result.normal[2] >= 0
    angle = np.degrees(
        np.arccos(np.clip(np.dot(result.normal, reference["normal"]), -1, 1))
    )
    assert angle <= reference["tolerances"]["normal_angular_deg"]
    assert result.ptv <= 2 * reference["tolerances"]["distance_to_plane"]
    assert result.signed_distances.shape == (len(points),)


def test_fit_plane_uses_orthogonal_distances() -> None:
    points = np.array([[0, 0, 2], [1, 0, 3], [0, 1, 3], [1, 1, 4]], dtype=float)
    result = fit_plane(points)
    np.testing.assert_allclose(result.normal, [-1 / np.sqrt(3)] * 2 + [1 / np.sqrt(3)])
    assert result.ptv == pytest.approx(0.0)


def test_fit_plane_requires_three_non_collinear_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_plane(np.array([[0, 0, 0], [1, 1, 1]]))
