from pathlib import Path

import numpy as np

from gdt_engine.diagnostics import analyze_point_cloud, detect_feature_type
from gdt_engine.io import load_point_cloud


def test_plane_sample_is_classified_from_pca_and_has_bounding_box() -> None:
    points = load_point_cloud(Path("data/nist/plane_sample.xyz"))

    result = analyze_point_cloud(points)

    assert result.geometry == "plane"
    np.testing.assert_allclose(result.bounding_box_min, [0.0, 0.0, -0.001])
    np.testing.assert_allclose(result.bounding_box_max, [100.0, 100.0, 0.001])
    np.testing.assert_allclose(result.extents, [100.0, 100.0, 0.002])
    assert result.eigenvalues.shape == (3,)
    assert result.eigenvalues[0] >= result.eigenvalues[1] >= result.eigenvalues[2]
    assert result.eigenvalues[2] < 1e-6


def test_sphere_sample_is_classified_from_pca() -> None:
    points = load_point_cloud(Path("data/nist/sphere_sample.xyz"))

    result = analyze_point_cloud(points)

    assert result.geometry == "sphere"
    assert result.eigenvalues.shape == (3,)
    assert np.all(result.eigenvalues > 0)
    assert result.eigenvalues[0] / result.eigenvalues[2] > 1.01


def test_diagnostics_rejects_invalid_point_cloud() -> None:
    with np.testing.assert_raises(ValueError):
        analyze_point_cloud(np.ones((2, 3)))


def test_detect_feature_type_classifies_synthetic_plane_and_line() -> None:
    x, y = np.meshgrid(np.linspace(-5.0, 5.0, 12), np.linspace(-4.0, 4.0, 10))
    plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    line = np.column_stack((np.linspace(-20.0, 20.0, 30), np.zeros(30), np.zeros(30)))

    plane_type, plane_confidence = detect_feature_type(plane)
    line_type, line_confidence = detect_feature_type(line)

    assert plane_type == "plane"
    assert line_type == "line"
    assert 0.0 <= plane_confidence <= 100.0
    assert 0.0 <= line_confidence <= 100.0


def test_detect_feature_type_handles_duplicate_points_without_division_error() -> None:
    feature_type, confidence = detect_feature_type(np.ones((8, 3)))

    assert feature_type in {"sphere", "cylinder"}
    assert confidence == 0.0
