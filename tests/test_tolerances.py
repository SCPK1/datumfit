import numpy as np
import pytest

from gdt_engine.app import _comparison_rows, _datum_metrics
from gdt_engine.tolerances.form import (
    evaluate_circularity,
    evaluate_cylindricity,
    evaluate_flatness,
    evaluate_straightness,
)
from gdt_engine.tolerances.orientation import (
    evaluate_angularity,
    evaluate_coaxiality,
    evaluate_parallelism,
    evaluate_perpendicularity,
)


def test_form_tolerances_for_ideal_geometries_are_zero() -> None:
    plane = np.array([[x, y, 2.0] for x in (-1, 0, 1) for y in (-1, 0, 1)])
    line = np.array([[0.0, 0.0, z] for z in (-1, 0, 1)])
    circle = np.array(
        [
            [3 * np.cos(t), 3 * np.sin(t)]
            for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ]
    )
    cylinder = np.array(
        [
            [4 * np.cos(t), 4 * np.sin(t), z]
            for z in (0, 5)
            for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ]
    )

    assert evaluate_flatness(plane) == pytest.approx(0.0)
    assert evaluate_straightness(line) == pytest.approx(0.0)
    assert evaluate_circularity(circle) == pytest.approx(0.0, abs=1e-10)
    assert evaluate_cylindricity(cylinder) == pytest.approx(0.0, abs=1e-10)


def test_form_tolerances_measure_peak_to_valley_deviation() -> None:
    plane = np.array([[0, 0, 0], [1, 0, 0.2], [0, 1, 0], [1, 1, 0.0]], dtype=float)
    line = np.array([[0, 0, 0], [1, 0.3, 0], [2, 0, 0]], dtype=float)
    circle = np.array([[2, 0], [-2, 0], [0, 2.2], [0, -2.2]], dtype=float)

    assert evaluate_flatness(plane) > 0.0
    assert evaluate_straightness(line) > 0.0
    assert evaluate_circularity(circle) > 0.0


def test_orientation_tolerances() -> None:
    z = np.array([0.0, 0.0, 1.0])
    x = np.array([1.0, 0.0, 0.0])
    tilted = np.array([np.sin(np.deg2rad(5)), 0.0, np.cos(np.deg2rad(5))])

    assert evaluate_parallelism(z, z, 100.0) == pytest.approx(0.0)
    assert evaluate_parallelism(z, tilted, 100.0) == pytest.approx(
        100 * np.sin(np.deg2rad(5))
    )
    assert evaluate_perpendicularity(z, x, 100.0) == pytest.approx(0.0)
    assert evaluate_perpendicularity(z, tilted, 100.0) == pytest.approx(
        100 * np.sin(np.deg2rad(85))
    )
    assert evaluate_angularity(z, tilted, 0.0, 100.0) == pytest.approx(
        100 * np.sin(np.deg2rad(5))
    )
    assert evaluate_angularity(z, x, 90.0, 100.0) == pytest.approx(0.0)
    assert evaluate_coaxiality(
        np.zeros(3), z, np.array([2.0, 0.0, 0.0]), z, 10.0
    ) == pytest.approx(2.0)
    assert evaluate_coaxiality(np.zeros(3), z, np.zeros(3), tilted, 10.0) > 0.0


def test_comparison_rows_only_include_selected_tolerances_and_compute_margin() -> None:
    metrics = {
        "measured_deviation": 0.012,
        "ptv": 0.012,
        "residuals": np.array([-0.006, 0.006]),
        "properties": {"max_peak": 0.006, "max_valley": -0.006},
    }

    rows = _comparison_rows(
        "plane", metrics, ["Flatness (PtV)", "Peak-to-Reference"], 0.01
    )

    assert [row["Characteristic"] for row in rows] == [
        "Flatness (PtV)",
        "Peak-to-Reference",
    ]
    assert rows[0]["Deviation from Limit [mm]"] == pytest.approx(0.002)
    assert rows[0]["Status"] == "FAIL"
    assert rows[1]["Status"] == "PASS"


def test_datum_metrics_reports_explicit_measured_nominal_and_extent() -> None:
    datum = np.column_stack((np.linspace(-10.0, 10.0, 12), np.zeros(12), np.zeros(12)))
    feature = np.column_stack((np.zeros(12), np.linspace(-10.0, 10.0, 12), np.zeros(12)))

    metrics = _datum_metrics(datum, feature, "Perpendicularity (⊥)", 50.0, 0.01)

    assert metrics["measured_angle_deg"] == pytest.approx(90.0)
    assert metrics["nominal_angle_deg"] == 90.0
    assert metrics["angular_error_deg"] == pytest.approx(0.0)
    assert metrics["length"] == 50.0
