import json

import numpy as np

from gdt_engine.app import (
    _datum_figure,
    _datum_metrics,
    _inspection_metrics,
    _report_csv,
    _summary_json,
)


def test_datum_view_contains_both_clouds_and_orientation_metrics() -> None:
    datum = np.column_stack((np.linspace(0.0, 100.0, 21), np.zeros(21), np.zeros(21)))
    feature = np.column_stack((np.linspace(0.0, 100.0, 21), np.zeros(21), np.linspace(0.0, 5.0, 21)))
    metrics = _datum_metrics(datum, feature, "Parallelism (//)", 100.0, 10.0)
    figure = _datum_figure(datum, feature, metrics)

    assert len(figure.data) == 3
    assert figure.data[0].marker.color == "lightgray"
    assert metrics["nominal_angle_deg"] == 0.0


def test_line_metrics_include_straightness_and_axis_direction() -> None:
    points = np.loadtxt("data/nist/line_sample.xyz")
    metrics = _inspection_metrics(points, "line")

    assert metrics["measured_deviation"] == np.max(metrics["residuals"])
    assert metrics["properties"]["axis"].shape == (3,)
    assert metrics["properties"]["point_on_axis"].shape == (3,)


def test_report_csv_contains_metadata_and_point_rows() -> None:
    points = np.loadtxt("data/nist/line_sample.xyz")
    metrics = _inspection_metrics(points, "line")
    report = _report_csv(
        points,
        metrics,
        mode="Single Feature (Form)",
        target="line",
        tolerance=0.5,
    )

    assert "# Mode,Single Feature (Form)" in report
    assert "# Verdict,PASS" in report
    assert "X,Y,Z,Residual_Deviation_mm,In_Tolerance_Flag" in report
    assert report.count("\n") == len(points) + 6


def test_summary_json_serializes_numpy_metrics() -> None:
    points = np.loadtxt("data/nist/line_sample.xyz")
    metrics = _inspection_metrics(points, "line")

    payload = json.loads(_summary_json(metrics, "line"))

    assert payload["target_feature"] == "line"
    assert payload["metrics"]["ptv"] == metrics["ptv"]
