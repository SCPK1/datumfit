from pathlib import Path

import numpy as np

from gdt_engine.app import _pill_configuration, _sync_selected_feature

APP_PATH = Path(__file__).parents[1] / "src" / "gdt_engine" / "app.py"


def test_app_has_no_deprecated_results_heading_and_uses_iso_symbols() -> None:
    content = APP_PATH.read_text(encoding="utf-8")

    assert "Metrology Results" not in content, "Zakazany nagłówek nadal istnieje w kodzie!"
    assert all(symbol in content for symbol in ("⏥", "⏤", "⌭", "S○"))


def test_line_cloud_auto_syncs_selector_to_suggested_mode() -> None:
    points = np.loadtxt("data/nist/line_sample.xyz")
    session_state = {}

    detected, confidence = _sync_selected_feature(
        session_state, points, "line_sample.xyz"
    )

    assert detected == "line"
    assert session_state["selected_target_feature"] == "Auto-Detect (Suggested)"
    assert 0.0 <= confidence <= 100.0


def test_manual_feature_selection_survives_same_cached_cloud() -> None:
    points = np.loadtxt("data/nist/line_sample.xyz")
    session_state = {}
    _sync_selected_feature(session_state, points, "line_sample.xyz")
    session_state["selected_target_feature"] = "sphere"

    _sync_selected_feature(session_state, points.copy(), "line_sample.xyz")

    assert session_state["selected_target_feature"] == "sphere"


def test_pill_labels_exclude_diameter_as_a_tolerance() -> None:
    for geometry in ("plane", "cylinder", "sphere", "line"):
        labels, _ = _pill_configuration(geometry)
        assert all("Diameter" not in label and "⌀" not in label for label in labels)
