from pathlib import Path

import numpy as np
import pytest

from gdt_engine.app import (
    _format_bounding_box,
    _format_deviation,
    _inspection_figure,
    _inspection_metrics,
    _pill_configuration,
)

APP_PATH = Path(__file__).parents[1] / "src" / "gdt_engine" / "app.py"


def test_deviation_is_formatted_in_millimeters_and_micrometers() -> None:
    assert _format_deviation(0.002) == "0.0020 mm (2.0 µm)"


def test_bounding_box_is_formatted_with_explicit_unit() -> None:
    assert _format_bounding_box(np.array([100.0, 100.0, 0.002])) == (
        "100.00 x 100.00 x 0.002 mm"
    )


def test_plane_figure_has_aspect_corrected_viewport_and_one_color_scale() -> None:
    points = np.loadtxt("data/nist/plane_sample.xyz")
    figure, _ = _inspection_figure(points, "plane", point_size=11, plane_opacity=0.25)

    assert figure.layout.height == 650
    assert figure.layout.margin.l == 10
    assert figure.layout.margin.r == 95
    assert figure.layout.margin.b == 10
    assert figure.layout.margin.t == 10
    assert figure.layout.scene.aspectmode == "manual"
    assert figure.layout.scene.aspectratio.to_plotly_json() == {
        "x": 1.0,
        "y": 1.0,
        "z": 0.35,
    }
    assert figure.layout.scene.camera.eye.to_plotly_json() == {
        "x": 1.4,
        "y": 1.4,
        "z": 0.8,
    }
    assert figure.data[0].marker.size == 11
    assert figure.data[0].marker.colorscale[0][1] == "#30123b"
    assert figure.data[0].marker.colorbar.title.text == "<b>Deviation</b><br>[mm]"
    colorbar = figure.data[0].marker.colorbar
    assert colorbar.title.side == "top"
    assert colorbar.title.font.size == 11
    assert colorbar.title.font.color == "#020617"
    assert colorbar.title.font.family == "Arial, sans-serif"
    assert colorbar.thickness == 14
    assert colorbar.len == 0.65
    assert colorbar.y == 0.50
    assert colorbar.yanchor == "middle"
    assert colorbar.x == 1.05
    assert colorbar.xanchor == "left"
    assert colorbar.ypad == 15
    assert colorbar.tickfont.size == 10
    assert colorbar.tickfont.color == "#020617"
    assert colorbar.tickfont.family == "Arial, sans-serif"
    assert colorbar.outlinewidth == 0
    assert figure.layout.legend.x == 0.02
    assert figure.layout.legend.y == 0.88
    assert figure.layout.legend.xanchor == "left"
    assert figure.layout.legend.yanchor == "top"
    assert figure.layout.showlegend is False
    assert figure.data[1].opacity == 0.25
    assert figure.data[1].showscale is False


def test_3d_viewport_shows_high_contrast_frame_edges() -> None:
    points = np.loadtxt("data/nist/plane_sample.xyz")
    for theme, expected_edge in (("Light Mode", "#000000"), ("Dark Mode", "#f8fafc")):
        figure, _ = _inspection_figure(points, "plane", theme=theme)
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(figure.layout.scene, axis_name)
            assert axis.showline is True
            assert axis.linecolor == expected_edge
            assert axis.linewidth == 3.0
        assert all(trace.name != "Point cloud bounds" for trace in figure.data)


def test_plotly_chart_disables_modebar_and_enables_mouse_navigation() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert '"displayModeBar": False' in source
    assert '"scrollZoom": True' in source
    assert '"staticPlot": False' in source


def test_viewport_title_and_plot_are_inside_one_bordered_panel() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    start = source.index('with viewport:\n        with st.container(key="viewport_panel"):')
    end = source.index('\n\nif __name__', start)
    panel = source[start:end]

    assert 'st.markdown("### 3D Viewport")' in panel
    assert 'st.plotly_chart(' in panel
    assert 'key="viewport_panel"' in panel
    assert '.st-key-viewport_panel' in source
    assert 'border: 1px solid {panel_border_color} !important;' in source


def test_dashboard_reduces_top_whitespace_and_aligns_content() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert '[data-testid="stMainBlockContainer"] {{' in source
    assert 'padding-top: 1rem !important;' in source
    assert '[data-testid="stSidebarContent"] > div,' in source
    assert 'margin-bottom: 1rem !important;' in source


def test_sidebar_collapse_icon_has_contrast_in_both_interface_themes() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert '[data-testid="stSidebarCollapseButton"] svg' in source
    assert 'color: {text_color} !important;' in source
    assert 'fill: {text_color} !important;' in source


def test_pure_white_theme_has_white_plot_scene_walls_and_contrasting_markers() -> None:
    points = np.loadtxt("data/nist/plane_sample.xyz")
    figure, _ = _inspection_figure(points, "plane", theme="Pure White (Lab Report)")

    assert figure.layout.paper_bgcolor == "#ffffff"
    assert figure.layout.plot_bgcolor == "#ffffff"
    assert figure.layout.template.to_plotly_json()["layout"]["paper_bgcolor"] == "white"
    assert figure.layout.font.color == "#020617"
    assert figure.layout.scene.bgcolor == "#ffffff"
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        axis = getattr(figure.layout.scene, axis_name)
        assert axis.backgroundcolor == "#e2e8f0"
        assert axis.gridcolor == "#64748b"
        assert axis.gridwidth == 1.5
        assert axis.zerolinecolor == "#1e293b"
        assert axis.zerolinewidth == 2.0
        assert axis.linecolor == "#000000"
        assert axis.linewidth == 3.0
        assert axis.tickfont.color == "#020617"
        assert axis.tickfont.size == 11
        assert axis.tickfont.family == "Arial, sans-serif"
        assert axis.title.text == f"{axis_name[0].upper()} [mm]"
        assert axis.title.font.color == "#020617"
        assert axis.title.font.size == 13
        assert axis.title.font.family == "Arial, sans-serif"
    assert figure.data[0].marker.colorbar.title.font.color == "#020617"
    assert figure.data[0].marker.colorbar.tickfont.color == "#020617"
    assert figure.layout.scene.aspectratio.to_plotly_json() == {
        "x": 1.0,
        "y": 1.0,
        "z": 0.35,
    }
    assert figure.data[0].marker.line.color == "#111827"


def test_figure_themes_explicitly_configure_scene_and_canvas() -> None:
    points = np.loadtxt("data/nist/plane_sample.xyz")
    for theme, expected_bg in (
        ("Pure White (Lab Report)", "#ffffff"),
        ("Dark Industrial", "#0e1117"),
    ):
        figure, _ = _inspection_figure(points, "plane", theme=theme)

        assert figure.layout.paper_bgcolor == expected_bg
        assert figure.layout.scene.bgcolor == expected_bg
        assert figure.layout.scene.aspectmode == "manual"


def test_dark_viewport_theme_uses_requested_canvas_and_axis_palette() -> None:
    points = np.loadtxt("data/nist/plane_sample.xyz")
    figure, _ = _inspection_figure(points, "plane", theme="Force Dark")

    assert figure.layout.paper_bgcolor == "#0e1117"
    assert figure.layout.plot_bgcolor == "#0e1117"
    assert figure.layout.font.color == "#f8fafc"
    assert figure.layout.scene.bgcolor == "#0e1117"
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        axis = getattr(figure.layout.scene, axis_name)
        assert axis.backgroundcolor == "#1e293b"
        assert axis.gridcolor == "#475569"
        assert axis.gridwidth == 1.5
        assert axis.zerolinecolor == "#94a3b8"
        assert axis.zerolinewidth == 2.0
        assert axis.linecolor == "#f8fafc"
        assert axis.linewidth == 3.0
        assert axis.tickfont.color == "#f8fafc"
        assert axis.tickfont.size == 11
        assert axis.tickfont.family == "Arial, sans-serif"
        assert axis.title.font.color == "#f8fafc"
        assert axis.title.font.size == 13
        assert axis.title.font.family == "Arial, sans-serif"
    assert figure.data[0].marker.colorbar.title.font.color == "#f8fafc"
    assert figure.data[0].marker.colorbar.tickfont.color == "#f8fafc"
    assert figure.data[0].marker.line.color == "white"


def test_legacy_auto_theme_falls_back_to_explicit_light_contrast() -> None:
    points = np.loadtxt("data/nist/plane_sample.xyz")
    figure, _ = _inspection_figure(points, "plane", theme="Auto (Inherit)")

    assert figure.layout.paper_bgcolor == "#ffffff"
    assert figure.layout.plot_bgcolor == "#ffffff"
    assert figure.layout.scene.bgcolor == "#ffffff"
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        axis = getattr(figure.layout.scene, axis_name)
        assert axis.showbackground is True
        assert axis.gridcolor == "#64748b"
        assert axis.zerolinecolor == "#1e293b"
        assert axis.tickfont.color == "#020617"


def test_viewport_defaults_to_white_and_white_theme_is_explicitly_configured() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert '"Color Theme"' not in source
    assert '"Inspection Mode"' in source
    assert 'figure, use_container_width=True, theme="streamlit"' not in source
    assert 'bg_axis_wall = "#1e293b" if is_dark else "#e2e8f0"' in source
    assert 'scene=scene_options' in source


def test_plotly_render_calls_disable_streamlit_theme() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert source.count('theme=None,') == 1


def test_dashboard_disables_streamlit_plotly_theme_and_uses_clean_headers() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert '                figure,\n                use_container_width=True,\n                theme=None,' in source
    assert 'st.markdown("### 3D Viewport")' in source
    assert '"Selected Controls:"' in source
    assert 'options = ["Auto-Detect (Suggested)", "plane", "line", "cylinder", "sphere"]' in source
    assert 'index=0' in source
    assert '"plotly_dark" if is_dark else "plotly_white"' in source
    assert 'geometry = (\n                    detected\n                    if geometry_selection == "Auto-Detect (Suggested)"' in source
    for emoji in ("🌙", "☀️", "⚡", "📁", "✕", "✔", "✖"):
        assert emoji not in source
    assert 'st.caption("LOADED ENTITY")' in source
    assert 'Heuristic classification: **{detected.upper()}** (PCA confidence: {confidence:.1f}%)' in source
    assert 'st.success("STATUS: PASS")' in source
    assert 'st.error("STATUS: FAIL")' in source


def test_iso_1101_pill_options_cover_form_and_datum_features() -> None:
    assert _pill_configuration("plane")[0] == ["⏥ Flatness (PtV)"]
    assert _pill_configuration("cylinder")[0] == [
        "⌭ Cylindricity (PtV)",
        "⏤ Axis Straightness",
    ]
    assert _pill_configuration("sphere")[0] == ["S○ Sphericity (PtV)"]
    assert _pill_configuration("line")[0] == ["⏤ Straightness (PtV)"]
    assert _pill_configuration("datum")[0] == [
        "⟂ Perpendicularity",
        "// Parallelism",
        "◎ Coaxiality",
    ]


def test_cylinder_figure_has_nominal_mesh_and_centered_points() -> None:
    points = np.loadtxt("data/nist/cylinder_sample.xyz")
    figure, _ = _inspection_figure(points, "cylinder")

    assert figure.layout.scene.aspectmode == "manual"
    np.testing.assert_allclose(np.mean(figure.data[0].x), 0.0, atol=1e-12)
    assert figure.data[0].marker.size == 9
    assert figure.data[1].type == "surface"
    assert figure.data[1].showscale is False


def test_inspection_metrics_expose_rms_and_geometry_properties() -> None:
    points = np.loadtxt("data/nist/plane_sample.xyz")
    metrics = _inspection_metrics(points, "plane")

    assert metrics["measured_deviation"] == pytest.approx(0.0021176470588)
    assert metrics["residual_rms"] == np.sqrt(np.mean(np.square(metrics["residuals"])))
    np.testing.assert_allclose(metrics["normal"], [0.0, 0.0, 1.0], atol=2e-6)
    np.testing.assert_allclose(metrics["centroid"], [50.0, 50.0, 0.0])


def test_dashboard_uses_english_metrology_labels_without_json() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert "st.json" not in source
    assert 'st.set_page_config(page_title="GD&T Metrology Inspector", layout="wide")' in source
    assert 'st.title("GD&T Metrology Inspector")' in source
    assert "Upload Point Cloud" in source
    assert "Point Count" in source
    assert "Target Feature" in source
    assert "Tolerance Limit (t) [mm]" in source
    assert 'st.expander("Display Settings")' not in source
    assert "#MainMenu {" not in source
    assert "st.columns([3, 2])" in source
    assert 'st.success("STATUS: PASS")' in source
    assert 'st.error("STATUS: FAIL")' in source
    assert 'st.metric("Peak Deviation", f"{worst_val:.4f} mm"' in source
    assert "Residual RMS" in source
    assert 'st.columns(3)' in source
    assert 'st.container(border=True)' in source
    assert 'Characteristic' in source
    assert 'Specified Limit [mm]' in source
    assert 'Measured Value [mm / µm]' in source
    assert 'Deviation from Limit [mm]' in source
    assert 'col_csv, col_html, col_json = st.columns(3)' in source
    assert 'use_container_width=True' in source
    assert "Fitted Normal Vector" in source
    assert "st.pills(" in source
    assert 'st.caption("LOADED ENTITY")' in source


def test_css_rules_present() -> None:
    with open(APP_PATH, encoding="utf-8") as source_file:
        code = source_file.read()

    assert ".stDeployButton" in code
    assert '[data-testid="stAppDeployButton"]' in code
    assert '[data-testid="stDeployButton"]' in code
    assert "display: none !important;" in code
    assert "stHeaderActionElements" in code
    assert 'header[data-testid="stHeader"]' in code
    assert 'header {visibility: hidden' not in code
    assert '#MainMenu, [data-testid="stMainMenu"]' in code
    assert 'display: none !important;' in code


def test_sidebar_upload_is_hidden_while_cloud_is_loaded() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    start = source.index("def _sidebar_cloud_input(")
    end = source.index("\ndef _cached_upload(", start)
    sidebar_input = source[start:end]

    assert 'st.expander("Upload / Replace File")' not in sidebar_input
    assert "if points is not None:" in sidebar_input
    assert "    else:" in sidebar_input
    assert "uploaded_file = st.file_uploader(" in sidebar_input
    assert 'st.caption("LOADED ENTITY")' in sidebar_input


def test_sidebar_exposes_separate_interface_and_viewport_themes() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    start = source.index('if st.sidebar.button("Clear All Cached Clouds"):')
    end = source.index("        if mode ==", start)
    sidebar = source[start:end]

    assert '"Interface Theme"' in sidebar
    assert '"Viewport Theme"' in sidebar
    assert sidebar.index('st.radio("Inspection Mode"') > sidebar.index(
        'st.sidebar.button("Clear All Cached Clouds")'
    )


def test_dashboard_uses_sidebar_themes_and_applies_interface_css() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'interface_theme = st.radio(' in source
    assert 'viewport_theme = st.radio(' in source
    assert 'theme=viewport_theme' in source
    assert 'if interface_theme == "Dark Mode":' in source
    assert 'app_background, sidebar_background = "#0e1117", "#161b22"' in source
    assert 'app_background, sidebar_background = "#ffffff", "#f8fafc"' in source
    assert 'background-color: {app_background} !important;' in source
    assert '.stApp button {{' in source
    assert 'background-color: {input_background} !important;' in source
    assert 'def _comparison_dataframe(' in source
    assert 'st: Any, rows: list[dict[str, Any]], theme: str' in source
    assert 'set_table_styles(' in source
    assert 'st.table(styled)' in source
    assert '[data-testid="stFileUploaderDropzone"]' in source
    assert '[data-testid="stFileUploader"] section' in source
