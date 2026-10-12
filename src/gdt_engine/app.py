"""Interactive Streamlit point-cloud inspection dashboard."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from gdt_engine.diagnostics import detect_feature_type
from gdt_engine.fitting.cylinder import fit_cylinder
from gdt_engine.fitting.line import fit_line_3d
from gdt_engine.fitting.plane import fit_plane
from gdt_engine.fitting.sphere import fit_sphere
from gdt_engine.io import load_point_cloud
from gdt_engine.tolerances.orientation import (
    evaluate_coaxiality,
    evaluate_parallelism,
    evaluate_perpendicularity,
)


def _load_uploaded_file(uploaded_file) -> np.ndarray:
    suffix = Path(uploaded_file.name).suffix or ".xyz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(uploaded_file.getvalue())
        temporary_path = Path(handle.name)
    try:
        return load_point_cloud(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _initialize_cloud_cache(session_state: Any) -> None:
    for cache_key in ("data_single", "data_datum", "data_feature"):
        session_state.setdefault(cache_key, {"points": None, "filename": None})


def _sidebar_cloud_input(
    st: Any, cache_key: str, uploader_label: str
) -> np.ndarray | None:
    entry = st.session_state.get(cache_key, {"points": None, "filename": None})
    points = entry.get("points")
    if points is not None:
        with st.container(border=True):
            st.caption("LOADED ENTITY")
            st.markdown(f"**{entry['filename']}** · {len(points)} pts")
            if st.button(
                "Unload / Replace",
                key=f"btn_replace_{cache_key}",
                use_container_width=True,
            ):
                st.session_state[cache_key] = {"points": None, "filename": None}
                generation_key = f"{cache_key}_uploader_generation"
                st.session_state[generation_key] = (
                    st.session_state.get(generation_key, 0) + 1
                )
                st.rerun()
    else:
        generation_key = f"{cache_key}_uploader_generation"
        generation = st.session_state.get(generation_key, 0)
        uploaded_file = st.file_uploader(
            uploader_label,
            type=["csv", "xyz"],
            key=f"{cache_key}_uploader_{generation}",
        )
        if uploaded_file is not None:
            try:
                points, _ = _cached_upload(st.session_state, uploaded_file, cache_key)
            except ValueError as exc:
                st.error(str(exc))
                return None
            st.rerun()
    return points


def _cached_upload(
    session_state: Any, uploaded_file: Any, state_key: str
) -> tuple[np.ndarray | None, str | None]:
    if uploaded_file is not None:
        points = _load_uploaded_file(uploaded_file)
        entry = {"points": points, "filename": uploaded_file.name}
        session_state[state_key] = entry
        return points, uploaded_file.name
    cached = session_state.get(state_key, {"points": None, "filename": None})
    return cached.get("points"), cached.get("filename")


def _pill_configuration(geometry: str) -> tuple[list[str], dict[str, str]]:
    if geometry == "datum":
        mapping = {
            "⟂ Perpendicularity": "Perpendicularity (⊥)",
            "// Parallelism": "Parallelism (//)",
            "◎ Coaxiality": "Coaxiality (◎)",
        }
    else:
        characteristics = {
            "plane": [
                ("⏥ Flatness (PtV)", "Flatness (PtV)"),
            ],
            "cylinder": [
                ("⌭ Cylindricity (PtV)", "Cylindricity (PtV)"),
                ("⏤ Axis Straightness", "Axis Straightness"),
            ],
            "sphere": [
                ("S○ Sphericity (PtV)", "Sphericity (PtV)"),
            ],
            "line": [("⏤ Straightness (PtV)", "Straightness (PtV)")],
        }
        mapping = dict(characteristics[geometry])
    return list(mapping), mapping


def _sync_selected_feature(
    session_state: Any, points: np.ndarray, filename: str | None = None
) -> tuple[str, float]:
    """Update the geometry suggestion when the active cloud changes."""
    cloud = np.ascontiguousarray(points, dtype=np.float64)
    digest = hashlib.sha256(cloud.tobytes()).hexdigest()
    signature = f"{filename or ''}:{cloud.shape}:{digest}"
    valid_features = {
        "Auto-Detect (Suggested)", "plane", "line", "cylinder", "sphere"
    }
    changed = session_state.get("_target_feature_cloud_signature") != signature
    invalid_selection = session_state.get("selected_target_feature") not in valid_features
    suggestion = session_state.get("_target_feature_suggestion")
    if changed or invalid_selection or suggestion is None:
        detected, confidence = detect_feature_type(cloud)
        session_state["selected_target_feature"] = "Auto-Detect (Suggested)"
        session_state["_target_feature_cloud_signature"] = signature
        session_state["_target_feature_suggestion"] = (detected, confidence)
        return detected, confidence
    return suggestion


def _comparison_rows(
    geometry: str,
    metrics: dict[str, Any],
    selected_tolerances: list[str],
    tolerance: float,
) -> list[dict[str, Any]]:
    if "datum_axis" in metrics:
        return [
            {
                "Characteristic": geometry,
                "Specified Limit [mm]": float(tolerance),
                "Measured Value [mm / µm]": _format_deviation(
                    float(metrics["measured_deviation"])
                ),
                "Deviation from Limit [mm]": float(metrics["measured_deviation"])
                - tolerance,
                "Status": "PASS" if metrics["verdict"] else "FAIL",
            }
        ] if selected_tolerances else []
    residuals = np.asarray(metrics["residuals"], dtype=np.float64)
    measured_by_characteristic = {
        "Flatness (PtV)": float(metrics.get("ptv", np.ptp(residuals))),
        "Peak-to-Reference": float(metrics.get("properties", {}).get("max_peak", np.max(residuals))),
        "Valley-to-Reference": abs(float(metrics.get("properties", {}).get("max_valley", np.min(residuals)))),
        "Cylindricity (PtV)": float(metrics.get("ptv", np.ptp(residuals))),
        "Axis Straightness": float(metrics.get("properties", {}).get("axis_straightness", np.max(np.abs(residuals)))),
        "Sphericity (PtV)": float(metrics.get("ptv", np.ptp(residuals))),
        "Straightness (PtV)": float(metrics.get("measured_deviation", np.max(residuals))),
    }
    rows = []
    for characteristic in selected_tolerances:
        measured = measured_by_characteristic[characteristic]
        rows.append(
            {
                "Characteristic": characteristic,
                "Specified Limit [mm]": float(tolerance),
                "Measured Value [mm / µm]": _format_deviation(measured),
                "Deviation from Limit [mm]": measured - tolerance,
                "Status": "PASS" if measured <= tolerance else "FAIL",
            }
        )
    return rows


def _comparison_dataframe(
    st: Any, rows: list[dict[str, Any]], theme: str
) -> None:
    import pandas as pd

    dataframe = pd.DataFrame(rows)
    is_dark = theme == "Dark Mode"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    header_background = "#1f2937" if is_dark else "#e2e8f0"
    row_background = "#111827" if is_dark else "#ffffff"
    border_color = "#334155" if is_dark else "#cbd5e1"
    styled = dataframe.style.set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("background-color", row_background),
                    ("color", text_color),
                    ("border-collapse", "collapse"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", header_background),
                    ("color", text_color),
                    ("border", f"1px solid {border_color}"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", row_background),
                    ("color", text_color),
                    ("border", f"1px solid {border_color}"),
                ],
            },
        ]
    ).map(
        lambda value: "color: #16803c; font-weight: bold"
        if value == "PASS"
        else "color: #c62828; font-weight: bold",
        subset=["Status"],
    )
    st.table(styled)


def _format_bounding_box(extents: np.ndarray) -> str:
    x, y, z = (float(value) for value in extents)
    return f"{x:.2f} x {y:.2f} x {z:.3f} mm"


def _format_deviation(value: float) -> str:
    return f"{value:.4f} mm ({value * 1000:.1f} µm)"


def _inspection_metrics(points: np.ndarray, geometry: str) -> dict[str, Any]:
    if geometry == "plane":
        fit = fit_plane(points)
        residuals = fit.signed_distances
        centroid = fit.centroid
        properties = {"normal": fit.normal, "max_peak": float(np.max(residuals)), "max_valley": float(np.min(residuals))}
    elif geometry == "sphere":
        fit = fit_sphere(points)
        residuals = fit.radial_residuals
        centroid = fit.center
        properties = {"center": fit.center, "radius": fit.radius}
    elif geometry == "cylinder":
        p0, axis, radius, residuals = fit_cylinder(points)
        fit = (p0, axis, radius)
        centroid = p0
        properties = {"axis": axis, "radius": radius, "diameter": 2.0 * radius, "point_on_axis": p0, "axis_straightness": float(np.max(np.abs(residuals)))}
    elif geometry == "line":
        p0, axis, residuals = fit_line_3d(points)
        fit = (p0, axis)
        centroid = p0
        properties = {"axis": axis, "point_on_axis": p0}
    else:
        raise ValueError(f"unsupported geometry: {geometry}")
    ptv = float(np.ptp(residuals))
    measured = float(np.max(residuals)) if geometry == "line" else ptv
    return {"fit": fit, "residuals": residuals, "ptv": ptv, "measured_deviation": measured, "residual_rms": float(np.sqrt(np.mean(np.square(residuals)))), "normal": properties.get("normal"), "properties": properties, "centroid": centroid, "extents": np.ptp(points, axis=0)}


def _scene_theme(theme: str) -> dict[str, Any]:
    is_dark = theme in {"Dark Mode", "Force Dark", "Dark Industrial"}
    bg_canvas = "#0e1117" if is_dark else "#ffffff"
    bg_axis_wall = "#1e293b" if is_dark else "#e2e8f0"
    grid_color = "#475569" if is_dark else "#64748b"
    zero_line = "#94a3b8" if is_dark else "#1e293b"
    frame_edge = "#f8fafc" if is_dark else "#000000"
    font_color = "#f8fafc" if is_dark else "#020617"
    axis_common = {
        "showbackground": True,
        "backgroundcolor": bg_axis_wall,
        "showgrid": True,
        "gridcolor": grid_color,
        "gridwidth": 1.5,
        "zeroline": True,
        "zerolinecolor": zero_line,
        "zerolinewidth": 2.0,
        "showline": True,
        "linecolor": frame_edge,
        "linewidth": 3.0,
        "tickfont": {"color": font_color, "size": 11, "family": "Arial, sans-serif"},
        "title": {"font": {"color": font_color, "size": 13, "family": "Arial, sans-serif"}},
    }
    return {
        "template": "plotly_dark" if is_dark else "plotly_white",
        "paper_bgcolor": bg_canvas,
        "plot_bgcolor": bg_canvas,
        "font": {"color": font_color},
        "scene": {
            "bgcolor": bg_canvas,
            "xaxis": {**axis_common, "title": {**axis_common["title"], "text": "X [mm]"}},
            "yaxis": {**axis_common, "title": {**axis_common["title"], "text": "Y [mm]"}},
            "zaxis": {**axis_common, "title": {**axis_common["title"], "text": "Z [mm]"}},
        },
    }


def _inspection_figure(points: np.ndarray, geometry: str, point_size: int = 9, plane_opacity: float = 0.35, theme: str = "Pure White (Lab Report)") -> tuple[Any, float]:
    import plotly.graph_objects as go

    origin = points.mean(axis=0)
    centered = points - origin
    metrics = _inspection_metrics(points, geometry)
    residuals = metrics["residuals"]
    is_dark = theme in {"Dark Mode", "Force Dark", "Dark Industrial"}
    c_font = "#020617" if not is_dark else "#f8fafc"
    marker_line_color = "white" if is_dark else "gray" if theme == "Auto (Inherit)" else "#111827"
    colorbar = {
        "title": {
            "text": "<b>Deviation</b><br>[mm]",
            "side": "top",
            "font": {"size": 11, "color": c_font, "family": "Arial, sans-serif"},
        },
        "thickness": 14,
        "len": 0.65,
        "y": 0.50,
        "yanchor": "middle",
        "x": 1.05,
        "xanchor": "left",
        "ypad": 15,
        "tickfont": {"size": 10, "color": c_font, "family": "Arial, sans-serif"},
        "outlinewidth": 0,
    }
    figure = go.Figure(go.Scatter3d(x=centered[:, 0], y=centered[:, 1], z=centered[:, 2], mode="markers", marker={"size": point_size, "line": {"width": 1, "color": marker_line_color}, "color": residuals, "colorscale": "Turbo", "showscale": True, "colorbar": colorbar}, name="Measured points"))
    if geometry == "line":
        p0, axis = metrics["fit"]
        extent = max(float(np.ptp(points @ axis)), 1.0) / 2.0
        line = np.array([p0 - extent * axis, p0 + extent * axis]) - origin
        figure.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode="lines", line={"color": "red", "width": 5}, name="Fitted line axis"))
    elif geometry == "plane":
        fit = metrics["fit"]
        span = np.ptp(points, axis=0).max() * 0.1 or 1.0
        c = fit.centroid - origin
        u = np.linspace(centered[:, 0].min() - span, centered[:, 0].max() + span, 12)
        v = np.linspace(centered[:, 1].min() - span, centered[:, 1].max() + span, 12)
        xx, yy = np.meshgrid(u, v)
        zz = c[2] - (fit.normal[0] * (xx - c[0]) + fit.normal[1] * (yy - c[1])) / fit.normal[2]
        figure.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=plane_opacity, showscale=False, name="Fitted plane"))
    elif geometry == "sphere":
        fit = metrics["fit"]
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 30)
        c = fit.center - origin
        sx = c[0] + fit.radius * np.outer(np.sin(phi), np.cos(theta))
        sy = c[1] + fit.radius * np.outer(np.sin(phi), np.sin(theta))
        sz = c[2] + fit.radius * np.outer(np.cos(phi), np.ones(theta.size))
        figure.add_trace(go.Surface(x=sx, y=sy, z=sz, opacity=plane_opacity, showscale=False, name="Fitted sphere"))
    else:
        p0, axis, radius = metrics["fit"]
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(axis, helper); e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
        t = np.linspace(-np.ptp(points @ axis) / 2, np.ptp(points @ axis) / 2, 20)
        a = np.linspace(0, 2 * np.pi, 30)
        mesh = np.array([p0 - origin + ti * axis + radius * (np.cos(ai) * e1 + np.sin(ai) * e2) for ti in t for ai in a]).reshape(len(t), len(a), 3)
        figure.add_trace(go.Surface(x=mesh[:, :, 0], y=mesh[:, :, 1], z=mesh[:, :, 2], opacity=plane_opacity, showscale=False, name="Fitted cylinder"))
    themed_scene = _scene_theme(theme)
    themed_scene["scene"].update({"aspectmode": "manual", "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.35}, "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 0.8}, "center": {"x": 0, "y": 0, "z": 0}}})
    scene_options = themed_scene.pop("scene")
    figure.update_layout(
        height=650,
        margin={"l": 10, "r": 95, "b": 10, "t": 10},
        legend={"x": 0.02, "y": 0.88, "xanchor": "left", "yanchor": "top"},
        showlegend=False,
        scene=scene_options,
        **themed_scene,
    )
    return figure, metrics["measured_deviation"]


def _datum_figure(datum: np.ndarray, feature: np.ndarray, metrics: dict[str, Any], theme: str = "Pure White (Lab Report)") -> Any:
    """Render datum and tolerated feature in one centered orientation view."""
    import plotly.graph_objects as go

    origin = feature.mean(axis=0)
    datum_centered = datum - origin
    feature_centered = feature - origin
    is_dark = theme in {"Dark Mode", "Force Dark", "Dark Industrial"}
    c_font = "#020617" if not is_dark else "#f8fafc"
    figure = go.Figure(go.Scatter3d(
        x=datum_centered[:, 0], y=datum_centered[:, 1], z=datum_centered[:, 2],
        mode="markers", marker={"size": 9, "color": "lightgray", "opacity": 0.35,
                                 "line": {"color": "white" if is_dark else "#111827", "width": 1},
                                 "showscale": False}, name="Datum [A]"))
    figure.add_trace(go.Scatter3d(
        x=feature_centered[:, 0], y=feature_centered[:, 1], z=feature_centered[:, 2],
        mode="markers", marker={"size": 9, "color": metrics["residuals"],
                                 "line": {"color": "white" if is_dark else "#111827", "width": 1},
                                 "colorscale": "Turbo", "showscale": True,
                                 "colorbar": {"title": {"text": "<b>Deviation</b><br>[mm]", "side": "top", "font": {"size": 11, "color": c_font, "family": "Arial, sans-serif"}}, "thickness": 14, "len": 0.65, "y": 0.50, "yanchor": "middle", "x": 1.05, "xanchor": "left", "ypad": 15, "tickfont": {"size": 10, "color": c_font, "family": "Arial, sans-serif"}, "outlinewidth": 0}},
        name="Tolerated feature"))
    vector = metrics["datum_axis"]
    extent = max(float(np.ptp(datum @ vector)), 1.0) / 2.0
    axis = np.array([datum.mean(axis=0) - extent * vector,
                     datum.mean(axis=0) + extent * vector]) - origin
    figure.add_trace(go.Scatter3d(
        x=axis[:, 0], y=axis[:, 1], z=axis[:, 2], mode="lines",
        line={"color": "gray", "width": 5}, name="Datum vector"))
    themed_scene = _scene_theme(theme)
    themed_scene["scene"].update({"aspectmode": "manual", "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.35},
                                  "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 0.8},
                                             "center": {"x": 0, "y": 0, "z": 0}}})
    scene_options = themed_scene.pop("scene")
    figure.update_layout(
        height=650,
        margin={"l": 10, "r": 95, "b": 10, "t": 10},
        scene=scene_options,
        **themed_scene,
    )
    return figure


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summary_json(metrics: dict[str, Any], target: str) -> str:
    payload = {"target_feature": target, "metrics": {key: value for key, value in metrics.items() if key not in {"fit", "residuals"}}}
    return json.dumps(_jsonable(payload), indent=2, default=str)


def _report_csv(points: np.ndarray, metrics: dict[str, Any], mode: str, target: str, tolerance: float) -> str:
    if metrics.get("evaluated_characteristics") == []:
        verdict = "NOT EVALUATED"
    else:
        verdict = "PASS" if metrics["measured_deviation"] <= tolerance else "FAIL"
    output = io.StringIO()
    output.write(f"# Mode,{mode}\n# Target Feature / Relation,{target}\n# Nominal Tolerance,{tolerance}\n# Measured Deviation,{metrics['measured_deviation']}\n# Verdict,{verdict}\n")
    writer = csv.writer(output)
    writer.writerow(["X", "Y", "Z", "Residual_Deviation_mm", "In_Tolerance_Flag"])
    for point, residual in zip(points, metrics["residuals"]):
        writer.writerow([*point, float(residual), bool(abs(residual) <= tolerance)])
    return output.getvalue()


def _render_result_panel(
    st: Any,
    points: np.ndarray,
    geometry: str,
    tolerance: float,
    selected_tolerances: list[str],
    rows: list[dict[str, Any]],
    theme: str,
) -> dict[str, Any]:
    metrics = _inspection_metrics(points, geometry)
    if rows:
        metrics["measured_deviation"] = max(
            row["Deviation from Limit [mm]"] + tolerance for row in rows
        )
    metrics["evaluated_characteristics"] = selected_tolerances
    if rows:
        _comparison_dataframe(st, rows, theme)
    worst_val = float(metrics["measured_deviation"])
    rms_val = float(metrics["residual_rms"])
    properties = metrics["properties"]
    radius = float(properties.get("radius", 0.0))
    has_radius = "radius" in properties
    length_extent = float(np.max(metrics["extents"]))
    with st.container(border=True):
        st.markdown("### Inspection Report")
        peak_col, rms_col, size_col = st.columns(3)
        with peak_col:
            st.metric("Peak Deviation", f"{worst_val:.4f} mm", delta=f"{worst_val*1000:.1f} µm", delta_color="off")
        with rms_col:
            st.metric("Residual RMS", f"{rms_val:.4f} mm", delta=f"{rms_val*1000:.1f} µm", delta_color="off")
        with size_col:
            st.metric("Fitted Size", f"R {radius:.3f} mm" if has_radius else f"L {length_extent:.1f} mm")
        with st.container(border=True):
            if geometry == "plane":
                st.write(f"**Fitted Normal Vector:** {properties['normal'].round(4).tolist()}")
                st.write(f"**Centroid:** {metrics['centroid'].round(4).tolist()}")
            elif geometry == "cylinder":
                st.write(f"**Axis Direction Vector V:** {properties['axis'].round(4).tolist()}")
                st.write(f"**Point on Axis:** {properties['point_on_axis'].round(4).tolist()}")
            elif geometry == "sphere":
                st.write(f"**Center Coordinates:** {properties['center'].round(4).tolist()}")
            else:
                axis, point_on_axis = metrics["fit"][:2]
                st.write(f"**Axis Direction Vector V:** {axis.round(4).tolist()}")
                st.write(f"**Point on Axis:** {point_on_axis.round(4).tolist()}")
            st.write(f"**Point Count:** {len(points)}")
            st.write(f"**Bounding Box Dimensions:** {_format_bounding_box(metrics['extents'])}")
    return metrics


def _datum_metrics(datum: np.ndarray, feature: np.ndarray, relation: str, length: float, tolerance: float) -> dict[str, Any]:
    datum_line = fit_line_3d(datum)
    feature_line = fit_line_3d(feature)
    datum_axis, feature_axis = datum_line[1], feature_line[1]
    if relation == "Perpendicularity (⊥)":
        nominal, deviation = 90.0, evaluate_perpendicularity(datum_axis, feature_axis, length)
    elif relation == "Parallelism (//)":
        nominal, deviation = 0.0, evaluate_parallelism(datum_axis, feature_axis, length)
    else:
        nominal, deviation = 0.0, evaluate_coaxiality(datum_line[0], datum_axis, feature_line[0], feature_axis, length)
    measured_angle = float(
        np.rad2deg(
            np.arccos(np.clip(abs(np.dot(datum_axis, feature_axis)), 0.0, 1.0))
        )
    )
    return {
        "nominal_angle_deg": nominal,
        "measured_angle_deg": measured_angle,
        "angular_error_deg": abs(measured_angle - nominal),
        "measured_deviation": float(deviation),
        "datum_axis": datum_axis,
        "feature_axis": feature_axis,
        "verdict": deviation <= tolerance,
        "length": length,
        "relation": relation,
        "residuals": feature_line[2],
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GD&T Metrology Inspector", layout="wide")
    st.markdown(
        """
        <style>
        /* Ukryj wylacznie przycisk Deploy */
        .stDeployButton,
        [data-testid="stAppDeployButton"],
        [data-testid="stDeployButton"] {
            display: none !important;
        }

        #MainMenu, [data-testid="stMainMenu"] {
            display: none !important;
        }

        /* Ukryj kotwice linku przy tytulach */
        [data-testid="stHeaderActionElements"], a.anchor-link {
            display: none !important;
        }

        /* Przezroczysty naglowek z zachowaniem kontrolek */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("GD&T Metrology Inspector")
    _initialize_cloud_cache(st.session_state)
    if st.sidebar.button("Clear All Cached Clouds"):
        for cache_key in ("data_single", "data_datum", "data_feature"):
            st.session_state.pop(cache_key, None)
            generation_key = f"{cache_key}_uploader_generation"
            st.session_state[generation_key] = (
                st.session_state.get(generation_key, 0) + 1
            )
        st.rerun()
    with st.sidebar:
        mode = st.radio("Inspection Mode", ["Single Feature (Form)", "Datum-Referenced (Orientation)"])
        interface_theme = st.radio(
            "Interface Theme",
            options=["Light Mode", "Dark Mode"],
            horizontal=True,
            key="interface_theme_selector",
        )
        viewport_theme = st.radio(
            "Viewport Theme",
            options=["Light Mode", "Dark Mode"],
            horizontal=True,
            key="viewport_theme_selector",
        )
        if mode == "Single Feature (Form)":
            points = _sidebar_cloud_input(
                st, "data_single", "Upload Point Cloud (.csv, .xyz)"
            )
        else:
            datum = _sidebar_cloud_input(
                st, "data_datum", "1. Upload Datum Feature [A] (.csv, .xyz)"
            )
            feature = _sidebar_cloud_input(
                st, "data_feature", "2. Upload Tolerated Feature (.csv, .xyz)"
            )
    if interface_theme == "Dark Mode":
        app_background, sidebar_background = "#0e1117", "#161b22"
        surface_background, text_color = "#111827", "#f8fafc"
        muted_color, border_color, input_background = "#cbd5e1", "#334155", "#1f2937"
        panel_border_color = "#94a3b8"
    else:
        app_background, sidebar_background = "#ffffff", "#f8fafc"
        surface_background, text_color = "#ffffff", "#0f172a"
        muted_color, border_color, input_background = "#475569", "#cbd5e1", "#ffffff"
        panel_border_color = "#64748b"
    st.markdown(
        f"""
        <style>
        [data-testid="stMainBlockContainer"] {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }}
        [data-testid="stSidebarContent"] > div,
        [data-testid="stSidebar"] .block-container {{
            padding-top: 0.5rem !important;
        }}
        .stApp h1 {{
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
        }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background-color: {app_background} !important;
            color: {text_color} !important;
        }}
        [data-testid="stSidebar"] > div {{
            background-color: {sidebar_background} !important;
            color: {text_color} !important;
        }}
        .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"] {{
            color: {text_color} !important;
        }}
        .stApp [data-testid="stVerticalBlockBorderWrapper"],
        .stApp [data-testid="stExpander"] {{
            background-color: {surface_background} !important;
            border: 1px solid {border_color} !important;
            border-radius: 0.5rem !important;
        }}
        .stApp .st-key-viewport_panel {{
            background-color: {surface_background} !important;
            border: 1px solid {panel_border_color} !important;
            border-radius: 0.75rem !important;
            padding: 0.5rem 0.75rem !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }}
        .stApp input, .stApp textarea,
        .stApp [data-baseweb="select"] > div {{
            background-color: {input_background} !important;
            color: {text_color} !important;
            border-color: {border_color} !important;
        }}
        .stApp button {{
            background-color: {input_background} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}
        .stApp button:hover {{
            background-color: {surface_background} !important;
            border-color: {text_color} !important;
        }}
        [data-testid="stSidebarCollapseButton"] button {{
            background-color: {sidebar_background} !important;
            border-color: {border_color} !important;
            color: {text_color} !important;
        }}
        [data-testid="stSidebarCollapseButton"] button * {{
            color: {text_color} !important;
        }}
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stSidebarCollapseButton"] svg path {{
            color: {text_color} !important;
            fill: {text_color} !important;
            stroke: {text_color} !important;
        }}
        .stApp [data-testid="stFileUploaderDropzone"],
        .stApp [data-testid="stFileUploader"] section {{
            background-color: {input_background} !important;
            color: {text_color} !important;
            border-color: {border_color} !important;
        }}
        .stApp [data-testid="stFileUploader"] p,
        .stApp [data-testid="stFileUploader"] small,
        .stApp [data-testid="stFileUploader"] span {{
            color: {text_color} !important;
        }}
        .stApp [data-testid="stCaptionContainer"] {{ color: {muted_color} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    point_size = 9
    plane_opacity = 0.35
    if mode == "Single Feature (Form)":
        if points is None:
            st.info("Upload a point cloud to begin coordinate verification.")
            return
    elif datum is None or feature is None:
        st.info("Upload both datum and tolerated point clouds to begin orientation verification.")
        return
    viewport, results = st.columns([3, 2])
    with results:
        with st.container(border=True):
            st.markdown("### Inspection Specification")
            if mode == "Single Feature (Form)":
                cache_entry = st.session_state.get("data_single", {})
                detected, confidence = _sync_selected_feature(
                    st.session_state, points, cache_entry.get("filename")
                )
                options = ["Auto-Detect (Suggested)", "plane", "line", "cylinder", "sphere"]
                geometry_selection = st.selectbox(
                    "Target Feature",
                    options=options,
                    index=0,
                    key="selected_target_feature",
                )
                st.caption(
                    f"Heuristic classification: **{detected.upper()}** (PCA confidence: {confidence:.1f}%)"
                )
                geometry = (
                    detected
                    if geometry_selection == "Auto-Detect (Suggested)"
                    else geometry_selection
                )
                options, tolerance_mapping = _pill_configuration(geometry)
            else:
                options, tolerance_mapping = _pill_configuration("datum")
            selected_pills = st.pills(
                "Selected Controls:",
                options=options,
                default=options[:1],
                selection_mode="multi",
            )
            selected_tolerances = [
                tolerance_mapping[label] for label in (selected_pills or [])
            ]
            limit_column, extent_column = st.columns(2)
            with limit_column:
                tol_limit = st.number_input(
                    "Tolerance Limit (t) [mm]",
                    value=0.0500,
                    step=0.005,
                    format="%.4f",
                )
            with extent_column:
                if mode == "Datum-Referenced (Orientation)":
                    ref_extent = st.number_input(
                        "Reference Extent (L) [mm]", value=100.0, step=10.0
                    )
                else:
                    ref_extent = None

    try:
        if mode == "Single Feature (Form)":
            metrics = _inspection_metrics(points, geometry)
            rows = _comparison_rows(
                geometry, metrics, selected_tolerances, tol_limit
            )
            figure, _ = _inspection_figure(
                points, geometry, point_size, plane_opacity, theme=viewport_theme
            )
        else:
            points = feature
            primary_relation = (
                selected_tolerances[0]
                if selected_tolerances
                else "Perpendicularity (⊥)"
            )
            metrics = _datum_metrics(
                datum, points, primary_relation, ref_extent, tol_limit
            )
            relation_metrics = [
                _datum_metrics(datum, points, relation_name, ref_extent, tol_limit)
                for relation_name in selected_tolerances
            ]
            rows = [
                row
                for relation_name, result in zip(
                    selected_tolerances, relation_metrics
                )
                for row in _comparison_rows(
                    relation_name, result, [relation_name], tol_limit
                )
            ]
            metrics["evaluated_characteristics"] = selected_tolerances
            metrics["relation_results"] = relation_metrics
            if relation_metrics:
                metrics["measured_deviation"] = max(
                    result["measured_deviation"] for result in relation_metrics
                )
                metrics["verdict"] = all(
                    result["verdict"] for result in relation_metrics
                )
            figure = _datum_figure(datum, points, metrics, viewport_theme)
    except ValueError as exc:
        st.error(str(exc))
        return

    with results:
        if rows and all(row["Status"] == "PASS" for row in rows):
            st.success("STATUS: PASS")
        elif rows:
            st.error("STATUS: FAIL")
        else:
            st.warning("Select at least one tolerance to evaluate conformance.")
        if mode == "Single Feature (Form)":
            metrics = _render_result_panel(
                st, points, geometry, tol_limit, selected_tolerances, rows,
                interface_theme,
            )
            metrics["evaluated_characteristics"] = selected_tolerances
        else:
            if rows:
                _comparison_dataframe(st, rows, interface_theme)
            st.write(
                f"Measured Angle: {metrics['measured_angle_deg']:.2f}°, "
                f"Nominal Angle: {metrics['nominal_angle_deg']:.2f}°, "
                f"Angular Error: |{metrics['measured_angle_deg']:.2f} - "
                f"{metrics['nominal_angle_deg']:.2f}| = "
                f"{metrics['angular_error_deg']:.2f}°"
            )
            st.write(f"Evaluated over Extent L = {ref_extent:.2f} mm")
            st.metric(
                "Linear deviation", _format_deviation(metrics["measured_deviation"])
            )
            st.write(f"**Datum vector:** {metrics['datum_axis'].round(4).tolist()}")
            st.write(f"**Feature vector:** {metrics['feature_axis'].round(4).tolist()}")
            for relation_result in metrics["relation_results"]:
                st.caption(
                    f"{relation_result['relation']}: measured "
                    f"{relation_result['measured_angle_deg']:.2f}°, nominal "
                    f"{relation_result['nominal_angle_deg']:.2f}°, angular "
                    f"error {relation_result['angular_error_deg']:.2f}°"
                )
        target = geometry if mode == "Single Feature (Form)" else "Datum-Referenced"
        col_csv, col_html, col_json = st.columns(3)
        with col_csv:
            st.download_button(
            "Download Inspection Report (CSV)",
            _report_csv(points, metrics, mode, target, tol_limit),
            "inspection_report.csv",
            "text/csv",
            use_container_width=True,
        )
        with col_html:
            st.download_button(
            "Download Interactive 3D Model (HTML)",
            figure.to_html(include_plotlyjs="cdn"),
            "inspection_model.html",
            "text/html",
            use_container_width=True,
        )
        with col_json:
            st.download_button(
            "Download Summary (JSON)",
            _summary_json(metrics, target),
            "inspection_summary.json",
            "application/json",
            use_container_width=True,
        )
    with viewport:
        with st.container(key="viewport_panel"):
            st.markdown("### 3D Viewport")
            st.plotly_chart(
                figure,
                use_container_width=True,
                theme=None,
                config={
                    "displayModeBar": False,
                    "scrollZoom": True,
                    "staticPlot": False,
                },
            )


if __name__ == "__main__":
    main()
