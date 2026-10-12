from pathlib import Path

from click.testing import CliRunner

from gdt_engine.cli import cli


def test_inspect_plane_renders_report_table() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["inspect", "data/nist/plane_sample.xyz", "--type", "plane"]
    )
    assert result.exit_code == 0, result.output
    assert "Plane inspection" in result.output
    assert "Centroid" in result.output
    assert "PtV" in result.output
    assert "0.002" in result.output


def test_inspect_sphere_renders_report_table() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["inspect", "data/nist/sphere_sample.xyz", "--type", "sphere"]
    )
    assert result.exit_code == 0, result.output
    assert "Sphere inspection" in result.output
    assert "Radius" in result.output
    assert "50" in result.output


def test_inspect_rejects_unknown_type() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["inspect", "data/nist/plane_sample.xyz", "--type", "line"]
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_inspect_rejects_unsupported_file_extension(tmp_path: Path) -> None:
    sample = tmp_path / "sample.dat"
    sample.write_text("1 2 3\n4 5 6\n7 8 9\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["inspect", str(sample), "--type", "plane"]
    )

    assert result.exit_code != 0
    assert "Unsupported file format: .dat" in result.output
    assert "Supported extensions: .xyz, .csv, .txt" in result.output
