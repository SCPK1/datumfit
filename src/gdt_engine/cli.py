"""Command-line interface for geometric inspection."""

from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from gdt_engine.fitting.plane import fit_plane
from gdt_engine.fitting.sphere import fit_sphere
from gdt_engine.io import load_point_cloud


@click.group()
def cli() -> None:
    """Inspect GD&T point-cloud measurements."""


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--type", "fit_type", type=click.Choice(("plane", "sphere")), required=True
)
def inspect(path: Path, fit_type: str) -> None:
    """Fit a PLANE or SPHERE to a point-cloud file."""
    try:
        suffix = Path(path).suffix.lower()
        if suffix not in {".xyz", ".csv", ".txt"}:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                "Supported extensions: .xyz, .csv, .txt"
            )
        points = load_point_cloud(path)
        table = Table(title=f"{fit_type.title()} inspection")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        if fit_type == "plane":
            result = fit_plane(points)
            table.add_row("Centroid", _format_vector(result.centroid))
            table.add_row("Normal", _format_vector(result.normal))
            table.add_row("PtV", f"{result.ptv:.6g}")
        else:
            result = fit_sphere(points)
            table.add_row("Center", _format_vector(result.center))
            table.add_row("Radius", f"{result.radius:.6g}")
            table.add_row(
                "Max radial error", f"{np.max(np.abs(result.radial_residuals)):.6g}"
            )
        Console(file=click.get_text_stream("stdout"), force_terminal=False).print(table)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _format_vector(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in vector) + "]"


if __name__ == "__main__":
    cli()
