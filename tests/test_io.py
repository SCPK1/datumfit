from pathlib import Path

import numpy as np
import pytest

from gdt_engine.io import load_point_cloud


@pytest.fixture
def temp_data_dir() -> Path:
    path = Path(".test-data")
    path.mkdir(exist_ok=True)
    return path


def test_load_xyz_returns_float64_nx3() -> None:
    points = load_point_cloud(Path("data/nist/plane_sample.xyz"))
    assert points.shape == (7, 3)
    assert points.dtype == np.float64


@pytest.mark.parametrize("separator", [",", ";", " "])
def test_load_csv_autodetects_separator(temp_data_dir: Path, separator: str) -> None:
    path = temp_data_dir / "points.csv"
    path.write_text(
        "\n".join(
            separator.join(row)
            for row in (("1", "2", "3"), ("4", "5", "6"), ("7", "8", "9"))
        ),
        encoding="utf-8",
    )
    points = load_point_cloud(path)
    np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.mark.parametrize(
    "content, message",
    [("1 2 3\n1 nan 3\n1 2 3", "finite"), ("1 2 3\n1 2 3", "at least 3")],
)
def test_load_point_cloud_validates_input(
    temp_data_dir: Path, content: str, message: str
) -> None:
    path = temp_data_dir / "invalid.xyz"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_point_cloud(path)


def test_load_point_cloud_rejects_wrong_number_of_columns(temp_data_dir: Path) -> None:
    path = temp_data_dir / "bad.xyz"
    path.write_text("1 2 3\n4 5 6 7\n8 9 10", encoding="utf-8")
    with pytest.raises(ValueError, match="three columns"):
        load_point_cloud(path)
