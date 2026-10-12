from pathlib import Path


def test_nist_sample_files_are_present_and_readable() -> None:
    data_dir = Path(__file__).parents[1] / "data" / "nist"
    expected_files = (
        "plane_sample.xyz",
        "plane_sample_ref.json",
        "sphere_sample.xyz",
        "sphere_sample_ref.json",
    )

    for filename in expected_files:
        path = data_dir / filename
        assert path.is_file()
        assert path.read_text(encoding="utf-8").strip()
