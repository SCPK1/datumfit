import json

import numpy as np

from gdt_engine.app import (
    _cached_upload,
    _initialize_cloud_cache,
    _jsonable,
)


def test_jsonable_cleans_numpy_values_and_non_string_keys_recursively() -> None:
    payload = {
        True: np.bool_(True),
        7: {
            "deviation": np.float64(0.002),
            "count": np.int64(3),
            "points": np.array([[1.0, 2.0, 3.0]]),
            "flags": (np.bool_(False), True),
        },
    }

    decoded = json.loads(json.dumps(_jsonable(payload)))

    assert decoded["True"] is True
    assert decoded["7"]["deviation"] == 0.002
    assert decoded["7"]["count"] == 3
    assert decoded["7"]["points"] == [[1.0, 2.0, 3.0]]
    assert decoded["7"]["flags"] == [False, True]


def test_cloud_cache_initializes_all_structured_session_entries() -> None:
    session_state = {}

    _initialize_cloud_cache(session_state)

    assert session_state == {
        "data_single": {"points": None, "filename": None},
        "data_datum": {"points": None, "filename": None},
        "data_feature": {"points": None, "filename": None},
    }


def test_cached_upload_parses_and_retains_filename_and_points() -> None:
    class UploadedFile:
        name = "sample.xyz"

        @staticmethod
        def getvalue() -> bytes:
            return b"0 0 0\n1 0 0\n0 1 0\n"

    session_state = {}
    _initialize_cloud_cache(session_state)

    points, filename = _cached_upload(session_state, UploadedFile(), "data_single")
    reused_points, reused_filename = _cached_upload(
        session_state, None, "data_single"
    )

    np.testing.assert_array_equal(points, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(reused_points, points)
    assert filename == reused_filename == "sample.xyz"
