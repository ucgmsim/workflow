import os
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
import shapely

from workflow import utils


def test_get_available_cores_slurm_cpus_on_node() -> None:
    with patch.dict(os.environ, {"SLURM_CPUS_ON_NODE": "4"}):
        assert utils.get_available_cores() == 4


def test_get_available_cores_slurm_nprocs() -> None:
    with patch.dict(os.environ, {"SLURM_NPROCS": "8"}):
        assert utils.get_available_cores() == 8


def test_uses_cpu_affinity() -> None:
    with patch("psutil.Process.cpu_affinity", return_value=[1, 2]):
        assert utils.get_available_cores() == 2


def test_uses_cpu_count() -> None:
    mock_process = Mock()
    del mock_process.cpu_affinity
    with (
        patch("psutil.Process", return_value=mock_process),
        patch("psutil.cpu_count", return_value=12),
    ):
        result = utils.get_available_cores()
    assert result == 12


def test_raises_when_no_cpu_info() -> None:
    mock_process = Mock()
    del mock_process.cpu_affinity
    with (
        patch("psutil.Process", return_value=mock_process),
        patch("psutil.cpu_count", return_value=None),
        pytest.raises(RuntimeError, match="Cannot determine CPU count"),
    ):
        utils.get_available_cores()


@pytest.mark.slow
def test_read_nz_coastline() -> None:
    gdf = utils.read_nz_coastline()
    assert isinstance(gdf, gpd.GeoDataFrame)


def make_square(x: float, y: float, size: float) -> shapely.Polygon:
    """Helper to create a square polygon with bottom-left corner at (x,y)."""
    return shapely.Polygon(
        [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]
    )


def test_get_nz_outline_polygon_selects_two_largest() -> None:
    # Create dummy polygon outlines with different sizes
    x = 5180826
    y = 1567688
    x2 = x + 1000
    x3 = x + 2000
    line1_nztm = shapely.box(x, y, x + 100, y + 100).exterior
    line2_nztm = shapely.box(x2, y, x2 + 200, y + 100).exterior
    line3_nztm = shapely.box(x3, y, x3 + 400, y + 100).exterior

    outlines_nztm = [line1_nztm, line2_nztm, line3_nztm]
    dummy_gdf = gpd.GeoDataFrame(geometry=outlines_nztm, crs="2193")
    dummy_gdf = dummy_gdf.to_crs("4326")

    with patch("workflow.utils.read_nz_coastline", return_value=dummy_gdf):
        result = utils.get_nz_outline_polygon()

    expected_union = shapely.union(
        shapely.Polygon(line2_nztm.coords), shapely.Polygon(line3_nztm.coords)
    ).simplify(100)
    expected_union = shapely.transform(expected_union, lambda x: x[:, ::-1])
    # Check the result is equal to the union of the two largest polygons
    assert shapely.area(shapely.symmetric_difference(result, expected_union)) < 1e-4


def test_dict_zip_basic_two_dicts() -> None:
    """Test standard zipping of two dictionaries with matching keys."""
    d1 = {"a": 1, "b": 2}
    d2 = {"a": "apple", "b": "banana"}

    expected = {"a": (1, "apple"), "b": (2, "banana")}
    assert utils.dict_zip(d1, d2) == expected


def test_dict_zip_three_dicts() -> None:
    """Test standard zipping of three dictionaries."""
    d1 = {"a": 1}
    d2 = {"a": 2}
    d3 = {"a": 3}

    assert utils.dict_zip(d1, d2, d3) == {"a": (1, 2, 3)}


def test_dict_zip_strict_mismatch_raises_error() -> None:
    """Test that strict=True raises ValueError when keys don't match exactly."""
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1}  # Missing 'b'

    with pytest.raises(ValueError, match="Keys in dictionaries are not all the same"):
        utils.dict_zip(d1, d2, strict=True)


def test_dict_zip_non_strict_intersection() -> None:
    """Test that strict=False returns the intersection of keys."""
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"a": 10, "b": 20, "d": 40}

    # Only 'a' and 'b' are in both
    result = utils.dict_zip(d1, d2, strict=False)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"] == (1, 10)
    assert result["b"] == (2, 20)


def test_dict_zip_empty_input() -> None:
    """Test behaviour with no dictionaries provided."""
    assert utils.dict_zip() == {}


def test_dict_zip_single_dict() -> None:
    """Test behaviour with a single dictionary."""
    d1 = {"a": 1, "b": 2}
    assert utils.dict_zip(d1) == {"a": (1,), "b": (2,)}


def test_dict_zip_identical_keys_different_order() -> None:
    """Test that key order in input doesn't cause strict mode to fail."""
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 20, "a": 10}

    # Should not raise ValueError even though insertion order differs
    result = utils.dict_zip(d1, d2, strict=True)
    assert result["a"] == (1, 10)
    assert result["b"] == (2, 20)
