import hashlib
import io
import os
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
import shapely
from hypothesis import assume, given
from hypothesis import strategies as st

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


@given(
    value=st.text(min_size=0, max_size=64, alphabet=st.characters(codec="ascii")),
    size=st.sampled_from([16, 32, 64]),
)
def test_stable_hash_bounds(value: str, size: int) -> None:
    """Check that stable_hash output is always a valid ``size``-byte integer"""
    assert (
        -(1 << (size - 1))
        <= utils.stable_hash(value, size=size // 8)
        <= (1 << (size - 1)) - 1
    )


@given(
    value=st.text(min_size=0, max_size=64, alphabet=st.characters(codec="ascii")),
    size=st.sampled_from([16, 32, 64]),
)
def test_stable_hash_determinism(value: str, size: int) -> None:
    """Check that stable_hash output is deterministic"""
    hash_a = utils.stable_hash(value, size=size // 8)
    hash_b = utils.stable_hash(value, size=size // 8)
    assert hash_a == hash_b


@given(
    value_a=st.text(min_size=0, max_size=64, alphabet=st.characters(codec="ascii")),
    value_b=st.text(min_size=1, max_size=64, alphabet=st.characters(codec="ascii")),
    size=st.sampled_from([16, 32, 64]),
)
def test_stable_hash_collision(value_a: str, value_b: str, size: int) -> None:
    """Check that stable_hash output lacks collision"""
    # Combine value with
    assume(value_a != value_b)
    hash_a = utils.stable_hash(value_a, size=size // 8)
    hash_b = utils.stable_hash(value_b, size=size // 8)
    assert hash_a != hash_b


def _bumped_rectangle(
    x: float, y: float, width: float, height: float, bump: float
) -> shapely.LinearRing:
    """A rectangle whose top edge has a vertex pushed out by ``bump`` metres."""
    return shapely.LinearRing(
        [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x + width / 2, y + height + bump),
            (x, y + height),
        ]
    )


@pytest.mark.parametrize("bumped_island", ["south", "north"])
def test_get_nz_outline_polygon_simplify_tolerance_is_100m(bumped_island: str) -> None:
    """Each island is simplified with a 100 m tolerance: a 100.5 m feature survives.

    A tolerance of even 101 m would remove the bump and shrink the outline by
    at least half a square kilometre.
    """
    x = 5180826
    y = 1567688
    # The larger polygon is treated as the South Island.
    south = _bumped_rectangle(
        x, y, 20000, 10000, 100.5 if bumped_island == "south" else 0
    )
    north = _bumped_rectangle(
        x + 50000, y, 10000, 10000, 100.5 if bumped_island == "north" else 0
    )
    # A third, smaller island that must be dropped.
    tiny = shapely.box(x + 100000, y, x + 100500, y + 500).exterior

    dummy_gdf = gpd.GeoDataFrame(geometry=[tiny, north, south], crs="2193").to_crs(
        "4326"
    )
    with patch("workflow.utils.read_nz_coastline", return_value=dummy_gdf):
        result = utils.get_nz_outline_polygon()

    expected = shapely.union(
        shapely.Polygon(south.coords).simplify(100),
        shapely.Polygon(north.coords).simplify(100),
    )
    expected = shapely.transform(expected, lambda c: c[:, ::-1])
    # The bump alone covers 0.5 * width * 100.5 m^2 (>= 500,000 m^2).
    assert shapely.area(shapely.symmetric_difference(result, expected)) < 10
    assert shapely.area(result) == pytest.approx(shapely.area(expected), abs=10)


def test_get_available_cores_slurm_cpus_on_node_takes_priority() -> None:
    """SLURM_CPUS_ON_NODE wins over SLURM_NPROCS and the CPU affinity."""
    with (
        patch.dict(os.environ, {"SLURM_CPUS_ON_NODE": "3", "SLURM_NPROCS": "17"}),
        patch("psutil.Process.cpu_affinity", return_value=list(range(5))),
    ):
        assert utils.get_available_cores() == 3


def test_get_available_cores_slurm_nprocs_over_affinity() -> None:
    """SLURM_NPROCS is used ahead of the CPU affinity when present alone."""
    env = {k: v for k, v in os.environ.items() if k != "SLURM_CPUS_ON_NODE"}
    env["SLURM_NPROCS"] = "17"
    with (
        patch.dict(os.environ, env, clear=True),
        patch("psutil.Process.cpu_affinity", return_value=list(range(5))),
    ):
        assert utils.get_available_cores() == 17


def test_raises_exact_message_when_no_cpu_info() -> None:
    mock_process = Mock()
    del mock_process.cpu_affinity
    slurm_vars = ("SLURM_CPUS_ON_NODE", "SLURM_NPROCS")
    env = {k: v for k, v in os.environ.items() if k not in slurm_vars}
    with (
        patch.dict(os.environ, env, clear=True),
        patch("psutil.Process", return_value=mock_process),
        patch("psutil.cpu_count", return_value=None),
        pytest.raises(RuntimeError, match=r"^Cannot determine CPU count\.$"),
    ):
        utils.get_available_cores()


def test_dict_zip_is_strict_by_default() -> None:
    """Without an explicit ``strict`` argument mismatched keys are an error."""
    with pytest.raises(
        ValueError, match=r"^Keys in dictionaries are not all the same\.$"
    ):
        utils.dict_zip({"a": 1, "b": 2}, {"a": 3})


def test_dict_zip_non_strict_disjoint_keys_is_empty() -> None:
    assert utils.dict_zip({"a": 1}, {"b": 2}, {"a": 3}, strict=False) == {}


def test_merge_dictionaries_deep_merges_in_place() -> None:
    dict_a = {"x": 1, "nested": {"keep": 1, "override": 2}, "replaced": {"a": 1}}
    dict_b = {"nested": {"override": 20, "new": 30}, "replaced": 5, "y": 2}
    original_b = {"nested": {"override": 20, "new": 30}, "replaced": 5, "y": 2}

    assert utils.merge_dictionaries(dict_a, dict_b) is None
    assert dict_a == {
        "x": 1,
        "nested": {"keep": 1, "override": 20, "new": 30},
        "replaced": 5,
        "y": 2,
    }
    assert dict_b == original_b


@given(value=st.text(max_size=64))
def test_stable_hash_default_size_is_four_bytes(value: str) -> None:
    """The default digest size is 4 bytes (a signed 32-bit integer)."""
    result = utils.stable_hash(value)
    assert result == utils.stable_hash(value, size=4)
    assert -(1 << 31) <= result < (1 << 31)


@pytest.mark.parametrize("value", ["", "realisation", "Ōtautahi"])
@pytest.mark.parametrize("size", [1, 4, 8])
def test_stable_hash_matches_signed_blake2b_of_utf8(value: str, size: int) -> None:
    """The hash is the signed big-endian integer of the UTF-8 BLAKE2b digest."""
    expected = int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=size).digest(),
        byteorder="big",
        signed=True,
    )
    assert utils.stable_hash(value, size=size) == expected


def test_stable_hash_known_values() -> None:
    """Pin exact outputs so hashes stay stable across releases."""
    assert utils.stable_hash("") == 309448485
    assert utils.stable_hash("realisation") == 1579675602


GMT_TRIANGLE = b"""# @VGMT1.0 @GPOLYGON
# @R170/171/-42/-41
# FEATURE_DATA
>
170 -42
171 -42
171 -41
170 -42
"""


@pytest.mark.xfail(
    strict=True,
    reason="read_nz_coastline reads the temp file before flushing it (#251)",
)
def test_read_nz_coastline_parses_downloaded_gmt() -> None:
    """The downloaded GMT bytes are parsed into a GeoDataFrame of polygons."""
    source = io.BytesIO(GMT_TRIANGLE)
    with patch("urllib.request.urlopen", return_value=source) as urlopen:
        gdf = utils.read_nz_coastline()
    urlopen.assert_called_once_with(utils.NZ_COASTLINE_URL)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].equals(
        shapely.Polygon([(170, -42), (171, -42), (171, -41)])
    )
