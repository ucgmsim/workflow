"""Tests for the vendored `workflow.bounding_box` module."""

import numpy as np
import pytest
import shapely
from hypothesis import given
from hypothesis import strategies as st

from workflow import bounding_box
from workflow.bounding_box import BoundingBox

CENTROIDS = st.tuples(
    st.floats(min_value=-46.0, max_value=-35.0),
    st.floats(min_value=167.0, max_value=178.0),
)
BEARINGS = st.floats(min_value=0.0, max_value=360.0, exclude_max=True)
EXTENTS = st.floats(min_value=1.0, max_value=500.0)


@given(centroid=CENTROIDS, bearing=BEARINGS, extent_x=EXTENTS, extent_y=EXTENTS)
def test_matches_velocity_modelling(
    centroid: tuple[float, float], bearing: float, extent_x: float, extent_y: float
) -> None:
    """The vendored class should agree exactly with the upstream one."""
    upstream = pytest.importorskip("velocity_modelling.bounding_box")
    ours = BoundingBox.from_centroid_bearing_extents(
        centroid, bearing, extent_x, extent_y
    )
    theirs = upstream.BoundingBox.from_centroid_bearing_extents(
        centroid, bearing, extent_x, extent_y
    )
    np.testing.assert_array_equal(ours.bounds, theirs.bounds)
    np.testing.assert_array_equal(ours.corners, theirs.corners)
    np.testing.assert_array_equal(ours.origin, theirs.origin)
    for attribute in ["extent_x", "extent_y", "bearing", "great_circle_bearing"]:
        assert getattr(ours, attribute) == getattr(theirs, attribute)
    assert ours.area == theirs.area
    assert ours.polygon.equals_exact(theirs.polygon, tolerance=0)
    np.testing.assert_array_equal(
        ours.pad((1, 2), (3, 4)).bounds, theirs.pad((1, 2), (3, 4)).bounds
    )
    np.testing.assert_array_equal(
        BoundingBox.from_wgs84_coordinates(ours.corners).bounds,
        upstream.BoundingBox.from_wgs84_coordinates(theirs.corners).bounds,
    )


def test_from_wgs84_coordinates_round_trip() -> None:
    """Corners written to a realisation should read back to the same box."""
    box = BoundingBox.from_centroid_bearing_extents([-43.5, 172.6], 30.0, 100.0, 80.0)
    round_trip = BoundingBox.from_wgs84_coordinates(box.corners.tolist())
    np.testing.assert_allclose(round_trip.bounds, box.bounds, atol=1e-6)
    assert round_trip.extent_x == pytest.approx(100.0)
    assert round_trip.extent_y == pytest.approx(80.0)
    assert round_trip.area == pytest.approx(8000.0)
    assert round_trip.great_circle_bearing == pytest.approx(30.0, abs=1.0)
    assert "BoundingBox(" in repr(round_trip)


def test_bounding_box_for_geometry() -> None:
    """Oriented and axis-aligned boxes enclose the geometry."""
    geometry = shapely.Polygon([(0, 0), (1000, 1000), (0, 2000), (-1000, 1000)])
    oriented = BoundingBox.bounding_box_for_geometry(geometry)
    assert oriented.area == pytest.approx(2.0)
    axis_aligned = BoundingBox.bounding_box_for_geometry(geometry, axis_aligned=True)
    assert axis_aligned.area == pytest.approx(4.0)
    with pytest.raises(ValueError, match="Ill-defined geometry"):
        BoundingBox.bounding_box_for_geometry(shapely.Point(0, 0))


def test_minimum_area_bounding_box_for_polygons_masked() -> None:
    """The box covers the required polygon and the masked part of the optional one."""
    must_include = shapely.box(0, 0, 1000, 1000)
    may_include = shapely.box(0, 0, 3000, 1000)
    mask = shapely.box(-10000, -10000, 2000, 10000)
    box = bounding_box.minimum_area_bounding_box_for_polygons_masked(
        [must_include], [may_include], mask
    )
    assert isinstance(box, BoundingBox)
    assert box.area == pytest.approx(2.0)
