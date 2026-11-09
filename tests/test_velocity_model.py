"""Tests for `workflow.velocity_model`, the realisation's NZCVM layers as a query."""

import dataclasses

import numpy as np
import pyproj
import pytest
import shapely
from nzcvm.config.layers.backus import BackusAveragedLayerConfig

from workflow import velocity_model
from workflow.realisations import NZCVMSettings

# Inland in NZCVM's synthetic world, west of its coastline.
EASTING, NORTHING = pyproj.Transformer.from_crs(
    4326, velocity_model.NZTM_EPSG, always_xy=True
).transform(172.2, -43.6)


@pytest.fixture(scope="module")
def query(nzcvm_settings: NZCVMSettings) -> velocity_model.PointQuery:
    """The default layers, less Backus averaging, around the inland point."""
    return velocity_model.point_query(
        nzcvm_settings,
        shapely.Point(EASTING, NORTHING).buffer(5000.0),
        exclude=(BackusAveragedLayerConfig,),
    )


def test_layers_exclude_by_type(nzcvm_settings: NZCVMSettings) -> None:
    """Excluding a layer type drops it and keeps the rest in order."""
    kept = velocity_model.layers(nzcvm_settings, (BackusAveragedLayerConfig,))
    assert kept == [
        layer
        for layer in nzcvm_settings.layers
        if not isinstance(layer, BackusAveragedLayerConfig)
    ]
    assert len(kept) == len(nzcvm_settings.layers) - 1


def test_layers_refuse_an_empty_pipeline(nzcvm_settings: NZCVMSettings) -> None:
    """A pipeline with no layers left is refused."""
    with pytest.raises(ValueError, match="at least one"):
        velocity_model.layers(dataclasses.replace(nzcvm_settings, layers=[]))


def test_query_keeps_the_points_shape(query: velocity_model.PointQuery) -> None:
    """Every component comes back shaped like the points."""
    shape = (3, 4)
    qualities = query(
        np.full(shape, EASTING),
        np.full(shape, NORTHING),
        np.linspace(0.0, 20000.0, 12).reshape(shape),
    )
    assert {"vs", "rho"} <= set(qualities)
    assert all(values.shape == shape for values in qualities.values())


def test_query_stiffens_with_depth(query: velocity_model.PointQuery) -> None:
    """Down a column, the synthetic tomography gets faster and denser, and the
    clamp's 500 m/s floor holds at the surface."""
    depths = np.linspace(2000.0, 20000.0, 10)
    qualities = query(
        np.full_like(depths, EASTING), np.full_like(depths, NORTHING), depths
    )
    assert np.all(np.diff(qualities["vs"]) > 0.0)
    assert np.all(np.diff(qualities["rho"]) > 0.0)
    surface = query(np.array([EASTING]), np.array([NORTHING]), np.array([-5000.0]))
    assert surface["vs"][0] >= 500.0


def test_query_above_ground_reads_the_ground(
    query: velocity_model.PointQuery,
) -> None:
    """A point in the air reads the rock at the surface beneath it."""
    elevations = np.array([-5000.0, -4000.0])
    qualities = query(np.full(2, EASTING), np.full(2, NORTHING), elevations)
    assert qualities["vs"][0] == qualities["vs"][1]
