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
    """Excluding a layer type drops it and keeps the rest."""
    kept = velocity_model.layers(nzcvm_settings, (BackusAveragedLayerConfig,))
    assert len(kept) == len(nzcvm_settings.layers) - 1
    assert not any(isinstance(layer, BackusAveragedLayerConfig) for layer in kept)


def test_layers_refuse_an_empty_pipeline(nzcvm_settings: NZCVMSettings) -> None:
    """A pipeline with no layers left is refused."""
    with pytest.raises(ValueError, match="at least one"):
        velocity_model.layers(dataclasses.replace(nzcvm_settings, layers=[]))


def test_query_reads_a_column(query: velocity_model.PointQuery) -> None:
    """Down a column, shaped as a grid, the synthetic tomography gets faster and
    denser. Points in the air read the ground beneath them, where the clamp's
    500 m/s floor holds."""
    elevations = np.array([-5000.0, -4000.0, *np.linspace(2000.0, 20000.0, 10)])
    shape = (3, 4)
    qualities = query(
        np.full(shape, EASTING), np.full(shape, NORTHING), elevations.reshape(shape)
    )
    assert all(values.shape == shape for values in qualities.values())
    vs, rho = qualities["vs"].ravel(), qualities["rho"].ravel()
    assert vs[0] == vs[1] >= 500.0
    assert np.all(np.diff(vs[2:]) > 0.0)
    assert np.all(np.diff(rho[2:]) > 0.0)
