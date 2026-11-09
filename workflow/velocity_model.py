"""The realisation's NZCVM velocity model, as layers and as a point query.

`create-nzvm-input` hands the realisation's layers to `nzcvm generate`, which samples
them onto a simulator's grid. Other steps need the same model somewhere else, such as
on a fault, and :func:`point_query` builds the same layer pipeline and reads it at any
set of points instead.
"""

from collections.abc import Callable

import numpy as np
import pyproj
import shapely
from nzcvm.config.grids.model import Projection
from nzcvm.config.layers import LayerConfig
from nzcvm.grids.points import points_grid
from nzcvm.layers.pipeline import build_pipeline
from nzcvm.models.surface import Surface

from workflow.realisations import NZCVMSettings

NZTM_EPSG = 2193
"""The projected CRS (NZTM2000) the workflow places NZCVM grids in."""

type PointQuery = Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, np.ndarray]]
"""Given NZTM x (easting) and y (northing) and elevation z, all in metres with z
positive down, each component (`vs`, `rho`, ...) in NZCVM's units, shaped like x."""


def layers(
    settings: NZCVMSettings, exclude: tuple[type[LayerConfig], ...] = ()
) -> list[LayerConfig]:
    """The realisation's layers, less any of the excluded types.

    Parameters
    ----------
    settings : NZCVMSettings
        The realisation's NZCVM settings.
    exclude : tuple of type
        Layer config types to leave out.

    Returns
    -------
    list of LayerConfig
        The layers, top of the pipeline first.

    Raises
    ------
    ValueError
        If no layers are left.
    """
    kept = [layer for layer in settings.layers if not isinstance(layer, exclude)]
    if not kept:
        raise ValueError("NZCVM requires at least one defined layer.")
    return kept


def point_query(
    settings: NZCVMSettings,
    geometry: shapely.Geometry,
    exclude: tuple[type[LayerConfig], ...] = (),
    below_ground: bool = False,
) -> PointQuery:
    """Build the realisation's layer pipeline once, to read at arbitrary points.

    Each point is a column of one sample (`nzcvm.grids.points.points_grid`), so a
    layer that reads a node's vertical neighbours (Backus averaging) sees none:
    exclude it. A point above the ground reads the ground beneath it.

    Parameters
    ----------
    settings : NZCVMSettings
        The realisation's NZCVM settings.
    geometry : shapely.Geometry
        NZTM outline every queried point lies in, which picks the models to load.
    exclude : tuple of type
        Layer config types to leave out of the pipeline.
    below_ground : bool
        Read the query's z as depth below the topography rather than as elevation.
        SW4 places a rupture's points by their depth below the topography, so a
        rupture's rock is read this way.

    Returns
    -------
    PointQuery
        The query.
    """
    pipeline = build_pipeline(geometry, layers(settings, exclude))
    surface = Surface.load(settings.surface)
    projection = Projection(crs=pyproj.CRS(NZTM_EPSG))

    def query(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
        shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))
        if below_ground:
            # Elevations are positive down, so the ground's elevation plus a depth
            # below it is the point's elevation.
            x, y = np.broadcast_to(x, shape), np.broadcast_to(y, shape)
            ground = surface.transform(
                x.ravel().astype(np.float64),
                y.ravel().astype(np.float64),
                extrapolate=True,
            )
            z = np.asarray(z) + ground.reshape(shape)
        qualities = pipeline(points_grid(x, y, z, surface, projection))
        return {
            str(name): qualities[name].values.reshape(shape)
            for name in qualities.data_vars
        }

    return query
