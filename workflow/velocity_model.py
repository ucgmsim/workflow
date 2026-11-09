"""The realisation's NZCVM velocity model, as layers and as a point query.

`create-nzvm-input` hands the realisation's layers to `nzcvm generate`, which samples
them onto a simulator's grid. Other steps need the same model somewhere else, such as
on a fault, and :func:`point_query` builds the same layer pipeline and reads it at any
set of points instead.
"""

from collections.abc import Callable

import numpy as np
import shapely
from nzcvm.config.layers import LayerConfig
from nzcvm.grids.grid import Grid
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
) -> PointQuery:
    """Build the realisation's layer pipeline once, to read at arbitrary points.

    Each point is a column of one sample, so a layer that reads a node's vertical
    neighbours (Backus averaging) sees none: exclude it.

    Parameters
    ----------
    settings : NZCVMSettings
        The realisation's NZCVM settings.
    geometry : shapely.Geometry
        NZTM outline every queried point lies in, which picks the models to load.
    exclude : tuple of type
        Layer config types to leave out of the pipeline.

    Returns
    -------
    PointQuery
        The query.
    """
    pipeline = build_pipeline(geometry, layers(settings, exclude))
    surface = Surface.load(settings.surface)

    def query(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
        shape = np.shape(x)
        x, y, z = (np.asarray(v, dtype=np.float32).ravel() for v in (x, y, z))
        # Elevations are positive down, so depth below the topography is z less
        # the surface's elevation. A point above the ground reads the ground.
        depth = np.maximum(z - surface.transform(x, y, extrapolate=True), 0.0)
        grid = Grid(
            {
                name: (("i", "j", "k"), values.reshape(-1, 1, 1))
                for name, values in (("x", x), ("y", y), ("z", z), ("depth", depth))
            },
            # The Ely taper reads, then drops, each column's top level by `k`.
            coords={"k": [0]},
        )
        qualities = pipeline(grid)
        return {
            str(name): qualities[name].values.reshape(shape)
            for name in qualities.data_vars
        }

    return query
