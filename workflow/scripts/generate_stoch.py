"""Stoch Generation.

Description
-----------
Generate the stoch slip model for HF simulation. This is a down-sampled version of the SRF on
the uniform subfault grid the HF code simulates, with a stress parameter for every subfault.

Inputs
------
A realisation file containing a metadata configuration, and a generated SRF file.

Outputs
-------
An HDF5 file, written by xarray as a `DataTree`, holding the down-sampled SRF. The root group
carries the subfault size shared by every plane (`dx` and `dy`, km) and has one child group
per plane, `plane_0`, `plane_1`, ..., in SRF order. Each plane group has the plane geometry
as attributes (`longitude`, `latitude`, `strike`, `dip`, `rake`, `dtop`, `shypo`, `dhypo`)
and four `(dip, strike)` grids: `slip` (cm), `rise_time` (s), `rupture_time` (s) and
`stress_drop` (bars). The `strike` coordinate is the centre of each subfault along strike
from the top-centre of the plane, and `dip` the centre down dip from its top edge (km). Read
it with `xarray.open_datatree`.

Usage
-----
`generate-stoch [OPTIONS] REALISATION_FFP SRF_FFP STOCH_FFP`

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `generate-stoch` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

For More Help
-------------
See the output of `generate-stoch --help` or `workflow.scripts.generate_stoch`.
"""

import math
from pathlib import Path
from typing import Annotated

import numpy as np
import scipy.sparse as sp
import typer
import xarray as xr

from qcore import cli
from source_modelling import srf
from source_modelling.srf import SrfFile
from workflow import log_utils, realisations
from workflow.realisations import HFConfig, RealisationMetadata, StochConfig

app = typer.Typer()


def _box_average_matrix(
    n_fine: int, n_coarse: int, fine_dx: float, coarse_dx: float, *, centred: bool
) -> sp.csr_array[np.float32, tuple[int, int]]:
    """Build an area-pooling kernel for averaging high-resolution data into lower-resolution data.

    Assuming we have `n_fine` fine gridpoints, and `n_coarse` coarse gridpoints,
    Row j of the returned matrix gives the fractional-overlap weights (summing
    to 1) between coarse bin j and the fine cells it spans. This is equivalent
    to the ``adaptive_avg_pool2d`` kernel in pytorch with padding.

    If the coarse grid is longer than the fine grid, the bins that hang off it
    have weights summing to the covered fraction rather than to 1. This is what
    makes the kernel conserve the total (slip * area) rather than the cell
    value. See ``convert_srf_to_stoch`` for how we handle rise and rupture time,
    where this is not what we want.

    Parameters
    ----------
    n_fine : int
        The number of elements in the original, high-resolution grid dimension.
    n_coarse : int
        The number of elements in the target, downsampled grid dimension.
    fine_dx : float
        The physical resolution of the fine cells.
    coarse_dx : float
        The physical resolution of the coarse cells.
    centred : bool
        If True, the coarse grid is centred on the fine grid. Otherwise both
        grids start at the same point and any excess hangs off the far end.

    Returns
    -------
    scipy.sparse.csr_array
        A float32 sparse array of shape (n_coarse, n_fine) containing the area-weighted
        fractional overlap coefficients.

    Notes
    -----
    The weights correspond exactly to the fractional overlap of coarse bins over
    fine bins. This is mathematically equivalent to upsampling both grids to
    their Least Common Multiple (LCM) base units, padding the geometry with
    zeros in these units and computing a standard block average. The special
    case where the fine grid and coarse grid span the same length is implemented
    by srf2stoch.c. The main advantage of our approach is that a sparse matrix
    does not have to materialise all the empty cell overlaps in memory. A
    secondary advantage is that we handle the padded case, which lets us set a
    uniform dx/dy for all SRF segments as the HF code demands without changing
    total moment.

    For example, downsampling 5 fine cells to 3 coarse cells implies an LCM of
    15 base units. The 5 fine cells (A-E) take up 3 units each, while the 3
    coarse cells (C0-C2) take up 5 units each.

    The visual alignment of this overlap is as follows:

    ::

        THE LCM GRID (15 Base Units)
        |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

        FINE INPUT GRID (5 cells, each = 3 base units)
        |-----------|-----------|-----------|-----------|-----------|
        |     A     |     B     |     C     |     D     |     E     |
        |-----------|-----------|-----------|-----------|-----------|

        COARSE OUTPUT GRID (3 cells, each = 5 base units)
        |-------------------|-------------------|-------------------|
        |         C0        |         C1        |         C2        |
        |-------------------|-------------------|-------------------|

    Each row in the returned sparse matrix corresponds to the fractional
    makeup of a single coarse bin:

    * **Row 0 (Coarse Bin 0):** Spans 5 base units. Covers all 3 units of A
      (3/5) and 2 units of B (2/5).
    * **Row 1 (Coarse Bin 1):** Spans 5 base units. Covers the remaining 1
      unit of B (1/5), all 3 units of C (3/5), and 1 unit of D (1/5).
    * **Row 2 (Coarse Bin 2):** Spans 5 base units. Covers the remaining 2
      units of D (2/5) and all 3 units of E (3/5).

    """
    bin_width = coarse_dx / fine_dx
    overhang = (n_coarse * bin_width - n_fine) / 2 if centred else 0.0
    edges = np.arange(n_coarse + 1) * bin_width - overhang
    rows, cols, weights = [], [], []
    for j in range(n_coarse):
        lo, hi = edges[j], edges[j + 1]
        idx = np.arange(max(int(np.floor(lo)), 0), min(int(np.ceil(hi)), n_fine))
        weights.append((np.minimum(idx + 1, hi) - np.maximum(idx, lo)) / bin_width)
        rows.append(np.full(len(idx), j))
        cols.append(idx)
    return sp.csr_array(
        (
            np.concatenate(weights).astype(np.float32),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n_coarse, n_fine),
    )


def _weighted_box_mean(
    values: np.ndarray,
    weights: np.ndarray,
    wy: sp.csr_array,
    wx: sp.csr_array,
    empty: float = np.nan,
) -> np.ndarray:
    """Box-average `values` in proportion to `weights`, rather than by area.

    Dividing by the box-averaged weights cancels the covered fraction that the
    kernels carry at the edge of the plane. Cells with no weight get `empty`.
    """
    total = wy @ weights @ wx.T
    return np.divide(
        wy @ (values * weights) @ wx.T,
        total,
        out=np.full_like(total, empty),
        where=total > 0,
    )


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Take the circular mean of `angles` with respect to `weights`."""

    mean = np.average(
        np.exp(1j * np.radians(np.ravel(angles))),
        weights=np.ravel(weights) if np.any(weights) else None,
    )
    return float(np.angle(mean, deg=True) % 360.0)


def convert_srf_to_stoch(
    srf_file: SrfFile, dx: float, dy: float, stress_drop_bars: float
) -> xr.DataTree:
    """Down-sample an SRF file into a stoch slip model by box-averaging slip, tinit and rise.

    Parameters
    ----------
    srf_file : SrfFile
        The SRF file to convert.
    dx : float
        The desired strike-resolution for the output slip model.
    dy : float
        The desired dip-resolution for the output slip model.
    stress_drop_bars : float
        The stress parameter given to every subfault.

    Returns
    -------
    xr.DataTree
        The slip model downsampled from ``srf_file``, laid out as the module docstring
        describes.
    """
    planes = {}
    for i, segment in enumerate(srf_file.segments):
        header = srf_file.header.iloc[i].astype(np.float32)
        nstk, ndip = int(header["nstk"]), int(header["ndip"])

        slip, rake, rise, tinit = (
            segment[column].to_numpy(dtype=np.float32).reshape(ndip, nstk)
            for column in ("slip", "rake", "rise", "tinit")
        )

        # Read the extent at full precision, so that a dx dividing it exactly
        # gives a whole number of cells.
        length = float(srf_file.header["len"].iloc[i])
        width = float(srf_file.header["wid"].iloc[i])
        nx = _n_cells(length, dx)
        ny = _n_cells(width, dy)
        # The HF code centres the stoch grid along strike on (elon, elat), the
        # top-centre of the plane, but hangs it down-dip from the top edge at
        # dtop (with dhypo measured from that edge). So along strike the
        # padding is split between both ends, but down-dip it all goes at the
        # bottom.
        wx = _box_average_matrix(nstk, nx, length / nstk, dx, centred=True)
        wy = _box_average_matrix(ndip, ny, width / ndip, dy, centred=False)

        # Slip is spread over the cell, so partially covered edge cells keep
        # their lower area average to conserve moment. Rupture time and rise
        # time are not, so they are averaged over the covered part only: by
        # area for rupture time, by slip for rise time. Every cell is at least
        # partially covered (nx and ny round up), so only cells without slip
        # fall back to srf2stoch's nominal rise time.
        slip_grid = wy @ slip @ wx.T
        trup_grid = _weighted_box_mean(tinit, np.ones_like(tinit), wy, wx)
        rise_grid = _weighted_box_mean(rise, slip, wy, wx, empty=1e-5)

        dims = ("dip", "strike")
        # The plane geometry at full precision, which the stoch text format rounded.
        geometry = srf_file.header.iloc[i]
        planes[f"plane_{i}"] = xr.Dataset(
            {
                "slip": (dims, slip_grid, {"units": "cm"}),
                "rise_time": (dims, rise_grid, {"units": "s"}),
                "rupture_time": (dims, trup_grid, {"units": "s"}),
                "stress_drop": (
                    dims,
                    np.full((ny, nx), stress_drop_bars, np.float32),
                    {"units": "bar"},
                ),
            },
            coords={
                # Subfault centres, matching the grid alignment above.
                "strike": (
                    "strike",
                    (np.arange(nx) + 0.5 - nx / 2) * dx,
                    {"units": "km"},
                ),
                "dip": ("dip", (np.arange(ny) + 0.5) * dy, {"units": "km"}),
            },
            attrs={
                "longitude": float(geometry["elon"]),
                "latitude": float(geometry["elat"]),
                "strike": float(geometry["stk"]) % 360.0,
                "dip": float(geometry["dip"]),
                "rake": _circular_mean(rake, slip),
                "dtop": float(geometry["dtop"]),
                "shypo": float(geometry["shyp"]),
                "dhypo": float(geometry["dhyp"]),
            },
        )
    return xr.DataTree.from_dict(
        {"/": xr.Dataset(attrs={"dx": dx, "dy": dy}), **planes}
    )


_CELL_TOLERANCE = 1e-9


def _n_cells(extent: float, resolution: float) -> int:
    """Count the cells of size `resolution` needed to cover `extent`.

    A cell that would cover less than a billionth of `resolution` is float
    error from dividing the extent exactly, and is not counted.
    """
    return max(1, math.ceil(extent / resolution - _CELL_TOLERANCE))


def stoch_resolution(
    extents: np.ndarray,
    srf_resolutions: np.ndarray,
    target_resolution: float,
    padding_weight: float,
    min_resolution: float,
    max_resolution: float | None,
) -> float:
    """Choose the cell size shared by a set of planes along one axis.

    Every plane of a stoch file shares one cell size, so planes whose extent is
    not a multiple of it are padded out by part of a cell. This trades the
    padding against drift from a target cell size, both counted in cells, and
    picks the cell size within the bounds that minimises

        drift + padding_weight * padding.

    A weight of 0 gives the target, however much the planes are padded. For a
    single plane, any weight above 1 gives srf2stoch's ``target_dx``
    behaviour: the whole number of cells closest to the target, with no
    padding. The cell size is never finer than the coarsest SRF resolution
    (srf2stoch does the same), unless `max_resolution` forces it.

    Parameters
    ----------
    extents : np.ndarray
        The extent of each plane along the axis (length or width, km).
    srf_resolutions : np.ndarray
        The SRF resolution of each plane along the axis (km).
    target_resolution : float
        The preferred cell size (km).
    padding_weight : float
        The cost of one padded cell, in cells of drift from the target.
    min_resolution : float
        The smallest allowed cell size (km).
    max_resolution : float or None
        The largest allowed cell size (km), or None for no limit.

    Returns
    -------
    float
        The cell size within the bounds with the least cost. Among equals,
        the one closest to the target, then the finest.

    Notes
    -----
    For cell size d and target t, the padding and drift are

        padding(d) = sum_i (ceil(L_i / d) - L_i / d),
        drift(d)   = sum_i L_i * |1 / d - 1 / t|,

    that is, the cells hanging off the planes, and how many more or fewer
    cells the planes have than at the target. Measuring drift in cells, rather
    than in km, is what makes a single plane round its cell count to the
    nearest whole number, as srf2stoch does.

    In terms of u = 1 / d, both are linear wherever the ceilings are constant,
    apart from a kink at u = 1 / t. The cost is therefore smallest at a size
    d = L_i / n where some plane fits exactly into n cells, at the target, or
    at a bound, and we only have to check those.
    """
    upper = math.inf if max_resolution is None else max_resolution
    lower = min(max(min_resolution, float(srf_resolutions.max())), upper)
    candidates = [lower, min(max(target_resolution, lower), upper)]
    if math.isfinite(upper):
        candidates.append(upper)
    for extent in extents:
        # Plane fits exactly into n cells of extent / n, for each n in bounds.
        fewest = max(1, math.ceil(extent / upper - _CELL_TOLERANCE))
        most = math.floor(extent / lower + _CELL_TOLERANCE)
        candidates.extend(extent / n for n in range(fewest, most + 1))
    candidates = np.array(candidates)

    cells = extents / candidates[:, np.newaxis]
    n_cells = np.maximum(1, np.ceil(cells - _CELL_TOLERANCE))
    padding = (n_cells - cells).sum(axis=1)
    drift = np.abs(cells - extents / target_resolution).sum(axis=1)
    cost = drift + padding_weight * padding
    # Among equal costs, prefer the least drift from the target, then the finest.
    tolerance = _CELL_TOLERANCE * len(extents)
    cheapest = cost <= cost.min() + tolerance
    closest = cheapest & (drift <= drift[cheapest].min() + tolerance)
    return float(candidates[closest].min())


@cli.from_docstring(app)
@log_utils.log_call()
def generate_stoch(
    realisation_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    srf_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    stoch_ffp: Annotated[Path, typer.Argument(dir_okay=False)],
) -> None:
    """Generate a stoch slip model from an SRF file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation.
    srf_ffp : Path
        Path to the SRF file which is used as input for the slip model generation.
    stoch_ffp : Path
        Path to the output HDF5 file where the slip model will be saved.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    stoch_config = StochConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    hf_config = HFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    srf_file = srf.read_srf(srf_ffp)
    lengths = srf_file.header["len"].to_numpy(dtype=np.float64)
    widths = srf_file.header["wid"].to_numpy(dtype=np.float64)
    dx = stoch_resolution(
        lengths,
        lengths / srf_file.header["nstk"].to_numpy(),
        stoch_config.stoch_target_dx,
        stoch_config.stoch_padding_weight,
        stoch_config.stoch_min_dx,
        stoch_config.stoch_max_dx,
    )
    dy = stoch_resolution(
        widths,
        widths / srf_file.header["ndip"].to_numpy(),
        stoch_config.stoch_target_dy,
        stoch_config.stoch_padding_weight,
        stoch_config.stoch_min_dy,
        stoch_config.stoch_max_dy,
    )
    # The realisation has one stress parameter for the whole rupture, for now.
    stoch = convert_srf_to_stoch(srf_file, dx, dy, hf_config.source["stress_drop_bars"])
    stoch.to_netcdf(stoch_ffp, engine="h5netcdf")

    realisations.append_log_entry(realisation_ffp)
