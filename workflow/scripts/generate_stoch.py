"""Stoch Generation.

Description
-----------
Generate Stoch file for HF simulation. This file is just a down-sampled version of the SRF.

Inputs
------
A realisation file containing a metadata configuration, and a generated SRF file.

Outputs
-------
A [Stoch](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-Stochformat) file containing a down-sampled version of the SRF.

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

from pathlib import Path
from typing import Annotated

import numpy as np
import scipy.sparse as sp
import typer

from qcore import cli
from source_modelling import sources, srf
from source_modelling.srf import SrfFile
from source_modelling.stoch import StochFile, StochHeader, StochPlane
from workflow import log_utils, realisations
from workflow.realisations import HFConfig, RealisationMetadata, SourceConfig

app = typer.Typer()


def _box_average_matrix(
    n_fine: int, n_coarse: int, fine_dx: float, coarse_dx: float
) -> sp.csr_matrix:
    """Build an area-pooling kernel for averaging high-resolution data into lower-resolution data.

    Assuming we have `n_fine` fine gridpoints, and `n_coarse` coarse gridpoints,
    Row j of the returned matrix gives the fractional-overlap weights (summing
    to 1) between coarse bin j and the fine cells it spans. This is equivalent
    to the ``adaptive_avg_pool2d`` kernel in pytorch with padding.

    The two grids are centred on each other. If the coarse grid is longer than
    the fine grid, the bins at either end hang off the fine grid and their
    weights sum to the covered fraction rather than to 1. This is what makes
    the kernel conserve the total (slip * area) rather than the cell value. See
    ``convert_srf_to_stoch`` for how we handle trise where this is not what we want.

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

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse matrix of shape (n_coarse, n_fine) containing the area-weighted
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
    total moment. Because the stoch format records only a centre point and an
    ``nx * dx`` extent, the padding has to be centred too, or the slip
    distribution would sit off-centre on the plane the HF code reconstructs.

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
    # The coarse grid may be longer than the fine grid it covers. Split the
    # excess evenly between the two ends so that both grids stay centred on
    # the same point (see the Notes above).
    overhang = (n_coarse * bin_width - n_fine) / 2
    edges = np.arange(n_coarse + 1) * bin_width - overhang
    rows, cols, weights = [], [], []
    for j in range(n_coarse):
        lo, hi = edges[j], edges[j + 1]
        idx = np.arange(max(int(np.floor(lo)), 0), min(int(np.ceil(hi)), n_fine))
        weights.append((np.minimum(idx + 1, hi) - np.maximum(idx, lo)) / bin_width)
        rows.append(np.full(len(idx), j))
        cols.append(idx)
    return sp.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_coarse, n_fine),
    )


def circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Take the circular mean of `angles` with respect to `weights`.

    Parameters
    ----------
    angles : array of floats
        The angles to average, in degrees.
    weights : array of floats
        The weights to apply to the average.

    Returns
    -------
    float
        The weighted circular mean of angles.
    """

    rad = np.radians(np.ravel(angles))
    x = np.cos(rad)
    y = np.sin(rad)
    weights = np.ravel(weights)
    if not weights.any():
        # A plane with no slip anywhere has no slip-weighted mean, so
        # fall back to an unweighted average of the angles.
        weights = np.ones_like(weights)
    avg_vector = np.average(np.c_[x, y], weights=weights, axis=0)
    return np.degrees(np.arctan2(avg_vector[1], avg_vector[0])).item() % 360.0


def convert_srf_to_stoch(srf_file: SrfFile, dx: float, dy: float) -> StochFile:
    """Convert an SRF file into a Stoch file by box-averaging slip, tinit and tinit * rise.

    Parameters
    ----------
    srf_file : SrfFile
        The SRF file to convert.
    dx : float
        The desired strike-resolution for the output stoch file.
    dy : float
        The desired dip-resolution for the output stoch file.

    Returns
    -------
    StochFile
        An output stoch file downsampled from ``srf_file``.
    """
    planes = []
    for i, segment in enumerate(srf_file.segments):
        header = srf_file.header.iloc[i].astype(np.float32)
        nstk, ndip = int(header["nstk"]), int(header["ndip"])

        slip = segment["slip"].to_numpy(dtype=np.float32).reshape(ndip, nstk)
        rake = segment["rake"].to_numpy(dtype=np.float32).reshape(
            ndip, nstk
        ) % np.float32(360.0)
        rise = segment["rise"].to_numpy(dtype=np.float32).reshape(ndip, nstk)
        tinit = segment["tinit"].to_numpy(dtype=np.float32).reshape(ndip, nstk)

        dstk = float(header["len"]) / nstk
        ddip = float(header["wid"]) / ndip

        nx = int(np.ceil(float(header["len"]) / dx))
        ny = int(np.ceil(float(header["wid"]) / dy))
        wx = _box_average_matrix(nstk, nx, dstk, dx).astype(np.float32)
        wy = _box_average_matrix(ndip, ny, ddip, dy).astype(np.float32)

        def box_average(values: np.ndarray) -> np.ndarray:
            return wy @ values @ wx.T

        slip_grid = box_average(slip)
        # Cells at the edge of the plane are only partially covered by the SRF,
        # so their weights sum to less than one. That is what conserves the
        # moment for slip, but rupture time is a time rather than a quantity
        # spread over the cell, so it has to be divided by the covered
        # fraction. Every cell is partially covered because nx and ny are
        # rounded up, so this never divides by zero.
        #
        # NOTE: This does materialise an array of order (ny, nx) but it is the
        # *coarse* ny, nx. Unless we have ruptures larger than Hikurangi this is
        # unlikely to ever be an issue.
        coverage = np.outer(
            np.asarray(wy.sum(axis=1)).ravel(), np.asarray(wx.sum(axis=1)).ravel()
        )
        trup_grid = box_average(tinit) / coverage

        # The rise grid is a slip-averaged rise in each cell. Note that because
        # we are dividing rise by slip the coverage factor conveniently cancels
        # out and we do not have to (should not) similarly divide for the rise
        # sum.
        rise_sum = box_average(rise * slip)
        rise_grid = np.where(
            slip_grid > 0,
            rise_sum / np.where(slip_grid > 0, slip_grid, 1),
            np.float32(1e-5),
        )

        stoch_header = StochHeader(
            longitude=header["elon"],
            latitude=header["elat"],
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            strike=header["stk"] % np.float32(360.0),
            dip=header["dip"],
            average_rake=circular_mean(rake, slip),
            dtop=header["dtop"],
            shypo=header["shyp"],
            dhypo=header["dhyp"],
        )
        planes.append(StochPlane(stoch_header, slip_grid, rise_grid, trup_grid))
    return StochFile(planes)


@cli.from_docstring(app)
@log_utils.log_call()
def generate_stoch(
    realisation_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    srf_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    stoch_ffp: Annotated[Path, typer.Argument(dir_okay=False)],
) -> None:
    """Generate a stoch file from an SRF file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation.
    srf_ffp : Path
        Path to the SRF file which is used as input for the stoch file generation.
    stoch_ffp : Path
        Path to the output file where the generated stoch file will be saved.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    hf_config = HFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    srf_file = srf.read_srf(srf_ffp)
    if all(
        isinstance(fault, sources.Point)
        for fault in source_config.source_geometries.values()
    ):
        source = srf_file.header.iloc[0]
        srf_nstk = int(source["nstk"])
        srf_len = float(source["len"])
        dx = srf_len / srf_nstk
        srf_ndip = int(source["ndip"])
        srf_wid = float(source["wid"])
        dy = srf_wid / srf_ndip
    else:
        dx = hf_config.stoch_dx
        dy = hf_config.stoch_dy

    stoch_file = convert_srf_to_stoch(srf_file, dx, dy)
    with open(stoch_ffp, "w") as f:
        stoch_file.dump(f)

    realisations.append_log_entry(realisation_ffp)
