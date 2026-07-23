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
Can be run in the cybershake container. Can also be run from your own computer using the `generate-stoch` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you are executing on your own computer you also need to specify the `srf2stoch` path (`--srf2stoch-path`).

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
    to the ``adaptive_avg_pool2d`` kernel in pytorch.

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
    their Least Common Multiple (LCM) base units and computing a standard block
    average (which is the approach that srf2stoch.c takes). The main advantage
    of this approach is that a sparse matrix does not have to materialise all
    the empty cell overlaps in memory.

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
    bin_width = fine_dx / coarse_dx
    edges = np.arange(n_coarse + 1) * bin_width
    rows, cols, weights = [], [], []
    for j in range(n_coarse):
        lo, hi = edges[j], edges[j + 1]
        idx = np.arange(int(lo), min(int(np.ceil(hi)), n_fine))
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

    rad = np.radians(angles)
    x = np.cos(rad)
    y = np.sin(rad)
    avg_vector = np.average(np.c_[x, y], weights=weights, axis=0)
    return np.arctan2(avg_vector[1], avg_vector[0]).item()


def convert_srf_to_stoch(srf_file: SrfFile, dx: float, dy: float) -> StochFile:
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

        nx = np.ceil(header["nstk"] / dx)
        ny = np.ceil(header["ndip"] / dy)
        wx = _box_average_matrix(nstk, nx, header["dstk"], dx).astype(np.float32)
        wy = _box_average_matrix(ndip, ny, header["ddip"], dy).astype(np.float32)

        def box_average(values: np.ndarray) -> np.ndarray:
            return wy @ values @ wx.T

        slip_grid = box_average(slip)
        trup_grid = box_average(tinit)
        rise_sum = box_average(rise * slip)

        # The rise grid is a slip-averaged rise in each cell. This is performing
        # the normalisation for the average.
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

    This function uses the `srf2stoch` binary to generate a stoch file from the provided SRF file.

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
        geometries = list(source_config.source_geometries.values())
        min_length = min(fault.length for fault in geometries)
        min_width = min(fault.width for fault in geometries)
        # If the stoch dx is greater than the length (resp. dy and width), we
        # might get an empty stoch file.
        dx = min(hf_config.stoch_dx, min_length / 2)
        dy = min(hf_config.stoch_dy, min_width / 2)

    stoch_file = convert_srf_to_stoch(srf_file, dx, dy)
    with open(stoch_ffp, "w") as f:
        stoch_file.dump(f)

    realisations.append_log_entry(realisation_ffp)
