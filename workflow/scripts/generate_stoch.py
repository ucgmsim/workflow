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
from source_modelling import srf
from source_modelling.srf import SrfFile
from source_modelling.stoch import StochFile, StochHeader, StochPlane
from workflow import log_utils, realisations
from workflow.realisations import RealisationMetadata, StochConfig

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

        slip, rake, rise, tinit = (
            segment[column].to_numpy(dtype=np.float32).reshape(ndip, nstk)
            for column in ("slip", "rake", "rise", "tinit")
        )

        length, width = float(header["len"]), float(header["wid"])
        nx = int(np.ceil(length / dx))
        ny = int(np.ceil(width / dy))
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

        stoch_header = StochHeader(
            longitude=header["elon"],
            latitude=header["elat"],
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            strike=round(header["stk"]) % 360,
            dip=round(header["dip"]),
            average_rake=round(_circular_mean(rake, slip)),
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
    stoch_config = StochConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    stoch_file = convert_srf_to_stoch(
        srf.read_srf(srf_ffp), stoch_config.stoch_dx, stoch_config.stoch_dy
    )
    with open(stoch_ffp, "w") as f:
        stoch_file.dump(f)

    realisations.append_log_entry(realisation_ffp)
