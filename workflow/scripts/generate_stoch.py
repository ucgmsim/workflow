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


def _box_average_matrix(n_fine: int, n_coarse: int) -> sp.csr_matrix:
    """Build a (n_coarse, n_fine) exact box-average operator.

    Row j gives the fractional-overlap weights (summing to 1) between
    coarse bin j and the fine cells it spans, so `matrix @ fine_values`
    computes the exact area-weighted average srf2stoch.c computes via its
    LCM upsample-then-block-average trick, without ever materialising an
    upsampled array.
    """
    bin_width = n_fine / n_coarse
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


def _target_grid(
    length: np.float32, n_fine: int, target: float
) -> tuple[int, np.float32]:
    """Compute (n_coarse, spacing), matching srf2stoch.c's nx=(len/dx+0.5) logic."""
    dx = np.float32(target)
    nx = int(length / dx + 0.5)
    if nx > n_fine:
        nx = n_fine
        dx = length / np.float32(nx)
    return nx, dx


def convert_srf_to_stoch(
    srf_file: SrfFile, target_dx: float, target_dy: float
) -> StochFile:
    planes = []
    for i, segment in enumerate(srf_file.segments):
        header = srf_file.header.iloc[i].astype(np.float32)
        nstk, ndip = int(header["nstk"]), int(header["ndip"])

        # srf2stoch.c does all arithmetic in 32-bit float; matching that
        # precision (rather than numpy's float64 default) keeps our output
        # within float32 rounding distance of the reference tool's.
        slip = segment["slip"].to_numpy(dtype=np.float32).reshape(ndip, nstk)
        rake = segment["rake"].to_numpy(dtype=np.float32).reshape(
            ndip, nstk
        ) % np.float32(360.0)
        rise = segment["rise"].to_numpy(dtype=np.float32).reshape(ndip, nstk)
        tinit = segment["tinit"].to_numpy(dtype=np.float32).reshape(ndip, nstk)
        dx = np.float32(target_dx)
        dy = np.float32(target_dy)
        # EMOD3D consistent rounding behaviour.
        nx = int(header["len"] / dx + 0.5)
        ny = int(header["wid"] / dy + 0.5)
        wx = _box_average_matrix(nstk, nx).astype(np.float32)
        wy = _box_average_matrix(ndip, ny).astype(np.float32)

        def box_average(values: np.ndarray) -> np.ndarray:
            return wy @ values @ wx.T

        slip_grid = box_average(slip)
        trup_grid = box_average(tinit)
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
            dx=float(dx),
            dy=float(dy),
            strike=header["stk"] % np.float32(360.0),
            dip=header["dip"],
            # TODO: This is preserved for consistency with srf2stoch, but averaging rakes like this is incorrect!
            average_rake=np.sum(rake * slip) / np.sum(slip),
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
        # If the stoch dx is greater than the length (resp. dy and width), we might get an empty stoch file
        dx = min(hf_config.stoch_dx, min_length / 2)
        dy = min(hf_config.stoch_dy, min_width / 2)

    stoch_file = convert_srf_to_stoch(srf_file, dx, dy)
    with open(stoch_ffp, "w") as f:
        stoch_file.dump(f)

    realisations.append_log_entry(realisation_ffp)
