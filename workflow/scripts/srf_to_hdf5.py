#!/usr/bin/env python3
from pathlib import Path

import typer

from qcore import cli
from source_modelling import srf

app = typer.Typer()


@cli.from_docstring(app)
def srf_to_hdf5(srf_path: Path, hdf5_path: Path) -> None:
    """
    Convert an SRF file to HDF5 format.

    Parameters
    ----------
    srf_path : Path
        Path to the input SRF file.
    hdf5_path : Path
        Path where the output HDF5 file will be written.

    Examples
    --------
    >>> from pathlib import Path
    >>> srf_to_hdf5(Path("input.srf"), Path("output.h5"))
    """
    srf_file = srf.read_srf(srf_path)
    srf_file.write_sw4_hdf5(hdf5_path)
