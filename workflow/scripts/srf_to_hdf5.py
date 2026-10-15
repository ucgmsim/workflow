#!/usr/bin/env python3
"""SRF to HDF5.

Description
-----------
Convert an SRF file into SW4's SRF-HDF5 source format. This is a thin wrapper over `source_modelling.srf.SrfFile.write_sw4_hdf5` so the conversion can be run as a workflow stage.

Inputs
------
An SRF file, for example one produced by `realisation-to-srf`.

Outputs
-------
The same rupture written as an [SRF-HDF5](https://github.com/geodynamics/sw4/blob/master/doc/SW4_UsersGuide.pdf) file, which is the source description SW4 reads.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `srf-to-hdf5` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`srf-to-hdf5 [OPTIONS] SRF_PATH HDF5_PATH`

For More Help
-------------
See the output of `srf-to-hdf5 --help`.
"""

from pathlib import Path

import typer

from qcore import cli
from source_modelling import srf

app = typer.Typer()


@cli.from_docstring(app)
def srf_to_hdf5(srf_path: Path, hdf5_path: Path) -> None:
    """Convert an SRF file to SW4's SRF-HDF5 format.

    Parameters
    ----------
    srf_path : Path
        Path to the input SRF file.
    hdf5_path : Path
        Path where the output HDF5 file will be written.
    """
    srf_file = srf.read_srf(srf_path)
    srf_file.write_sw4_hdf5(hdf5_path)
