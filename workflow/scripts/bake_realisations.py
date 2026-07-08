#!/usr/bin/env python3
"""Bake NSHM-2022 minimal realisations into complete simulation-ready realisations.

Description
-----------
Adapts ``felipe_scripts/gen_FF_realisations_MP.py`` to NSHM inputs: instead of
generating a source from a GCMT solution, it reuses the sources already present
in each minimal NSHM realisation and materialises every remaining section
(velocity model, domain, intensity measures, EMOD3D, 1D velocity models, HF, BB,
resolution, rupture velocity) so the file is a complete, self-contained
realisation identical in structure and parameter values to Felipe's reference.

Usage
-----
``bake-realisations INPUT_DIR OUTPUT_DIR [--defaults-version 24.2.2.1] ...``
"""

import dataclasses
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer

app = typer.Typer()


@dataclasses.dataclass
class Overrides:
    """Custom parameter overrides applied on top of the scientific defaults.

    Attributes
    ----------
    vm_version : str
        Velocity-model version string to force (e.g. "2.09").
    rrup_interpolants : numpy.ndarray
        Shape ``(2, N)`` float32 array of magnitude / rrup interpolant rows.
    valid_periods : numpy.ndarray
        float64 array of pSA/SDI vibration periods.
    fas_frequencies : numpy.ndarray
        float64 array of Fourier amplitude spectrum frequencies.
    """

    vm_version: str
    rrup_interpolants: npt.NDArray[np.float32]
    valid_periods: npt.NDArray[np.float64]
    fas_frequencies: npt.NDArray[np.float64]


def load_overrides(felipe_scripts_dir: Path, vm_version: str = "2.09") -> Overrides:
    """Load the velocity-model and intensity-measure overrides from input files.

    Parameters
    ----------
    felipe_scripts_dir : Path
        Directory containing ``Mw_rrup_mod.txt``, ``periods.csv`` and
        ``frequencies.csv``.
    vm_version : str
        Velocity-model version to record in the overrides.

    Returns
    -------
    Overrides
        The loaded overrides.

    Raises
    ------
    FileNotFoundError
        If any of the three override files is missing.
    """
    felipe_scripts_dir = Path(felipe_scripts_dir)
    rrup_txt = felipe_scripts_dir / "Mw_rrup_mod.txt"
    periods_csv = felipe_scripts_dir / "periods.csv"
    frequencies_csv = felipe_scripts_dir / "frequencies.csv"
    for override_file in (rrup_txt, periods_csv, frequencies_csv):
        if not override_file.exists():
            raise FileNotFoundError(f"Required override file not found: {override_file}")

    mag_vec, rrup_vec = np.loadtxt(rrup_txt, unpack=True)
    rrup_interpolants = np.array([mag_vec, rrup_vec], dtype=np.float32)
    valid_periods = pd.read_csv(periods_csv)["valid_periods"].to_numpy(dtype=np.float64)
    fas_frequencies = pd.read_csv(frequencies_csv)["fas_frequencies"].to_numpy(
        dtype=np.float64
    )
    return Overrides(
        vm_version=vm_version,
        rrup_interpolants=rrup_interpolants,
        valid_periods=valid_periods,
        fas_frequencies=fas_frequencies,
    )


if __name__ == "__main__":
    app()
