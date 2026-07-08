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
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer

app = typer.Typer()


# Canonical top-level key order: matches felipe_3528839_realisation.json, with
# rupture_velocity (absent from Felipe's file) appended last.
FELIPE_SECTION_ORDER: list[str] = [
    "metadata",
    "sources",
    "rupture_propagation",
    "magnitudes",
    "rakes",
    "log_trail",
    "velocity_model",
    "domain",
    "im",
    "seeds",
    "emod3d",
    "resolution",
    "srf",
    "velocity_model_1d",
    "hf_velocity_model_1d",
    "hf",
    "bb",
    "rupture_velocity",
]

# A minimal file can be baked only if it carries the source-side sections.
REQUIRED_MINIMAL_SECTIONS: tuple[str, ...] = (
    "sources",
    "magnitudes",
    "rakes",
    "rupture_propagation",
)


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


def is_valid_minimal(realisation: dict[str, Any]) -> bool:
    """Return whether a minimal realisation carries all source-side sections.

    Parameters
    ----------
    realisation : dict
        The parsed realisation JSON.

    Returns
    -------
    bool
        True if every section in ``REQUIRED_MINIMAL_SECTIONS`` is present.
    """
    return all(section in realisation for section in REQUIRED_MINIMAL_SECTIONS)


def normalize_key_order(realisation: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the realisation with top-level keys in canonical order.

    Keys listed in ``FELIPE_SECTION_ORDER`` come first, in that order; any
    unexpected keys are preserved afterwards in their original order.

    Parameters
    ----------
    realisation : dict
        The parsed realisation JSON.

    Returns
    -------
    dict
        A new dict with keys reordered.
    """
    ordered = {
        key: realisation[key] for key in FELIPE_SECTION_ORDER if key in realisation
    }
    for key in realisation:
        if key not in ordered:
            ordered[key] = realisation[key]
    return ordered


if __name__ == "__main__":
    app()
