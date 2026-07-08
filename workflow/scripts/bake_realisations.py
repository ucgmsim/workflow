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
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer

from workflow.defaults import DefaultsVersion
from workflow.realisations import (
    BroadbandParameters,
    EMOD3DParameters,
    HFConfig,
    HFVelocityModel1D,
    IntensityMeasureCalculationParameters,
    RealisationMetadata,
    Resolution,
    RuptureVelocity,
    VelocityModel1D,
    VelocityModelParameters,
)
from workflow.scripts.generate_domain import (
    generate_domain_from_realisation,
    total_magnitude,
)

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

# Sections materialised verbatim from the scientific defaults (order irrelevant;
# normalize_key_order fixes the final layout).
_DEFAULTS_SECTION_CLASSES = (
    EMOD3DParameters,
    Resolution,
    BroadbandParameters,
    HFConfig,
    VelocityModel1D,
    HFVelocityModel1D,
    RuptureVelocity,
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


def bake_one(
    src: Path, dst: Path, defaults_version: DefaultsVersion, overrides: Overrides
) -> None:
    """Bake a single minimal realisation into a complete realisation at ``dst``.

    The source file is copied, never modified. Section-write order matters:
    the velocity model is written before domain generation, which reads it.

    Parameters
    ----------
    src : Path
        The minimal realisation to read.
    dst : Path
        Where to write the complete realisation.
    defaults_version : DefaultsVersion
        Scientific defaults version to materialise (and record in metadata).
    overrides : Overrides
        Velocity-model and intensity-measure overrides.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)

    # 1. Point metadata at the target defaults version.
    metadata = RealisationMetadata.read_from_realisation(dst)
    metadata.defaults_version = defaults_version
    metadata.write_to_realisation(dst)

    # 2. Velocity model (defaults + overrides) -- MUST precede domain generation.
    velocity_model = VelocityModelParameters.read_from_defaults(defaults_version)
    velocity_model.version = overrides.vm_version
    velocity_model.rrup_interpolants = overrides.rrup_interpolants
    velocity_model.write_to_realisation(dst)

    # 3. Domain (computed; reads the velocity model written above).
    generate_domain_from_realisation(dst)

    # 4. Intensity measures (defaults + overrides).
    intensity_measures = IntensityMeasureCalculationParameters.read_from_defaults(
        defaults_version
    )
    intensity_measures.valid_periods = overrides.valid_periods
    intensity_measures.fas_frequencies = overrides.fas_frequencies
    intensity_measures.write_to_realisation(dst)

    # 5. Remaining sections verbatim from the scientific defaults.
    for section_cls in _DEFAULTS_SECTION_CLASSES:
        section_cls.read_from_defaults(defaults_version).write_to_realisation(dst)

    # 6. Normalise key order for clean diffing against the reference.
    with open(dst, encoding="utf-8") as handle:
        realisation = json.load(handle)
    realisation = normalize_key_order(realisation)
    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(realisation, handle, indent=4)


def summary_row(realisation: dict[str, Any], rupture_id: str) -> dict[str, Any]:
    """Extract a one-row scrutiny summary from a baked realisation.

    Parameters
    ----------
    realisation : dict
        A complete (baked) realisation.
    rupture_id : str
        The rupture identifier for this realisation.

    Returns
    -------
    dict
        Summary fields suitable for a review CSV. ``total_magnitude_mw`` is the
        moment-summed total in the Mw convention (as used for domain sizing).
    """
    sources = realisation["sources"]["source_geometries"]
    magnitudes = realisation["magnitudes"]["magnitudes"]
    domain = realisation["domain"]
    return {
        "rupture_id": rupture_id,
        "n_faults": len(sources),
        "fault_names": ";".join(sources),
        "total_magnitude_mw": round(
            float(total_magnitude(list(magnitudes.values()))), 4
        ),
        "domain_depth_km": domain["depth"],
        "domain_duration_s": round(float(domain["duration"]), 2),
        "n_valid_periods": len(realisation["im"]["valid_periods"]),
        "n_fas_frequencies": len(realisation["im"]["fas_frequencies"]),
        "defaults_version": realisation["metadata"]["defaults_version"],
        "vm_version": realisation["velocity_model"]["version"],
    }


if __name__ == "__main__":
    app()
