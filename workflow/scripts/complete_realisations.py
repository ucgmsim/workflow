#!/usr/bin/env python3
"""Complete NSHM-2022 minimal realisations into simulation-ready realisations.

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
``complete-realisations INPUT_DIR OUTPUT_DIR [--defaults-version 24.2.2.1] ...``
"""

import dataclasses
import json
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer
from tqdm import tqdm

from workflow.defaults import DefaultsVersion, load_defaults
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
from workflow.scripts.reconcile_parameters import (
    Decision,
    flatten_sections,
    load_decisions,
    read_deployed_parameters,
    resolve_value,
    value_fingerprint,
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

# A minimal file can be completed only if it carries the source-side sections.
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


def resolve_parameters(
    decisions: dict[str, Decision],
    defaults_version: DefaultsVersion,
    felipe_scripts_dir: Path,
    events_dir: Path | None,
) -> dict[str, Any]:
    """Resolve each recorded decision to the value it selects.

    The decision file records a choice of source rather than a value, so the
    three candidates are rebuilt here exactly as the reconciler built them, and
    every resolved value is checked against the fingerprint its decision carries.

    Parameters
    ----------
    decisions : dict of str to Decision
        Recorded decisions, keyed by dotted path.
    defaults_version : DefaultsVersion
        Scientific defaults version supplying the ``defaults`` candidate.
    felipe_scripts_dir : Path
        Directory supplying the ``felipe`` candidate.
    events_dir : Path or None
        Directory supplying the ``deployed`` candidate. Required only when some
        decision names ``deployed``.

    Returns
    -------
    dict
        Maps dotted path to the decided value.

    Raises
    ------
    ValueError
        If a decision names ``deployed`` without an events directory, or if a
        resolved value no longer matches the fingerprint the decision recorded.
    """
    if any(decision.source == "deployed" for decision in decisions.values()) and (
        events_dir is None
    ):
        raise ValueError(
            "A decision names source 'deployed'; pass --deployed-from <events dir> "
            "so the deployed candidate can be read."
        )

    defaults = flatten_sections(load_defaults(defaults_version))
    overrides = load_overrides(felipe_scripts_dir)
    felipe = {
        "im.valid_periods": overrides.valid_periods.tolist(),
        "im.fas_frequencies": overrides.fas_frequencies.tolist(),
        "velocity_model.version": overrides.vm_version,
        "velocity_model.rrup_interpolants": overrides.rrup_interpolants.tolist(),
    }
    deployed: dict[str, Any] = {}
    if events_dir is not None:
        deployed, _ = read_deployed_parameters(events_dir)

    resolved: dict[str, Any] = {}
    for path, decision in decisions.items():
        candidates = {
            source: flat[path]
            for source, flat in (
                ("defaults", defaults), ("felipe", felipe), ("deployed", deployed)
            )
            if path in flat
        }
        value = resolve_value(decision.source, candidates)
        if value_fingerprint(value) != decision.sha256:
            raise ValueError(
                f"Decision for {path} names source '{decision.source}', but that "
                f"source has moved since the decision was recorded "
                f"({decision.decided}). Re-run reconcile-parameters."
            )
        resolved[path] = value
    return resolved


def apply_parameters(realisation: dict[str, Any], resolved: dict[str, Any]) -> None:
    """Write decided parameter values into a realisation, in place.

    Only the decided keys are touched; every other key of the same section is
    left as the defaults and overrides produced it.

    Parameters
    ----------
    realisation : dict
        The realisation to modify.
    resolved : dict
        Maps dotted path to the decided value.
    """
    for path, value in resolved.items():
        section, _, key = path.partition(".")
        realisation.setdefault(section, {})[key] = value


def complete_one(
    src: Path,
    dst: Path,
    defaults_version: DefaultsVersion,
    overrides: Overrides,
    resolved: dict[str, Any] | None = None,
) -> None:
    """Complete a single minimal realisation, writing the full file to ``dst``.

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
    resolved : dict, optional
        Decided parameter values to apply last, overriding the defaults and the
        override files. Omit to keep the pre-existing behaviour.
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

    # 6. Apply the campaign's recorded parameter decisions, then normalise key
    #    order for clean diffing against the reference. Decisions are applied
    #    last so they win over both the defaults and the override files.
    with open(dst, encoding="utf-8") as handle:
        realisation = json.load(handle)
    if resolved:
        apply_parameters(realisation, resolved)
    realisation = normalize_key_order(realisation)
    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(realisation, handle, indent=4)


def summary_row(realisation: dict[str, Any], rupture_id: str) -> dict[str, Any]:
    """Extract a one-row scrutiny summary from a completed realisation.

    Parameters
    ----------
    realisation : dict
        A complete (fully materialised) realisation.
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


@dataclasses.dataclass
class CompletionResult:
    """Outcome of completing a single realisation."""

    rupture_id: str
    ok: bool
    error: str | None = None
    summary: dict[str, Any] | None = None


def _rupture_id_from_path(path: Path) -> str:
    """Return the rupture id from a ``realisation_<id>.json`` path.

    Parameters
    ----------
    path : Path
        Path to a ``realisation_<id>.json`` file.

    Returns
    -------
    str
        The rupture id: the filename stem with the ``realisation_`` prefix
        removed.
    """
    return path.stem.removeprefix("realisation_")


def _complete_worker(
    args: tuple[Path, Path, DefaultsVersion, Overrides, dict[str, Any] | None],
) -> CompletionResult:
    """Complete one realisation, capturing any error for aggregate reporting.

    Parameters
    ----------
    args : tuple
        ``(src, dst, defaults_version, overrides, resolved)``.

    Returns
    -------
    CompletionResult
        Success carries a ``summary``; failure carries an ``error`` and the
        partial output is removed.
    """
    src, dst, defaults_version, overrides, resolved = args
    rupture_id = _rupture_id_from_path(src)
    try:
        complete_one(src, dst, defaults_version, overrides, resolved)
        with open(dst, encoding="utf-8") as handle:
            completed = json.load(handle)
        return CompletionResult(
            rupture_id, ok=True, summary=summary_row(completed, rupture_id)
        )
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the batch
        if dst.exists():
            dst.unlink()
        return CompletionResult(
            rupture_id, ok=False, error=f"{type(exc).__name__}: {exc}"
        )


@app.command()
def complete_realisations(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument()],
    defaults_version: Annotated[
        DefaultsVersion, typer.Option()
    ] = DefaultsVersion.v24_2_2_1,
    felipe_scripts_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False)
    ] = Path("felipe_scripts"),
    vm_version: Annotated[str, typer.Option()] = "2.09",
    parameters: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Campaign decision file recording which parameters to adopt.",
        ),
    ] = None,
    deployed_from: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Events directory supplying the 'deployed' candidate, needed "
            "only when a decision names it.",
        ),
    ] = None,
    workers: Annotated[int, typer.Option(min=1)] = min(8, cpu_count()),
) -> None:
    """Complete every minimal realisation in ``input_dir`` into a full file.

    Parameters
    ----------
    input_dir : Path
        Directory of ``realisation_<id>.json`` minimal stubs (read-only).
    output_dir : Path
        Directory to write complete realisations, ``completion_summary.csv`` and
        ``error_log.txt``.
    defaults_version : DefaultsVersion
        Scientific defaults version to materialise.
    felipe_scripts_dir : Path
        Directory containing the override files.
    vm_version : str
        Velocity-model version to force.
    parameters : Path, optional
        Campaign decision file recording which parameters to adopt.
    deployed_from : Path, optional
        Events directory supplying the 'deployed' candidate.
    workers : int
        Number of parallel processes (1 = serial).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(felipe_scripts_dir, vm_version)

    resolved: dict[str, Any] = {}
    if parameters is not None:
        resolved = resolve_parameters(
            load_decisions(parameters), defaults_version, felipe_scripts_dir,
            deployed_from,
        )
        print(f"Applying {len(resolved)} recorded parameter decision(s).")

    valid_files: list[Path] = []
    broken_ids: list[str] = []
    for realisation_ffp in sorted(input_dir.glob("realisation_*.json")):
        with open(realisation_ffp, encoding="utf-8") as handle:
            realisation = json.load(handle)
        if is_valid_minimal(realisation):
            valid_files.append(realisation_ffp)
        else:
            broken_ids.append(_rupture_id_from_path(realisation_ffp))

    work = [
        (src, output_dir / src.name, defaults_version, overrides, resolved)
        for src in valid_files
    ]
    results: list[CompletionResult] = []
    if workers == 1:
        for job in tqdm(work, desc="Completing realisations"):
            results.append(_complete_worker(job))
    else:
        with Pool(processes=workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_complete_worker, work),
                total=len(work),
                desc="Completing realisations",
            ):
                results.append(result)

    completed = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]

    if completed:
        summary_df = pd.DataFrame(
            [result.summary for result in completed]
        ).sort_values("rupture_id")
        summary_df.to_csv(output_dir / "completion_summary.csv", index=False)

    with open(output_dir / "error_log.txt", "w", encoding="utf-8") as handle:
        handle.write(
            f"Skipped {len(broken_ids)} broken minimal stub(s) (no source sections):\n"
        )
        for rupture_id in broken_ids:
            handle.write(f"  SKIPPED rupture {rupture_id}\n")
        handle.write(f"\nFailed to complete {len(failed)} realisation(s):\n")
        for result in failed:
            handle.write(f"  FAILED rupture {result.rupture_id}: {result.error}\n")

    print(f"\nCompleted {len(completed)} realisation(s) -> {output_dir}")
    print(
        f"Skipped {len(broken_ids)} broken stub(s); {len(failed)} failed to complete."
    )
    print(f"Summary : {output_dir / 'completion_summary.csv'}")
    print(f"Errors  : {output_dir / 'error_log.txt'}")


if __name__ == "__main__":
    app()
