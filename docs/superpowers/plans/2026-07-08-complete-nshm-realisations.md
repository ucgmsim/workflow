# Complete NSHM-2022 Realisations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn each valid minimal NSHM-2022 realisation stub into a complete, self-contained realisation file — matching Felipe's reference (`felipe_3528839_realisation.json`) section-for-section — written to a new folder, leaving the originals untouched.

**Architecture:** A single `typer` CLI module, `workflow/scripts/complete_realisations.py`, adapted from `felipe_scripts/gen_FF_realisations_MP.py`. It drops the GCMT/nodal-plane source generation (we already have NSHM sources) and adds full materialisation of every default section. It reuses the workflow's `read_from_defaults` / `write_to_realisation` machinery and `generate_domain_from_realisation`. Pure helpers (`load_overrides`, `is_valid_minimal`, `normalize_key_order`, `summary_row`) are unit-tested; `complete_one` and the driver are covered by slow integration tests that diff rupture-independent sections against Felipe's committed reference.

**Tech Stack:** Python 3.12, typer, numpy, pandas, tqdm, multiprocessing; workflow package (`workflow.realisations`, `workflow.defaults`, `workflow.scripts.generate_domain`); pytest.

## Global Constraints

- Target scientific defaults version: **24.2.2.1** (100 m resolution; `bb.flo` 1.0). Copied verbatim from spec.
- Velocity-model version override: **2.09**.
- Overrides come from `felipe_scripts/{Mw_rrup_mod.txt, periods.csv, frequencies.csv}` — reused as-is.
- **Never modify** any file under `realisations_from_nshm2022_to_realisation/` — inputs are read-only.
- Output must be **deterministic / idempotent**: no RNG; `seeds` are copied, not regenerated.
- Final files must contain exactly these **18 sections** in this order (`normalize_key_order`): `metadata, sources, rupture_propagation, magnitudes, rakes, log_trail, velocity_model, domain, im, seeds, emod3d, resolution, srf, velocity_model_1d, hf_velocity_model_1d, hf, bb, rupture_velocity`.
- Follow repo conventions: typer CLI under `workflow/scripts/`, tests under `tests/`, numpydoc docstrings, ruff formatting.
- Run tests with: `.venv/bin/python -m pytest tests/test_complete_realisations.py -v` (pytest 9.0.2 is in `.venv`).
- Current branch is `andrew-cs-2022` (not the default `pegasus`); commit here.

## File Structure

- **Create** `workflow/scripts/complete_realisations.py` — the whole feature: constants, `Overrides`, `load_overrides`, `is_valid_minimal`, `normalize_key_order`, `complete_one`, `summary_row`, `CompletionResult`, `_complete_worker`, and the `complete_realisations` typer command.
- **Create** `tests/test_complete_realisations.py` — unit + integration tests.
- **Create** `tests/data/minimal_realisation_sample.json`, `tests/data/felipe_reference_realisation.json`, `tests/data/broken_minimal_stub.json` — committed fixtures.
- **Modify** `pyproject.toml` `[project.scripts]` — add the `complete-realisations` entry point.

---

### Task 1: Module scaffold, test fixtures, and `load_overrides`

**Files:**
- Create: `workflow/scripts/complete_realisations.py`
- Create: `tests/test_complete_realisations.py`
- Create: `tests/data/{minimal_realisation_sample.json, felipe_reference_realisation.json, broken_minimal_stub.json}`

**Interfaces:**
- Produces: `Overrides` dataclass with fields `vm_version: str`, `rrup_interpolants: np.ndarray[float32] (2, N)`, `valid_periods: np.ndarray[float64]`, `fas_frequencies: np.ndarray[float64]`; and `load_overrides(felipe_scripts_dir: Path, vm_version: str = "2.09") -> Overrides`.

- [ ] **Step 1: Create committed test fixtures from real files**

```bash
mkdir -p tests/data
cp realisations_from_nshm2022_to_realisation/realisation_114741.json tests/data/minimal_realisation_sample.json
cp realisations_from_nshm2022_to_realisation/realisation_59421.json  tests/data/broken_minimal_stub.json
cp felipe_3528839_realisation.json                                    tests/data/felipe_reference_realisation.json
```

Verify: `python -c "import json; assert 'sources' in json.load(open('tests/data/minimal_realisation_sample.json')); assert 'sources' not in json.load(open('tests/data/broken_minimal_stub.json')); print('fixtures ok')"`

- [ ] **Step 2: Write the failing test**

Create `tests/test_complete_realisations.py`:

```python
"""Tests for the complete_realisations campaign tool."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from workflow.defaults import DefaultsVersion
from workflow.scripts import complete_realisations as br

DATA = Path(__file__).parent / "data"
FELIPE = DATA / "felipe_reference_realisation.json"
SAMPLE = DATA / "minimal_realisation_sample.json"
BROKEN = DATA / "broken_minimal_stub.json"
FELIPE_SCRIPTS = Path(__file__).parents[1] / "felipe_scripts"


def test_load_overrides_shapes_and_dtypes() -> None:
    overrides = br.load_overrides(FELIPE_SCRIPTS, vm_version="2.09")
    assert overrides.vm_version == "2.09"
    assert overrides.rrup_interpolants.shape == (2, 29)
    assert overrides.rrup_interpolants.dtype == np.float32
    assert overrides.valid_periods.shape == (111,)
    assert overrides.fas_frequencies.shape == (389,)


def test_load_overrides_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        br.load_overrides(tmp_path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError` (module `complete_realisations` or `load_overrides` not defined).

- [ ] **Step 4: Write minimal implementation**

Create `workflow/scripts/complete_realisations.py`:

```python
#!/usr/bin/env python3
"""Complete NSHM-2022 minimal realisations into complete simulation-ready realisations.

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add workflow/scripts/complete_realisations.py tests/test_complete_realisations.py tests/data/
git commit -m "feat(complete): scaffold complete_realisations with load_overrides + fixtures"
```

---

### Task 2: `is_valid_minimal`, canonical order, `normalize_key_order`

**Files:**
- Modify: `workflow/scripts/complete_realisations.py`
- Test: `tests/test_complete_realisations.py`

**Interfaces:**
- Produces: `FELIPE_SECTION_ORDER: list[str]`, `REQUIRED_MINIMAL_SECTIONS: tuple[str, ...]`, `is_valid_minimal(realisation: dict) -> bool`, `normalize_key_order(realisation: dict) -> dict`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_complete_realisations.py`:

```python
def test_is_valid_minimal_true_for_full_stub() -> None:
    assert br.is_valid_minimal(json.loads(SAMPLE.read_text())) is True


def test_is_valid_minimal_false_for_broken_stub() -> None:
    assert br.is_valid_minimal(json.loads(BROKEN.read_text())) is False


def test_normalize_key_order_matches_canonical() -> None:
    scrambled = {"bb": 1, "sources": 2, "metadata": 3, "surprise": 4, "domain": 5}
    assert list(br.normalize_key_order(scrambled)) == [
        "metadata",
        "sources",
        "domain",
        "bb",
        "surprise",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -k "valid_minimal or normalize" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'is_valid_minimal'`.

- [ ] **Step 3: Write minimal implementation**

In `workflow/scripts/complete_realisations.py`, add below the imports (before `Overrides`):

```python
from typing import Any

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
```

And add these functions (below `load_overrides`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -k "valid_minimal or normalize" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/complete_realisations.py tests/test_complete_realisations.py
git commit -m "feat(complete): add is_valid_minimal and normalize_key_order"
```

---

### Task 3: `complete_one` (full materialisation)

**Files:**
- Modify: `workflow/scripts/complete_realisations.py`
- Test: `tests/test_complete_realisations.py`

**Interfaces:**
- Consumes: `Overrides`, `normalize_key_order`, `DefaultsVersion`.
- Produces: `complete_one(src: Path, dst: Path, defaults_version: DefaultsVersion, overrides: Overrides) -> None` — writes a complete 18-section realisation to `dst`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_complete_realisations.py`:

```python
@pytest.mark.slow
def test_complete_one_produces_complete_realisation(tmp_path: Path) -> None:
    overrides = br.load_overrides(FELIPE_SCRIPTS)
    dst = tmp_path / "completed.json"

    br.complete_one(SAMPLE, dst, DefaultsVersion.v24_2_2_1, overrides)
    completed = json.loads(dst.read_text())

    # Exactly the 18 canonical sections, in canonical order.
    assert list(completed) == br.FELIPE_SECTION_ORDER
    # Domain computed and sane.
    assert completed["domain"]["depth"] > 0
    assert completed["domain"]["duration"] > 0
    # Overrides applied.
    assert completed["velocity_model"]["version"] == "2.09"
    assert len(completed["im"]["valid_periods"]) == 111
    assert len(completed["im"]["fas_frequencies"]) == 389
    assert completed["metadata"]["defaults_version"] == "24.2.2.1"
    # Rupture-independent sections identical to Felipe's reference.
    felipe = json.loads(FELIPE.read_text())
    for section in [
        "velocity_model",
        "im",
        "emod3d",
        "hf",
        "bb",
        "resolution",
        "srf",
        "velocity_model_1d",
        "hf_velocity_model_1d",
    ]:
        assert completed[section] == felipe[section], f"{section} differs from Felipe"


def test_complete_one_does_not_touch_source(tmp_path: Path) -> None:
    # Guard the read-only-inputs constraint: complete_one must copy, never edit src.
    src = tmp_path / "realisation_114741.json"
    shutil.copy(SAMPLE, src)
    before = src.read_bytes()
    dst = tmp_path / "completed.json"
    try:
        br.complete_one(src, dst, DefaultsVersion.v24_2_2_1, br.load_overrides(FELIPE_SCRIPTS))
    except Exception:  # noqa: BLE001 -- even on failure src must be untouched
        pass
    assert src.read_bytes() == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -k complete_one -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'complete_one'`.

- [ ] **Step 3: Write minimal implementation**

In `workflow/scripts/complete_realisations.py`, extend the imports:

```python
import json
import shutil

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
```

Add this constant next to the others:

```python
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
```

Add the function:

```python
def complete_one(
    src: Path, dst: Path, defaults_version: DefaultsVersion, overrides: Overrides
) -> None:
    """Complete a single minimal realisation into a complete realisation at ``dst``.

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -k complete_one -v`
Expected: PASS (2 passed). The slow test takes ~5–8 s (domain generation).

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/complete_realisations.py tests/test_complete_realisations.py
git commit -m "feat(complete): add complete_one full-materialisation of a realisation"
```

---

### Task 4: `summary_row` (scrutiny aid)

**Files:**
- Modify: `workflow/scripts/complete_realisations.py`
- Test: `tests/test_complete_realisations.py`

**Interfaces:**
- Consumes: `total_magnitude` (from `generate_domain`).
- Produces: `summary_row(realisation: dict, rupture_id: str) -> dict` with keys `rupture_id, n_faults, fault_names, total_magnitude_mw, domain_depth_km, domain_duration_s, n_valid_periods, n_fas_frequencies, defaults_version, vm_version`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_complete_realisations.py` (uses Felipe's complete reference, so it is fast):

```python
def test_summary_row_fields() -> None:
    felipe = json.loads(FELIPE.read_text())
    row = br.summary_row(felipe, "3528839")
    assert row["rupture_id"] == "3528839"
    assert row["n_faults"] == 1
    assert row["fault_names"] == "3528839"
    assert row["vm_version"] == "2.09"
    assert row["n_valid_periods"] == 111
    assert row["n_fas_frequencies"] == 389
    assert isinstance(row["total_magnitude_mw"], float)
    assert row["domain_depth_km"] == felipe["domain"]["depth"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -k summary_row -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'summary_row'`.

- [ ] **Step 3: Write minimal implementation**

Add to `workflow/scripts/complete_realisations.py`:

```python
def summary_row(realisation: dict[str, Any], rupture_id: str) -> dict[str, Any]:
    """Extract a one-row scrutiny summary from a completed realisation.

    Parameters
    ----------
    realisation : dict
        A complete (completed) realisation.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -k summary_row -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/complete_realisations.py tests/test_complete_realisations.py
git commit -m "feat(complete): add summary_row scrutiny extraction"
```

---

### Task 5: CLI driver, parallel worker, reporting, entry point

**Files:**
- Modify: `workflow/scripts/complete_realisations.py`
- Modify: `pyproject.toml` (add `[project.scripts]` entry)
- Test: `tests/test_complete_realisations.py`

**Interfaces:**
- Consumes: `load_overrides`, `is_valid_minimal`, `complete_one`, `summary_row`.
- Produces: `CompletionResult` dataclass (`rupture_id: str, ok: bool, error: str | None, summary: dict | None`), `_complete_worker(args) -> CompletionResult`, and the `complete_realisations(input_dir, output_dir, defaults_version, felipe_scripts_dir, vm_version, workers)` typer command. Writes `<output_dir>/completion_summary.csv` and `<output_dir>/error_log.txt`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_complete_realisations.py`:

```python
@pytest.mark.slow
def test_complete_realisations_end_to_end(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    shutil.copy(SAMPLE, input_dir / "realisation_114741.json")
    shutil.copy(BROKEN, input_dir / "realisation_59421.json")

    br.complete_realisations(
        input_dir,
        output_dir,
        defaults_version=DefaultsVersion.v24_2_2_1,
        felipe_scripts_dir=FELIPE_SCRIPTS,
        vm_version="2.09",
        workers=1,
    )

    completed = output_dir / "realisation_114741.json"
    assert completed.exists()
    assert list(json.loads(completed.read_text())) == br.FELIPE_SECTION_ORDER
    # Broken stub skipped, not written.
    assert not (output_dir / "realisation_59421.json").exists()
    # Reports written.
    assert (output_dir / "completion_summary.csv").exists()
    assert "114741" in (output_dir / "completion_summary.csv").read_text()
    assert "59421" in (output_dir / "error_log.txt").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -k end_to_end -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'complete_realisations'`.

- [ ] **Step 3: Write minimal implementation**

Extend imports in `workflow/scripts/complete_realisations.py`:

```python
from multiprocessing import Pool, cpu_count
from typing import Annotated

from tqdm import tqdm
```

Add the worker, result type, and command (place the `complete_realisations` command above the `if __name__` block):

```python
@dataclasses.dataclass
class CompletionResult:
    """Outcome of completing a single realisation."""

    rupture_id: str
    ok: bool
    error: str | None = None
    summary: dict[str, Any] | None = None


def _rupture_id_from_path(path: Path) -> str:
    """Return the rupture id from a ``realisation_<id>.json`` path."""
    return path.stem.removeprefix("realisation_")


def _complete_worker(
    args: tuple[Path, Path, DefaultsVersion, Overrides],
) -> CompletionResult:
    """Complete one realisation, capturing any error for aggregate reporting.

    Parameters
    ----------
    args : tuple
        ``(src, dst, defaults_version, overrides)``.

    Returns
    -------
    CompletionResult
        Success carries a ``summary``; failure carries an ``error`` and the
        partial output is removed.
    """
    src, dst, defaults_version, overrides = args
    rupture_id = _rupture_id_from_path(src)
    try:
        complete_one(src, dst, defaults_version, overrides)
        with open(dst, encoding="utf-8") as handle:
            completed = json.load(handle)
        return CompletionResult(rupture_id, ok=True, summary=summary_row(completed, rupture_id))
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the batch
        if dst.exists():
            dst.unlink()
        return CompletionResult(rupture_id, ok=False, error=f"{type(exc).__name__}: {exc}")


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
    workers: Annotated[int, typer.Option(min=1)] = min(8, cpu_count()),
) -> None:
    """Complete every minimal realisation in ``input_dir`` into a complete file.

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
    workers : int
        Number of parallel processes (1 = serial).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(felipe_scripts_dir, vm_version)

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
        (src, output_dir / src.name, defaults_version, overrides)
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
        summary_df = pd.DataFrame([result.summary for result in completed]).sort_values(
            "rupture_id"
        )
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
    print(f"Skipped {len(broken_ids)} broken stub(s); {len(failed)} failed to complete.")
    print(f"Summary : {output_dir / 'completion_summary.csv'}")
    print(f"Errors  : {output_dir / 'error_log.txt'}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -k end_to_end -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Add the console-script entry point**

In `pyproject.toml`, under `[project.scripts]`, add a line next to `generate-realisations-from-csv`:

```toml
complete-realisations = "workflow.scripts.complete_realisations:app"
```

- [ ] **Step 6: Run the full test module**

Run: `.venv/bin/python -m pytest tests/test_complete_realisations.py -v`
Expected: PASS (all tests). ~10–20 s total (two slow tests do domain generation).

- [ ] **Step 7: Commit**

```bash
git add workflow/scripts/complete_realisations.py tests/test_complete_realisations.py pyproject.toml
git commit -m "feat(complete): add parallel CLI driver, reporting, and entry point"
```

---

### Task 6: Run the completion over the real campaign data and verify

**Files:**
- Output only: `realisations_completed_24.2.2.1/` (283 completed files + `completion_summary.csv` + `error_log.txt`). Not committed unless requested.

**Interfaces:**
- Consumes: the finished `complete-realisations` tool.

- [ ] **Step 1: Run the tool on all minimal files**

Run:
```bash
.venv/bin/python workflow/scripts/complete_realisations.py \
  realisations_from_nshm2022_to_realisation \
  realisations_completed_24.2.2.1
```
Expected: progress bar over 283 files; final print `Completed 283 realisation(s)` (or fewer if some domain generations fail), `Skipped 10 broken stub(s)`. Runtime ~3–4 min at 8 workers.

- [ ] **Step 2: Verify counts and structure**

Run:
```bash
.venv/bin/python - <<'PY'
import json, glob
files = sorted(glob.glob("realisations_completed_24.2.2.1/realisation_*.json"))
print("completed files:", len(files))
from collections import Counter
c = Counter()
for f in files:
    d = json.load(open(f))
    c[tuple(d.keys())] += 1
    assert d["metadata"]["defaults_version"] == "24.2.2.1", f
    assert d["velocity_model"]["version"] == "2.09", f
    assert "domain" in d and d["domain"]["duration"] > 0, f
print("distinct key layouts:", len(c), "(expect 1)")
print("sections per file:", len(files) and len(files[0] and json.load(open(files[0]))))
PY
```
Expected: all files have `defaults_version 24.2.2.1`, `velocity_model.version 2.09`, a positive-duration domain, and one distinct 18-key layout.

- [ ] **Step 3: Confirm consistency with Felipe on a spot file**

Run:
```bash
.venv/bin/python - <<'PY'
import json
felipe = json.load(open("felipe_3528839_realisation.json"))
import glob
one = json.load(open(sorted(glob.glob("realisations_completed_24.2.2.1/realisation_*.json"))[0]))
for s in ["velocity_model","im","emod3d","hf","bb","resolution","srf",
          "velocity_model_1d","hf_velocity_model_1d"]:
    print(s, "OK" if one[s] == felipe[s] else "DIFFERS")
PY
```
Expected: every listed section prints `OK`.

- [ ] **Step 4: Review the summary and error log**

Run:
```bash
head -5 realisations_completed_24.2.2.1/completion_summary.csv
echo "---"
cat realisations_completed_24.2.2.1/error_log.txt
echo "--- rows:"; wc -l realisations_completed_24.2.2.1/completion_summary.csv
```
Expected: a CSV with one row per completed file (rupture id, #faults, magnitude, domain depth/duration, #periods…); `error_log.txt` lists the 10 skipped broken stubs and any per-file completion failures with reasons. Report any failures to the user for follow-up.

---

## Self-Review

**1. Spec coverage.**
- New folder / originals untouched → Task 3 `complete_one` copies src (guarded by `test_complete_one_does_not_touch_source`); Task 6 writes `realisations_completed_24.2.2.1/`. ✓
- `defaults_version` → 24.2.2.1 (patch) → Task 3 step 1; verified Task 6 step 2. ✓
- velocity_model 2.09 + rrup; im dense periods/FAS → Task 3; asserted Task 3 test. ✓
- Computed domain → Task 3 (`generate_domain_from_realisation`). ✓
- Remaining sections from defaults incl. `rupture_velocity` → `_DEFAULTS_SECTION_CLASSES`, Task 3. ✓
- 18-section canonical order → `FELIPE_SECTION_ORDER` (Task 2), asserted Task 3/5. ✓
- Consistency with Felipe (identical shared sections) → Task 3 test diff; Task 6 step 3. ✓
- Skip 10 broken stubs; per-file failure logging → Task 5 driver + test. ✓
- Summary CSV scrutiny aid → Task 4 + Task 5. ✓
- Deterministic/idempotent → no RNG in `complete_one`; seeds copied. ✓
- Parallel, ~3–4 min → Task 5 `Pool`; Task 6 run. ✓

**2. Placeholder scan.** No TBD/TODO; every code step shows complete code; every test step shows the assertions. ✓

**3. Type consistency.** `Overrides` fields (`vm_version, rrup_interpolants, valid_periods, fas_frequencies`) are consumed identically in `complete_one`. `complete_one(src, dst, defaults_version, overrides)` signature matches every caller (`_complete_worker`, tests). `CompletionResult(rupture_id, ok, error, summary)` matches its constructions and the driver's `.ok/.summary/.error` reads. `summary_row(realisation, rupture_id)` matches its two call sites. `FELIPE_SECTION_ORDER` referenced consistently. ✓

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-08-complete-nshm-realisations.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
