# Point-Source SRF v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make point-source ruptures emit SRF version 2.0 (per-point `vs` and `den`) by augmenting the working v1 output of `generic_slip2srf` in Python.

**Architecture:** `generate_point_source_srf` keeps calling `generic_slip2srf` to produce a v1 SRF (the binary's own v2 path is broken). When `srf_config.srf_version == "2.0"`, we then read that SRF, look up `vs`/`den` per point from the same 1-D velocity model already used for the slip calculation, and rewrite it as v2 via `source_modelling`'s v2 ASCII writer. Two small helpers are added to the script; no multi-fault/stitch code changes.

**Tech Stack:** Python, numpy, pandas, `source_modelling` (SRF read/write, bumped to 2026.6.6), `uv` (env + lock), pytest, ruff, ty.

## Global Constraints

(Every task implicitly includes these. Values copied verbatim from the spec.)

- Floor `source-modelling` at `2026.6.6` (the v2 ASCII writer; installed is effectively `2026.6.2`).
- Change **only** `generate_point_source_srf` in `workflow/scripts/realisation_to_srf.py` plus the dependency files. **No** changes to `stitch_srf_files` / multi-fault / rupture-combination code, nor to `generic_slip2srf.c` / `genslip` / `point_source_slip`.
- Units in the SRF v2 file: `vs` = `Vs_model[km/s] × 1e5` (cm/s); `den` = `rho_model` (g/cm³, unchanged).
- Layer selection reuses `source_modelling.moment.point_source_slip`'s convention: layer **top**-depths, `searchsorted(side="right") - 1`, exact boundary → deeper layer.
- `version` and the `vs`/`den` columns must be set together (the v2 writer infers the version from the column count; a mismatch corrupts the file silently).
- SRF versions other than `"1.0"`/`"2.0"` for point sources raise `NotImplementedError`.
- Docstrings follow numpydoc. Run `ruff format` + `ruff check` + `ty check` before each commit. Use `uv run --no-sync` for python/console commands (so uv does not re-sync mid-task), except in Task 1 where we deliberately re-lock and sync.
- The native binaries for manual verification (the container does not run here): `genslip` = `/home/arr65/src/EMOD3D/tools/genslip_v5.4.2`, `generic_slip2srf` = `/home/arr65/src/EMOD3D/tools/generic_slip2srf`.
- Commit messages: short imperative subject (no body), plus the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

- **Modify** `pyproject.toml` (line 19) and `uv.lock` — bump the `source-modelling` floor.
- **Modify** `workflow/scripts/realisation_to_srf.py` — add `_velocity_model_vs_den` and `_rewrite_point_source_srf_as_v2` (just before `generate_point_source_srf`, current line 621) and the version gate at the end of `generate_point_source_srf` (after current line 716).
- **Modify** `tests/test_realisation_to_srf.py` — add `test_velocity_model_vs_den` and `test_rewrite_point_source_srf_as_v2`, plus the imports they need.

---

### Task 1: Bump `source_modelling` to 2026.6.6 and confirm the suite is green

**Files:**
- Modify: `pyproject.toml:19`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `source_modelling==2026.6.6` available in the venv, providing `srf.read_srf` / `srf.write_srf` with v2 ASCII support (consumed by Task 3).

- [ ] **Step 1: Edit the dependency floor**

In `pyproject.toml`, change line 19 from:
```
  "source_modelling>=2026.6.2",
```
to:
```
  "source_modelling>=2026.6.6",
```
(Leave the unversioned `"source_modelling",` reference at line 122 untouched — it inherits the resolved version.)

- [ ] **Step 2: Re-lock and sync the environment**

Run:
```bash
cd /home/arr65/src/workflow
uv lock --upgrade-package source-modelling
uv sync
```
Expected: `uv.lock` updates the `source-modelling` entry to `2026.6.6`; `uv sync` installs it.

- [ ] **Step 3: Verify the installed version**

Run:
```bash
uv run --no-sync python -c "import importlib.metadata as m; print(m.version('source-modelling'))"
```
Expected output: `2026.6.6`

- [ ] **Step 4: Run the full existing test suite (regression guard on the multi-fault read/write interaction)**

Run:
```bash
uv run --no-sync pytest -q
```
Expected: all tests pass. **If any test fails because of the bump (e.g. in a `stitch`/multi-fault path), STOP and report it — do not fix multi-fault code (out of scope).**

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Bump source_modelling floor to 2026.6.6 for SRF v2 writer" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add the `vs`/`den` velocity-model lookup (critical calculation)

**Files:**
- Modify: `workflow/scripts/realisation_to_srf.py` (add `_velocity_model_vs_den` immediately before `def generate_point_source_srf(` at current line 621)
- Test: `tests/test_realisation_to_srf.py`

**Interfaces:**
- Produces: `_velocity_model_vs_den(velocity_model_df: pd.DataFrame, depths_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning `(vs_cm_s, den_g_cm3)`. Consumed by Task 3. Requires `velocity_model_df` to have columns `depth_km` (layer top depths, km), `Vs` (km/s), `rho` (g/cm³).

- [ ] **Step 1: Add the imports the new tests need**

Ensure the top of `tests/test_realisation_to_srf.py` contains these imports (add the missing ones; keep import groups ordered stdlib / third-party / first-party):
```python
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from workflow import schemas
from workflow.realisations import RuptureVelocity, SRFConfig
from workflow.scripts import realisation_to_srf
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_realisation_to_srf.py`:
```python
def test_velocity_model_vs_den() -> None:
    velocity_model_df = pd.DataFrame(
        {
            "thickness": [3.0, 5.0, 5.0, 5.0, 100.0],
            "Vs": [0.73, 1.57, 2.91, 3.64, 4.18],
            "rho": [1.93, 2.34, 2.76, 3.11, 3.42],
        }
    )
    velocity_model_df["depth_km"] = (
        velocity_model_df["thickness"].cumsum() - velocity_model_df["thickness"]
    )

    # Layer tops: [0, 3, 8, 13, 18] km. vs is cm/s (Vs km/s * 1e5); den is g/cm^3 unchanged.
    # An exact-boundary depth (8.0) takes the deeper layer, matching point_source_slip.
    depths_km = np.array([0.0, 5.0, 8.0, 8.06, 25.0])
    vs, den = realisation_to_srf._velocity_model_vs_den(velocity_model_df, depths_km)

    np.testing.assert_allclose(vs, [0.73e5, 1.57e5, 2.91e5, 2.91e5, 4.18e5])
    np.testing.assert_allclose(den, [1.93, 2.34, 2.76, 2.76, 3.42])
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
uv run --no-sync pytest tests/test_realisation_to_srf.py::test_velocity_model_vs_den -v
```
Expected: FAIL/ERROR — `AttributeError: module 'workflow.scripts.realisation_to_srf' has no attribute '_velocity_model_vs_den'`.

- [ ] **Step 4: Implement the function**

In `workflow/scripts/realisation_to_srf.py`, immediately before `def generate_point_source_srf(` (current line 621), insert:
```python
def _velocity_model_vs_den(
    velocity_model_df: pd.DataFrame, depths_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Look up shear-wave velocity and density at given depths.

    Uses the same layer selection as ``source_modelling.moment.point_source_slip`` (which
    computes this point source's slip): the layer containing the depth, looked up on layer
    *top* depths, with an exact-boundary depth assigned to the deeper layer. Sharing the
    convention keeps a point source's slip and its vs/den on the same layer.

    Parameters
    ----------
    velocity_model_df : pd.DataFrame
        The 1-D velocity model with a ``depth_km`` column of layer *top* depths (as built
        for the ``point_source_slip`` call), plus ``Vs`` (km/s) and ``rho`` (g/cm^3).
    depths_km : np.ndarray
        Point depths, in kilometres.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``vs`` in cm/s and ``den`` in g/cm^3, one value per input depth.
    """
    layer = np.maximum(
        np.searchsorted(velocity_model_df["depth_km"].to_numpy(), depths_km, side="right")
        - 1,
        0,
    )
    return (
        velocity_model_df["Vs"].to_numpy()[layer] * 1e5,
        velocity_model_df["rho"].to_numpy()[layer],
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
uv run --no-sync pytest tests/test_realisation_to_srf.py::test_velocity_model_vs_den -v
```
Expected: PASS.

- [ ] **Step 6: Format, lint, type-check**

Run:
```bash
uv run --no-sync ruff format workflow/scripts/realisation_to_srf.py tests/test_realisation_to_srf.py
uv run --no-sync ruff check workflow/scripts/realisation_to_srf.py tests/test_realisation_to_srf.py
uv run --no-sync ty check --exclude workflow/schemas.py --exclude setup.py
```
Expected: ruff reports no errors; ty reports no new errors.

- [ ] **Step 7: Commit**

```bash
git add workflow/scripts/realisation_to_srf.py tests/test_realisation_to_srf.py
git commit -m "Add velocity-model vs/den lookup for point-source SRF v2" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Rewrite the point-source SRF as v2 and gate it on `srf_version`

**Files:**
- Modify: `workflow/scripts/realisation_to_srf.py` (add `_rewrite_point_source_srf_as_v2` immediately after `_velocity_model_vs_den`; add the version gate at the end of `generate_point_source_srf`, after current line 716)
- Test: `tests/test_realisation_to_srf.py`

**Interfaces:**
- Consumes: `_velocity_model_vs_den` (Task 2); `srf.read_srf` / `srf.write_srf` from `source_modelling` 2026.6.6 (Task 1).
- Produces: `_rewrite_point_source_srf_as_v2(srf_ffp: Path, velocity_model_df: pd.DataFrame) -> None`, and the gate that calls it for `srf_version == "2.0"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_realisation_to_srf.py`:
```python
def test_rewrite_point_source_srf_as_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    points = pd.DataFrame(
        [
            {
                "lon": 172.8, "lat": -43.5, "dep": 8.06, "stk": 64.0, "dip": 58.0,
                "area": 1.0e8, "tinit": 0.0, "dt": 0.02, "rake": 131.0, "slip": 12.5,
                "rise": 0.5,
            }
        ]
    )
    fake_srf = SimpleNamespace(points=points, version="1.0")
    monkeypatch.setattr(realisation_to_srf.srf, "read_srf", lambda _ffp: fake_srf)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        realisation_to_srf.srf,
        "write_srf",
        lambda ffp, srf_file: captured.update(ffp=ffp, srf_file=srf_file),
    )

    velocity_model_df = pd.DataFrame(
        {
            "thickness": [3.0, 5.0, 5.0, 5.0, 100.0],
            "Vs": [0.73, 1.57, 2.91, 3.64, 4.18],
            "rho": [1.93, 2.34, 2.76, 3.11, 3.42],
        }
    )
    velocity_model_df["depth_km"] = (
        velocity_model_df["thickness"].cumsum() - velocity_model_df["thickness"]
    )

    realisation_to_srf._rewrite_point_source_srf_as_v2(Path("unused.srf"), velocity_model_df)

    written = captured["srf_file"]
    assert written.version == "2.0"
    assert list(written.points.columns) == [
        "lon", "lat", "dep", "stk", "dip", "area", "tinit", "dt",
        "vs", "den", "rake", "slip", "rise",
    ]
    assert written.points["vs"].iloc[0] == pytest.approx(2.91e5)
    assert written.points["den"].iloc[0] == pytest.approx(2.76)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
uv run --no-sync pytest tests/test_realisation_to_srf.py::test_rewrite_point_source_srf_as_v2 -v
```
Expected: FAIL/ERROR — `AttributeError: module 'workflow.scripts.realisation_to_srf' has no attribute '_rewrite_point_source_srf_as_v2'`.

- [ ] **Step 3: Implement the rewrite function**

In `workflow/scripts/realisation_to_srf.py`, immediately after `_velocity_model_vs_den` (and before `def generate_point_source_srf(`), insert:
```python
def _rewrite_point_source_srf_as_v2(
    srf_ffp: Path, velocity_model_df: pd.DataFrame
) -> None:
    """Rewrite a version-1.0 SRF in place as version 2.0 with per-point vs and den.

    Parameters
    ----------
    srf_ffp : Path
        Path to the version-1.0 SRF written by ``generic_slip2srf``; overwritten as v2.0.
    velocity_model_df : pd.DataFrame
        The 1-D velocity model (with ``depth_km`` top depths, ``Vs``, ``rho``) used for the
        slip calculation, supplying vs and den by depth.
    """
    srf_file = srf.read_srf(srf_ffp)
    vs, den = _velocity_model_vs_den(velocity_model_df, srf_file.points["dep"].to_numpy())
    dt_index = srf_file.points.columns.get_loc("dt")
    srf_file.points.insert(dt_index + 1, "vs", vs)
    srf_file.points.insert(dt_index + 2, "den", den)
    srf_file.version = "2.0"
    srf.write_srf(srf_ffp, srf_file)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
uv run --no-sync pytest tests/test_realisation_to_srf.py::test_rewrite_point_source_srf_as_v2 -v
```
Expected: PASS.

- [ ] **Step 5: Wire the gate into `generate_point_source_srf`**

In `workflow/scripts/realisation_to_srf.py`, at the **end** of `generate_point_source_srf` (immediately after the existing final line `logger.info("command completed", stderr=proc.stderr.decode("utf-8"))`, current line 716), append:
```python

    if params.srf_config.srf_version == "2.0":
        _rewrite_point_source_srf_as_v2(
            environment.srf_directory / (normalise_name(name) + ".srf"),
            velocity_model_df,
        )
    elif params.srf_config.srf_version != "1.0":
        raise NotImplementedError(
            f"Point sources support SRF versions 1.0 and 2.0, not "
            f"{params.srf_config.srf_version!r}"
        )
```
(`velocity_model_df` is the variable already built earlier in this function for the
`point_source_slip` call, so the lookup uses the identical layer basis as the slip.)

- [ ] **Step 6: Run the focused tests + the full suite**

Run:
```bash
uv run --no-sync pytest tests/test_realisation_to_srf.py -v
uv run --no-sync pytest -q
```
Expected: all pass.

- [ ] **Step 7: Format, lint, type-check**

Run:
```bash
uv run --no-sync ruff format workflow/scripts/realisation_to_srf.py tests/test_realisation_to_srf.py
uv run --no-sync ruff check workflow/scripts/realisation_to_srf.py tests/test_realisation_to_srf.py
uv run --no-sync ty check --exclude workflow/schemas.py --exclude setup.py
```
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add workflow/scripts/realisation_to_srf.py tests/test_realisation_to_srf.py
git commit -m "Write SRF v2 (vs/den) for point sources" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: End-to-end verification with the native binaries (no commit)

Confirms the real pipeline produces a valid v2 SRF whose `vs`/`den` match the velocity model at the source depth. Uses a scratch directory; nothing in the repo or example data is modified.

**Files:** none modified (verification only).

- [ ] **Step 1: Generate a point-source realisation**

Run (network is available; `2012p001887` is a known GeoNet event sitting at exactly 8.0 km, a layer boundary in the default model — a good consistency check):
```bash
cd /home/arr65/src/workflow
SC=$(mktemp -d)
uv run --no-sync gcmt-to-realisation 2012p001887 24.2.2.4 "$SC/ps.json" point-source
```
Expected: `ps.json` written (a `UserWarning` about Leonard width is expected, not an error).

- [ ] **Step 2: Run the point-source SRF generation (default `srf_version` is 2.0)**

Run:
```bash
uv run --no-sync realisation-to-srf "$SC/ps.json" "$SC/ps.srf" \
  --work-directory "$SC/work" \
  --genslip-path /home/arr65/src/EMOD3D/tools/genslip_v5.4.2 \
  --generic-slip2srf-path /home/arr65/src/EMOD3D/tools/generic_slip2srf
```
Expected: exit 0; `ps.srf` written.

- [ ] **Step 3: Confirm the SRF is v2 with vs/den, and the values match the model layer**

Run:
```bash
uv run --no-sync python - "$SC/ps.srf" "$SC/work/velocity_model" <<'PY'
import sys
import numpy as np
from source_modelling import srf
s = srf.read_srf(sys.argv[1])
print("version:", s.version)
print("columns:", list(s.points.columns))
print(s.points[["dep", "vs", "den"]])
# Independently look up the expected layer value at the point depth.
import pandas as pd
vm = pd.read_csv(sys.argv[2], sep=r"\s+", skiprows=1, header=None,
                 names=["thickness", "Vp", "Vs", "rho", "Qp", "Qs"])
vm["top"] = vm["thickness"].cumsum() - vm["thickness"]
dep = float(s.points["dep"].iloc[0])
layer = max(0, np.searchsorted(vm["top"].to_numpy(), dep, side="right") - 1)
print("expected vs (cm/s):", vm["Vs"].iloc[layer] * 1e5, "expected den:", vm["rho"].iloc[layer])
PY
```
Expected: `version: 2.0`; `columns` include `vs` and `den` between `dt` and `rake`; the printed `vs`/`den` equal `expected vs`/`expected den`.

- [ ] **Step 4: Record the result**

Note the observed `version`, `dep`, `vs`, `den` and confirm they match the independent layer lookup. No commit.

---

## Self-Review

**Spec coverage:**
- §3 / §4.1 gate on `srf_version` → Task 3 Step 5. ✓
- §4.2 `_velocity_model_vs_den` (units + layer rule) → Task 2. ✓
- §4.3 `_rewrite_point_source_srf_as_v2` (read/insert/version/write together) → Task 3 Steps 3–4. ✓
- §5 dependency bump + suite green → Task 1. ✓
- §6 tests (critical calc; light expected-output; suite green) → Task 2 Step 2, Task 3 Step 1, Tasks 1/3 suite runs. ✓
- §6 boundary expectation (deeper layer) → Task 2 test, depth `8.0` → `2.91e5`. ✓
- §7 manual e2e verification → Task 4. ✓
- §8 out-of-scope (no multi-fault edits; reject v3.0) → Global Constraints + Task 3 `NotImplementedError`. ✓

**Placeholder scan:** No TBD/TODO; every code/command step shows full content. ✓

**Type consistency:** `_velocity_model_vs_den(velocity_model_df, depths_km) -> (np.ndarray, np.ndarray)` and `_rewrite_point_source_srf_as_v2(srf_ffp, velocity_model_df) -> None` are used identically in Task 3 and the gate. Column order asserted in Task 3 matches the v2 layout in spec §2.5. ✓
