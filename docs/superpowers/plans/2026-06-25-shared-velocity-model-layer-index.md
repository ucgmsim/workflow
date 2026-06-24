# Shared Velocity-Model Layer Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `point_source_slip` and the workflow's `_velocity_model_vs_den` select the velocity-model layer through one shared `source_modelling.moment.velocity_model_layer_index`, so the depth→layer convention can never drift.

**Architecture:** Extract the 0-km guard + `searchsorted(side="right") - 1` + clamp into `velocity_model_layer_index` in `source_modelling/moment.py` (overloaded: scalar→`np.intp`, array→`npt.NDArray[np.intp]`). `point_source_slip` calls it (behaviour unchanged). The workflow's `_velocity_model_vs_den` calls `moment.velocity_model_layer_index` and drops its duplicated lookup + precondition comment. Two repos, two commits; the workflow side is validated against a locally-installed `source_modelling`.

**Tech Stack:** Python, numpy/pandas, `source_modelling`, `uv`, pytest, ruff, ty.

## Global Constraints

- The shared function `velocity_model_layer_index(velocity_model_df, depths_km)` lives in `source_modelling/moment.py`, beside `point_source_slip`. It owns: the 0-km guard (`ValueError`, message `"Velocity model does not begin at 0km depth (are you using bottom depth instead of top depth)?"`) and the lookup `np.maximum(np.searchsorted(depth_km, depths, side="right") - 1, 0)`. It is overloaded scalar→`np.intp`, array→`npt.NDArray[np.intp]`, matching the file's existing `@typing.overload ... -> T: ...  # numpydoc ignore=GL08` style (docstring only on the implementation).
- `point_source_slip` behaviour is unchanged (the slip arithmetic is untouched); its existing tests must still pass.
- The workflow's `_velocity_model_vs_den` uses `moment.velocity_model_layer_index` (qualified call — `moment` is already imported), drops the duplicated lookup and the 0-km precondition comment, and keeps `vs = Vs×1e5` (cm/s) / `den = rho` (g/cm³).
- The `source_modelling` change lands on the existing `point-source-srf-v2` branch in `/home/arr65/src/source_modelling` (already created and checked out). The workflow change is validated by `uv pip install /home/arr65/src/source_modelling` into the workflow venv, then `uv run --no-sync` (so uv does not re-sync back to the locked 2026.6.6).
- The workflow `pyproject.toml` `source_modelling` floor stays `>=2026.6.6` with a PENDING comment (bump to the release containing `velocity_model_layer_index` is a maintainer action, out of scope).
- numpydoc docstrings; run `ruff format` + `ruff check` + `ty` before each commit. Commit messages: short imperative subject (no body) + trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

- **`source_modelling` repo** (`/home/arr65/src/source_modelling`): `source_modelling/moment.py` (add the function + refactor `point_source_slip`), `tests/test_moment.py` (add two tests).
- **`workflow` repo** (`/home/arr65/src/workflow`): `workflow/scripts/realisation_to_srf.py` (refactor `_velocity_model_vs_den`), `pyproject.toml` (PENDING comment). The existing `tests/test_realisation_to_srf.py::test_velocity_model_vs_den` is unchanged and must still pass.

---

### Task 1: Extract `velocity_model_layer_index` in `source_modelling` (branch `point-source-srf-v2` in `/home/arr65/src/source_modelling`)

**Files:**
- Modify: `/home/arr65/src/source_modelling/source_modelling/moment.py` (insert function immediately before `def point_source_slip(` at line 307; refactor `point_source_slip` body lines 338-352)
- Test: `/home/arr65/src/source_modelling/tests/test_moment.py`

**Interfaces:**
- Produces: `velocity_model_layer_index(velocity_model_df: pd.DataFrame, depths_km: float | npt.NDArray[np.floating]) -> np.intp | npt.NDArray[np.intp]` — scalar in→`np.intp`, array in→`npt.NDArray[np.intp]`. Consumed by `point_source_slip` (this task) and the workflow (Task 2).

- [ ] **Step 1: Ensure you're on the `point-source-srf-v2` branch in the source_modelling repo**

```bash
git -C /home/arr65/src/source_modelling checkout point-source-srf-v2
```
(This branch already exists and is current — a no-op confirmation.)

- [ ] **Step 2: Write the failing tests** in `/home/arr65/src/source_modelling/tests/test_moment.py` (append; `np`, `pd`, `pytest`, `moment` are already imported there)

```python
def test_velocity_model_layer_index():
    """Deepest layer whose top <= depth; boundary -> deeper; clamp at ends."""
    vm = pd.DataFrame({"depth_km": [0.0, 1.0, 2.0]})

    assert moment.velocity_model_layer_index(vm, 0.0) == 0
    assert moment.velocity_model_layer_index(vm, 0.5) == 0
    assert moment.velocity_model_layer_index(vm, 1.0) == 1  # boundary -> deeper layer
    assert moment.velocity_model_layer_index(vm, 1.5) == 1
    assert moment.velocity_model_layer_index(vm, 5.0) == 2  # below last top -> last layer

    np.testing.assert_array_equal(
        moment.velocity_model_layer_index(vm, np.array([0.0, 1.0, 1.5, 2.0, 5.0])),
        np.array([0, 1, 1, 2, 2]),
    )


def test_velocity_model_layer_index_top_depth():
    """A velocity model not beginning at 0 km depth raises."""
    bad_vm = pd.DataFrame({"depth_km": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Velocity model does not begin at 0km depth"):
        moment.velocity_model_layer_index(bad_vm, 1.0)
```

- [ ] **Step 3: Run the new tests to verify they fail**

```bash
cd /home/arr65/src/source_modelling && uv run --no-sync pytest tests/test_moment.py -k velocity_model_layer_index -v
```
Expected: FAIL/ERROR — `AttributeError: module 'source_modelling.moment' has no attribute 'velocity_model_layer_index'`.

- [ ] **Step 4: Add `velocity_model_layer_index`** to `source_modelling/moment.py`, immediately before `def point_source_slip(` (line 307)

```python
@typing.overload
def velocity_model_layer_index(
    velocity_model_df: pd.DataFrame, depths_km: float
) -> np.intp: ...  # numpydoc ignore=GL08
@typing.overload
def velocity_model_layer_index(
    velocity_model_df: pd.DataFrame, depths_km: npt.NDArray[np.floating]
) -> npt.NDArray[np.intp]: ...  # numpydoc ignore=GL08
def velocity_model_layer_index(
    velocity_model_df: pd.DataFrame, depths_km: float | npt.NDArray[np.floating]
) -> np.intp | npt.NDArray[np.intp]:
    """Return the velocity-model layer index containing each depth.

    Selects the deepest layer whose top depth does not exceed the query depth (a depth
    exactly on a layer boundary takes the deeper layer); a depth above the first layer top
    clamps to layer 0. Scalar in -> scalar out; array in -> array out.

    Parameters
    ----------
    velocity_model_df : pd.DataFrame
        Velocity model with a ``depth_km`` column of layer *top* depths in kilometres, the
        first of which must be 0.
    depths_km : float or npt.NDArray[np.floating]
        Query depth(s) in kilometres.

    Returns
    -------
    np.intp or npt.NDArray[np.intp]
        The layer index for each query depth.

    Raises
    ------
    ValueError
        If the velocity model does not begin at 0 km depth (a sign that bottom depths were
        passed instead of top depths).
    """
    if not np.isclose(velocity_model_df["depth_km"].iloc[0], 0.0):
        raise ValueError(
            "Velocity model does not begin at 0km depth (are you using bottom depth instead of top depth)?"
        )
    return np.maximum(
        np.searchsorted(
            velocity_model_df["depth_km"].to_numpy(), depths_km, side="right"
        )
        - 1,
        0,
    )
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
cd /home/arr65/src/source_modelling && uv run --no-sync pytest tests/test_moment.py -k velocity_model_layer_index -v
```
Expected: 2 passed.

- [ ] **Step 6: Refactor `point_source_slip`** — replace its guard + index block (`moment.py` lines 338-352, i.e. the `# While this is not strictly necessary...` comment through `idx = max(0, idx)`) with a single call. The lines below it (`vs_km_per_s = velocity_model_df.iloc[idx]["Vs"]` onward) and the slip arithmetic stay exactly as they are.

Replace:
```python
    # While this is not strictly necessary, it does act as a sanity check to
    # ensure that the bug does not reoccur in the future.
    if not np.isclose(velocity_model_df["depth_km"].iloc[0], 0.0):
        raise ValueError(
            "Velocity model does not begin at 0km depth (are you using bottom depth instead of top depth)?"
        )
    # Finds the first index i in the velocity model such that depth[i] <= source depth < depth[i + 1]
    # At a boundary therefore, it returns the bottom-most layer index instead of the top.
    idx = (
        np.searchsorted(
            velocity_model_df["depth_km"].to_numpy(), source_depth_km, side="right"
        )
        - 1
    )
    idx = max(0, idx)
```
with:
```python
    idx = velocity_model_layer_index(velocity_model_df, source_depth_km)
```

- [ ] **Step 7: Run the full moment test suite (regression guard on the refactor)**

```bash
cd /home/arr65/src/source_modelling && uv run --no-sync pytest tests/test_moment.py -v
```
Expected: all pass — including `test_point_source_slip_top_depth`, `_simple`, `_middle`, `_boundary`, `_bad_dataframe` (behaviour unchanged), plus the two new tests.

- [ ] **Step 8: Format, lint, type-check**

```bash
cd /home/arr65/src/source_modelling
uv run --no-sync ruff format source_modelling/moment.py tests/test_moment.py
uv run --no-sync ruff check source_modelling/moment.py tests/test_moment.py
uv run --no-sync ty check
```
Expected: ruff clean; `ty` reports no new errors in `moment.py`/`test_moment.py`. (If `ty` is not in the env, run `uv run ty check`.)

- [ ] **Step 9: Commit (in the source_modelling repo)**

```bash
git -C /home/arr65/src/source_modelling add source_modelling/moment.py tests/test_moment.py
git -C /home/arr65/src/source_modelling commit -m "Extract shared velocity_model_layer_index from point_source_slip" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Use the shared function in the workflow (in `/home/arr65/src/workflow`)

**Files:**
- Modify: `workflow/scripts/realisation_to_srf.py` (replace the `_velocity_model_vs_den` function)
- Modify: `pyproject.toml` (PENDING comment by the `source_modelling` floor)

**Interfaces:**
- Consumes: `moment.velocity_model_layer_index` from Task 1 (`moment` already imported in `realisation_to_srf.py`).

- [ ] **Step 1: Install the local (Task 1) source_modelling into the workflow venv**

```bash
cd /home/arr65/src/workflow
uv pip install /home/arr65/src/source_modelling
uv run --no-sync python -c "from source_modelling import moment; print('has fn:', hasattr(moment, 'velocity_model_layer_index'))"
```
Expected: `has fn: True`. (From here on use `uv run --no-sync` for everything so uv does not revert the venv to the locked 2026.6.6.)

- [ ] **Step 2: Replace the `_velocity_model_vs_den` function** in `workflow/scripts/realisation_to_srf.py` (the whole function — docstring, the `# depth_km must be top depths...` comment, and the `np.maximum(np.searchsorted(...))` block) with:

```python
def _velocity_model_vs_den(
    velocity_model_df: pd.DataFrame, depths_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Look up shear-wave velocity (cm/s) and density (g/cm^3) at given depths.

    The layer for each depth is chosen by ``moment.velocity_model_layer_index`` (shared with
    ``point_source_slip``); vs is converted km/s -> cm/s and den passes through g/cm^3.

    Parameters
    ----------
    velocity_model_df : pd.DataFrame
        The 1-D velocity model with a ``depth_km`` column of layer top depths, plus ``Vs``
        (km/s) and ``rho`` (g/cm^3).
    depths_km : np.ndarray
        Point depths, in kilometres.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``vs`` in cm/s and ``den`` in g/cm^3, one value per input depth.
    """
    layer = moment.velocity_model_layer_index(velocity_model_df, depths_km)
    return (
        velocity_model_df["Vs"].to_numpy()[layer] * 1e5,
        velocity_model_df["rho"].to_numpy()[layer],
    )
```

- [ ] **Step 3: Run the workflow tests to verify they still pass**

```bash
cd /home/arr65/src/workflow
uv run --no-sync pytest tests/test_realisation_to_srf.py -v
uv run --no-sync pytest -q
```
Expected: `test_velocity_model_vs_den` and `test_rewrite_point_source_srf_as_v2` pass (their velocity-model DataFrames start at depth 0, so the shared guard is satisfied), and the full suite stays green (117 passed). The behaviour is identical, so no test changes are needed.

- [ ] **Step 4: Add the PENDING floor marker** in `pyproject.toml` — insert a comment line immediately above the `"source_modelling>=2026.6.6",` entry (line 19):

```
  # PENDING: bump to the source_modelling release containing moment.velocity_model_layer_index
  "source_modelling>=2026.6.6",
```

- [ ] **Step 5: Format, lint, type-check**

```bash
cd /home/arr65/src/workflow
uv run --no-sync ruff format workflow/scripts/realisation_to_srf.py
uv run --no-sync ruff check workflow/scripts/realisation_to_srf.py
uv tool run ty check --exclude workflow/schemas.py --exclude setup.py
```
Expected: ruff clean; zero `ty` errors in `realisation_to_srf.py` (pre-existing errors elsewhere are fine).

- [ ] **Step 6: Commit (in the workflow repo)**

```bash
cd /home/arr65/src/workflow
git add workflow/scripts/realisation_to_srf.py pyproject.toml
git commit -m "Use shared moment.velocity_model_layer_index for point-source vs/den" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §2 shared function → Task 1 Steps 4 (impl) + 2/5 (tests). ✓
- §3 `point_source_slip` refactor (behaviour unchanged) → Task 1 Step 6 + Step 7 regression run. ✓
- §4 workflow `_velocity_model_vs_den` refactor + comment removal → Task 2 Step 2. ✓
- §5 coordination (branch, local install, `--no-sync`, PENDING floor) → Task 1 Step 1, Task 2 Steps 1 + 4 + Global Constraints. ✓
- §6 testing (new source_modelling tests; existing point_source_slip + workflow tests still pass) → Task 1 Steps 2/7, Task 2 Step 3. ✓

**Placeholder scan:** No TBD/TODO; the only "PENDING" is the deliberate, real coordination marker. All code/commands are complete. ✓

**Type consistency:** `velocity_model_layer_index(velocity_model_df, depths_km)` overload signatures (scalar→`np.intp`, array→`npt.NDArray[np.intp]`) are used consistently: `point_source_slip` passes the scalar `source_depth_km` (→`np.intp`, valid for `iloc`), the workflow passes the array `depths_km` (→array, valid for fancy indexing). ✓
