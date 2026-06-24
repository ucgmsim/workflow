# Shared Velocity-Model Layer Index — Design

Date: 2026-06-25
Branch: `point-source-srf-v2` (workflow); a new branch in `/home/arr65/src/source_modelling`
Author: Andrew Ridden-Harper (with Claude)

## 1. Problem

Two places select a velocity-model layer from a depth, using the *same* convention but as *separate copies* of the logic:

- `source_modelling.moment.point_source_slip` (`moment.py:340-352`) — for the point source's slip.
- `workflow.scripts.realisation_to_srf._velocity_model_vs_den` — for the point source's SRF v2 `vs`/`den`.

They must stay identical (a divergence would put a point's slip and its `vs`/`den` on different layers). Today nothing enforces that — a future edit to one could silently diverge. Goal: make both call **one** function so they cannot drift.

## 2. The shared function (`source_modelling.moment.velocity_model_layer_index`)

Add to `source_modelling/moment.py`, next to `point_source_slip`:

```python
def velocity_model_layer_index(
    velocity_model_df: pd.DataFrame, depths_km: float | np.ndarray
) -> np.intp | np.ndarray:
    """Return the velocity-model layer index containing each depth.

    Selects the deepest layer whose top depth does not exceed the query depth (a depth
    exactly on a layer boundary takes the deeper layer); a depth above the first layer top
    clamps to layer 0. Scalar in -> scalar out; array in -> array out.

    Parameters
    ----------
    velocity_model_df : pd.DataFrame
        Velocity model with a ``depth_km`` column of layer *top* depths in kilometres, the
        first of which must be 0.
    depths_km : float or np.ndarray
        Query depth(s) in kilometres.

    Returns
    -------
    np.intp or np.ndarray
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
        np.searchsorted(velocity_model_df["depth_km"].to_numpy(), depths_km, side="right")
        - 1,
        0,
    )
```

This owns the whole convention: the **0-km guard** and the **`searchsorted(side="right") - 1` + clamp**. `np.searchsorted`/`np.maximum` return a scalar for a scalar `depths_km` and an array for an array, so both call sites get the index type they expect.

## 3. `point_source_slip` refactor (behaviour unchanged)

Replace the guard + index block (`moment.py:338-352`) with a single call; the slip arithmetic is untouched:

```python
    idx = velocity_model_layer_index(velocity_model_df, source_depth_km)
    vs_km_per_s = velocity_model_df.iloc[idx]["Vs"]
    rho_g_per_cm3 = velocity_model_df.iloc[idx]["rho"]
    ...  # unchanged unit math
```

`idx` is now `np.intp` (was a Python `int` from `max(0, idx)`); `df.iloc[np.intp]` behaves identically. Existing `point_source_slip` tests must still pass.

## 4. Workflow refactor (`_velocity_model_vs_den`)

`moment` is already imported in `realisation_to_srf.py`. Use the qualified call (consistent with repo style of preferring `module.symbol` when the module is already imported), and **remove the 0-km precondition comment** (commit `59d1114`) — the guard now travels with the function:

```python
def _velocity_model_vs_den(
    velocity_model_df: pd.DataFrame, depths_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Look up shear-wave velocity (cm/s) and density (g/cm^3) at given depths.

    The layer for each depth is chosen by ``moment.velocity_model_layer_index`` (shared with
    ``point_source_slip``); vs is converted km/s -> cm/s, den passes through g/cm^3.

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

## 5. Cross-repo coordination (approach 1)

`source_modelling`'s version is dynamic from git tags; the current floor `2026.6.6` will **not** contain `velocity_model_layer_index`. So:

- The `source_modelling` change lands on the existing `point-source-srf-v2` branch in `/home/arr65/src/source_modelling` (already created and checked out; based on `main`, 5 commits ahead of `v2026.06.6`).
- The workflow change is validated locally by installing the local `source_modelling` into the workflow venv: `uv pip install /home/arr65/src/source_modelling`, then `uv run --no-sync ...` so uv does not re-sync back to the locked `2026.6.6` mid-work.
- The workflow `pyproject.toml` floor stays `>=2026.6.6` with a **PENDING** marker: it must be bumped to the `source_modelling` release that contains `velocity_model_layer_index` before this workflow change is merged/published. (Same coordination pattern as the `nshmdb` floor in the current campaign.) Cutting that release is a maintainer action, out of scope here.

## 6. Testing

- **`source_modelling`**: add `test_velocity_model_layer_index` (in `tests/test_moment.py`) covering the convention — mid-layer selection in several layers, an exact-boundary depth → the deeper layer, the surface (depth 0 → layer 0), scalar vs array input returning the matching shape, and the 0-km guard raising `ValueError`. Existing `point_source_slip` tests must still pass (behaviour unchanged).
- **workflow**: the existing `test_velocity_model_vs_den` must still pass unchanged (it now exercises the shared function and asserts the workflow's units/values — the workflow's own responsibility, not a re-test of the library). The full workflow suite must stay green against the locally-installed `source_modelling`.

## 7. Out of scope / non-goals

- No behaviour change to `point_source_slip` or to the produced SRFs.
- No `source_modelling` release (maintainer action) and no final workflow floor bump (PENDING until release).
- No new module — the function lives in `moment.py` beside `point_source_slip`.

## 8. Risks

- Until `source_modelling` is released with the function and the workflow floor is bumped, the workflow change is not mergeable/publishable (it imports a symbol absent from `2026.6.6`). Mitigated by the local path-install for development and the explicit PENDING marker.
