# Point-Source SRF v2 — Design

Date: 2026-06-24
Branch: `point-source-srf-v2`
Author: Andrew Ridden-Harper (with Claude)

## 1. Problem

`realisation_to_srf.py` recently gained SRF **version 2.0** support for finite-fault
sources (via `genslip`). Point sources, which use the `generic_slip2srf` binary, still
only ever produce **version 1.0** SRFs. We need point sources to also produce v2 SRFs.

SRF v2 differs from v1 only by adding, to each rupture point, a shear-wave velocity
`vs` and a density `den`.

## 2. Key facts established by verification (not assumption)

All of the following were verified empirically against the native binaries
(`/home/arr65/src/EMOD3D/tools/{generic_slip2srf,genslip_v5.4.2}`) and, where noted,
against the C source. The container `runner.sif` cannot be launched in this environment,
but the native binaries are byte-faithful builds of the same source.

1. **`generic_slip2srf version=2.0` is broken — there are no "junk vs/den" to replace.**
   It never initialises the V2-only struct fields (`nseg`, `np_seg`, `srf_hcmnt`), so
   `write_srf2`'s ASCII path writes **only the header and zero rupture points** (empirically
   a 14-line, point-less file) or segfaults (uninitialised-stack dependent). Therefore the
   binary's own v2 path must **not** be used. Instead: keep producing v1 (which works
   perfectly), then add `vs`/`den` in Python. The C source is `generic_slip2srf.c` /
   `srf_subs.c` under `…/StandRupFormat/`; we do not modify it.

2. **A point source is exactly one point** at the GCMT centroid depth (`nstk=ndip=1`).
   Confirmed end-to-end: GeoNet `2012p001887` → `POINTS 1`, `dep = 8.0 km`. `genslip` is
   never invoked for point sources.

3. **Units in the SRF v2 file** (verified two independent ways + C source
   `gslip_srf_subs.c:720-721`):
   - `vs` is in **cm/s** = `Vs_model[km/s] × 1e5`.
   - `den` is in **g/cm³** = `rho_model` unchanged (note the asymmetry: vs is scaled, den is not).

4. **Depth → layer rule.** Two conventions exist, differing *only* for a depth that lands
   exactly on a layer boundary:
   - `source_modelling.moment.point_source_slip` — which already computes this point
     source's **slip** — uses layer **top-depths** with `searchsorted(side="right") - 1`:
     the layer that *contains* the depth, sending an exact-boundary depth to the **deeper**
     layer. This was deliberately fixed and documented in source_modelling **PR #60
     (tag v2026.05.1**, present in 2026.6.2 and 2026.6.6); the fix replaced a buggy
     `argmin(|top − depth|)` "closest top" lookup and added a guard that raises
     *"are you using bottom depth instead of top depth?"* if the model does not start at 0 km.
   - `genslip` (finite-fault vs/den) uses cumulative **bottom-depths** with a strict `>`
     comparison, sending an exact-boundary depth to the **shallower** layer
     (`ruptime.c:load_vsden`, `gslip_srf_subs.c:709-711`).

   **Decision: use `point_source_slip`'s convention for vs/den** (see §4.2), so a point
   source's slip and its vs/den always resolve to the *same* layer. This matters in
   practice — it is **not** measure-zero for point sources: GCMT centroid depths frequently
   land exactly on a model boundary (the baseline event `2012p001887` sits at *exactly*
   8.0 km, a boundary in the default 24.2.2.4 velocity model). genslip's opposite tiebreak
   is not meaningfully reproducible here: genslip never processes point sources, and its
   finite-fault subfault depths are internally high-precision and essentially never coincide
   with a boundary, so there is no real finite-fault "vs/den on a boundary" to match.

5. **`source_modelling==2026.6.6` can read v1 and write v2 ASCII** (the installed `2026.6.2`
   cannot — its `write_srf` hardcodes `"1.0"`). The v2 `points` DataFrame column order is
   `lon, lat, dep, stk, dip, area, tinit, dt, vs, den, rake, slip, rise`; the Rust writer
   infers v1 vs v2 from the array width. **Gotcha:** setting `version="2.0"` without
   inserting `vs`/`den` (or vice-versa) corrupts the file *silently* — they must be changed
   together. Round-tripping is value-faithful for our data (the v1 SRF is already
   float32-derived).

## 3. Design overview

Change **only** `generate_point_source_srf` in
`workflow/scripts/realisation_to_srf.py`, plus a dependency bump. After
`generic_slip2srf` writes the v1 SRF, if `srf_config.srf_version == "2.0"`, rewrite the
file as v2 by looking up `vs`/`den` per point from the **same** `velocity_model_df` already
built for the slip calculation, and re-emitting with `source_modelling`'s v2 writer.

No change to any multi-fault / `stitch_srf_files` / rupture-combination code.

## 4. Detailed design

### 4.1 Gate at the end of `generate_point_source_srf`

`velocity_model_df` (with a `depth_km` top-depth column) is already built at current lines
656-659 and passed to `point_source_slip`. After the existing `subprocess.run(...)` block
(current line 716):

```python
if params.srf_config.srf_version == "2.0":
    _rewrite_point_source_srf_as_v2(
        environment.srf_directory / (normalise_name(name) + ".srf"),
        velocity_model_df,
    )
elif params.srf_config.srf_version != "1.0":
    raise NotImplementedError(
        f"Point sources support SRF versions 1.0 and 2.0, not {params.srf_config.srf_version!r}"
    )
```

(The v1.0 case is the existing, unchanged behaviour. SRF 3.0 — Vp + moment tensor — is out
of scope and rejected explicitly rather than silently mis-emitted.)

### 4.2 Critical calculation — `_velocity_model_vs_den`

A pure, vectorised lookup that mirrors `point_source_slip`'s layer selection and applies the
verified units. This is the only "critical calculation" and is the focus of unit testing.

```python
def _velocity_model_vs_den(
    velocity_model_df: pd.DataFrame, depths_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Look up shear-wave velocity and density at given depths.

    Uses the same layer selection as ``source_modelling.moment.point_source_slip`` (which
    computes this point source's slip): the layer that contains the depth, looked up on
    layer *top* depths, with an exact-boundary depth assigned to the deeper layer. Sharing
    the convention keeps a point source's slip and its vs/den on the same layer.

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

### 4.3 I/O wrapper — `_rewrite_point_source_srf_as_v2`

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

Keying the lookup on the SRF's own `dep` column (rather than `source_depth_km`) keeps it
correct if a point source ever produces more than one point, and keeps it self-consistent
with the depths actually written into the SRF. `version` and the `vs`/`den` columns are set
together to avoid the silent-corruption gotcha. No new imports are required (`np`, `pd`,
`srf`, `Path` are already imported).

## 5. Dependency change

- `pyproject.toml`: floor `source-modelling` at `2026.6.6` (currently effectively `2026.6.2`).
- Regenerate `uv.lock`.

**Repo-wide effect to check (not change):** bumping `source_modelling` also swaps the SRF
reader/writer used by `stitch_srf_files`. Per scope, that code is left untouched; we will
run the existing test suite after the bump and flag (not fix) any multi-fault interaction.

## 6. Testing

Per project guidance: test only the critical calculation and the expected output; do not
test the well-established `source_modelling` library itself.

1. **`test_velocity_model_vs_den` (critical calculation).** Build a velocity-model
   DataFrame (with a `depth_km` top-depth column) whose layer Vs/rho values were exercised
   in the genslip verification runs (e.g. `Vs = [0.73, 1.57, 2.91, …]` km/s,
   `rho = [1.93, 2.34, 2.76, …]` g/cm³, layer tops at 0, 3, 8, 13, … km). Assert:
   - **units:** `vs = Vs × 1e5` cm/s (e.g. depth 8.06 → `2.91e5`) and `den = rho` g/cm³
     (these interior values equal what genslip actually wrote, so the test is a genuine
     regression guard on the unit factors);
   - a mid-layer depth resolves to the correct containing layer in each of several layers;
   - **boundary case (Option A):** a depth exactly on a layer top (e.g. 8.0 km, the top of
     the layer whose `Vs = 2.91`) resolves to that **deeper** layer (`2.91e5`) — i.e. the
     same layer `point_source_slip` would pick — not the shallower one;
   - depth 0 → first layer; depth below the last layer top → last layer.

2. **`test_point_source_srf_v2_output` (expected output, light).** Given a small v1 SRF and
   a known velocity model, run `_rewrite_point_source_srf_as_v2` and assert the rewritten
   file has `version == "2.0"` and the point's `vs`/`den` equal the expected lookup values.
   Kept minimal — it confirms our wrapper wires version + columns together correctly, not
   that the library parses SRFs.

3. **Existing suite still green after the dependency bump** (`uv run pytest`), as a guard on
   the multi-fault read/write interaction.

## 7. Development verification (manual, not CI)

- Run the point path end-to-end with `srf_version=2.0` using the native binaries; read the
  result back with `source_modelling 2026.6.6` and confirm it is v2 with `vs`/`den` matching
  the velocity-model layer at the source depth (the baseline event sits at the 8.0 km
  boundary, a useful check that slip and vs/den agree on the layer).
- Confirm `_velocity_model_vs_den` reproduces the genslip v2 `(depth → vs, den)` tables for
  all interior depths across every layer (units + interior layer selection).

## 8. Out of scope

- Multi-fault `stitch_srf_files` (it hardcodes `version="1.0"` and would drop `vs`/`den`).
- SRF 3.0 (Vp + moment tensor).
- Any change to `generic_slip2srf.c` / `genslip` / the C sources.
- Changing `point_source_slip` (we reuse its existing, fixed convention).

## 9. Risks / open items

- **source_modelling bump vs. multi-fault stitch.** Mitigation: run the existing suite;
  surface any breakage rather than editing stitch.
- **Boundary convention.** Resolved: we reuse `point_source_slip`'s convention so slip and
  vs/den share a layer; this is the maintained, guarded convention and the right choice
  given point-source depths really do land on boundaries. Documented in §2.4.
