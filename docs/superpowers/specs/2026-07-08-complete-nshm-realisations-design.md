# Complete NSHM-2022 minimal realisations to the 24.2.2.1 simulation spec

- **Date:** 2026-07-08
- **Status:** Approved (design)
- **Author:** Andrew Ridden-Harper (with Claude)
- **Branch:** `andrew-cs-2022`

## Context

We are preparing a large earthquake-simulation campaign. Each scenario is
defined by a realisation JSON file. We currently have minimal ("stub")
realisation files produced by `generate_realisations_from_csv.py` →
`nshm2022-to-realisation` for a set of NSHM-2022 rupture IDs. These stubs
contain only the source-side sections and rely on scientific defaults for
everything else.

A researcher (Felipe) produced a fully-populated reference realisation
(`felipe_3528839_realisation.json`) for historical (GCMT) earthquakes using a
custom script (`felipe_scripts/gen_FF_realisations_MP.py`) plus three input
data files. We want our NSHM stubs to become **complete, self-contained
realisations that match Felipe's reference** — every section materialised — so
the files can be **fully scrutinised before any HPC time is spent**.

## Objective

For each valid minimal NSHM realisation, produce a complete realisation file,
identical in structure and parameter values to Felipe's reference, written to a
**new folder** so the existing minimal files are never overwritten.

## Key findings (verified during design)

1. **`nshm2022-to-realisation` vs `gcmt-to-realisation`.** Our stubs already
   contain `srf` and `seeds` (persisted as a side-effect of
   `read_from_realisation_or_defaults` / `read_from_realisation_or_random`),
   whereas Felipe's GCMT stubs do not. Felipe's script's GCMT/nodal-plane source
   generation is **not needed** for us — we already have NSHM sources.

2. **What Felipe's script actually customises** beyond defaults:
   `velocity_model` (`version` 2.07→**2.09**; `rrup_interpolants` from
   `Mw_rrup_mod.txt`), `im` (`valid_periods` from `periods.csv`,
   `fas_frequencies` from `frequencies.csv`), and a per-rupture computed
   `domain`. Everything else in his full file comes from the scientific
   defaults, materialised by later workflow stages.

3. **Defaults version.** The only differences between `24.2.2.4` (our stubs) and
   `24.2.2.1` (Felipe) are `resolution` **0.4→0.1** (400 m → 100 m) and
   `bb.flo` **0.25→1.0**. In a *minimal* file the only version-dependent field
   is `metadata.defaults_version` itself (sources/magnitudes/rakes/rupture tree
   come from the NSHM DB + persisted seeds; `srf` is version-independent root
   defaults). Therefore setting `metadata.defaults_version = 24.2.2.1` in a copy
   is **provably equivalent** to regenerating at 24.2.2.1 with the same seeds,
   and it **preserves the exact curated ruptures** (regeneration would
   re-randomise — there is no seed CLI option).

4. **Consistency with Felipe is exact.** Materialising every section from the
   *current* 24.2.2.1 defaults (plus the `velocity_model`/`im` overrides above)
   reproduces Felipe's `felipe_3528839_realisation.json` **byte-for-byte** for
   all shared sections. Verified by direct diff:
   `velocity_model`, `im`, `emod3d`, `hf`, `bb`, `resolution`, `srf`,
   `velocity_model_1d`, `hf_velocity_model_1d` all identical; our existing
   minimal `srf` is identical to his `srf`.

5. **`rupture_velocity`** is present in the current 24.2.2.1 defaults but absent
   from Felipe's file (his predates it). It **is consumed** by
   `realisation_to_srf.py` and `hf_sim.py`, so we include it — giving an
   18-section file (a superset of Felipe's 17).

## Inputs

- `realisations_from_nshm2022_to_realisation/` — 293 minimal files:
  **283 valid**, **10 broken stubs** (only `metadata/srf/seeds`; these are the
  10 `nshm2022-to-realisation` failures recorded in `error_log.txt`).
- Felipe's override files, reused as-is:
  `felipe_scripts/{Mw_rrup_mod.txt, periods.csv, frequencies.csv}`.
- Scientific defaults version **24.2.2.1** (via `workflow.defaults`).
- No `nshmdb.db` required — we reuse the existing sources.

## Output

- New folder (default `realisations_completed_24.2.2.1/`), one complete realisation
  per successfully-completed rupture, each with these **18 sections**:
  `metadata, sources, rupture_propagation, magnitudes, rakes, log_trail, srf,
  seeds, velocity_model, domain, im, emod3d, resolution, bb, hf,
  velocity_model_1d, hf_velocity_model_1d, rupture_velocity`.
- Top-level keys normalised to Felipe's ordering for easy diffing.
- `error_log.txt` — broken stubs skipped + any per-file failures, with reasons.
- `completion_summary.csv` — scrutiny aid: one row per output file with rupture id,
  #faults, total magnitude, domain depth / duration / area, #periods, #FAS.

## Per-file algorithm (deterministic, idempotent; order matters)

For each **valid** minimal file:

1. Copy the minimal file into the output folder.
2. Set `metadata.defaults_version = 24.2.2.1`.
3. Write `velocity_model` from 24.2.2.1 defaults, overriding `version = "2.09"`
   and `rrup_interpolants` (float32, from `Mw_rrup_mod.txt`).
   **Must precede step 4** — domain generation reads `velocity_model`.
4. Compute + write `domain` via `generate_domain_from_realisation`.
5. Write `im` from defaults, overriding `valid_periods` (`periods.csv`) and
   `fas_frequencies` (`frequencies.csv`).
6. Write `emod3d, resolution, bb, hf, velocity_model_1d, hf_velocity_model_1d,
   rupture_velocity` straight from 24.2.2.1 defaults.
7. Leave `sources, rupture_propagation, magnitudes, rakes, srf, seeds` as copied
   (all verified equal to what defaults/regeneration would give).
8. Normalise top-level key order; append a provenance log entry recording the
   override files and versions used.

### Section provenance

| From the existing stub | Overridden (defaults + custom) | Computed | Straight from 24.2.2.1 defaults |
|---|---|---|---|
| sources, rupture_propagation, magnitudes, rakes, srf, seeds | velocity_model (2.09 + rrup), im (dense periods/FAS), metadata.defaults_version | domain | emod3d, resolution, bb, hf, velocity_model_1d, hf_velocity_model_1d, rupture_velocity |

## Implementation

A standalone, parallelised **completion script**, adapted from
`gen_FF_realisations_MP.py` with the GCMT/nodal-plane source generation removed
and full-defaults materialisation added.

- **Location (default):** `workflow/scripts/complete_realisations.py`, alongside the
  existing `generate_realisations_from_csv.py`. `typer` CLI.
- **Arguments:** input dir, output dir, defaults version (default 24.2.2.1),
  path to the three override files, velocity-model version (default 2.09),
  worker count.
- **Parallelism:** `multiprocessing.Pool` (default `min(8, cpu_count())`).
  ~5–7 s/file (domain-dominated) ⇒ ~3–4 min for 283 files at 8 workers.
- **Robustness:** skip the 10 broken stubs; wrap each file in try/except (some
  deep-fault ruptures may fail domain generation where `rrup < ztor`) and log
  the reason rather than dropping silently; write `error_log.txt`.
- **Determinism:** no RNG in the augmentation; seeds are copied, not
  regenerated. Re-running reproduces byte-identical output.
- **Reusability:** helper functions factored so the script could later graduate
  into a general `complete-realisation-defaults` workflow utility if wanted.

## Verification already performed

- Diff of current-24.2.2.1-defaults-derived sections vs Felipe's file: identical
  for all shared sections (finding 4).
- End-to-end prototype on a single-fault (114741) and a multi-fault (113534)
  rupture: each produced a valid 18-section file with a sensible domain
  (depth 48–50 km, duration 155–190 s), `velocity_model.version = 2.09`, and
  111 periods / 389 FAS frequencies.

## Out of scope

- Fixing the 10 failed rupture generations (pre-existing; separate task).
- Running the actual EMOD3D/HF/BB simulations.
- Any change to the workflow package's default parameters or generators.

## Open items / assumptions (confirmed with user)

- **Fully materialise** every section (vs sparse overrides) — confirmed.
- **Patch `defaults_version`** in copies (vs regenerate) — confirmed.
- **Include `rupture_velocity`** (18 sections) — confirmed.
- Output folder name and script location per defaults above — may be renamed.
