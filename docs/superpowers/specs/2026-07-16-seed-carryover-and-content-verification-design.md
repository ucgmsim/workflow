# Seed carry-over and content verification for the NSHM-2022 realisation regeneration

Status: approved design, not yet implemented.
Date: 2026-07-16.
Branch: `cs-nshm2022-prep` (main branch is `pegasus`).
Amends: `2026-07-14-traceable-realisation-regeneration-design.md` — reverses Decision 1
and two Non-goals; see "What this changes" below. Everything not named there still stands.

## Why this amendment

The 2026-07-14 design chose to **re-draw** the five seeds per realisation, accepting that
hypocentres, initial faults and rupture trees would differ from the current files, and that
the SRFs and animations already built from them would be invalidated and regenerated
(Decision 1; Non-goals).

That trade is no longer worth making. Substantial downstream work already exists — the SRF
set on BSC and in Dropbox, the local repaired copies, the animations — all built from the
exact 291 files now committed at `5d3e149` in `cybershake_nshm_2022` (branch
`add-srf-helper-scripts`). Re-drawing discards that for no scientific gain: the seed *values*
carry no significance, so there is nothing to prefer about a fresh draw. Reproducing the
existing seeds keeps every downstream artefact valid while still giving each file the honest,
commit-pinned `log_trail` that the regeneration exists to provide.

The target therefore changes from "same science, new draw" to **"same files, honest
provenance"**: each regenerated `realisation.json` must match its committed original in every
scientific field, differing only in `log_trail`.

## What this changes in the 2026-07-14 design

- **Decision 1 ("Seeds are re-drawn, not carried over") is reversed.** Seeds are now carried
  over — reproduced exactly from the committed originals, via a committed manifest.
- **Non-goal "Preserving the current 291 realisations' scientific content" becomes a Goal**,
  and it is *verified mechanically*, not asserted.
- **Non-goal "Bit-reproducibility of the set from inputs alone" is relaxed.** With the seed
  manifest as a recorded input, the whole set becomes reproducible from its inputs (manifest
  + rupture list + `nshmdb.db` + args + commit N) — contingent on the pilot confirming that
  commit-N code reproduces the original content.

Unchanged: the provenance goals, the two excluded ruptures (Decision 2), the `nshmdb.db`
derivation and checksum, the CI gates, and the mechanical provenance checker.

## Established facts (grounding)

Verified against the committed originals and the current tree:

1. **All 291 files carry a complete five-seed block** — `nshm_to_realisation_seed`,
   `rupture_propagation_seed`, `genslip_seed`, `srfgen_seed`, `hf_seed`. No gaps.
2. **The generation invocation is uniform.** Every event ran
   `nshm2022-to-realisation <db> <rupture_id> <out.json> 24.2.2.1 --dip-delta 20`, then
   `<baker> <out.json> full_realisations`. The only per-event variables are the rupture id
   and the five seeds — no per-event hypocentre or initial-fault overrides.
3. **`metadata` carries no build or timestamp stamp** — only `name`, `version: "1"`,
   `defaults_version`, `tag`. A content comparison therefore excludes exactly one key,
   `log_trail`; everything else, `metadata` included, must match.
4. **The seed-replay path already exists.** `Seeds.read_from_realisation_or_random`
   (`workflow/realisations.py:261`) reads seeds from the file when present and only draws
   fresh ones when the block is absent. Reproducing seeds needs **no change to the generation
   engine** — only that the seed block is in place before `nshm2022-to-realisation` runs.
5. **The area-weighted fault-selection fix is committed** (`9f35c90`, branch-only). The
   originals were made with this change *uncommitted*; it now lives at HEAD. So commit-N code
   is *likely* to reproduce the original content — but not provably, because:
6. **`bake_realisations.py` no longer exists.** The original baker was an ad-hoc script of
   that name; its committed successor is `complete-realisations`. Same name ≠ same code.
   Together with the magnitude-convention (BoldM) changes that arrived via the pegasus rebase,
   this means content reproduction must be **proven, not assumed** — hence the pilot.

## Design (Approach A)

Five pieces. The generic pieces live in `workflow`; the campaign data lives in
`cybershake_nshm_2022` (see "Placement").

### 1. Seed manifest, built once from the originals

A tested `workflow` helper reads the committed `events/<rupture_id>/realisation.json` tree and
emits a sorted CSV:

```
rupture_id,nshm_to_realisation_seed,rupture_propagation_seed,genslip_seed,srfgen_seed,hf_seed
```

Run once against the originals at `5d3e149`, it **asserts facts 1 and 2** (291 complete
blocks; uniform args), and that each directory name matches the rupture id in its own
`log_trail` so the manifest's `rupture_id` is provably the id used to generate the file. It
fails loudly otherwise, so the manifest is faithful the day it is written. It holds 291 rows — the two ruptures excluded by Decision 2 never produced
a file and have no seeds to carry. The CSV is committed to `cybershake_nshm_2022`.

The manifest records only the rupture id and five seeds. A per-event content hash was
considered — to make the manifest self-verifying — and deferred: verification diffs against
the committed originals directly (piece 4), so a hash would be redundant tamper-evidence, not
a load-bearing input. It can be added later if wanted.

### 2. README, the provenance narrative

Committed beside the manifest. It states plainly that the five integers per event were drawn
from OS entropy by `Seeds.random_seeds()` (`random.randint(0, 2**31 - 1)`) during the initial
ad-hoc phase; that they carry no intrinsic meaning; and that they are recorded verbatim
**solely** to reproduce the existing realisation files exactly, keeping them consistent with
the SRFs and results already built from them. It documents the one fixed generation command
(fact 2) and the verification guarantee (piece 4).

### 3. Injection — opt-in, no engine change

`generate-realisations-from-csv` gains an optional `--seed-manifest <path.csv>`. When given,
for each rupture id it writes the real five seeds into the stub —
`{"metadata": {}, "seeds": {<five seeds>}}` — instead of the empty `seeds: {}` the 2026-07-14
plan writes. `nshm2022-to-realisation` then reads them through the existing
`read_from_realisation_or_random` path. When the flag is absent, the driver keeps its current
fresh-draw behaviour. A rupture id absent from the manifest falls back to the empty stub,
which is harmless: the only absentees are the two Decision-2 exclusions, which fail before any
seed is consumed.

This keeps the tool **generic** — seed replay is an opt-in feature any campaign can use, not a
CyberShake hard-code — so it is safe to upstream to `pegasus`. An implementation test must
confirm that `metadata.write_to_realisation` preserves the pre-written `seeds` key: the stub
must survive the metadata write at `nshm2022_to_realisation.py:292`.

### 4. Verification — content diff, hard-stop

A generic `workflow` checker compares two realisations (or two directories) for deep equality
**excluding only `log_trail`**, and reports the first differing field per event. The campaign
regenerates the full set into a **fresh directory**, diffs it against the still-in-place
originals, and replaces the originals only once the diff is clean — so a botched run can never
destroy the comparison target. Any mismatch **aborts the campaign** and is reconciled before
anything is committed.

The bar is **exact** structural equality: same seeds + same code + same platform yields
identical JSON. If floats ever force a tolerance, that is a finding to surface, not silently
absorb.

### 5. Pilot — prove content reproduction before the full run

Before the 291-event run *and before finalising the plan edits*, run
extract → inject → regenerate → content-diff on ~4 events spanning at least one multi-fault
and one single-fault (e.g. `149379` and a single-fault event). An exact match confirms that
commit-N code reproduces the originals from their seeds. A mismatch stops everything for code
archaeology — the candidates are `complete-realisations` versus the vanished
`bake_realisations.py`, the BoldM magnitude changes, and any residual difference in the
area-weighting fix — before the campaign proceeds.

## Placement — the code-vs-data boundary

One rule governs the split: **`workflow` ships versioned, reusable, provenance-stamped code;
`cybershake_nshm_2022` holds this campaign's data, narrative, and glue.**
`cybershake_nshm_2022` is not a package and cannot version-stamp anything; the `log_trail`
version string exists only because the engine lives in `workflow`.

| Artefact | Home |
| --- | --- |
| `--seed-manifest` injection, the content checker, the manifest builder | `workflow` (generic, tested, pegasus-mergeable) |
| `seed_manifest.csv` + `README.md` | `cybershake_nshm_2022/cybershake_nshm_2022/seed_manifest/` (beside `events/`) |

The manifest is a campaign **input**, read by path exactly as `nshmdb.db` and
`annealed_minimal_ruptures.csv` are — not data embedded in the engine.

## Provenance — commit N is tagged, not merged first

Commit N is the SHA stamped into all 291 `log_trail`s; it must stay permanently reachable, or
the untraceability this project exists to kill returns. This branch was already rebased once
(its old base is orphaned); a later squash-merge would orphan commit N too.

Resolution: pin commit N on `cs-nshm2022-prep` and protect it with an **annotated, pushed git
tag**. The tag guarantees reachability regardless of how or whether the branch is later
merged, so no `pegasus` merge is required before the campaign runs. Building pieces 3 and 4
generically keeps that future merge clean when it happens.

## Deferred — not in this work

- **Separating the branch for a `pegasus` PR** — relocating campaign glue, removing
  `felipe_scripts/` (Felipe's reference inputs; needs team sign-off), and cherry-picking the
  generic commits. Captured as a late plan task, not done now: it changes neither commit N's
  stamp (git-state-derived, not file-inventory-derived) nor the campaign's output.
- **Co-locating `annealed_minimal_ruptures.csv`** with the seed manifest — coherent, since
  both are campaign inputs, but optional; the campaign runs whether it moves or not.

## Impact on the 2026-07-14 plan

Targeted edits, to be made when the plan is revised:

- **Task 3** (stub on failure): the stub gains the real seed block when `--seed-manifest` is
  supplied.
- **New task**: build and commit the seed manifest and README, using the asserting builder.
- **New task**: the content checker and the pilot, gating the full run.
- **New task**: tag commit N (annotated, pushed).
- **New, deferred task**: the `pegasus`-PR separation above.
- **Decision 1, the two Non-goals, and the "Seeds" section** are updated to match "What this
  changes".

## Done looks like

- `seed_manifest.csv` and `README.md` committed in `cybershake_nshm_2022`, the manifest proven
  faithful by its asserting builder.
- The pilot shows exact content match (excluding `log_trail`) on multi- and single-fault
  events.
- All 291 regenerated files differ from their originals **only** in `log_trail`, confirmed by
  the content checker.
- Every `log_trail` names commit N; commit N is tagged and pushed.
- The seed-injection flag and the content checker are generic and carry tests — ready to
  upstream to `pegasus` later.
