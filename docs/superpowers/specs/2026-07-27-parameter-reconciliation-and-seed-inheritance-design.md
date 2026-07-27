# Parameter reconciliation and seed inheritance for the NSHM-2022 realisation set

Status: approved design, not yet implemented.
Date: 2026-07-27.
Branch: `cs-nshm2022-prep` (main branch is `pegasus`).
Amends: `2026-07-14-traceable-realisation-regeneration-design.md` and
`2026-07-16-seed-carryover-and-content-verification-design.md`. It replaces the seed manifest,
widens the campaign's purpose, and relaxes the content-verification bar. Everything not named
in "What this changes" still stands.

## Why this amendment

Two things happened after the 2026-07-16 design was approved.

**The campaign's purpose widened.** The set is no longer being regenerated *only* to give each
file an honest, commit-pinned `log_trail`. `pegasus` has since gained a scientific default the
set does not carry — `PGD`, added to `im.ims` by `ec2fb25` (2026-07-21) — and the set must now
pick it up. The regeneration therefore has to *change* the files in a specific, intended way,
not merely re-provenance them.

**That will keep happening.** `pegasus` is under active development, so defaults will keep
moving, and the campaign's own overrides (`felipe_scripts/`) will keep needing to be weighed
against them. Some overrides should eventually be superseded by defaults. A one-off patch for
`PGD` would leave the same trap armed for the next change.

So the campaign needs a **repeatable reconciliation**: a way to see every place the campaign's
parameters disagree with `pegasus`, decide each one deliberately, record the decision, and have
the recorded decisions drive regeneration without further prompting.

## What this changes

- **The seed manifest is deleted.** Seeds are inherited directly from the deployed
  `realisation.json` files instead of from a committed CSV. Plan Tasks 7a and 10a go away.
- **"Same files, honest provenance" becomes "same science, updated parameters, honest
  provenance."** The 2026-07-16 target — differ *only* in `log_trail` — is relaxed: the
  regenerated set may also differ in parameters whose change was explicitly decided and
  recorded.
- **Verification is driven by the decision record, not by a hard-coded rule.** The decision file
  *is* the allowlist.
- **Deployment becomes opt-in and guarded** by two independent flags.

Unchanged: the provenance goals, the two excluded ruptures, the `nshmdb.db` derivation and
checksum, the CI gates, the mechanical provenance checker, the tag-not-merge resolution for
commit N, and the code-vs-data placement rule.

## Established facts (verified 2026-07-27)

1. **The campaign repo was renamed.** `cybershake_nshm_2022` → `cs_nshm_2022` (commit `37b2ea1`,
   2026-07-22, mirroring the GitHub rename). Branch is now `main`, not `add-srf-helper-scripts`.
   The events tree moved to `cs_nshm_2022/cs_nshm_2022/events/`. All 291 `realisation.json`
   files survived as a **pure `git mv`** — zero content change — and the working tree is clean.
2. **`pegasus` is already fully merged.** `origin/pegasus` is `7e465c5`; `dee2f1b` merged it into
   `cs-nshm2022-prep`, and `git merge-base origin/pegasus HEAD` equals pegasus's tip. The branch
   is **0 behind / 25 ahead**. Plan Task 1 is therefore already satisfied.
3. **Ten realisation sections are uniform across all 291 events** — `im`, `velocity_model`,
   `emod3d`, `resolution`, `srf`, `velocity_model_1d`, `hf_velocity_model_1d`, `hf`, `bb`,
   `rupture_velocity`. Exactly one distinct value each. These are *parameters*.
   The remaining seven — `metadata`, `sources`, `rupture_propagation`, `magnitudes`, `rakes`,
   `domain`, `seeds` — vary per event. These are *derivations*, from `nshmdb.db` plus seeds.
4. **Only `im` drifts from HEAD defaults.** Eight of the ten parameter sections — `emod3d`,
   `resolution`, `srf`, `velocity_model_1d`, `hf_velocity_model_1d`, `hf`, `bb` and
   `rupture_velocity` — match HEAD defaults exactly, across all 291 files. `velocity_model` was
   not compared statically because `felipe_scripts` overrides it; the reconciler is what will
   compare it. The drift is confined to two keys of `im`:
   - `im.ims` — deployed has 8, HEAD defaults have 9. The difference is exactly `{PGD}`;
     defaults are a strict superset. `felipe_scripts` has no opinion on this key.
   - `im.fas_frequencies` — defaults hold a 100-point grid over 0.1–100 Hz (ratio 1.0722672);
     `felipe_scripts` holds a 389-point grid over 0.0132–100 Hz (ratio 1.0232930). They share
     exactly one value, 100.0 Hz. Genuinely different grids, and **incommensurate**: only 1 of
     the 100 defaults values lies within 1e-9 of any felipe value, and the nearest-neighbour
     gap reaches 1.1e-2. They are not nestable — felipe's ratio cubed is 1.0715193 against
     defaults' 1.0722672 — so a tolerance-based merge of the two returns felipe's grid
     unchanged, and a tolerance loose enough to match more would substitute frequencies
     differing by 0.1%. Choose one grid; do not synthesise a third.
   - `im.valid_periods` — **identical** across defaults, felipe and deployed (111 values).
     `ec2fb25` absorbed felipe's list into the defaults, making that override a no-op.
5. **The deployed `fas_frequencies` differ from felipe's by float noise only.** 165 of 389
   values differ, by at most 57 ULP (≤ 6.69e-15 relative). Both are shortest-round-trip
   representations; the two lists are the same intended log-spaced grid produced by different
   floating-point evaluation paths. This is not a scientific difference.
6. **Set-union is unsafe on float grids.** `union(felipe, deployed)` yields 554 values
   containing 165 near-duplicate pairs separated by ~1e-16, inside a grid whose real spacing is
   2.3%. Union preserves rounding artefacts as spurious grid points.
7. **`im` does not affect SRF generation.** `IntensityMeasureCalculationParameters` is read only
   by `workflow/scripts/im_calc.py`. Neither `realisation_to_srf` nor `generate_stoch` reads it.
   Changing `im` cannot invalidate an already-computed SRF.
8. **The seed-replay path already exists.** `Seeds.read_from_realisation_or_random`
   (`workflow/realisations.py:262`) reads seeds from the target file when present and only draws
   fresh ones when absent. Inheriting seeds needs **no change to the generation engine**.
9. **`rich` and `pyyaml` are already dependencies.** The interactive presentation and the
   decision file need nothing new.
10. **`Overrides` replaces whole lists.** `complete_one` (`complete_realisations.py:202`) assigns
    `valid_periods` and `fas_frequencies` wholesale from `load_overrides`
    (`complete_realisations.py:118`). Today that loses nothing, because defaults and felipe are
    identical for `valid_periods`. If `pegasus` later extends that list, the override would
    silently discard the extension. This is the hazard the reconciler exists to expose.

## Design

Five pieces. Generic, tested code lives in `workflow`; campaign data lives in `cs_nshm_2022`.

### 1. `reconcile-parameters` — the pre-flight reconciler

A new `workflow` CLI that compares three sources. It works **key by key** within each of the ten
watched parameter sections (fact 3) — `im.ims` and `im.fas_frequencies` are separate decisions,
not one decision about `im` — so a section is never accepted or rejected wholesale.

| source | where it comes from |
| --- | --- |
| **defaults** | `load_defaults(version)` at HEAD |
| **felipe** | `load_overrides(felipe_scripts_dir)` — the existing loader, reused |
| **deployed** | the values in `events/*/realisation.json` |

Each parameter is classified:

- **identical** — bitwise equal across all sources that have an opinion. Silent.
- **equivalent within tolerance** — numeric values agreeing to within a relative tolerance
  (default `1e-9`, overridable). Reported for the record, but **never prompts**. This is what
  keeps fact 5 from becoming a permanent nag.
- **genuine conflict** — different lengths, different sets, or values beyond tolerance.
  **Prompts.**

Conflicts are presented with `rich`, showing each candidate's provenance, element count, and
range, so the choice is made on visible evidence rather than recall. Resolutions:

- `defaults` — adopt the pegasus value
- `felipe` — keep the campaign override
- `keep-deployed` — pin the value currently in the files, changing nothing
- `union` — **only** for discrete, set-valued parameters such as `im.ims`. Suppressed for
  float-valued grids, with the reason shown (fact 6).

Because parameter sections are uniform (fact 3), **one decision is the bulk apply** — it
necessarily covers all 291 events. If the deployed set has partially diverged, the reconciler
groups events by value and shows the split before asking, so a half-finished deployment is
visible rather than silently averaged over.

`--non-interactive` makes the tool exit non-zero on any unresolved conflict instead of
prompting, so CI and unattended reruns can never guess.

Applied to today's tree, the reconciler prompts for exactly two conflicts — `im.ims` and
`im.fas_frequencies` — reports `im.fas_frequencies` deployed-vs-felipe as equivalent within
tolerance, and is silent on everything else.

### 2. `campaign_parameters.yaml` — the decision record

YAML, for symmetry with `defaults.yaml` and because `pyyaml` is already a dependency.
Committed to `cs_nshm_2022/cs_nshm_2022/`, beside `events/`, as campaign data.

```yaml
im:
  ims:
    source: defaults
    reason: "adopt PGD, added to pegasus defaults by ec2fb25"
    decided: 2026-07-27
    sha256: <fingerprint of the chosen value>
  valid_periods:
    source: defaults
    reason: "felipe's list was absorbed into defaults by ec2fb25; the override is a no-op"
    decided: 2026-07-27
    sha256: ...
  fas_frequencies:
    source: felipe
    reason: "richer grid: 389 points from 0.0132 Hz, against 100 points from 0.1 Hz"
    decided: 2026-07-27
    sha256: ...
```

The **fingerprint is what makes this survive active development.** On a later run:

- a decision whose chosen source still hashes the same → **silent**, stays settled;
- a decision whose chosen source **changed underneath** → **re-prompts**, flagged as
  *previously resolved, source has changed*;
- a newly conflicting parameter with no decision → **prompts**;
- no conflict → no entry needed.

So the second and subsequent pegasus merges only ask about what actually moved. The `reason`
field is required, not optional: an unexplained decision is the failure mode this campaign
exists to eliminate.

`complete-realisations` gains `--parameters campaign_parameters.yaml`. When supplied, it
resolves each watched parameter from the recorded source instead of applying `Overrides`
blindly. When absent, current behaviour is unchanged, so the tool stays generic and every other
campaign is unaffected.

### 3. Seed inheritance — the manifest is replaced

`generate-realisations-from-csv` gains `--inherit-seeds-from EVENTS_DIR`. For each rupture id it
reads `EVENTS_DIR/<rupture_id>/realisation.json`, takes **only** its `seeds` block, and writes
it into the stub — `{"metadata": {}, "seeds": {…}}` — before invoking
`nshm2022-to-realisation`, which replays it through the existing path (fact 8). Every other
field is derived fresh and is entirely independent of the existing file.

A missing directory, missing file, or missing seed block falls back to a fresh random draw, so
new events and the two Decision-2 exclusions need no special handling.

This deletes the committed seed manifest and its README: the deployed realisations already carry
their seeds, so a second copy is redundant state to keep in sync. An implementation test must
confirm the pre-written `seeds` key survives the metadata write in
`nshm2022_to_realisation.py`.

### 4. Deployment — opt-in, with an independent overwrite gate

Two flags, both off by default, on `complete-realisations` and on the standalone
`copy-realisations-to-event-dirs`, sharing one helper so the semantics cannot diverge:

| flags | behaviour |
| --- | --- |
| *(neither)* | writes only to the output directory; touches nothing under `events/` |
| `--deploy-dir` | creates **new** event directories; **refuses** to replace any existing `realisation.json`, reports how many it refused, exits non-zero |
| `--deploy-dir --overwrite-existing` | replaces existing files |

Refusing by default is the point: the destructive step must be named explicitly. This reverses
the current Task 4 test that asserts unconditional overwrite.

### 5. Verification — the decision file is the allowlist

`verify-realisation-content` stops hard-coding "everything but `log_trail`". It takes the
decision file and asserts the regenerated set differs from the deployed set in **exactly**:

- `log_trail` — always expected; it is the point of the campaign; and
- those parameters whose recorded decision changes the deployed value.

Two failure modes, both hard stops:

- an **unexpected** difference — a source, a magnitude, a domain, an unresolved parameter —
  means regeneration did something nobody decided;
- a **missing** expected difference means a recorded decision silently failed to apply.

Checking both directions is what makes the decision file load-bearing rather than decorative.
Numeric comparison uses the same tolerance as the reconciler, so fact 5 does not resurface here.

## Pilot — still a mandatory gate

The 2026-07-16 pilot stands, with its bar restated: regenerate a handful of events spanning at
least one multi-fault and one single-fault case, and verify against the deployed originals. The
expected result is now *"differs only in `log_trail` and the decided parameters"*, not *"differs
only in `log_trail`"*.

The pilot remains load-bearing because facts 3 and 4 constrain only the **parameter** sections.
The seven derived sections — `sources`, `magnitudes`, `rupture_propagation`, `rakes`, `domain`,
`seeds`, `metadata` — cannot be checked statically; whether commit-N code reproduces them from
inherited seeds is exactly what the pilot proves. The candidates for a mismatch are unchanged:
`complete-realisations` versus the vanished `bake_realisations.py`, the BoldM magnitude-
convention changes, and the area-weighted fault selection.

## Placement

| Artefact | Home |
| --- | --- |
| `reconcile-parameters`, `--inherit-seeds-from`, the deploy flags, the content checker | `workflow` — generic, tested, pegasus-mergeable |
| `campaign_parameters.yaml` | `cs_nshm_2022/cs_nshm_2022/` — campaign data |

`felipe_scripts/` stays in `workflow` for now. Relocating it remains deferred (it needs team
sign-off) and does not block this work: it is read by path, and the reconciler takes that path
as an argument.

## Deferred — not in this work

- **Retiring the redundant `valid_periods` override.** Fact 4 shows it is a no-op today. Once
  recorded as `source: defaults`, the override file itself can be dropped — but that is a
  `felipe_scripts` change needing sign-off, and the decision record already captures the intent.
- **A surgical in-place parameter update.** Considered and rejected for now: full regeneration is
  idempotent under seed inheritance and is always verified, so a second write path would be
  unverified duplication. Revisit only if regeneration runtime becomes a real constraint.
- **Separating the branch for a `pegasus` PR** — unchanged from the 2026-07-16 design.

## Impact on the 2026-07-14 plan

| | |
| --- | --- |
| **Delete** | Task 7a (`build_seed_manifest`), Task 10a (build and commit the manifest) |
| **Rewrite** | Task 3 (`--inherit-seeds-from` in place of `--seed-manifest`), Task 4 (deploy + overwrite gates), Task 7b (decision-file-driven verification), Task 10b and 11a (pilot and full verify, restated bar) |
| **Add** | `reconcile-parameters` and its decision file; a task to run it and commit the result |
| **Reduce** | Task 1 — `pegasus` is already merged (fact 2); becomes verify-only |
| **Mechanical** | `cybershake_nshm_2022` → `cs_nshm_2022`, and branch `add-srf-helper-scripts` → `main`, throughout |

## Done looks like

- `reconcile-parameters` exists, is tested, and reports today's tree as exactly two conflicts,
  one tolerance-equivalence, and silence elsewhere.
- `campaign_parameters.yaml` is committed to `cs_nshm_2022`, every entry carrying a source, a
  reason, a date and a fingerprint.
- The pilot shows the regenerated events differing from their originals only in `log_trail` and
  the decided parameters.
- All 291 regenerated files pass the same check against the full deployed set, in both
  directions.
- Every regenerated `log_trail` names commit N; commit N is tagged and pushed.
- The 291 deployed files carry `PGD`.
- Deployment happened through `--deploy-dir --overwrite-existing`, deliberately, after
  verification passed.
- Seed inheritance, the deploy gates, the reconciler and the content checker are generic and
  carry tests — ready to upstream to `pegasus`.
