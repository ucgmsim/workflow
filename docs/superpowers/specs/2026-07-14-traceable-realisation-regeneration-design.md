# Traceable regeneration of the CyberShake NSHM-2022 realisation set

Status: approved design, not yet implemented.
Date: 2026-07-14.
Branch: `cs-nshm2022-prep` (main branch is `pegasus`).

## Problem

The 291 `realisation.json` files currently committed in `cybershake_nshm_2022`
were produced by an ad-hoc shake-down run. Their provenance is not merely
incomplete — every field of `log_trail` is misleading:

| field | recorded value | why it is wrong |
| --- | --- | --- |
| `utility` | `bake_realisations.py` | the script's pre-rename name, and it was run as `python <file>` rather than through its entry point |
| `version` | `0.1.dev1277+g41974dfa1.d20260709` | a **stale build stamp**, not a statement about the code that ran. The dirty suffix is a symptom; the disease is that `importlib.metadata` reads a cached `.dist-info` frozen at the last *reinstall*. The code that ran demonstrably was **not** `41974df` — the area-weighted fault-selection change was uncommitted at the time, confirmed by regenerating from the persisted seeds. See "How the SHA gets into `log_trail`, and why it lies". |
| `args` | `["minimal_realisations/", "full_realisations"]` | `full_realisations` does not exist; the stub step likewise records `realisations/`, which does not exist either |

Shell history shows the run was assembled over several attempts that even mixed
defaults versions (`24.2.2.4` in early attempts, `24.2.2.1` in the kept files).
The set cannot be defended as reproducible, and no amount of after-the-fact
annotation will fix that. It must be regenerated.

Complicating this, `origin/pegasus` has advanced by three commits that must be
incorporated before the campaign is pinned.

## Goals

1. Every `realisation.json` carries a `log_trail` whose `version` names a
   **definite commit SHA** — no dirty suffix — that is pushed, tagged, and
   resolvable by a third party.
2. `utility` and `args` in the `log_trail` are accurate and still resolve.
3. The inputs that git cannot hold (`nshmdb.db` above all) are pinned by
   checksum **and** by a reproducible derivation.
4. Provenance is *verified mechanically*, not asserted. A committed checker
   re-runs the argument at any time.

## Non-goals

- Preserving the current 291 realisations' scientific content. Seeds are
  re-drawn (see Decision 1); hypocentres, initial faults and rupture trees will
  differ. This invalidates the SRFs and animations on `cybershake_nshm_2022`'s
  `add-srf-helper-scripts` branch, which must be regenerated afterwards.
- Changing the science. Connectivity parameters, defaults version, velocity
  model version and magnitude conventions are all held at their current values.
- Bit-reproducibility of the *set* from inputs alone (see Decision 1).

## The provenance chain

The campaign is auditable only if every link holds:

```
CRU_fault_system_solution.zip          sha256 5975568e5ccd05c1…  (from Jake Faulkner's NSHM 2022 release)
        │
        │  nshm_db_generator.py @ NSHM2022DB 95a005a
        ▼
nshmdb.db                              sha256 00e256480618cd15…  (to be REBUILT and verified)
        │
        │  ← annealed_minimal_ruptures.csv   sha256 b1e4a40d0556…  (293 ids, from Jake Faulkner)
        │  ← defaults 24.2.2.1, --dip-delta 20
        │  nshm2022-to-realisation  @ workflow <PINNED SHA>
        ▼
minimal_realisations/    291 stubs (2 excluded, see Decision 2)
        │
        │  ← felipe_scripts/ overrides, --vm-version 2.09
        │  complete-realisations   @ workflow <PINNED SHA>
        ▼
complete_realisations/   291 complete realisations
        │
        │  copy_realisations_to_event_dirs.py
        ▼
cybershake_nshm_2022/cybershake_nshm_2022/events/<id>/realisation.json
```

Dependency versions are pinned by `uv.lock` (sha256 `27358cba5f0ade50…`):
`nshmdb` 2025.12.1, `source_modelling` 2026.6.2, `velocity-modelling` 2026.2.1,
`qcore-utils` 2025.12.2, `im-calculation` 2025.12.5, `oq-wrapper` 2025.12.5.

### How the SHA gets into `log_trail`, and why it lies

`LogEntry.from_utility` calls `importlib.metadata.version("workflow")`. This
reads the **cached `.dist-info/METADATA` on disk**, which `setuptools-scm`
stamped when the package was last *built*. For an editable install, source
changes are picked up at import — but **the version metadata is not**. It stays
frozen until the package is reinstalled.

`uv run` re-syncs only when `pyproject.toml` or `uv.lock` change. It does **not**
rebuild when source files change, or when `HEAD` moves. Demonstrated on this
repo, on a clean tree at `b541da03a`:

```
git HEAD                      b541da03a
git status --porcelain        (empty — clean)
.dist-info Version:           0.1.dev1285+g12172130f.d20260714
```

The recorded version named the wrong commit **and** carried a phantom dirty
flag, on a clean tree. It was simply the stamp from an earlier build.

**This is the true root cause of the original defect.** The 291 existing files
record `g41974dfa1`, yet regenerating from their persisted seeds proved the code
that ran contained the area-weighted fault-selection change, which was *not* in
`41974df`. The tree being dirty was a symptom; the disease is that the version
string was never a claim about the code that ran at all. It was a claim about
whenever someone last happened to rebuild.

Consequences for this design, and they are not optional:

1. **The package must be force-rebuilt immediately before the run**, on a clean
   tree, at the pinned commit:

   ```
   uv sync --reinstall-package workflow --all-extras --dev
   ```

   Verified to re-stamp correctly:
   `0.1.dev1285+g12172130f.d20260714` → `0.1.dev1286+gb541da03a`.

2. **A pre-flight gate must assert the stamp matches reality before any
   realisation is written** — clean tree, and installed metadata equal to the
   version derived from `HEAD`. Without this, a stale stamp is undetectable at
   run time and silently poisons all 291 files.

3. Goal 4 stops being belt-and-braces and becomes essential. Had the campaign
   been run as originally specced — clean tree, `uv run`, no forced rebuild —
   every file would have been stamped `0.1.dev1285+g12172130f.d20260714`: wrong
   SHA, spurious dirty flag, and nobody the wiser.

One helpful fact does survive: **untracked and gitignored files do not dirty the
tree.** `minimal_realisations/` and `complete_realisations/` are gitignored, so
the run cannot dirty its own tree mid-campaign. Once the stamp is correct at the
start, it stays correct.

## Decisions

### 1. Seeds are re-drawn, not carried over

`Seeds.random_seeds()` is left exactly as it is: five integers per realisation
drawn from OS entropy and persisted into the file. Those five seeds anchor the
whole downstream chain — `nshm_to_realisation_seed` (initial fault, hypocentre,
rupture tree), `rupture_propagation_seed`, `genslip_seed` and `srfgen_seed`
(slip distribution), `hf_seed` (per-station HF).

Consequence, accepted explicitly: the realisation files are the **only** record
of the draw. The set cannot be regenerated from the recorded inputs alone; an
individual file can always be replayed, because `read_from_realisation_or_random`
reads persisted seeds back. A master-seed derivation was considered and
rejected as unnecessary — the previous values carry no significance, and
re-drawing them is acceptable.

### 2. Ruptures 59421 and 95011 are excluded

Both fail with `ValueError: The graph must be connected to find a spanning tree`.
Their fault sets are genuinely disconnected under the connectivity parameters:

- **59421** (6 faults) — `Alpine: Resolution - Charles / Dagg / Five Fingers`
  plus `Caswell High 1 / 4 / 5`: two clusters with no permitted jump between
  them.
- **95011** (4 faults) — `Alpine: Jacksons to Kaniere` and
  `Alpine: Kaniere to Springs Junction` plus `Awatere: Southwest` and
  `Hunter Valley`: Marlborough faults unreachable from the Alpine sections.

NSHM's inversion permits jumps that this workflow's connectivity model
(`--jump-cutoff 15 km`, `--separation-distance 5 km`, `--min-connected-depth 5 km`)
rejects. This is a property of the data, not a bug.

They are excluded rather than rescued, so that **every event in the campaign is
generated with identical parameters**. Loosening the cutoff for two events only
would leave them carrying parameters the other 291 do not, with no scientific
rationale beyond "they otherwise fail"; loosening it for all 293 would re-sample
every rupture tree in the set and is a different campaign. The exclusions and
their tracebacks are recorded in `PROVENANCE.md`. Final count: **291 of 293**.

### 3. `pegasus` is merged, not rebased

`origin/pegasus` (`8b39380`) is three commits ahead of the merge base
`72548bd`; `cs-nshm2022-prep` (`1217213`) is thirteen ahead.

Merge, because rebasing rewrites every commit on the branch and would leave
`41974df` — the SHA referenced by the *currently committed* 291 realisations in
`cybershake_nshm_2022` — unreachable and eventually garbage-collected. The
historical artifacts' provenance would become permanently undecipherable, which
is precisely the failure this work exists to prevent. A merge preserves it,
yields one definite checkout-able SHA spanning both lineages, and needs no
force-push.

The merge cannot alter realisation content: the incoming commits touch
`realisation_to_srf.py` (multi-segment SRF version fix), `container/Dockerfile`
(gdal), and delete a CI workflow. None touch the generation path. This is
verified mechanically after merging rather than taken on trust.

### 4. Defaults version 24.2.2.1 throughout

Confirmed by inspection rather than assumed: `srf` is defined **only** in
`default_parameters/root/defaults.yaml`, so `srf.resolution = 0.1` for every
version. The defaults version passed to `nshm2022-to-realisation` therefore does
**not** affect the stub's fault geometry — `simplify_fault` sees the same
resolution either way. The only version-specific values are
`resolution.resolution` (0.1 vs 0.4, the simulation grid) and `bb.flo`
(1.0 vs 0.25).

Using 24.2.2.1 for both steps makes `complete-realisations` setting
`metadata.defaults_version` a no-op rather than a rewrite, and matches Felipe's
reference realisation, which is the target of the completion step.

(The 2026-07-08 design doc's parenthetical describing "our stubs" as 24.2.2.4 is
stale; it makes no material difference, for the reason just given.)

### 5. `nshmdb.db` is rebuilt and verified

The database is the root of the chain and its origin was never recorded. It has
been reconstructed from evidence:

```
11:40:00  CRU_fault_system_solution.zip   69,413,618 B   → /home/arr65/data/cs_nshm_2022/
11:40:00  annealed_minimal_ruptures.csv   293 rupture ids
11:42:46  git clone NSHM2022DB            → 95a005a  (single reflog entry; never moved since)
11:44:03  nshmdb.db                       1,012,535,296 B
```

The schema matches `nshm_db_generator.py` exactly (`fault`, `fault_plane`,
`parent_fault`, `rupture`, `rupture_faults`,
`magnitude_frequency_distribution`), and it is the only generator in the repo.
But no shell-history entry records the command, so the chain is *reconstructed*,
not proven.

It will therefore be **rebuilt**: run `nshm_db_generator.py` at NSHM2022DB
`95a005a` against `CRU_fault_system_solution.zip`, and compare **logical**
content against the current database — per-table row counts and a hash over
sorted rows. Byte-identity is not expected and is not the test; SQLite page
layout and rowid allocation need not be stable.

- If the content matches, the chain is proven and recorded as fact.
- If it differs, that is a genuine finding. The **rebuilt** database is used for
  the campaign, since its provenance is then known by construction.

Either outcome ends with a database whose derivation is reproducible.

## Components

Code changes must land **before** the pinned commit, since the run's SHA has to
contain them.

1. **`workflow/scripts/generate_realisations_from_csv.py`** — delete the partial
   stub when the subprocess fails. `nshm2022-to-realisation` writes `metadata`
   and `seeds` before the failure point, so a crash currently leaves a
   source-less file behind; that is why `minimal_realisations/` held 293 files
   of which two were unusable. The output directory should hold exactly the
   valid stubs. Failures remain fully recorded in `error_log.txt`.

2. **`copy_realisations_to_event_dirs.py`** — currently committed with a
   hardcoded path (`/home/arr65/src/cybershake_nshm_2022/flow/events`) that no
   longer exists; the tree moved to `cybershake_nshm_2022/cybershake_nshm_2022/events/`.
   Take source and destination as CLI arguments so the script is not
   machine-specific.

3. **`workflow/scripts/verify_realisation_provenance.py`** (new, entry point
   `verify-realisation-provenance`) — two modes, because the stale-metadata bug
   means checking after the fact is too late.

   **Pre-flight** (`--preflight`), run *before* any realisation is written, and
   refusing to proceed unless all hold:
   - `git status --porcelain` is empty;
   - `importlib.metadata.version("workflow")` **equals the version derived from
     the current `HEAD`** — this is the check that catches a stale `.dist-info`,
     and the one whose absence would have silently poisoned the whole campaign;
   - the version carries no `.d` suffix.

   **Post-hoc** (default), over an output directory:
   - exactly two `log_trail` entries, in order;
   - `utility` ∈ {`nshm2022-to-realisation`, `complete-realisations`};
   - `version` equals the expected `0.1.dev<N>+g<sha>` string in every file;
   - all 18 sections present, in `FELIPE_SECTION_ORDER`.

   This is the component that discharges Goal 4. It exits non-zero on any
   violation and is re-runnable by anyone, later, as an independent audit.

4. **`PROVENANCE.md` + `manifest.csv`** — written after the run, committed both
   to `workflow` and alongside the artifacts in `cybershake_nshm_2022`.

5. **`pyproject.toml`** — exclude `felipe_scripts/` from the `ruff` and `ty`
   gates. Discovered while writing the implementation plan: committing
   `felipe_scripts/` as a reference input **broke two CI gates that had been
   passing** — `ruff` reports 17 errors in it and `ty` one. It is Felipe's
   original code, kept verbatim; it was never meant to be linted. A
   config-file exclusion (rather than a workflow-file change) applies both in CI
   and locally, since CI invokes `ty check --exclude workflow/schemas.py
   --exclude setup.py` and `ruff check` with no path arguments.

   Verified: with the exclusion, `ty` reports `All checks passed!` and `ruff` is
   left with a single `D103` in `copy_realisations_to_event_dirs.py`, which
   component 2 above removes.

## Run protocol

Ordering is load-bearing. The pinned SHA must exist before the run; the record
of the run can only exist after it. The tag is applied last, so that nothing
unproven is ever tagged.

1. Merge `origin/pegasus`; verify the generation path is untouched.
2. Land the three code changes.
3. **Run every CI gate**, exactly as CI runs them. `tests/test_complete_realisations.py`
   was renamed and has never been run; no commit is pinned on unverified code.

   ```
   uv sync --all-extras --dev      # required first: without pandas-stubs, ty false-passes
   uv run pytest -q
   uv run ruff check
   uv run ty check --exclude workflow/schemas.py --exclude setup.py
   uv run deptry .
   fdfind . workflow/ -E "__init__.py" --extension py | xargs numpydoc lint
   ```
4. Commit. Confirm `git status --porcelain` is empty, and push the branch — a
   SHA that exists only on one laptop is not auditable. **This is commit N.**
5. Rebuild and verify `nshmdb.db` (Decision 5).
6. **Force-rebuild the package so the version stamp is not stale:**

   ```
   uv sync --reinstall-package workflow --all-extras --dev
   ```

   `uv sync` alone is **not** sufficient — it will not rebuild when only `HEAD`
   has moved, and will happily serve a `.dist-info` stamped days ago at a
   different commit.
7. **Pre-flight gate:** `uv run verify-realisation-provenance --preflight`.
   Refuses to proceed unless the tree is clean and the installed metadata
   matches `HEAD`. This is the single check that stands between the campaign and
   291 files stamped with the wrong commit.
8. Smoke-test: run both steps over a three-rupture subset of the CSV into a
   scratch directory, and confirm `verify-realisation-provenance` passes on the
   output. Only then commit to the full run — a bad stamp discovered after 291
   files is 291 files wasted.
9. Run the campaign from the repo root, through the **entry points** — not
   `python <script>.py`, which is what produced `utility: bake_realisations.py`:

   ```
   uv run generate-realisations-from-csv nshmdb.db annealed_minimal_ruptures.csv \
           minimal_realisations 24.2.2.1
   uv run complete-realisations minimal_realisations complete_realisations \
           --defaults-version 24.2.2.1 --vm-version 2.09
   ```

   Paths are relative to the repo root, so the `args` recorded in `log_trail`
   remain meaningful — unlike the previous run's `realisations/` and
   `full_realisations`.
10. `uv run verify-realisation-provenance complete_realisations --expect-version <string>`.
11. Tag commit N as `cs-nshm2022-realisations-v1` (annotated) and push the tag.
    See the tagging risk below — verify the tag does not perturb the version
    scheme before relying on it.
12. Write `PROVENANCE.md` and `manifest.csv`; commit as **commit N+1**.
13. Distribute to `cybershake_nshm_2022`; commit the events, `PROVENANCE.md` and
    `manifest.csv` there.

## Acceptance criteria

- 291 realisations produced; exactly two documented exclusions; no other
  failures.
- `verify-realisation-provenance` exits zero.
- Every `log_trail` version is byte-identical to the expected clean string, and
  the SHA it names is reachable from `origin/cs-nshm2022-prep` and tagged.
- `PROVENANCE.md` pins: workflow SHA (full 40 chars), tag, remote URL, expected
  version string, `uv.lock` sha256 and resolved dependency versions,
  `nshmdb.db` sha256 with its rebuild result, `CRU_fault_system_solution.zip`
  sha256, CSV sha256, defaults 24.2.2.1 / vm 2.09 / dip-delta 20, both
  exclusions with tracebacks, host and Python version, and per-file sha256 of
  all 291 outputs.
- The test suite and `ty check --all-extras` pass at commit N.

## Risks

- **Stale version metadata — the one that nearly sank this.** See "How the SHA
  gets into `log_trail`, and why it lies". Mitigated by the forced reinstall
  (step 6) and the pre-flight gate (step 7), and caught by the post-hoc verifier
  even if both are somehow skipped. Treat any change to the run protocol that
  removes either as a defect.

- **Tagging may perturb the version scheme.** The repo currently has **no tags**
  — `git describe` returns nothing — which is why `setuptools-scm` falls back to
  `0.1.dev<count>+g<sha>`. Adding `cs-nshm2022-realisations-v1` may match
  `setuptools-scm`'s default `tag_regex` (which can strip a leading
  `[\w-]+-` prefix and read `v1` as version `1`), changing every subsequent
  version string. Tagging *after* the run keeps the recorded strings intact, but
  anyone who later checks out the tag and reinstalls could see a different
  version than the one in `log_trail`. Before relying on the tag: create it,
  reinstall, and check whether the version string moves. If it does, either pick
  a tag name `setuptools-scm` ignores or drop the tag — **the SHA is the anchor,
  the tag is only a human-readable pointer.**

- **Downstream invalidation.** New seeds mean new hypocentres, so the SRFs and
  slip animations on `cybershake_nshm_2022`'s `add-srf-helper-scripts` branch
  must be regenerated. They need regenerating anyway, because the multi-segment
  SRF version fix arrives with the `pegasus` merge.
- **The rebuilt database may differ.** Decision 5 accepts this and prefers the
  rebuilt database. If it differs substantially, stop and investigate before
  running the campaign rather than pressing on.
- **`felipe_scripts/` is a required input to `complete-realisations`** (it
  carries the velocity-model and intensity-measure overrides) and is resolved
  relative to the working directory. The campaign must be run from the repo
  root.

## Open items

- The CRU solution zip's own origin is not recorded beyond "Jake Faulkner's NSHM
  2022 release". If a citable source URL or release identifier exists, it should
  go in `PROVENANCE.md`; the chain otherwise terminates at a checksum.
- `annealed_minimal_ruptures.csv` was provided by Jake Faulkner as the first
  sample of ruptures for this campaign. The selection procedure implied by
  "annealed" is not documented. Worth asking him, if the rupture set ever needs
  to be defended or reproduced.
