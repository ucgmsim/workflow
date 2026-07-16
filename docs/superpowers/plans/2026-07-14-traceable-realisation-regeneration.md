# Traceable Regeneration of the NSHM-2022 Realisation Set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Amended 2026-07-16** by `docs/superpowers/specs/2026-07-16-seed-carryover-and-content-verification-design.md`. Seeds are now **carried over** from a committed manifest, and each regenerated file is **verified** to reproduce its original in every field except `log_trail` — reversing the original decision to re-draw them. The change lands in Task 3 (opt-in `--seed-manifest`), new Tasks 7a/7b (manifest builder and content checker), and new Tasks 10a/10b/11a (build the manifest, pilot, verify), with Tasks 13 and 14 updated to match.

**Goal:** Regenerate the 291 CyberShake NSHM-2022 realisations so every file's `log_trail` names a definite, pushed, tagged commit SHA — reproducing each file's existing scientific content exactly (seeds carried over from a committed manifest, content verified identical bar `log_trail`), with all inputs pinned by checksum and the result proved by committed checkers.

**Architecture:** Three phases. **Code** (Tasks 1–8, test-driven) merges `origin/pegasus`, restores the CI gates, fixes two campaign scripts, and adds four new tools — a provenance verifier, a database comparator, a seed-manifest builder, and a content checker — ending in the pinned commit ("commit N"). **Inputs** (Task 9) rebuilds and verifies `nshmdb.db` from the CRU solution zip. **Campaign** (Tasks 10–14) is a runbook, not TDD: it builds the seed manifest, proves content reproduction on a pilot, regenerates all 291, verifies each reproduces its original bar `log_trail`, and distributes — with exact commands, expected output, and explicit abort conditions.

**Tech Stack:** Python 3.12, `typer` CLIs with `numpydoc` docstrings, `pytest`, `uv` for environment management, `setuptools-scm` for versioning, SQLite.

## Global Constraints

Every task's requirements implicitly include this section.

- **Working directory is `/home/arr65/src/workflow`** for every command unless a task says otherwise. `complete-realisations` resolves `felipe_scripts/` relative to the working directory.
- **Always `uv run`; never bare `python`.** Invoking a tool as `python <script>.py` is what recorded `utility: bake_realisations.py` in the existing files.
- **Defaults version is `24.2.2.1`** at every step, for both stub generation and completion.
- **`--dip-delta 20`** and **`--vm-version 2.09`** — do not change these.
- **Excluded ruptures: `59421` and `95011`.** Disconnected fault graphs. Expected to fail; their failure is a pass condition, not an error.
- **Expected output count: 291** realisations from a 293-row input CSV.
- **Never commit generated realisations to the `workflow` repo.** `minimal_realisations/` and `complete_realisations/` are gitignored, and must stay that way — it is what stops the campaign dirtying its own tree.
- **Seeds are carried over, not re-drawn.** The campaign passes `--seed-manifest` so each realisation replays its original five seeds, and the regenerated content must match the committed original in every field except `log_trail` — verified mechanically before distribution. The manifest itself lives in `cybershake_nshm_2022`, not this repo.

### The CI gates, exactly as CI runs them

```bash
uv sync --all-extras --dev                                       # required first: without
                                                                 # pandas-stubs, ty false-passes
uv run pytest -q
uv run ruff check
uv run ty check --exclude workflow/schemas.py --exclude setup.py
uv run deptry .
fdfind . workflow/ -E "__init__.py" --extension py | xargs numpydoc lint
```

`ruff` enforces numpydoc-style docstrings (`D101`/`D102`/`D103`) and type annotations (`ANN001`/`ANN201`) on all non-test code. Tests are exempt from docstring rules.

## Reference: the bug this plan exists to defeat

`LogEntry.from_utility` (`workflow/realisations.py:1256`) calls `importlib.metadata.version("workflow")`, which reads a **cached `.dist-info/METADATA`** stamped by setuptools-scm at *install* time. An editable install does **not** refresh it when source files change or when `HEAD` moves. `uv run` re-syncs only when `pyproject.toml` or `uv.lock` change.

Observed live on this repo, on a **clean** tree:

```
git HEAD                a4a610c6f99b931fa9ae183ec439a41269ea91f4
git status --porcelain  (empty)
installed version       0.1.dev1286+gb541da03a       <- names commit b541da03a, not HEAD
```

Left unchecked, all 291 files would record a commit that has nothing to do with the code that ran. Tasks 5, 6 and 10 exist to make that impossible.

---

## Task 1: Merge `origin/pegasus` and prove the generation path is untouched

**Files:**
- Modify: none directly — a merge commit
- Verify: `workflow/scripts/nshm2022_to_realisation.py`, `workflow/scripts/complete_realisations.py`, `workflow/scripts/generate_realisations_from_csv.py`, `workflow/realisations.py`, `workflow/default_parameters/`

**Interfaces:**
- Consumes: nothing
- Produces: a merged branch tip containing the multi-segment SRF version fix

- [ ] **Step 1: Confirm the starting state**

```bash
git rev-parse --abbrev-ref HEAD        # expect: cs-nshm2022-prep
git status --porcelain                 # expect: empty
git fetch origin pegasus
git log --oneline cs-nshm2022-prep..origin/pegasus
```

Expected exactly three commits:
```
8b39380 Merge branch 'pegasus' of github.com:ucgmsim/workflow into pegasus
b9548f7 update dockerfile to install gdal
fa92c3b Multi-segment SRF version fix (#121)
```

If more than three appear, `origin/pegasus` has moved since this plan was written. **Stop** and re-check that none of the new commits touch the generation path before continuing.

- [ ] **Step 2: Record the pre-merge SHA, then merge**

```bash
PRE_MERGE=$(git rev-parse HEAD)
echo "pre-merge: $PRE_MERGE"
git merge origin/pegasus -m "merge: bring pegasus's multi-segment SRF version fix into cs-nshm2022-prep"
```

Expected: a clean merge. If git reports conflicts, **stop** — resolve deliberately and re-run Step 3, which is the whole point of this task.

- [ ] **Step 3: Prove the merge cannot have changed realisation content**

```bash
git diff --stat "$PRE_MERGE" HEAD -- \
    workflow/scripts/nshm2022_to_realisation.py \
    workflow/scripts/complete_realisations.py \
    workflow/scripts/generate_realisations_from_csv.py \
    workflow/realisations.py \
    workflow/defaults.py \
    workflow/default_parameters/
```

Expected: **no output**. Every path that determines what goes into a realisation is byte-identical across the merge.

If this prints anything at all, the merge changed the generation path and the spec's Decision 3 assumption is void. **Stop and report** rather than proceeding.

- [ ] **Step 4: Confirm the tests still pass after the merge**

```bash
uv sync --all-extras --dev
uv run pytest -q
```

Expected: all tests pass.

---

## Task 2: Exclude `felipe_scripts/` from the lint and type gates

`felipe_scripts/` was committed on this branch as a reference input — it is Felipe's original code, kept verbatim because `complete-realisations` reproduces its parameter values and the tests read its data files. Linting it was never intended, and committing it **broke two CI gates that used to pass**: `ruff` reports 17 errors in it and `ty` one. Restore them before anything else is built on top.

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing
- Produces: green `ruff` and `ty` gates (modulo the one `D103` that Task 4 fixes)

- [ ] **Step 1: Confirm the gates are currently broken**

```bash
uv run ruff check --output-format=concise 2>&1 | tail -2
uv run ty check --exclude workflow/schemas.py --exclude setup.py 2>&1 | tail -1
```

Expected: ruff `Found 18 errors.`; ty `Found 1 diagnostic`.

- [ ] **Step 2: Exclude the directory from `ruff`**

In `pyproject.toml`, immediately **before** the existing `[tool.ruff.lint]` section, insert:

```toml
[tool.ruff]
# Felipe's reference scripts are third-party inputs, kept verbatim because
# complete-realisations reproduces their parameter values and the tests read
# their data files. They are not our source and are not linted.
exclude = ["felipe_scripts"]
```

- [ ] **Step 3: Exclude the directory from `ty`**

At the **end** of `pyproject.toml`, append:

```toml
[tool.ty.src]
exclude = ["felipe_scripts"]
```

This is why it goes in `pyproject.toml` rather than the workflow file: CI runs `ty check --exclude workflow/schemas.py --exclude setup.py`, so a config-file exclusion applies both in CI and locally, and `.github/workflows/types.yml` needs no change.

- [ ] **Step 4: Confirm both gates are restored**

```bash
uv run ty check --exclude workflow/schemas.py --exclude setup.py
uv run ruff check --output-format=concise
uv run deptry .
```

Expected:
- ty: `All checks passed!`
- ruff: exactly **one** error — `copy_realisations_to_event_dirs.py:20:5: D103 Missing docstring in public function`. This is the file Task 4 rewrites; it will go away then. No other error may remain.
- deptry: `Success! No dependency issues found.`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: exclude felipe_scripts from the ruff and ty gates

felipe_scripts/ is Felipe's original code, committed verbatim as a reference
input because complete-realisations reproduces its parameter values and the
tests read its data files. It is not our source. Committing it broke two CI
gates that had been passing."
```

---

## Task 3: `generate_realisations_from_csv` — remove the partial stub on failure

When a rupture's fault graph is disconnected, `nshm2022-to-realisation` has already written `metadata` and `seeds` before it fails, leaving a source-less file behind. That is why `minimal_realisations/` previously held 293 files of which two were unusable. Make the output directory hold exactly the valid stubs.

**Files:**
- Modify: `workflow/scripts/generate_realisations_from_csv.py`
- Test: `tests/test_generate_realisations_from_csv.py` (create)

**Interfaces:**
- Consumes: `workflow.defaults.DefaultsVersion`
- Produces:
  - `generate_one(nshmdb_path: Path, rupture_id: int, realisation_ffp: Path, defaults_version: DefaultsVersion, seeds: dict[str, int] | None = None) -> str | None` — returns an error message on failure (having deleted any partial file), or `None` on success. When `seeds` is given, it is written into the stub before generation so `nshm2022-to-realisation` replays those seeds instead of drawing fresh ones.
  - `load_seed_manifest(seed_manifest: Path) -> dict[int, dict[str, int]]` — `{rupture_id: {column: value}}` read from the seed-manifest CSV. Every column except `rupture_id` is passed through as a seed; the realisation engine validates the block when it reads it back, so the driver stays schema-agnostic.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_realisations_from_csv.py`:

```python
"""Tests for the generate_realisations_from_csv campaign driver."""

import subprocess
from pathlib import Path

import pytest

from workflow.defaults import DefaultsVersion
from workflow.scripts import generate_realisations_from_csv as gr


def test_generate_one_deletes_the_partial_stub_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # nshm2022-to-realisation writes metadata and seeds before the point at which
    # a disconnected rupture graph fails, so a crash leaves this behind.
    target = tmp_path / "realisation_59421.json"
    target.write_text('{"metadata": {}, "seeds": {}}', encoding="utf-8")

    def fail(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            1, cmd, output="some stdout", stderr="graph must be connected"
        )

    monkeypatch.setattr(subprocess, "run", fail)

    message = gr.generate_one(
        Path("nshmdb.db"), 59421, target, DefaultsVersion.v24_2_2_1
    )

    assert message is not None
    assert "59421" in message
    assert "graph must be connected" in message
    assert not target.exists()


def test_generate_one_returns_none_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"

    def succeed(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[3]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", succeed)

    assert (
        gr.generate_one(Path("nshmdb.db"), 100932, target, DefaultsVersion.v24_2_2_1)
        is None
    )
    assert target.exists()


def test_generate_one_passes_the_scientific_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}
    target = tmp_path / "realisation_42.json"

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        Path(cmd[3]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", capture)

    gr.generate_one(Path("nshmdb.db"), 42, target, DefaultsVersion.v24_2_2_1)

    assert captured["cmd"] == [
        "nshm2022-to-realisation",
        "nshmdb.db",
        "42",
        str(target),
        "24.2.2.1",
        "--dip-delta",
        "20",
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_generate_realisations_from_csv.py -q
```

Expected: FAIL — `AttributeError: module 'workflow.scripts.generate_realisations_from_csv' has no attribute 'generate_one'`.

- [ ] **Step 3: Extract `generate_one` and delete the partial**

In `workflow/scripts/generate_realisations_from_csv.py`, add this function above the `@app.command()` definition:

```python
def generate_one(
    nshmdb_path: Path,
    rupture_id: int,
    realisation_ffp: Path,
    defaults_version: DefaultsVersion,
) -> str | None:
    """Generate one minimal realisation stub for a rupture.

    Parameters
    ----------
    nshmdb_path : Path
        Path to the NSHM 2022 database file.
    rupture_id : int
        The NSHM rupture id to generate.
    realisation_ffp : Path
        Path the realisation stub is written to.
    defaults_version : DefaultsVersion
        Scientific default parameters version to use.

    Returns
    -------
    str or None
        An error message if generation failed, otherwise None.
    """
    cmd = [
        "nshm2022-to-realisation",
        str(nshmdb_path),
        str(rupture_id),
        str(realisation_ffp),
        str(defaults_version),
        "--dip-delta",
        "20",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # metadata and seeds are written before the point at which a disconnected
        # rupture graph fails, so a crash leaves a source-less file behind.
        realisation_ffp.unlink(missing_ok=True)
        return (
            f"Failed to generate realisation for rupture {rupture_id}, skipping:\n"
            f"--- stdout ---\n{exc.stdout}\n"
            f"--- stderr ---\n{exc.stderr}\n"
            f"--- return code: {exc.returncode} ---\n"
        )
    return None
```

Then replace the body of the `for rupture_id in tqdm(...)` loop with:

```python
    for rupture_id in tqdm(rupture_ids, desc="Generating realisations"):
        realisation_ffp = output_dir / f"realisation_{rupture_id}.json"
        try:
            error_msg = generate_one(
                nshmdb_path, rupture_id, realisation_ffp, defaults_version
            )
        except FileNotFoundError:
            print(
                "\n'nshm2022-to-realisation' command not found. "
                "Is the workflow package installed?"
            )
            raise typer.Exit(code=1)
        if error_msg is not None:
            print(f"\n{error_msg}")
            error_log_handle.write(error_msg + "\n")
            error_log_handle.flush()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_generate_realisations_from_csv.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/generate_realisations_from_csv.py tests/test_generate_realisations_from_csv.py
git commit -m "fix(scripts): delete the partial stub when rupture generation fails

nshm2022-to-realisation writes metadata and seeds before the point at which a
disconnected fault graph fails, leaving a source-less file behind. The stub
directory should hold exactly the valid stubs, not 293 files of which two are
booby traps whose only defence is a downstream is_valid_minimal check."
```

- [ ] **Step 6: Write the failing seed-injection tests**

Add `import json` to the top of `tests/test_generate_realisations_from_csv.py` (beside `import subprocess`), then append these three tests:

```python
def test_generate_one_writes_seeds_into_the_stub_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    seeds = {
        "nshm_to_realisation_seed": 531798913,
        "rupture_propagation_seed": 31268976,
        "genslip_seed": 513004717,
        "srfgen_seed": 1837842819,
        "hf_seed": 1524796118,
    }
    seen: dict[str, object] = {}

    def capture_stub(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen["stub"] = json.loads(Path(cmd[3]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", capture_stub)

    result = gr.generate_one(
        Path("nshmdb.db"), 100932, target, DefaultsVersion.v24_2_2_1, seeds=seeds
    )

    assert result is None
    assert seen["stub"]["seeds"] == seeds


def test_generate_one_writes_no_stub_when_seeds_are_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    seen: dict[str, bool] = {}

    def note_absence(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen["existed_before"] = Path(cmd[3]).exists()
        Path(cmd[3]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", note_absence)

    gr.generate_one(Path("nshmdb.db"), 100932, target, DefaultsVersion.v24_2_2_1)

    assert seen["existed_before"] is False


def test_load_seed_manifest_reads_rows_keyed_by_rupture_id(tmp_path: Path) -> None:
    manifest = tmp_path / "seed_manifest.csv"
    manifest.write_text(
        "rupture_id,nshm_to_realisation_seed,rupture_propagation_seed,"
        "genslip_seed,srfgen_seed,hf_seed\n"
        "100932,1,2,3,4,5\n"
        "101084,6,7,8,9,10\n",
        encoding="utf-8",
    )

    seeds_by_rupture = gr.load_seed_manifest(manifest)

    assert seeds_by_rupture[100932] == {
        "nshm_to_realisation_seed": 1,
        "rupture_propagation_seed": 2,
        "genslip_seed": 3,
        "srfgen_seed": 4,
        "hf_seed": 5,
    }
    assert seeds_by_rupture[101084]["hf_seed"] == 10
```

- [ ] **Step 7: Run the tests to verify they fail**

```bash
uv run pytest tests/test_generate_realisations_from_csv.py -q
```

Expected: FAIL — `TypeError: generate_one() got an unexpected keyword argument 'seeds'` and `AttributeError: module '...' has no attribute 'load_seed_manifest'`.

- [ ] **Step 8: Carry seeds through when a manifest is supplied**

Add `import json` beside the existing `import subprocess` at the top of `workflow/scripts/generate_realisations_from_csv.py`.

Give `generate_one` a `seeds` parameter, and write the seed stub before generating. Replace the function with:

```python
def generate_one(
    nshmdb_path: Path,
    rupture_id: int,
    realisation_ffp: Path,
    defaults_version: DefaultsVersion,
    seeds: dict[str, int] | None = None,
) -> str | None:
    """Generate one minimal realisation stub for a rupture.

    Parameters
    ----------
    nshmdb_path : Path
        Path to the NSHM 2022 database file.
    rupture_id : int
        The NSHM rupture id to generate.
    realisation_ffp : Path
        Path the realisation stub is written to.
    defaults_version : DefaultsVersion
        Scientific default parameters version to use.
    seeds : dict of str to int, optional
        When given, written into the stub before generation so
        nshm2022-to-realisation replays these seeds instead of drawing fresh ones.

    Returns
    -------
    str or None
        An error message if generation failed, otherwise None.
    """
    if seeds is not None:
        realisation_ffp.write_text(
            json.dumps({"metadata": {}, "seeds": seeds}), encoding="utf-8"
        )
    cmd = [
        "nshm2022-to-realisation",
        str(nshmdb_path),
        str(rupture_id),
        str(realisation_ffp),
        str(defaults_version),
        "--dip-delta",
        "20",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # metadata and seeds are written before the point at which a disconnected
        # rupture graph fails, so a crash leaves a source-less file behind.
        realisation_ffp.unlink(missing_ok=True)
        return (
            f"Failed to generate realisation for rupture {rupture_id}, skipping:\n"
            f"--- stdout ---\n{exc.stdout}\n"
            f"--- stderr ---\n{exc.stderr}\n"
            f"--- return code: {exc.returncode} ---\n"
        )
    return None
```

Add the manifest loader directly above the `@app.command()` line:

```python
def load_seed_manifest(seed_manifest: Path) -> dict[int, dict[str, int]]:
    """Load a seed manifest CSV into per-rupture seed dictionaries.

    Parameters
    ----------
    seed_manifest : Path
        CSV with a ``rupture_id`` column and one column per seed.

    Returns
    -------
    dict of int to (dict of str to int)
        Maps each rupture id to its seed block. Every column except
        ``rupture_id`` is passed through unchanged; the realisation engine
        validates the block when it reads it back.
    """
    df = pd.read_csv(seed_manifest)
    if "rupture_id" not in df.columns:
        raise ValueError(
            f"Seed manifest {seed_manifest} must contain a 'rupture_id' column."
        )
    seed_columns = [column for column in df.columns if column != "rupture_id"]
    return {
        int(row["rupture_id"]): {column: int(row[column]) for column in seed_columns}
        for _, row in df.iterrows()
    }
```

Add the `--seed-manifest` option to the command signature, immediately after the `defaults_version` argument:

```python
    defaults_version: Annotated[DefaultsVersion, typer.Argument()],
    seed_manifest: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="CSV of rupture_id + seeds to replay, one row per rupture.",
        ),
    ] = None,
```

Load it near the top of the command body, just after the `df = pd.read_csv(csv_file)` block that builds `rupture_ids`:

```python
    seeds_by_rupture: dict[int, dict[str, int]] = {}
    if seed_manifest is not None:
        seeds_by_rupture = load_seed_manifest(seed_manifest)
```

And pass the per-rupture seeds into the call inside the loop:

```python
            error_msg = generate_one(
                nshmdb_path,
                rupture_id,
                realisation_ffp,
                defaults_version,
                seeds=seeds_by_rupture.get(rupture_id),
            )
```

- [ ] **Step 9: Run the tests to verify they pass**

```bash
uv run pytest tests/test_generate_realisations_from_csv.py -q
```

Expected: `6 passed`.

- [ ] **Step 10: Commit**

```bash
git add workflow/scripts/generate_realisations_from_csv.py tests/test_generate_realisations_from_csv.py
git commit -m "feat(scripts): add opt-in --seed-manifest to replay recorded seeds

When a seed manifest is supplied, each stub is pre-written with its rupture's
five seeds so nshm2022-to-realisation replays them via
read_from_realisation_or_random instead of drawing fresh ones. Without the flag,
behaviour is unchanged. The driver treats every non-rupture_id column as a seed
and lets the engine validate the block, so it needs no knowledge of the seed
schema."
```

---

## Task 4: Move `copy_realisations_to_event_dirs` into the package and give it a CLI

The script is committed with a hardcoded path — `/home/arr65/src/cybershake_nshm_2022/flow/events` — that **no longer exists**; the tree moved to `cybershake_nshm_2022/cybershake_nshm_2022/events/`. Move it alongside the other campaign tools and take both directories as arguments. This also clears the last outstanding `ruff` error.

**Files:**
- Move: `copy_realisations_to_event_dirs.py` → `workflow/scripts/copy_realisations_to_event_dirs.py`
- Modify: `pyproject.toml`
- Test: `tests/test_copy_realisations_to_event_dirs.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces: `copy_realisations(source_dir: Path, events_dir: Path) -> tuple[int, list[str]]` — returns the number copied and the names of files skipped for want of an integer id.

- [ ] **Step 1: Move the file with `git mv` to preserve history**

```bash
git mv copy_realisations_to_event_dirs.py workflow/scripts/copy_realisations_to_event_dirs.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_copy_realisations_to_event_dirs.py`:

```python
"""Tests for the copy_realisations_to_event_dirs campaign tool."""

from pathlib import Path

from workflow.scripts import copy_realisations_to_event_dirs as cre


def test_copy_realisations_creates_one_dir_per_rupture(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"a": 1}', encoding="utf-8")
    (source / "realisation_71220.json").write_text('{"b": 2}', encoding="utf-8")
    events = tmp_path / "events"

    copied, skipped = cre.copy_realisations(source, events)

    assert copied == 2
    assert skipped == []
    assert (events / "100932" / "realisation.json").read_text() == '{"a": 1}'
    assert (events / "71220" / "realisation.json").read_text() == '{"b": 2}'


def test_copy_realisations_skips_files_without_an_integer_id(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text("{}", encoding="utf-8")
    (source / "notes.json").write_text("{}", encoding="utf-8")
    events = tmp_path / "events"

    copied, skipped = cre.copy_realisations(source, events)

    assert copied == 1
    assert skipped == ["notes.json"]


def test_copy_realisations_overwrites_an_existing_realisation(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"new": 1}', encoding="utf-8")
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    copied, _ = cre.copy_realisations(source, events)

    assert copied == 1
    assert (events / "100932" / "realisation.json").read_text() == '{"new": 1}'
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run pytest tests/test_copy_realisations_to_event_dirs.py -q
```

Expected: FAIL — `AttributeError: ... has no attribute 'copy_realisations'`.

- [ ] **Step 4: Rewrite the script**

Replace the entire contents of `workflow/scripts/copy_realisations_to_event_dirs.py`:

```python
#!/usr/bin/env python3
"""Distribute completed realisations into per-event CyberShake directories.

Description
-----------
For every ``realisation_<id>.json`` in the source directory, create a directory
named after the rupture id under the events directory and copy the realisation
into it as ``realisation.json``.

Usage
-----
``copy-realisations-to-event-dirs SOURCE_DIR EVENTS_DIR``
"""

import re
import shutil
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


def copy_realisations(source_dir: Path, events_dir: Path) -> tuple[int, list[str]]:
    """Copy each realisation into ``events_dir/<rupture id>/realisation.json``.

    Parameters
    ----------
    source_dir : Path
        Directory of ``realisation_<id>.json`` files.
    events_dir : Path
        Directory to create per-rupture event directories under.

    Returns
    -------
    tuple of (int, list of str)
        The number of realisations copied, and the names of any files skipped
        because no integer id could be read from the filename.
    """
    copied = 0
    skipped: list[str] = []

    for realisation_ffp in sorted(source_dir.glob("*.json")):
        match = re.search(r"\d+", realisation_ffp.name)
        if match is None:
            skipped.append(realisation_ffp.name)
            continue

        event_dir = events_dir / match.group()
        event_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(realisation_ffp, event_dir / "realisation.json")
        copied += 1

    return copied, skipped


@app.command()
def copy_realisations_to_event_dirs(
    source_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    events_dir: Annotated[Path, typer.Argument(file_okay=False)],
) -> None:
    """Distribute completed realisations into per-event CyberShake directories.

    Parameters
    ----------
    source_dir : Path
        Directory of ``realisation_<id>.json`` files.
    events_dir : Path
        Directory to create per-rupture event directories under.
    """
    copied, skipped = copy_realisations(source_dir, events_dir)

    print(f"Copied {copied} realisation(s) into {events_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with no integer id:")
        for name in skipped:
            print(f"  {name}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Register the entry point**

In `pyproject.toml`, in `[project.scripts]`, immediately after the `complete-realisations` line, add:

```toml
copy-realisations-to-event-dirs = "workflow.scripts.copy_realisations_to_event_dirs:app"
```

- [ ] **Step 6: Run the tests and confirm `ruff` is now fully clean**

```bash
uv sync --all-extras --dev
uv run pytest tests/test_copy_realisations_to_event_dirs.py -q
uv run ruff check
```

Expected: `3 passed`, and ruff reports **`All checks passed!`** — the `D103` from Task 2 Step 4 is now gone.

- [ ] **Step 7: Commit**

```bash
git add workflow/scripts/copy_realisations_to_event_dirs.py tests/test_copy_realisations_to_event_dirs.py pyproject.toml
git commit -m "refactor(scripts): make copy-realisations-to-event-dirs a proper CLI

The script was committed with a hardcoded events path that no longer exists.
Move it in with the other campaign tools, take both directories as arguments,
and give it an entry point so the whole campaign runs through entry points."
```

---

## Task 5: `verify_realisation_provenance` — version parsing and the pre-flight gate

This is the component that makes the stale-metadata bug impossible to miss. It must exist before anything is generated.

**Files:**
- Create: `workflow/scripts/verify_realisation_provenance.py`
- Test: `tests/test_verify_realisation_provenance.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ScmVersion` — frozen dataclass with fields `raw: str`, `sha: str`, `dirty: bool`
  - `parse_scm_version(version: str) -> ScmVersion` — raises `ValueError` if there is no `+g<sha>` local segment
  - `git_head_sha(repo_root: Path) -> str` — full 40-character SHA
  - `git_is_clean(repo_root: Path) -> bool` — tracked modifications only
  - `preflight_problems(repo_root: Path, installed_version: str) -> list[str]` — empty means fit to run
  - `EXPECTED_UTILITIES: tuple[str, ...]`, `REINSTALL_COMMAND: str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verify_realisation_provenance.py`:

```python
"""Tests for the realisation provenance verifier."""

from pathlib import Path

import pytest

from workflow.scripts import verify_realisation_provenance as vp

# A real clean stamp observed on this repo.
CLEAN_VERSION = "0.1.dev1286+gb541da03a"
CLEAN_SHA = "b541da03a" + "0" * 31

# The stamp the existing (untrustworthy) 291 realisations carry.
DIRTY_VERSION = "0.1.dev1277+g41974dfa1.d20260709"


def test_parse_scm_version_reads_a_clean_stamp() -> None:
    version = vp.parse_scm_version(CLEAN_VERSION)
    assert version.sha == "b541da03a"
    assert version.dirty is False


def test_parse_scm_version_reads_a_dirty_stamp() -> None:
    version = vp.parse_scm_version(DIRTY_VERSION)
    assert version.sha == "41974dfa1"
    assert version.dirty is True


def test_parse_scm_version_rejects_a_stamp_naming_no_commit() -> None:
    with pytest.raises(ValueError, match="identifies no commit"):
        vp.parse_scm_version("0.1.dev1286")


def test_preflight_passes_when_the_stamp_matches_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vp, "git_is_clean", lambda repo_root: True)
    monkeypatch.setattr(vp, "git_head_sha", lambda repo_root: CLEAN_SHA)

    assert vp.preflight_problems(tmp_path, CLEAN_VERSION) == []


def test_preflight_catches_stale_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The exact failure observed live: clean tree, but the cached .dist-info
    # still names an older commit than HEAD.
    monkeypatch.setattr(vp, "git_is_clean", lambda repo_root: True)
    monkeypatch.setattr(
        vp, "git_head_sha", lambda repo_root: "a4a610c6f99b931fa9ae183ec439a41269ea91f4"
    )

    problems = vp.preflight_problems(tmp_path, CLEAN_VERSION)

    assert len(problems) == 1
    assert "STALE METADATA" in problems[0]
    assert "uv sync --reinstall-package workflow" in problems[0]


def test_preflight_catches_a_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vp, "git_is_clean", lambda repo_root: False)
    monkeypatch.setattr(vp, "git_head_sha", lambda repo_root: CLEAN_SHA)

    problems = vp.preflight_problems(tmp_path, CLEAN_VERSION)

    assert any("modified tracked files" in problem for problem in problems)


def test_preflight_catches_a_dirty_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vp, "git_is_clean", lambda repo_root: True)
    monkeypatch.setattr(vp, "git_head_sha", lambda repo_root: "41974dfa1" + "0" * 31)

    problems = vp.preflight_problems(tmp_path, DIRTY_VERSION)

    assert any("dirty suffix" in problem for problem in problems)
```

Import only what this task uses. Ruff's `F401` (unused import) is on by default and `tests/**.py` is exempt only from the `D` docstring rules, so an import added before it is needed is an error. Task 6 adds the two it needs.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_verify_realisation_provenance.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.scripts.verify_realisation_provenance'`.

- [ ] **Step 3: Write the module**

Create `workflow/scripts/verify_realisation_provenance.py`:

```python
#!/usr/bin/env python3
"""Verify the provenance recorded in realisation files.

Description
-----------
``log_trail`` records the ``workflow`` version reported by
``importlib.metadata.version``, which reads a **cached** ``.dist-info`` stamped
by setuptools-scm at *install* time. An editable install does not refresh it
when source files change or when ``HEAD`` moves, and ``uv run`` re-syncs only
when ``pyproject.toml`` or ``uv.lock`` change. The recorded version can
therefore name a commit that has nothing to do with the code that ran — which is
exactly what happened to the first, untraceable batch of realisations.

This tool exists to make that failure impossible to miss.

Usage
-----
``verify-realisation-provenance --preflight`` checks the environment *before* a
campaign, and refuses to proceed unless the tree is clean and the installed
metadata matches ``HEAD``.

``verify-realisation-provenance REALISATION_DIR`` audits finished realisations.
"""

import dataclasses
import re
import subprocess
from pathlib import Path

import typer

app = typer.Typer()


SCM_VERSION_RE = re.compile(
    r"^(?P<public>[^+]+)\+g(?P<sha>[0-9a-f]+)(?:\.d(?P<dirty>\d{8}))?$"
)

EXPECTED_UTILITIES: tuple[str, ...] = (
    "nshm2022-to-realisation",
    "complete-realisations",
)

REINSTALL_COMMAND = "uv sync --reinstall-package workflow --all-extras --dev"


@dataclasses.dataclass(frozen=True)
class ScmVersion:
    """A setuptools-scm version string, decomposed.

    Attributes
    ----------
    raw : str
        The version string as recorded.
    sha : str
        The abbreviated commit SHA from the local version segment.
    dirty : bool
        Whether the version carries a ``.d<YYYYMMDD>`` dirty suffix.
    """

    raw: str
    sha: str
    dirty: bool


def parse_scm_version(version: str) -> ScmVersion:
    """Decompose a setuptools-scm version string.

    Parameters
    ----------
    version : str
        A version of the form ``0.1.dev1286+gb541da03a``, optionally carrying a
        ``.d<YYYYMMDD>`` dirty suffix.

    Returns
    -------
    ScmVersion
        The decomposed version.

    Raises
    ------
    ValueError
        If the version has no ``+g<sha>`` local segment, and so identifies no
        commit at all.
    """
    match = SCM_VERSION_RE.match(version)
    if match is None:
        raise ValueError(
            f"Version {version!r} has no +g<sha> local segment: it identifies no commit."
        )
    return ScmVersion(raw=version, sha=match["sha"], dirty=match["dirty"] is not None)


def git_head_sha(repo_root: Path) -> str:
    """Return the full 40-character SHA of ``HEAD``.

    Parameters
    ----------
    repo_root : Path
        Root of the git repository.

    Returns
    -------
    str
        The full SHA of ``HEAD``.
    """
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_is_clean(repo_root: Path) -> bool:
    """Return whether the working tree has no modified tracked files.

    Untracked files are ignored, matching setuptools-scm's own dirty check,
    which considers only tracked modifications. This matters: the campaign writes
    into gitignored directories and so cannot dirty its own tree.

    Parameters
    ----------
    repo_root : Path
        Root of the git repository.

    Returns
    -------
    bool
        True if no tracked file is modified.
    """
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return status == ""


def preflight_problems(repo_root: Path, installed_version: str) -> list[str]:
    """Return the reasons this environment is unfit to run a traceable campaign.

    Parameters
    ----------
    repo_root : Path
        Root of the workflow git repository.
    installed_version : str
        The version reported by ``importlib.metadata.version("workflow")``.

    Returns
    -------
    list of str
        One message per problem. Empty means the environment is fit to run.
    """
    problems: list[str] = []

    if not git_is_clean(repo_root):
        problems.append(
            "Working tree has modified tracked files. setuptools-scm will stamp a "
            "'.d<date>' dirty suffix and the recorded SHA will not identify the code "
            "that ran."
        )

    try:
        version = parse_scm_version(installed_version)
    except ValueError as exc:
        problems.append(str(exc))
        return problems

    if version.dirty:
        problems.append(
            f"Installed version {version.raw!r} carries a dirty suffix. Commit or stash, "
            f"then run: {REINSTALL_COMMAND}"
        )

    head = git_head_sha(repo_root)
    if not head.startswith(version.sha):
        problems.append(
            f"STALE METADATA: the installed version names commit {version.sha}, but HEAD "
            f"is {head[: len(version.sha)]}. importlib.metadata reads a cached .dist-info "
            f"that is not refreshed when HEAD moves, so every realisation would record "
            f"the wrong commit. Run: {REINSTALL_COMMAND}"
        )

    return problems
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_verify_realisation_provenance.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/verify_realisation_provenance.py tests/test_verify_realisation_provenance.py
git commit -m "feat(provenance): add pre-flight gate catching stale version metadata

importlib.metadata reads a cached .dist-info that uv does not refresh when HEAD
moves, so a clean tree is no guarantee the recorded SHA names the code that ran.
Assert the installed stamp matches HEAD before any realisation is written."
```

---

## Task 6: `verify_realisation_provenance` — the post-hoc audit and the CLI

**Files:**
- Modify: `workflow/scripts/verify_realisation_provenance.py`
- Modify: `pyproject.toml`
- Test: `tests/test_verify_realisation_provenance.py`

**Interfaces:**
- Consumes: `parse_scm_version`, `preflight_problems`, `EXPECTED_UTILITIES` (Task 5); `FELIPE_SECTION_ORDER` from `workflow.scripts.complete_realisations`
- Produces: `realisation_problems(realisation_ffp: Path, expected_version: str) -> list[str]`; entry point `verify-realisation-provenance`

- [ ] **Step 1: Write the failing tests**

First extend the import block at the **top** of `tests/test_verify_realisation_provenance.py` — do not add imports mid-file, ruff's isort rule (`I`) rejects it:

```python
import json
from pathlib import Path

import pytest

from workflow.scripts import verify_realisation_provenance as vp
from workflow.scripts.complete_realisations import FELIPE_SECTION_ORDER
```

Then append to the same file:

```python
def write_realisation(
    path: Path,
    version: str = CLEAN_VERSION,
    utilities: tuple[str, ...] = ("nshm2022-to-realisation", "complete-realisations"),
    sections: list[str] | None = None,
) -> Path:
    sections = FELIPE_SECTION_ORDER if sections is None else sections
    realisation: dict[str, object] = {section: {} for section in sections}
    if "log_trail" in realisation:
        realisation["log_trail"] = {
            "log": [
                {
                    "utility": utility,
                    "version": version,
                    "timestamp": "2026-07-14T00:00:00",
                    "args": [],
                }
                for utility in utilities
            ]
        }
    path.write_text(json.dumps(realisation), encoding="utf-8")
    return path


def test_realisation_problems_accepts_a_good_file(tmp_path: Path) -> None:
    path = write_realisation(tmp_path / "realisation_100932.json")
    assert vp.realisation_problems(path, CLEAN_VERSION) == []


def test_realisation_problems_catches_a_dirty_recorded_version(tmp_path: Path) -> None:
    path = write_realisation(tmp_path / "realisation_100932.json", version=DIRTY_VERSION)

    problems = vp.realisation_problems(path, CLEAN_VERSION)

    # One per log entry.
    assert len(problems) == 2
    assert all(DIRTY_VERSION in problem for problem in problems)


def test_realisation_problems_catches_the_old_utility_name(tmp_path: Path) -> None:
    path = write_realisation(
        tmp_path / "realisation_100932.json",
        utilities=("nshm2022-to-realisation", "bake_realisations.py"),
    )

    problems = vp.realisation_problems(path, CLEAN_VERSION)

    assert any("bake_realisations.py" in problem for problem in problems)


def test_realisation_problems_catches_a_missing_log_entry(tmp_path: Path) -> None:
    path = write_realisation(
        tmp_path / "realisation_100932.json", utilities=("nshm2022-to-realisation",)
    )

    problems = vp.realisation_problems(path, CLEAN_VERSION)

    assert any("complete-realisations" in problem for problem in problems)


def test_realisation_problems_catches_a_missing_section(tmp_path: Path) -> None:
    sections = [s for s in FELIPE_SECTION_ORDER if s != "domain"]
    path = write_realisation(tmp_path / "realisation_100932.json", sections=sections)

    problems = vp.realisation_problems(path, CLEAN_VERSION)

    assert any("missing ['domain']" in problem for problem in problems)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_verify_realisation_provenance.py -q
```

Expected: FAIL — `AttributeError: ... has no attribute 'realisation_problems'`.

- [ ] **Step 3: Add the audit function and the CLI**

First extend the import block at the **top** of `workflow/scripts/verify_realisation_provenance.py` to add the four names this task needs (`json`, `metadata`, `Annotated`, `FELIPE_SECTION_ORDER`) — Task 5 deliberately left them out because ruff's `F401` rejects an import before its first use:

```python
import dataclasses
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from workflow.scripts.complete_realisations import FELIPE_SECTION_ORDER
```

Then append to the same file:

```python
def realisation_problems(realisation_ffp: Path, expected_version: str) -> list[str]:
    """Return the provenance defects in one finished realisation.

    Parameters
    ----------
    realisation_ffp : Path
        Path to a complete realisation file.
    expected_version : str
        The version string every ``log_trail`` entry must carry.

    Returns
    -------
    list of str
        One message per defect. Empty means the file's provenance is sound.
    """
    problems: list[str] = []
    realisation = json.loads(realisation_ffp.read_text(encoding="utf-8"))

    log = realisation.get("log_trail", {}).get("log", [])
    utilities = [entry.get("utility") for entry in log]
    if utilities != list(EXPECTED_UTILITIES):
        problems.append(
            f"log_trail records utilities {utilities}, expected {list(EXPECTED_UTILITIES)}"
        )

    for entry in log:
        if entry.get("version") != expected_version:
            problems.append(
                f"{entry.get('utility')} recorded version {entry.get('version')!r}, "
                f"expected {expected_version!r}"
            )

    sections = list(realisation)
    if sections != FELIPE_SECTION_ORDER:
        missing = sorted(set(FELIPE_SECTION_ORDER) - set(sections))
        unexpected = sorted(set(sections) - set(FELIPE_SECTION_ORDER))
        detail = ""
        if missing:
            detail += f"; missing {missing}"
        if unexpected:
            detail += f"; unexpected {unexpected}"
        problems.append(f"sections are not the canonical 18 in order{detail}")

    return problems


@app.command()
def verify_realisation_provenance(
    realisation_dir: Annotated[
        Path | None, typer.Argument(exists=True, file_okay=False)
    ] = None,
    preflight: Annotated[
        bool,
        typer.Option(
            "--preflight",
            help="Check the environment is fit to run, instead of auditing output.",
        ),
    ] = False,
    expect_version: Annotated[
        str | None,
        typer.Option(
            help="Version every log_trail entry must carry. "
            "Defaults to the installed workflow version."
        ),
    ] = None,
    repo_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("."),
) -> None:
    """Verify realisation provenance, or that the environment can produce it.

    Parameters
    ----------
    realisation_dir : Path, optional
        Directory of complete realisations to audit. Required unless
        ``--preflight`` is given.
    preflight : bool
        Check the environment rather than auditing an output directory.
    expect_version : str, optional
        Version every ``log_trail`` entry must carry. Defaults to the installed
        ``workflow`` version.
    repo_root : Path
        Root of the workflow git repository.
    """
    installed = metadata.version("workflow")

    if preflight:
        problems = preflight_problems(repo_root, installed)
        if problems:
            print("PREFLIGHT FAILED — refusing to run a campaign:")
            for problem in problems:
                print(f"  - {problem}")
            raise typer.Exit(code=1)
        print(f"Preflight OK. Realisations will record version {installed}")
        return

    if realisation_dir is None:
        print("Provide a realisation directory, or pass --preflight.")
        raise typer.Exit(code=1)

    expected = expect_version or installed
    realisations = sorted(realisation_dir.glob("realisation_*.json"))
    failures = {
        realisation_ffp: problems
        for realisation_ffp in realisations
        if (problems := realisation_problems(realisation_ffp, expected))
    }

    if failures:
        print(
            f"PROVENANCE FAILED for {len(failures)} of {len(realisations)} realisation(s):"
        )
        for realisation_ffp, problems in failures.items():
            print(f"  {realisation_ffp.name}")
            for problem in problems:
                print(f"    - {problem}")
        raise typer.Exit(code=1)

    print(f"Provenance OK: {len(realisations)} realisation(s), all recording {expected}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Register the entry point**

In `pyproject.toml`, in `[project.scripts]`, after the `copy-realisations-to-event-dirs` line, add:

```toml
verify-realisation-provenance = "workflow.scripts.verify_realisation_provenance:app"
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv sync --all-extras --dev
uv run pytest tests/test_verify_realisation_provenance.py -q
uv run ruff check
```

Expected: `12 passed`, and ruff `All checks passed!` — confirming the imports added in Steps 1 and 3 are all used.

- [ ] **Step 6: Prove the checker catches the real, existing bad files**

The 291 files from the untraceable run are still on disk. The verifier must reject them.

```bash
uv run verify-realisation-provenance complete_realisations --expect-version "0.1.dev9999+gdeadbeef"
```

Expected: exits **1**, reporting failures for all 291 — each naming `bake_realisations.py` as an unexpected utility and the dirty `0.1.dev1277+g41974dfa1.d20260709` version. This is the checker working: the files it rejects are exactly the ones this campaign exists to replace.

- [ ] **Step 7: Commit**

```bash
git add workflow/scripts/verify_realisation_provenance.py tests/test_verify_realisation_provenance.py pyproject.toml
git commit -m "feat(provenance): audit finished realisations against an expected version

Assert every log_trail carries exactly the two expected utilities, in order,
each stamped with the pinned version, and that all 18 canonical sections are
present. Verified to reject the existing untraceable 291."
```

---

## Task 7: `compare_nshmdb` — logical comparison of two NSHM databases

Needed by Task 9. Byte-identity is **not** the test: SQLite page layout and rowid allocation need not be stable across builds. What must match is the content.

**Files:**
- Create: `workflow/scripts/compare_nshmdb.py`
- Modify: `pyproject.toml`
- Test: `tests/test_compare_nshmdb.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces: `table_names(connection: sqlite3.Connection) -> list[str]`; `table_digest(connection: sqlite3.Connection, table: str) -> tuple[int, str]`; entry point `compare-nshmdb`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compare_nshmdb.py`:

```python
"""Tests for the NSHM database comparator."""

import sqlite3
from pathlib import Path

from workflow.scripts import compare_nshmdb as cn


def build_db(path: Path, rows: list[tuple[int, str]]) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE rupture (i INTEGER, s TEXT)")
    connection.executemany("INSERT INTO rupture VALUES (?, ?)", rows)
    connection.commit()
    return connection


def test_table_digest_is_independent_of_row_order(tmp_path: Path) -> None:
    left = build_db(tmp_path / "left.db", [(1, "x"), (2, "y"), (3, "z")])
    right = build_db(tmp_path / "right.db", [(3, "z"), (1, "x"), (2, "y")])

    assert cn.table_digest(left, "rupture") == cn.table_digest(right, "rupture")


def test_table_digest_detects_a_changed_row(tmp_path: Path) -> None:
    left = build_db(tmp_path / "left.db", [(1, "x"), (2, "y")])
    right = build_db(tmp_path / "right.db", [(1, "x"), (2, "CHANGED")])

    assert cn.table_digest(left, "rupture") != cn.table_digest(right, "rupture")


def test_table_digest_detects_a_duplicated_row(tmp_path: Path) -> None:
    # An XOR-based digest would cancel the duplicate out and call these equal.
    # Summing the row digests must not.
    left = build_db(tmp_path / "left.db", [(1, "x"), (2, "y")])
    right = build_db(tmp_path / "right.db", [(1, "x"), (2, "y"), (2, "y")])

    assert cn.table_digest(left, "rupture") != cn.table_digest(right, "rupture")


def test_table_names_are_sorted(tmp_path: Path) -> None:
    connection = build_db(tmp_path / "db.db", [])
    connection.execute("CREATE TABLE fault (i INTEGER)")
    connection.commit()

    assert cn.table_names(connection) == ["fault", "rupture"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_compare_nshmdb.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.scripts.compare_nshmdb'`.

- [ ] **Step 3: Write the module**

Create `workflow/scripts/compare_nshmdb.py`:

```python
#!/usr/bin/env python3
"""Compare two NSHM databases on logical content.

Description
-----------
Byte-identity is not expected and is not the test: SQLite page layout and rowid
allocation need not be stable across builds of the same data. What must match is
the content. Each table's rows are hashed individually and the digests summed,
which is independent of the order SQLite returns them in, sensitive to a row's
multiplicity (an XOR would not be), and uses constant memory — which matters,
because ``rupture_faults`` has around 20 million rows.

Usage
-----
``compare-nshmdb LEFT_DB RIGHT_DB``
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


def table_names(connection: sqlite3.Connection) -> list[str]:
    """Return the names of the tables in a database, sorted.

    Parameters
    ----------
    connection : sqlite3.Connection
        An open connection to the database.

    Returns
    -------
    list of str
        The sorted table names.
    """
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )
    return [name for (name,) in rows]


def table_digest(connection: sqlite3.Connection, table: str) -> tuple[int, str]:
    """Return the row count and an order-independent content hash for one table.

    Parameters
    ----------
    connection : sqlite3.Connection
        An open connection to the database.
    table : str
        Name of the table to digest. Must come from :func:`table_names`.

    Returns
    -------
    tuple of (int, str)
        The number of rows, and a 64-character hex content digest.
    """
    total = 0
    count = 0
    for row in connection.execute(f'SELECT * FROM "{table}"'):
        row_digest = hashlib.sha256(repr(row).encode("utf-8")).digest()
        total = (total + int.from_bytes(row_digest, "big")) % (2**256)
        count += 1
    return count, f"{total:064x}"


@app.command()
def compare_nshmdb(
    left: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    right: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Compare two NSHM databases on logical content, exiting 1 if they differ.

    Parameters
    ----------
    left : Path
        The first database.
    right : Path
        The second database.
    """
    left_connection = sqlite3.connect(left)
    right_connection = sqlite3.connect(right)

    left_tables = table_names(left_connection)
    right_tables = table_names(right_connection)

    differences: list[str] = []
    if left_tables != right_tables:
        differences.append(f"table sets differ: {left_tables} vs {right_tables}")

    for table in sorted(set(left_tables) & set(right_tables)):
        left_count, left_hash = table_digest(left_connection, table)
        right_count, right_hash = table_digest(right_connection, table)
        same = (left_count, left_hash) == (right_count, right_hash)
        print(
            f"{table:36s} {left_count:>11,} rows  {left_hash[:16]}  "
            f"{'same' if same else 'DIFFER'}"
        )
        if not same:
            differences.append(
                f"{table}: {left_count:,} rows / {left_hash[:16]} vs "
                f"{right_count:,} rows / {right_hash[:16]}"
            )

    if differences:
        print("\nDATABASES DIFFER:")
        for difference in differences:
            print(f"  - {difference}")
        raise typer.Exit(code=1)

    print("\nDatabases are logically identical.")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Register the entry point**

In `pyproject.toml`, in `[project.scripts]`, after the `verify-realisation-provenance` line, add:

```toml
compare-nshmdb = "workflow.scripts.compare_nshmdb:app"
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv sync --all-extras --dev
uv run pytest tests/test_compare_nshmdb.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add workflow/scripts/compare_nshmdb.py tests/test_compare_nshmdb.py pyproject.toml
git commit -m "feat(provenance): add logical comparison of two NSHM databases

Sum per-row hashes rather than compare bytes: SQLite page layout need not be
stable across builds of identical data, and rupture_faults has ~2e7 rows, so the
digest must use constant memory and be order-independent."
```

---

## Task 7a: `build_seed_manifest` — extract the seeds the originals were generated with

The seed manifest is what lets the campaign reproduce the existing files rather than re-draw them. This tool builds it from the committed originals, asserting as it goes that every file has a complete seed block and that each file's own `log_trail` names the rupture id its directory claims — so the manifest is provably faithful.

**Files:**
- Create: `workflow/scripts/build_seed_manifest.py`
- Modify: `pyproject.toml` (add a `[project.scripts]` entry)
- Test: `tests/test_build_seed_manifest.py` (create)

**Interfaces:**
- Consumes: `workflow.realisations.Seeds` (only for the canonical field list)
- Produces:
  - `SEED_FIELDS: tuple[str, ...]` — the five seed field names, in `Seeds` definition order
  - `seed_row(realisation_ffp: Path, rupture_id: int) -> dict[str, int]` — `{rupture_id + five seeds}`; raises `ValueError` on an incomplete block, a `log_trail` that names a different rupture, or non-uniform generation args
  - `build_seed_manifest(events_dir: Path, output_csv: Path) -> int` — writes the sorted CSV, returns the row count

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_seed_manifest.py`:

```python
"""Tests for the seed manifest builder."""

import json
from pathlib import Path

import pytest

from workflow.scripts import build_seed_manifest as bsm

SEEDS_A = {
    "nshm_to_realisation_seed": 531798913,
    "rupture_propagation_seed": 31268976,
    "genslip_seed": 513004717,
    "srfgen_seed": 1837842819,
    "hf_seed": 1524796118,
}
SEEDS_B = {
    "nshm_to_realisation_seed": 11,
    "rupture_propagation_seed": 22,
    "genslip_seed": 33,
    "srfgen_seed": 44,
    "hf_seed": 55,
}


def _write_event(events_dir: Path, rupture_id: int, seeds: dict[str, int]) -> Path:
    event_dir = events_dir / str(rupture_id)
    event_dir.mkdir(parents=True)
    realisation = {
        "seeds": seeds,
        "log_trail": {
            "log": [
                {
                    "utility": "nshm2022-to-realisation",
                    "version": "0.1.dev1+gdeadbeef",
                    "args": [
                        "nshmdb.db",
                        str(rupture_id),
                        "out.json",
                        "24.2.2.1",
                        "--dip-delta",
                        "20",
                    ],
                }
            ]
        },
    }
    ffp = event_dir / "realisation.json"
    ffp.write_text(json.dumps(realisation), encoding="utf-8")
    return ffp


def test_seed_row_extracts_the_five_seeds(tmp_path: Path) -> None:
    ffp = _write_event(tmp_path, 100932, SEEDS_A)
    assert bsm.seed_row(ffp, 100932) == {"rupture_id": 100932, **SEEDS_A}


def test_seed_row_rejects_an_incomplete_block(tmp_path: Path) -> None:
    incomplete = dict(SEEDS_A)
    del incomplete["hf_seed"]
    ffp = _write_event(tmp_path, 100932, incomplete)
    with pytest.raises(ValueError, match="missing"):
        bsm.seed_row(ffp, 100932)


def test_seed_row_rejects_a_mismatched_rupture_id(tmp_path: Path) -> None:
    ffp = _write_event(tmp_path, 100932, SEEDS_A)
    with pytest.raises(ValueError, match="does not name rupture 999"):
        bsm.seed_row(ffp, 999)


def test_build_seed_manifest_writes_a_sorted_csv(tmp_path: Path) -> None:
    events = tmp_path / "events"
    _write_event(events, 101084, SEEDS_B)
    _write_event(events, 100932, SEEDS_A)
    out = tmp_path / "seed_manifest.csv"

    count = bsm.build_seed_manifest(events, out)

    assert count == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "rupture_id,nshm_to_realisation_seed,rupture_propagation_seed,"
        "genslip_seed,srfgen_seed,hf_seed"
    )
    assert lines[1].startswith("100932,")
    assert lines[2].startswith("101084,")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_build_seed_manifest.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.scripts.build_seed_manifest'`.

- [ ] **Step 3: Write the module**

Create `workflow/scripts/build_seed_manifest.py`:

```python
#!/usr/bin/env python3
"""Build a seed manifest CSV from a tree of committed realisation files.

Description
-----------
Reads ``events/<rupture_id>/realisation.json`` for every event under an events
directory and writes one CSV row per event: the rupture id and its five seeds.
The manifest lets ``generate-realisations-from-csv --seed-manifest`` replay the
exact seeds a set was generated with, reproducing its content.

For each file it asserts the seed block is complete and that the file's own
``log_trail`` names the same rupture id, generated with the uniform campaign
arguments, so the manifest is provably faithful to the files it describes.
"""

import csv
import dataclasses
import json
from pathlib import Path
from typing import Annotated

import typer

from workflow.realisations import Seeds

app = typer.Typer()

SEED_FIELDS: tuple[str, ...] = tuple(field.name for field in dataclasses.fields(Seeds))


def seed_row(realisation_ffp: Path, rupture_id: int) -> dict[str, int]:
    """Extract ``{rupture_id + five seeds}`` from one realisation file.

    Parameters
    ----------
    realisation_ffp : Path
        Path to a ``realisation.json``.
    rupture_id : int
        The rupture id the file is expected to belong to.

    Returns
    -------
    dict of str to int
        ``rupture_id`` plus the five seed fields.

    Raises
    ------
    ValueError
        If the seed block is incomplete, or the file's ``log_trail`` does not
        name ``rupture_id`` generated with the uniform campaign arguments.
    """
    realisation = json.loads(realisation_ffp.read_text(encoding="utf-8"))
    seeds = realisation.get("seeds", {})
    missing = [field for field in SEED_FIELDS if field not in seeds]
    if missing:
        raise ValueError(f"{realisation_ffp}: seed block missing {missing}")

    log = realisation.get("log_trail", {}).get("log", [])
    nshm_entry = next(
        (entry for entry in log if entry.get("utility") == "nshm2022-to-realisation"),
        None,
    )
    if nshm_entry is None:
        raise ValueError(f"{realisation_ffp}: no nshm2022-to-realisation log entry")
    args = nshm_entry.get("args", [])
    if str(rupture_id) not in args:
        raise ValueError(
            f"{realisation_ffp}: log_trail does not name rupture {rupture_id}"
        )
    if "24.2.2.1" not in args or "--dip-delta" not in args:
        raise ValueError(f"{realisation_ffp}: non-uniform generation args {args}")

    row = {field: int(seeds[field]) for field in SEED_FIELDS}
    return {"rupture_id": rupture_id, **row}


def build_seed_manifest(events_dir: Path, output_csv: Path) -> int:
    """Write a sorted seed manifest for every event under ``events_dir``.

    Parameters
    ----------
    events_dir : Path
        Directory of ``<rupture_id>/realisation.json`` subdirectories.
    output_csv : Path
        Destination CSV path.

    Returns
    -------
    int
        The number of rows written.
    """
    rupture_dirs = sorted(
        (path for path in events_dir.iterdir() if (path / "realisation.json").is_file()),
        key=lambda path: int(path.name),
    )
    rows = [seed_row(path / "realisation.json", int(path.name)) for path in rupture_dirs]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["rupture_id", *SEED_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


@app.command()
def main(
    events_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_csv: Annotated[Path, typer.Argument()],
) -> None:
    """Build a seed manifest CSV from an events directory tree.

    Parameters
    ----------
    events_dir : Path
        Directory of ``<rupture_id>/realisation.json`` subdirectories.
    output_csv : Path
        Destination CSV path.
    """
    count = build_seed_manifest(events_dir, output_csv)
    print(f"Wrote {count} seed rows to {output_csv}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Register the entry point**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
build-seed-manifest = "workflow.scripts.build_seed_manifest:app"
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/test_build_seed_manifest.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add workflow/scripts/build_seed_manifest.py tests/test_build_seed_manifest.py pyproject.toml
git commit -m "feat(scripts): add build-seed-manifest

Extracts the five seeds from each committed realisation into a CSV keyed by
rupture id, asserting every block is complete and every file's log_trail names
the rupture its directory claims, generated with the uniform campaign args. This
is the input generate-realisations-from-csv --seed-manifest replays."
```

---

## Task 7b: `verify_realisation_content` — prove a regenerated file reproduces its original

Provenance verification (Tasks 5–6) proves each file records a clean commit. This proves the *content* is the same as before — identical in every field except `log_trail`, which is the one field a re-run is allowed to change.

**Files:**
- Create: `workflow/scripts/verify_realisation_content.py`
- Modify: `pyproject.toml` (add a `[project.scripts]` entry)
- Test: `tests/test_verify_realisation_content.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `diff_content(expected: object, actual: object, path: str = "") -> list[str]` — dotted paths at which two realisations differ, skipping the top-level `log_trail`; empty means equivalent
  - `compare_files(expected_ffp: Path, actual_ffp: Path) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verify_realisation_content.py`:

```python
"""Tests for the realisation content checker."""

import json
from pathlib import Path

from workflow.scripts import verify_realisation_content as vc


def test_diff_content_is_empty_for_identical_realisations() -> None:
    realisation = {"magnitudes": {"A": 7.1}, "seeds": {"hf_seed": 5}}
    assert vc.diff_content(realisation, realisation) == []


def test_diff_content_ignores_log_trail() -> None:
    expected = {"magnitudes": {"A": 7.1}, "log_trail": {"log": [{"version": "old"}]}}
    actual = {"magnitudes": {"A": 7.1}, "log_trail": {"log": [{"version": "new"}]}}
    assert vc.diff_content(expected, actual) == []


def test_diff_content_reports_a_scientific_difference() -> None:
    expected = {"magnitudes": {"A": 7.1}}
    actual = {"magnitudes": {"A": 7.2}}
    assert vc.diff_content(expected, actual) == ["magnitudes.A: 7.1 != 7.2"]


def test_diff_content_reports_list_differences_with_a_path() -> None:
    expected = {"rakes": [10, 20, 30]}
    actual = {"rakes": [10, 25, 30]}
    assert vc.diff_content(expected, actual) == ["rakes[1]: 20 != 25"]


def test_compare_files_ignores_only_log_trail(tmp_path: Path) -> None:
    expected_ffp = tmp_path / "original.json"
    actual_ffp = tmp_path / "regenerated.json"
    expected_ffp.write_text(
        json.dumps({"seeds": {"hf_seed": 5}, "log_trail": {"log": ["a"]}}),
        encoding="utf-8",
    )
    actual_ffp.write_text(
        json.dumps({"seeds": {"hf_seed": 5}, "log_trail": {"log": ["b"]}}),
        encoding="utf-8",
    )
    assert vc.compare_files(expected_ffp, actual_ffp) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_verify_realisation_content.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.scripts.verify_realisation_content'`.

- [ ] **Step 3: Write the module**

Create `workflow/scripts/verify_realisation_content.py`:

```python
#!/usr/bin/env python3
"""Verify regenerated realisations reproduce the originals, ignoring log_trail.

Description
-----------
Compares each ``realisation_<id>.json`` in a regenerated directory against the
committed original at ``events/<id>/realisation.json``, treating every field as
significant except ``log_trail`` (which legitimately changes when the code is
re-run). Any other difference is reported and makes the command exit non-zero.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


def diff_content(expected: object, actual: object, path: str = "") -> list[str]:
    """Return the dotted paths at which two realisations differ, ignoring log_trail.

    Parameters
    ----------
    expected : object
        The reference JSON value (initially the whole realisation dict).
    actual : object
        The value to compare against it.
    path : str
        The dotted path to the current value, used for reporting.

    Returns
    -------
    list of str
        One entry per differing leaf, empty when equivalent. The top-level
        ``log_trail`` key is skipped.
    """
    diffs: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if path == "" and key == "log_trail":
                continue
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                diffs.append(f"{child}: only in actual")
            elif key not in actual:
                diffs.append(f"{child}: only in expected")
            else:
                diffs.extend(diff_content(expected[key], actual[key], child))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(expected)} != {len(actual)}")
        else:
            for index, (exp, act) in enumerate(zip(expected, actual, strict=True)):
                diffs.extend(diff_content(exp, act, f"{path}[{index}]"))
    elif expected != actual:
        diffs.append(f"{path}: {expected!r} != {actual!r}")
    return diffs


def compare_files(expected_ffp: Path, actual_ffp: Path) -> list[str]:
    """Compare two realisation files, ignoring ``log_trail``.

    Parameters
    ----------
    expected_ffp : Path
        The original realisation file.
    actual_ffp : Path
        The regenerated realisation file.

    Returns
    -------
    list of str
        The differences, empty when equivalent.
    """
    expected = json.loads(expected_ffp.read_text(encoding="utf-8"))
    actual = json.loads(actual_ffp.read_text(encoding="utf-8"))
    return diff_content(expected, actual)


@app.command()
def main(
    events_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    regenerated_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Compare a regenerated realisation set against the committed originals.

    Parameters
    ----------
    events_dir : Path
        Directory of ``<rupture_id>/realisation.json`` originals.
    regenerated_dir : Path
        Directory of ``realisation_<rupture_id>.json`` regenerated files.
    """
    mismatched: dict[str, list[str]] = {}
    regenerated = sorted(regenerated_dir.glob("realisation_*.json"))
    for actual_ffp in regenerated:
        rupture_id = actual_ffp.stem.removeprefix("realisation_")
        expected_ffp = events_dir / rupture_id / "realisation.json"
        if not expected_ffp.is_file():
            mismatched[rupture_id] = ["no original to compare against"]
            continue
        diffs = compare_files(expected_ffp, actual_ffp)
        if diffs:
            mismatched[rupture_id] = diffs

    print(
        f"Compared {len(regenerated)} realisation(s); "
        f"{len(mismatched)} differ beyond log_trail"
    )
    for rupture_id, diffs in sorted(mismatched.items()):
        print(f"\n{rupture_id}:")
        for difference in diffs[:20]:
            print(f"  {difference}")
    if mismatched:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Register the entry point**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
verify-realisation-content = "workflow.scripts.verify_realisation_content:app"
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/test_verify_realisation_content.py -q
```

Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add workflow/scripts/verify_realisation_content.py tests/test_verify_realisation_content.py pyproject.toml
git commit -m "feat(scripts): add verify-realisation-content

Compares a regenerated realisation set against the committed originals field by
field, ignoring only log_trail, and exits non-zero on any other difference. This
is what turns 'we fed the seeds back in' into 'the content is provably the same'."
```

---

## Task 8: Run every gate, then pin the commit

This task produces **commit N** — the SHA every realisation will record.

**Files:**
- Modify: none (verification and push only)

- [ ] **Step 1: Run the entire test suite, including the slow tests**

```bash
uv sync --all-extras --dev
uv run pytest -q
```

Expected: all pass, **with no deselections**. The two `@pytest.mark.slow` end-to-end tests in `tests/test_complete_realisations.py` must run here — they have never been run, and no commit gets pinned on unverified code.

If anything fails, **stop**. Fix it and re-run before going further.

- [ ] **Step 2: Run every other CI gate**

```bash
uv run ruff check
uv run ty check --exclude workflow/schemas.py --exclude setup.py
uv run deptry .
fdfind . workflow/ -E "__init__.py" --extension py | xargs numpydoc lint
```

Expected: `All checks passed!` from ruff and ty; `Success! No dependency issues found.` from deptry; no output from numpydoc.

If `fdfind` is not installed, use `find workflow -name '*.py' ! -name '__init__.py' | xargs numpydoc lint` — the two new modules must lint clean, since CI will run them.

- [ ] **Step 3: Confirm the tree is clean and push**

```bash
git status --porcelain --untracked-files=no    # expect: empty
git push -u origin cs-nshm2022-prep
```

A SHA that exists only on one laptop is not auditable. This push is what makes the recorded commit resolvable by anyone else.

- [ ] **Step 4: Record commit N**

```bash
git rev-parse HEAD
```

Write the full 40-character SHA down. Every subsequent task refers to it as **commit N**. It goes into `PROVENANCE.md` in Task 13.

---

## Task 9: Rebuild `nshmdb.db` and test the provenance chain

**Files:**
- Create: `/home/arr65/data/cs_nshm_2022/nshmdb_rebuilt_20260714.db`
- Possibly modify: `/home/arr65/src/workflow/nshmdb.db`

- [ ] **Step 1: Confirm NSHM2022DB is where the reflog says it is**

```bash
git -C /home/arr65/src/NSHM2022DB rev-parse HEAD
git -C /home/arr65/src/NSHM2022DB status --porcelain --untracked-files=no
```

Expected: `95a005a...` and empty output. If HEAD is not `95a005a`, the reconstruction in the spec is void — **stop and report**.

- [ ] **Step 2: Rebuild the database**

This takes a couple of minutes and writes about 1 GB.

```bash
cd /home/arr65/src/NSHM2022DB
uv run nshm_db_generator \
    /home/arr65/data/cs_nshm_2022/CRU_fault_system_solution.zip \
    /home/arr65/data/cs_nshm_2022/nshmdb_rebuilt_20260714.db
cd /home/arr65/src/workflow
```

- [ ] **Step 3: Compare against the database that has been in use**

```bash
uv run compare-nshmdb \
    nshmdb.db \
    /home/arr65/data/cs_nshm_2022/nshmdb_rebuilt_20260714.db
```

Expected, if the reconstructed chain is correct:
```
fault                              2,325 rows  <hash>  same
fault_plane                        4,735 rows  <hash>  same
magnitude_frequency_distribution  41,905 rows  <hash>  same
parent_fault                         557 rows  <hash>  same
rupture                          411,270 rows  <hash>  same
rupture_faults                19,773,517 rows  <hash>  same

Databases are logically identical.
```

- [ ] **Step 4: Act on the result**

**If they match** (exit 0): the chain is proven. `nshmdb.db` stays as it is. Record in Task 13 that the rebuild reproduced it exactly.

**If they differ** (exit 1): this is a genuine finding, not a failure. The database in use was *not* what the reconstruction claimed. Adopt the rebuilt database, whose provenance is known by construction:

```bash
cp /home/arr65/data/cs_nshm_2022/nshmdb_rebuilt_20260714.db nshmdb.db
```

Then **report the differences before continuing** — a large discrepancy (for example a different rupture count) means the campaign's rupture ids may not mean what we think they mean, and Task 11 must not start until that is understood.

- [ ] **Step 5: Record the checksum of the database of record**

```bash
sha256sum nshmdb.db
```

This value goes into `PROVENANCE.md` in Task 13. For reference, the database in use before this task had sha256 `00e256480618cd15e11fbf744037d037bf3fc2d523fb977ee30e0b84a640bc57`.

---

## Task 10: Force the version stamp, gate on it, and smoke-test

Do not skip any step here. This is the sequence that stands between the campaign and 291 files stamped with the wrong commit.

- [ ] **Step 1: Force-rebuild the package**

```bash
uv sync --reinstall-package workflow --all-extras --dev
```

`uv sync` **alone is not sufficient** — it will not rebuild when only `HEAD` has moved, and will happily serve a `.dist-info` stamped days ago at a different commit.

Expected output includes a line like:
```
 - workflow==<old version>
 + workflow==0.1.dev<N>+g<sha of commit N>
```

- [ ] **Step 2: Run the pre-flight gate**

```bash
uv run verify-realisation-provenance --preflight
```

Expected:
```
Preflight OK. Realisations will record version 0.1.dev<N>+g<sha>
```

If it reports `STALE METADATA` or a dirty tree, **stop**. Fix the cause and re-run — do not proceed with a failing pre-flight, which is the exact mistake that produced the untraceable batch.

Record the version string it prints. Call it **EXPECTED_VERSION**; Task 11, Task 12 and Task 13 all need it.

- [ ] **Step 3: Smoke-test on three ruptures**

```bash
SMOKE=/tmp/claude-1000/-home-arr65-src-workflow/b518f6e9-16db-4ae1-a09d-e4bf7d6e1754/scratchpad/smoke
mkdir -p "$SMOKE"
printf 'chosen_nshm_id\n100932\n101084\n101091\n' > "$SMOKE/smoke.csv"

uv run generate-realisations-from-csv nshmdb.db "$SMOKE/smoke.csv" "$SMOKE/minimal" 24.2.2.1
uv run complete-realisations "$SMOKE/minimal" "$SMOKE/complete" \
        --defaults-version 24.2.2.1 --vm-version 2.09
```

Expected: 3 stubs, 3 complete realisations, no errors.

- [ ] **Step 4: Verify the smoke-test output**

```bash
uv run verify-realisation-provenance "$SMOKE/complete"
```

Expected:
```
Provenance OK: 3 realisation(s), all recording 0.1.dev<N>+g<sha>
```

If this fails, **stop**. A bad stamp discovered after 291 files is 291 files wasted; discovered after three, it costs nothing.

- [ ] **Step 5: Confirm the run did not dirty the tree**

```bash
git status --porcelain --untracked-files=no    # expect: empty
```

This must still be empty. The output directories are gitignored, which is what makes the stamp stable across the whole campaign. If this is not empty, something wrote into a tracked file — **stop and investigate**.

---

## Task 10a: Build and commit the seed manifest

The manifest is a campaign **input**, built once from the committed originals and committed to `cybershake_nshm_2022` beside the events it describes. It must be built before the originals are overwritten in Task 14, and before the pilot and run that consume it.

- [ ] **Step 1: Confirm the originals are the untouched source of truth**

```bash
git -C /home/arr65/src/cybershake_nshm_2022 rev-parse --abbrev-ref HEAD    # add-srf-helper-scripts
git -C /home/arr65/src/cybershake_nshm_2022 status --porcelain -- cybershake_nshm_2022/events    # empty
ls -d /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/events/*/ | wc -l    # 291
```

If the events tree is not clean, **stop** — the manifest must be built from the committed originals, not from edited files.

- [ ] **Step 2: Build the manifest**

```bash
mkdir -p /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/seed_manifest
uv run build-seed-manifest \
    /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/events \
    /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/seed_manifest/seed_manifest.csv
```

Expected: `Wrote 291 seed rows to ...`. If the builder raises — an incomplete seed block, a `log_trail` that names a different rupture than its directory, or non-uniform generation args — **stop**. A manifest that cannot be built cleanly is not one to trust.

- [ ] **Step 3: Write the README beside it**

Create `cybershake_nshm_2022/cybershake_nshm_2022/seed_manifest/README.md` (in the `cybershake_nshm_2022` repo):

```markdown
# Realisation seed manifest

`seed_manifest.csv` records the five random seeds of every event in this
CyberShake NSHM-2022 realisation set, one row per rupture:

    rupture_id, nshm_to_realisation_seed, rupture_propagation_seed,
    genslip_seed, srfgen_seed, hf_seed

## Why this file exists

The seeds were originally drawn from OS entropy by `Seeds.random_seeds()` in
`ucgmsim/workflow` (`random.randint(0, 2**31 - 1)`) during the first, ad-hoc
generation of this set. They carry **no intrinsic meaning** — any draw would
have been equally valid. They are recorded here verbatim for one reason: so the
realisation files can be regenerated *exactly*, keeping them consistent with the
SRFs and downstream results already built from them, while giving each file an
honest, commit-pinned `log_trail`.

## How the set was generated

Every event was produced with identical arguments, varying only the rupture id
and its seeds:

    nshm2022-to-realisation nshmdb.db <rupture_id> <out.json> 24.2.2.1 --dip-delta 20
    complete-realisations <minimal_dir> <out_dir> --defaults-version 24.2.2.1 --vm-version 2.09

To reproduce, pass this file to the batch driver:

    generate-realisations-from-csv nshmdb.db <ruptures.csv> <out_dir> 24.2.2.1 \
        --seed-manifest seed_manifest.csv

The driver writes each event's seeds into the stub before generation, so
`nshm2022-to-realisation` replays them via `read_from_realisation_or_random`.

## The guarantee

A regenerated file is verified to match the original it replaces in **every
field except `log_trail`** (`verify-realisation-content`). `log_trail` is the one
field a re-run is meant to change: it now records a definite, pushed, tagged
commit instead of the original set's stale build stamp.

Full provenance of the regenerated set is in the campaign's `PROVENANCE.md`.
```

- [ ] **Step 4: Commit the manifest and README to `cybershake_nshm_2022`**

```bash
cd /home/arr65/src/cybershake_nshm_2022
git add cybershake_nshm_2022/seed_manifest/seed_manifest.csv cybershake_nshm_2022/seed_manifest/README.md
git commit -m "feat: record the realisation seed manifest

The five seeds of each of the 291 events, extracted from the committed
realisations by build-seed-manifest. Recorded so the set can be regenerated
exactly — same content, honest log_trail — via
generate-realisations-from-csv --seed-manifest. The seeds carry no intrinsic
meaning; see README.md."
cd /home/arr65/src/workflow
```

- [ ] **Step 5: Confirm the `workflow` tree is still clean**

```bash
git status --porcelain --untracked-files=no    # expect: empty
```

Building the manifest reads the originals and writes into `cybershake_nshm_2022`; it must not have touched a tracked file in `workflow`. If this is not empty, **stop and investigate**.

---

## Task 10b: Pilot — prove content reproduction before the full run

This is the load-bearing gate. Replaying the seeds only reproduces the content if commit N's code derives the same realisation from them as the ad-hoc code did — and the ad-hoc baker (`bake_realisations.py`) no longer exists, `complete-realisations` is its successor, and the pegasus rebase brought magnitude-convention (BoldM) changes. Prove it on a handful of events before committing to 291.

- [ ] **Step 1: Build a small pilot rupture list**

Include at least one multi-fault event. `149379` is multi-fault; the other three round out the sample.

```bash
PILOT=/tmp/claude-1000/-home-arr65-src-workflow/b518f6e9-16db-4ae1-a09d-e4bf7d6e1754/scratchpad/pilot
mkdir -p "$PILOT"
printf 'chosen_nshm_id\n149379\n100932\n101084\n101091\n' > "$PILOT/pilot.csv"
SEEDS=/home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/seed_manifest/seed_manifest.csv
```

- [ ] **Step 2: Regenerate them, replaying the recorded seeds**

```bash
uv run generate-realisations-from-csv \
    nshmdb.db "$PILOT/pilot.csv" "$PILOT/minimal" 24.2.2.1 \
    --seed-manifest "$SEEDS"
uv run complete-realisations "$PILOT/minimal" "$PILOT/complete" \
    --defaults-version 24.2.2.1 --vm-version 2.09
```

Expected: 4 minimal stubs, 4 complete realisations, no errors.

- [ ] **Step 3: Verify the content matches the originals**

```bash
uv run verify-realisation-content \
    /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/events \
    "$PILOT/complete"
```

Expected: `Compared 4 realisation(s); 0 differ beyond log_trail`.

- [ ] **Step 4: Gate on the result**

If it reports **0 differ**, commit N reproduces the originals from their seeds; proceed to Task 11.

If **any** event differs beyond `log_trail`, **stop**. The command prints the exact differing fields. Reconcile before going further — the likely causes, in order, are: `complete-realisations` differing from the vanished `bake_realisations.py`; the BoldM magnitude-convention changes from the pegasus rebase; or a residual difference in the area-weighted fault selection (`9f35c90`). Any fix moves commit N, so re-run Task 8 (gates + pin) and Task 10 (force the stamp) afterwards. Do **not** run the full campaign against a failing pilot.

- [ ] **Step 5: Confirm the tree is still clean**

```bash
git status --porcelain --untracked-files=no    # expect: empty
git rev-parse HEAD                             # expect: commit N, unchanged
```

---

## Task 11: Run the campaign

- [ ] **Step 1: Clear the output of the previous, untraceable run**

```bash
rm -rf minimal_realisations complete_realisations
```

These are gitignored, so this touches nothing tracked. The old outputs are still preserved in the `cybershake_nshm_2022` repo's git history, and Task 14 replaces the working copies.

- [ ] **Step 2: Generate the minimal stubs**

```bash
uv run generate-realisations-from-csv \
    nshmdb.db \
    annealed_minimal_ruptures.csv \
    minimal_realisations \
    24.2.2.1 \
    --seed-manifest /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/seed_manifest/seed_manifest.csv
```

`--seed-manifest` makes each stub replay its recorded seeds (Task 10a), so the content reproduces the originals rather than drawing a fresh set. The two excluded ruptures are absent from the manifest and fall back to a fresh draw, which is moot — they fail before any seed is used.

Expected: `Done. Processed 293 rupture ID(s).`, with two failures printed — ruptures **59421** and **95011**, both `ValueError: The graph must be connected to find a spanning tree`. **These two failures are a pass condition.**

```bash
ls minimal_realisations/realisation_*.json | wc -l    # expect: 291
```

291, not 293 — Task 3 deletes the partial stubs. If you get 293, Task 3's change is not in effect and you are running stale code; go back to Task 10 Step 1.

- [ ] **Step 3: Complete the realisations**

```bash
uv run complete-realisations \
    minimal_realisations \
    complete_realisations \
    --defaults-version 24.2.2.1 \
    --vm-version 2.09
```

Expected: `Completed 291 realisation(s)`, no skips (there are no broken stubs left to skip), no failures.

```bash
ls complete_realisations/realisation_*.json | wc -l    # expect: 291
```

- [ ] **Step 4: Audit the whole set**

```bash
uv run verify-realisation-provenance complete_realisations
```

Expected:
```
Provenance OK: 291 realisation(s), all recording 0.1.dev<N>+g<sha>
```

The version must equal **EXPECTED_VERSION** from Task 10. If a single file fails, **stop** — do not distribute a partially-sound set.

- [ ] **Step 5: Confirm the tree is still clean**

```bash
git status --porcelain --untracked-files=no    # expect: empty
git rev-parse HEAD                             # expect: commit N, unchanged
```

---

## Task 11a: Verify the full set reproduces the originals

Task 11 Step 4 proved every file records commit N. This proves every file reproduces the *content* it replaces. Run it now, while the originals in `cybershake_nshm_2022` are still the pre-campaign files — Task 14 overwrites them.

- [ ] **Step 1: Compare the whole set against the committed originals**

```bash
uv run verify-realisation-content \
    /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/events \
    complete_realisations
```

Expected: `Compared 291 realisation(s); 0 differ beyond log_trail`.

- [ ] **Step 2: Gate on the result**

If **0 differ**, the regenerated set is content-identical to the originals bar `log_trail`; proceed to tag and distribute.

If **any** event differs, **stop** — do not tag, record, or distribute a set that does not reproduce what it replaces. The pilot (Task 10b) should have caught this; a difference surfacing only at full scale points to an event outside the pilot sample. Investigate the printed fields and reconcile exactly as in Task 10b Step 4. A fix moves commit N, so re-run Tasks 8, 10, 10b and 11.

- [ ] **Step 3: Confirm the tree is still clean**

```bash
git status --porcelain --untracked-files=no    # expect: empty
git rev-parse HEAD                             # expect: commit N, unchanged
```

---

## Task 12: Tag commit N, without perturbing the version scheme

The repo has **no tags** — which is why setuptools-scm falls back to `0.1.dev<count>+g<sha>`. Adding one may match its default `tag_regex` and change every version string thereafter, so that anyone checking out the tag and reinstalling would see a version that does not match `log_trail`. Probe before committing to a name.

- [ ] **Step 1: Probe the first candidate**

```bash
git tag -a cs-nshm2022-realisations-v1 -m "CyberShake NSHM-2022 realisation set, 291 events"
uv sync --reinstall-package workflow --all-extras --dev
uv run python -c "from importlib import metadata; print(metadata.version('workflow'))"
```

- [ ] **Step 2: Decide**

**If the printed version is unchanged** (still equal to EXPECTED_VERSION): the tag is inert. Keep it and push:

```bash
git push origin cs-nshm2022-realisations-v1
```

**If the version changed** (setuptools-scm parsed the tag as a version): delete it and try a name it cannot read as one:

```bash
git tag -d cs-nshm2022-realisations-v1
git tag -a campaign/cs-nshm2022-realisations-2026-07-14 \
    -m "CyberShake NSHM-2022 realisation set, 291 events"
uv sync --reinstall-package workflow --all-extras --dev
uv run python -c "from importlib import metadata; print(metadata.version('workflow'))"
```

If *that* is inert, push it. If it too perturbs the version, **delete it and do not tag at all**:

```bash
git tag -d campaign/cs-nshm2022-realisations-2026-07-14
```

The pushed commit N is the anchor; the tag is only a human-readable pointer, and a tag that corrupts the version scheme is worse than no tag. Record whichever outcome occurred in `PROVENANCE.md`.

- [ ] **Step 3: Restore the correct stamp**

Whatever happened above, the last `uv sync --reinstall-package` may have left a different stamp installed. Re-assert it:

```bash
uv run verify-realisation-provenance --preflight
```

Expected: `Preflight OK` reporting **EXPECTED_VERSION**. If it reports anything else, the tag is still perturbing the version — go back to Step 2 and remove it.

---

## Task 13: Write the provenance record

**Files:**
- Create: `docs/campaigns/2026-07-14-nshm2022-realisations/PROVENANCE.md`
- Create: `docs/campaigns/2026-07-14-nshm2022-realisations/manifest.csv`

This is **commit N+1**. It can only exist after the run, which is why it is a separate commit from the one the realisations record.

- [ ] **Step 1: Generate the per-file manifest**

It is committed under `docs/campaigns/`, not into `complete_realisations/`, which is gitignored.

```bash
mkdir -p docs/campaigns/2026-07-14-nshm2022-realisations
uv run python - <<'PY'
import csv
import hashlib
from pathlib import Path

rows = [
    {
        "rupture_id": realisation_ffp.stem.removeprefix("realisation_"),
        "file": realisation_ffp.name,
        "sha256": hashlib.sha256(realisation_ffp.read_bytes()).hexdigest(),
    }
    for realisation_ffp in sorted(
        Path("complete_realisations").glob("realisation_*.json")
    )
]
manifest = Path("docs/campaigns/2026-07-14-nshm2022-realisations/manifest.csv")
with manifest.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, ["rupture_id", "file", "sha256"])
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} rows")
PY
```

Expected: `291 rows`.

- [ ] **Step 2: Gather every value the record needs**

The version is read **from a realisation file**, not from installed metadata — Task 12's tag probing may have changed what is installed, and what matters is what the files actually say.

```bash
echo "commit N        : $(git rev-parse HEAD)"
echo "branch          : $(git rev-parse --abbrev-ref HEAD)"
echo "remote          : $(git remote get-url origin)"
echo "tag             : $(git tag --points-at HEAD)"     # empty if Task 12 chose not to tag
echo "recorded version: $(uv run python -c "
import json
from pathlib import Path
first = sorted(Path('complete_realisations').glob('realisation_*.json'))[0]
print(json.loads(first.read_text())['log_trail']['log'][0]['version'])
" 2>/dev/null | tail -1)"
echo "nshmdb.db       : $(sha256sum nshmdb.db | cut -d' ' -f1)"
echo "CRU zip         : $(sha256sum /home/arr65/data/cs_nshm_2022/CRU_fault_system_solution.zip | cut -d' ' -f1)"
echo "input CSV       : $(sha256sum annealed_minimal_ruptures.csv | cut -d' ' -f1)"
echo "uv.lock         : $(sha256sum uv.lock | cut -d' ' -f1)"
echo "NSHM2022DB      : $(git -C /home/arr65/src/NSHM2022DB rev-parse HEAD)"
echo "python          : $(uv run python -c 'import platform; print(platform.python_version())' 2>/dev/null | tail -1)"
echo "host            : $(uname -srm) / $(hostname)"
```

The `recorded version` must equal **EXPECTED_VERSION** from Task 10. If it does not, something is badly wrong — **stop**.

- [ ] **Step 3: Write `docs/campaigns/2026-07-14-nshm2022-realisations/PROVENANCE.md`**

Fill every `«…»` from Step 2's output, from `minimal_realisations/error_log.txt`, and from Task 9's comparison result. Leave nothing unfilled.

```markdown
# CyberShake NSHM-2022 realisation set — provenance

291 realisations, generated 2026-07-14. This record exists so that a third party
can establish exactly what produced these files, and reproduce or refute it.

## The artefact

- 291 files, at `cybershake_nshm_2022/cybershake_nshm_2022/events/<rupture id>/realisation.json`
- Per-file sha256: `manifest.csv`, alongside this file.
- Content-identical to the 2026-07-09 set bar `log_trail`, verified by `verify-realisation-content`; seeds carried over from `seed_manifest.csv`.
- Every file's `log_trail` records exactly two entries — `nshm2022-to-realisation`
  then `complete-realisations` — both stamped `«recorded version»`.
- Verify at any time:
  `uv run verify-realisation-provenance <dir> --expect-version «recorded version»`

## Code

| | |
| --- | --- |
| repository | `«remote»` |
| branch | `cs-nshm2022-prep` |
| commit | `«commit N, full 40 chars»` |
| tag | `«tag, or: none — see the tagging note»` |
| version stamped into every file | `«recorded version»` |

The version's `g<sha>` segment names the commit above and carries no `.d<date>`
dirty suffix. That was asserted before the run by
`verify-realisation-provenance --preflight`, and again afterwards across all 291
files. It is not a claim; it is a checked property.

**Tagging note.** «Either: the tag above is inert — reinstalling at it reproduces
the same version string. Or: setuptools-scm parses the candidate tag names as
versions, so no tag was applied; the commit SHA is the anchor.»

## Inputs

| input | sha256 | origin |
| --- | --- | --- |
| `nshmdb.db` | `«sha256»` | Built by `nshm_db_generator.py` at NSHM2022DB `«NSHM2022DB sha»` from the CRU zip below. |
| `CRU_fault_system_solution.zip` | `«sha256»` | The NSHM 2022 crustal fault system solution, from Jake Faulkner. Its own upstream release identifier is not recorded. |
| `annealed_minimal_ruptures.csv` | `«sha256»` | 293 rupture ids, provided by Jake Faulkner as the first sample of ruptures to simulate for this campaign. The selection procedure implied by "annealed" is not documented. |
| `seed_manifest.csv` | `«sha256»` | The five seeds of each of the 291 events, extracted from the 2026-07-09 set by `build-seed-manifest` and replayed via `--seed-manifest` so this set reproduces that one's content. Committed in `cybershake_nshm_2022`. |
| `uv.lock` | `«sha256»` | Pins `nshmdb` 2025.12.1, `source_modelling` 2026.6.2, `velocity-modelling` 2026.2.1, `qcore-utils` 2025.12.2, `im-calculation` 2025.12.5, `oq-wrapper` 2025.12.5. |

### The database rebuild

`nshmdb.db` had no recorded origin. It was reconstructed from evidence — the
NSHM2022DB reflog shows a single clone at `95a005a` on 2026-07-08 11:42:46, the
database was written at 11:44:03, and its schema matches the repo's only
generator — and then **tested by rebuilding it**:

```
uv run compare-nshmdb nshmdb.db nshmdb_rebuilt_20260714.db
```

Result: «either: *logically identical across all six tables — the chain is
proven*; or: *differed as follows: «detail». The rebuilt database was adopted, so
the database used here has a known derivation either way.*»

## Parameters

| | |
| --- | --- |
| scientific defaults | `24.2.2.1` (both steps) |
| `--dip-delta` | `20` |
| `--vm-version` | `2.09` |
| connectivity | `--jump-cutoff 15 km`, `--separation-distance 5 km`, `--min-connected-depth 5 km` (defaults) |

## Commands

Run from the repository root, on a clean tree at the commit above, after
`uv sync --reinstall-package workflow --all-extras --dev`:

```
uv run verify-realisation-provenance --preflight
uv run generate-realisations-from-csv nshmdb.db annealed_minimal_ruptures.csv minimal_realisations 24.2.2.1 --seed-manifest «cybershake_nshm_2022»/cybershake_nshm_2022/seed_manifest/seed_manifest.csv
uv run complete-realisations minimal_realisations complete_realisations --defaults-version 24.2.2.1 --vm-version 2.09
uv run verify-realisation-provenance complete_realisations
uv run verify-realisation-content «cybershake_nshm_2022»/cybershake_nshm_2022/events complete_realisations
```

## Exclusions: 291 of 293

Two ruptures in the input CSV cannot be realised. Their fault sets are
disconnected under the connectivity parameters above: NSHM's inversion permits
jumps that this workflow's connectivity model rejects, so
`sample_rupture_propagation` cannot find a spanning tree. This is a property of
the data, not a defect.

Both were excluded rather than rescued by loosening the jump cutoff, so that
**every event in this set was generated with identical parameters**.

**59421** — 6 faults, two clusters:
`Alpine: Resolution - Charles`, `Alpine: Resolution - Dagg`,
`Alpine: Resolution - Five Fingers` | `Caswell High 1`, `Caswell High 4`,
`Caswell High 5`

**95011** — 4 faults, Marlborough faults unreachable from the Alpine sections:
`Alpine: Jacksons to Kaniere`, `Alpine: Kaniere to Springs Junction` |
`Awatere: Southwest`, `Hunter Valley`

Both raise `ValueError: The graph must be connected to find a spanning tree`.
Full tracebacks are in `minimal_realisations/error_log.txt` from the run.

## Seeds

Each realisation's five seeds — `nshm_to_realisation_seed`,
`rupture_propagation_seed`, `genslip_seed`, `srfgen_seed`, `hf_seed` — were
**carried over** from the previous set, not re-drawn. They were originally drawn
from OS entropy by `Seeds.random_seeds()`; they carry no intrinsic meaning, and
were reproduced verbatim so this set is content-identical to the one the existing
SRFs and downstream results were built from.

They are recorded in `cybershake_nshm_2022/cybershake_nshm_2022/seed_manifest/`
(`seed_manifest.csv` + `README.md`), extracted from the previous set by
`build-seed-manifest` and replayed here via
`generate-realisations-from-csv --seed-manifest`. Because the seeds are now a
recorded input, **the whole set is reproducible from its inputs** — and every
file was verified content-identical to the file it replaced, bar `log_trail`, by
`verify-realisation-content`.

## Environment

| | |
| --- | --- |
| Python | `«version»` |
| host | `«uname / hostname»` |
| generated | 2026-07-14 |

## What this supersedes

An earlier, untraceable batch of 291 realisations generated 2026-07-09. Those
files recorded `utility: bake_realisations.py` (a script name, not an entry
point), version `0.1.dev1277+g41974dfa1.d20260709` (a stale build stamp naming a
commit that demonstrably did not contain the code that ran), and argument paths
that no longer existed. They should not be cited for provenance.

This set reproduces their **content** exactly — same seeds, verified identical in
every field but `log_trail` — so anything already derived from them (SRFs,
animations) remains valid. Only the provenance changed: `log_trail` now names
commit N, tagged and pushed.
```

- [ ] **Step 4: Commit the record**

```bash
git add docs/campaigns/2026-07-14-nshm2022-realisations/
git commit -m "docs: record the provenance of the 2026-07-14 NSHM-2022 realisation set

291 of 293 ruptures, generated at commit «commit N» with a version stamp
asserted clean before the run and verified across every file afterwards. Pins
each input by checksum, records the nshmdb.db rebuild result, and states the two
exclusions and their cause."
git push origin cs-nshm2022-prep
```

---

## Task 14: Distribute to `cybershake_nshm_2022`

- [ ] **Step 1: Confirm the destination**

```bash
git -C /home/arr65/src/cybershake_nshm_2022 rev-parse --abbrev-ref HEAD
git -C /home/arr65/src/cybershake_nshm_2022 status --porcelain
ls -d /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/events/*/ | wc -l
```

Expected: branch `add-srf-helper-scripts`, clean, 291 existing event directories. If the tree is not clean, **stop** — do not overwrite uncommitted work.

- [ ] **Step 2: Distribute**

```bash
uv run copy-realisations-to-event-dirs \
    complete_realisations \
    /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/events
```

Expected: `Copied 291 realisation(s) into ...`, no skips. `completion_summary.csv` and `error_log.txt` are not JSON, so they are not copied.

- [ ] **Step 3: Verify what landed**

```bash
cd /home/arr65/src/cybershake_nshm_2022
git status --short | awk '{print $1}' | sort | uniq -c
```

Expected exactly:
```
    291 M
```

That is, 291 **modified** files and nothing else — no `A` (added), no `D` (deleted), no `??` (untracked). The rupture ids are unchanged; only their contents are.

If any file is *added* or *deleted*, the rupture set changed, which it must not have. **Stop and investigate.**

- [ ] **Step 4: Copy the provenance record alongside the artefacts**

```bash
cp /home/arr65/src/workflow/docs/campaigns/2026-07-14-nshm2022-realisations/PROVENANCE.md \
   /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/PROVENANCE.md
cp /home/arr65/src/workflow/docs/campaigns/2026-07-14-nshm2022-realisations/manifest.csv \
   /home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022/manifest.csv
```

- [ ] **Step 5: Verify the manifest describes the files that actually landed**

A manifest is worth nothing unless it describes these files.

```bash
uv --directory /home/arr65/src/workflow run python - <<'PY'
import csv
import hashlib
from pathlib import Path

root = Path("/home/arr65/src/cybershake_nshm_2022/cybershake_nshm_2022")
events = root / "events"
with (root / "manifest.csv").open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

mismatches = [
    row["rupture_id"]
    for row in rows
    if hashlib.sha256((events / row["rupture_id"] / "realisation.json").read_bytes()).hexdigest()
    != row["sha256"]
]

print(f"{len(rows)} in manifest, {len(mismatches)} mismatched")
if mismatches:
    raise SystemExit(f"MISMATCH: {mismatches}")
PY
```

Expected: `291 in manifest, 0 mismatched`.

- [ ] **Step 6: Commit the artefacts**

```bash
cd /home/arr65/src/cybershake_nshm_2022
git add cybershake_nshm_2022/events cybershake_nshm_2022/PROVENANCE.md cybershake_nshm_2022/manifest.csv
git commit -m "feat: regenerate all 291 realisations with verified provenance

Replaces the 2026-07-09 batch, whose log_trail recorded a stale build stamp
naming a commit that did not contain the code that ran, an old script name, and
paths that no longer existed.

Every file now records commit «commit N» of ucgmsim/workflow, asserted clean
before the run and verified across all 291 afterwards. Inputs are pinned by
checksum in PROVENANCE.md; per-file hashes are in manifest.csv.

Seeds were carried over from seed_manifest.csv, so this set reproduces the
previous batch's content exactly: verify-realisation-content confirms every file
is identical bar log_trail. The SRFs and slip animations already built from the
previous batch therefore remain valid — only the provenance changed."
```

- [ ] **Step 7: Confirm what stays valid, and what is genuinely separate**

Because the seeds were carried over and the content is verified identical bar
`log_trail` (Task 11a), **nothing downstream is invalidated**. The 54 GB of SRFs,
the slip animations, and the derived scratch all correspond to these realisations
exactly as before — this change rewrote provenance, not science. No SRF
regeneration is required, and none is triggered by this plan.

The SRF version-mislabelling is a **separate** matter, unaffected either way. The
local SRFs under `/home/arr65/data/cs_nshm_2022` were already repaired in place by
`scripts/fix_srf_version.py` (a one-line version-header rewrite, no regeneration);
the BSC and Dropbox copies are still mislabelled and still need that script — keep
it. The audit baselines `srf_version_audit_BEFORE_20260714.csv` and
`mislabelled_multi_fault_srfs_BEFORE_20260714.txt`
(`cybershake_nshm_2022@a623667`) record the pre-fix state (**219 mislabelled, all
multi-fault; 72 correct, all single-fault** — the signature of `stitch_srf_files`
hardcoding `version="1.0"`) and remain the reference for that independent fix.

When you push `cs-nshm2022-prep` or open a PR, flag the good news explicitly: the
realisation provenance is now sound and the downstream artefacts are preserved,
not invalidated.

---

## Deferred: separating the generic tooling for a pegasus PR

Not part of this campaign, and captured here only so it is not lost. The generic tools this branch adds — `complete-realisations`, `generate-realisations-from-csv` (with `--seed-manifest`), `build-seed-manifest`, `verify-realisation-content`, `verify-realisation-provenance`, `compare-nshmdb`, and the area-weighted fault-selection fix `9f35c90` — are reusable and belong on `pegasus`. When upstreaming later:

- Cherry-pick only the generic commits. Leave behind the campaign data and personal inputs: `felipe_scripts/` (Felipe's reference inputs — needs his sign-off), `annealed_minimal_ruptures.csv`, the campaign docs, and `copy_realisations_to_event_dirs.py` / `render_all.sh` if they are not promoted into the package.
- The seed manifest and its README stay in `cybershake_nshm_2022` — they are campaign data, not tooling.

This does not affect commit N or the campaign: the version stamp is derived from git state, not file inventory, and commit N is pinned and tagged regardless of any later separation.

---

## Done

At completion:

- 291 realisations, every one recording commit N with a stamp asserted clean before the run and verified after.
- Content-identical to the previous set in every field but `log_trail`, proved on a pilot before the run and across all 291 after (`verify-realisation-content`) — so the SRFs and results already built from them stay valid.
- The seed manifest and its README committed in `cybershake_nshm_2022`, making the set reproducible from recorded inputs.
- `nshmdb.db` with a derivation that was tested, not assumed.
- Every CI gate green — including the two this branch had broken.
- A committed checker anyone can re-run to confirm all of the above.
- `PROVENANCE.md` pinning every input by checksum, and stating honestly what is *not* known: the CRU zip's upstream release, and what "annealed" means.
