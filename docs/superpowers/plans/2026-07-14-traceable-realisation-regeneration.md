# Traceable Regeneration of the NSHM-2022 Realisation Set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Amended 2026-07-16** by `docs/superpowers/specs/2026-07-16-seed-carryover-and-content-verification-design.md`: seeds are **carried over**, not re-drawn, and each regenerated file is **verified** against its original.
>
> **Amended 2026-07-27** by `docs/superpowers/specs/2026-07-27-parameter-reconciliation-and-seed-inheritance-design.md`, which supersedes parts of the above. Three changes:
> 1. **The seed manifest is gone.** Seeds are inherited directly from the deployed `realisation.json` files (`--inherit-seeds-from`), so there is no second copy to keep in sync. **Tasks 7a and 10a are replaced**, not merely edited.
> 2. **The campaign now updates parameters, deliberately.** `pegasus` added `PGD` to `im.ims` (`ec2fb25`) and the set must adopt it, so the regenerated files *will* differ from the originals — but only where a human decided they should. A new interactive reconciler records every decision in a committed `campaign_parameters.yaml`.
> 3. **Verification is driven by that decision file**, not by a hard-coded "everything but `log_trail`" rule.
>
> Also mechanical: the campaign repo was renamed `cybershake_nshm_2022` → `cs_nshm_2022` and its branch is now `main`; `origin/pegasus` is already merged (`dee2f1b`), so **Task 1 is verify-only**.
>
> **Task map after the amendment.** Task 7a is now the reconciler's comparison engine (it was the seed-manifest builder); 7b is the reconciler's decision record and CLI (new); 7c is the content checker (it was 7b); 7d and 7e are new, teaching `complete-realisations` to apply decisions and to deploy behind a gate. Task 10a now runs the reconciler and commits its decisions (it was building the seed manifest). Nothing named `build-seed-manifest` or `--seed-manifest` survives.

**Goal:** Regenerate the 291 CyberShake NSHM-2022 realisations so every file's `log_trail` names a definite, pushed, tagged commit SHA, reproducing each file's scientific content exactly — seeds inherited from the deployed files — while adopting the `pegasus` parameter updates the set is missing, with every such change chosen by a human, recorded with a reason, and mechanically verified.

**Architecture:** Three phases. **Code** (Tasks 1–8, test-driven) verifies `origin/pegasus` is merged, restores the CI gates, fixes two campaign scripts, and adds four new tools — a provenance verifier, a database comparator, a **parameter reconciler**, and a content checker — ending in the pinned commit ("commit N"). **Inputs** (Task 9) rebuilds `nshmdb.db` from the CRU solution zip and compares, turning a reconstruction into evidence. **Campaign** (Tasks 10–14) is a runbook, not TDD: it reconciles parameters against `pegasus` and commits the decisions, proves reproduction on a pilot, regenerates all 291, verifies each differs from its original only in `log_trail` and the decided parameters, and deploys behind an explicit overwrite gate — with exact commands, expected output, and explicit abort conditions.

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
- **Seeds are inherited, not re-drawn.** The campaign passes `--inherit-seeds-from <events dir>` so each realisation replays the five seeds already recorded in its deployed `realisation.json`. There is no seed manifest; the deployed files are the source of truth.
- **The regenerated set may differ from the originals only in `log_trail` and in decided parameters.** Every other difference is a bug and aborts the campaign. What counts as "decided" is whatever `campaign_parameters.yaml` records — that file is the allowlist, and it lives in `cs_nshm_2022`, not this repo.
- **`nshmdb.db` is the original database of record**, restored to the repo root on 2026-07-27 and confirmed byte-identical to the one the deployed set was generated from: sha256 `00e256480618cd15e11fbf744037d037bf3fc2d523fb977ee30e0b84a640bc57`, mtime 2026-07-08 11:44. It is gitignored (`.gitignore:197`), so re-check that checksum before the campaign rather than assuming the file present is the file wanted. The CRU solution zip Task 9 rebuilds from is git-tracked at `/home/arr65/src/NSHM2022DB/tests/CRU_fault_system_solution.zip`.
- **The campaign repo is `/home/arr65/src/cs_nshm_2022`, branch `main`.** Events live at `cs_nshm_2022/cs_nshm_2022/events/<rupture_id>/realisation.json`. Renamed from `cybershake_nshm_2022` on 2026-07-22 (`37b2ea1`), a pure `git mv` that left all 291 files byte-identical.

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

## Task 1: Confirm `origin/pegasus` is merged, and characterise what it changed

**This task no longer merges anything.** `dee2f1b` already merged `origin/pegasus` into `cs-nshm2022-prep`, and pegasus has not moved since. What remains is to prove that, and to write down exactly what the merge changed in the generation path — because it *did* change it, and that change is the reason the reconciler (Task 7a) exists.

**Files:**
- Modify: none — verification only
- Verify: `workflow/scripts/nshm2022_to_realisation.py`, `workflow/scripts/complete_realisations.py`, `workflow/scripts/generate_realisations_from_csv.py`, `workflow/realisations.py`, `workflow/default_parameters/`

**Interfaces:**
- Consumes: nothing
- Produces: a written list of generation-path changes, consumed by Task 7a's reconciliation and Task 10a's decisions

- [ ] **Step 1: Confirm the starting state**

```bash
git rev-parse --abbrev-ref HEAD        # expect: cs-nshm2022-prep
git status --porcelain                 # expect: empty
git fetch origin --prune
```

- [ ] **Step 2: Prove pegasus is fully contained, with nothing left to merge**

```bash
git merge-base --is-ancestor origin/pegasus HEAD && echo "CONTAINED" || echo "NOT CONTAINED"
git rev-list --left-right --count origin/pegasus...HEAD
```

Expected: `CONTAINED`, and a count whose **left number is 0** (nothing on pegasus is missing here). The right number is however many commits this branch is ahead — it grows as the plan proceeds and is not a fixed value.

If the left number is **not** 0, pegasus has moved since this plan was written. Merge it (`git merge origin/pegasus`), then continue to Step 3 — which is precisely the step that will tell you what the new commits did to the generation path.

- [ ] **Step 3: List what the merge changed in the generation path**

```bash
git diff --stat dee2f1b^ HEAD -- \
    workflow/scripts/nshm2022_to_realisation.py \
    workflow/scripts/complete_realisations.py \
    workflow/scripts/generate_realisations_from_csv.py \
    workflow/realisations.py \
    workflow/defaults.py \
    workflow/default_parameters/
```

Unlike the original version of this task, output here is **expected, not a failure**. As of 2026-07-27 the known changes are:

- `workflow/default_parameters/root/defaults.yaml` — the `im:` block, from `ec2fb25`. `im.ims` gained `PGD`; `im.valid_periods` was expanded from 31 to 111 entries, absorbing `felipe_scripts/periods.csv` exactly.
- `workflow/realisations.py` — two new read-only properties on `Magnitudes` (`total_moment`, `total_magnitude`). No behaviour change to seeds or to any written section.

Anything **beyond** these two is new since this plan was written. Do not treat it as benign: write it down and carry it into Task 10a, where the reconciler will surface any of it that reaches a realisation.

- [ ] **Step 4: Confirm the tests pass**

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
  - `read_inherited_seeds(events_dir: Path, rupture_id: int) -> dict[str, int] | None` — the `seeds` block of `events_dir/<rupture id>/realisation.json`, or `None` when that file is absent or carries no seeds. Only the seed block is read; the driver stays schema-agnostic and the realisation engine validates the block when it reads it back.
  - CLI option `--inherit-seeds-from <events dir>`, off by default. The campaign points it at `/home/arr65/src/cs_nshm_2022/cs_nshm_2022/events`.

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


def test_read_inherited_seeds_reads_the_deployed_seed_block(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    (events_dir / "100932").mkdir(parents=True)
    seeds = {
        "nshm_to_realisation_seed": 531798913,
        "rupture_propagation_seed": 31268976,
        "genslip_seed": 513004717,
        "srfgen_seed": 1837842819,
        "hf_seed": 1524796118,
    }
    (events_dir / "100932" / "realisation.json").write_text(
        json.dumps(
            {
                "metadata": {"name": "Rupture 100932"},
                "magnitudes": {"AlpineK2T": 7.1},
                "seeds": seeds,
            }
        ),
        encoding="utf-8",
    )

    # Only the seed block is taken; every other field is ignored.
    assert gr.read_inherited_seeds(events_dir, 100932) == seeds


def test_read_inherited_seeds_returns_none_when_the_event_is_absent(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()

    assert gr.read_inherited_seeds(events_dir, 999999) is None


def test_read_inherited_seeds_returns_none_when_the_seed_block_is_missing(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    (events_dir / "100932").mkdir(parents=True)
    (events_dir / "100932" / "realisation.json").write_text(
        json.dumps({"metadata": {"name": "Rupture 100932"}}), encoding="utf-8"
    )

    assert gr.read_inherited_seeds(events_dir, 100932) is None
```

- [ ] **Step 7: Run the tests to verify they fail**

```bash
uv run pytest tests/test_generate_realisations_from_csv.py -q
```

Expected: FAIL — `TypeError: generate_one() got an unexpected keyword argument 'seeds'` and `AttributeError: module '...' has no attribute 'read_inherited_seeds'`.

- [ ] **Step 8: Inherit seeds from the deployed realisations**

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
        nshm2022-to-realisation replays these seeds via
        ``Seeds.read_from_realisation_or_random`` instead of drawing fresh ones.

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

Add the seed reader directly above the `@app.command()` line:

```python
def read_inherited_seeds(events_dir: Path, rupture_id: int) -> dict[str, int] | None:
    """Read the seed block of an already-deployed realisation.

    Only the ``seeds`` key is taken. Every other field in the deployed file is
    ignored, so the regenerated realisation is derived fresh from the database
    and the current code rather than inherited.

    Parameters
    ----------
    events_dir : Path
        Directory holding one ``<rupture id>/realisation.json`` per event.
    rupture_id : int
        The NSHM rupture id to look up.

    Returns
    -------
    dict of str to int, or None
        The deployed seed block, or None when the event has no deployed
        realisation or that realisation carries no seeds. None makes the caller
        fall back to a fresh random draw, which is correct for a brand-new event.
    """
    realisation_ffp = events_dir / str(rupture_id) / "realisation.json"
    if not realisation_ffp.is_file():
        return None
    with open(realisation_ffp, encoding="utf-8") as handle:
        realisation = json.load(handle)
    return realisation.get("seeds") or None
```

Add the `--inherit-seeds-from` option to the command signature, immediately after the `defaults_version` argument:

```python
    defaults_version: Annotated[DefaultsVersion, typer.Argument()],
    inherit_seeds_from: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help=(
                "Events directory to inherit seeds from. Each rupture reuses the "
                "seeds recorded in <events dir>/<rupture id>/realisation.json; "
                "ruptures with no deployed file get a fresh random draw."
            ),
        ),
    ] = None,
```

Add the matching docstring entry to the command's `Parameters` section, or `ruff` will fail on `numpydoc`:

```
    inherit_seeds_from : Path, optional
        Events directory to inherit seeds from. Omit to draw fresh seeds.
```

Replace the loop body's `generate_one` call so each rupture looks up its own seeds, and count how many were inherited:

```python
    inherited = 0
    for rupture_id in tqdm(rupture_ids, desc="Generating realisations"):
        realisation_ffp = output_dir / f"realisation_{rupture_id}.json"
        seeds = (
            read_inherited_seeds(inherit_seeds_from, rupture_id)
            if inherit_seeds_from is not None
            else None
        )
        if seeds is not None:
            inherited += 1
        error_msg = generate_one(
            nshmdb_path, rupture_id, realisation_ffp, defaults_version, seeds=seeds
        )
```

And report the split alongside the existing summary, immediately before the `Error log written to` line:

```python
    if inherit_seeds_from is not None:
        print(
            f"Inherited seeds for {inherited} of {len(rupture_ids)} rupture(s); "
            f"{len(rupture_ids) - inherited} drew fresh seeds."
        )
```

That count is a campaign gate, not decoration: Task 10b and Task 11 both assert it equals the number of ruptures being regenerated. A silent fallback to fresh seeds is exactly the failure that would produce plausible-looking realisations inconsistent with the SRFs already built from them.

- [ ] **Step 9: Run the tests to verify they pass**

```bash
uv run pytest tests/test_generate_realisations_from_csv.py -q
```

Expected: `8 passed`.

- [ ] **Step 10: Commit**

```bash
git add workflow/scripts/generate_realisations_from_csv.py tests/test_generate_realisations_from_csv.py
git commit -m "feat(scripts): add opt-in --inherit-seeds-from to reuse deployed seeds

Each stub is pre-written with the seed block already recorded in the deployed
<events dir>/<rupture id>/realisation.json, so nshm2022-to-realisation replays
those seeds via read_from_realisation_or_random instead of drawing fresh ones.
Only the seeds are taken; every other field is derived fresh from the database
and the current code. A rupture with no deployed file falls back to a fresh
draw, and the run reports the inherited/fresh split so a silent fallback cannot
pass unnoticed. Without the flag, behaviour is unchanged."
```

---

## Task 4: Move `copy_realisations_to_event_dirs` into the package and give it a CLI

The script is committed with a hardcoded path — `/home/arr65/src/cybershake_nshm_2022/flow/events` (`copy_realisations_to_event_dirs.py:17`) — that **no longer exists**. It is doubly stale: the `flow/` layout was replaced by an inner package directory, and the repo itself was then renamed, so the tree now lives at `/home/arr65/src/cs_nshm_2022/cs_nshm_2022/events/`. Move the script alongside the other campaign tools and take both directories as arguments. This also clears the last outstanding `ruff` error.

**Files:**
- Move: `copy_realisations_to_event_dirs.py` → `workflow/scripts/copy_realisations_to_event_dirs.py`
- Modify: `pyproject.toml`
- Test: `tests/test_copy_realisations_to_event_dirs.py` (create)

This task also adds the **overwrite gate**. Deploying is destructive: it replaces the very files the content checker compares against, so once it has run there is no comparison target left. The default must therefore be to *refuse* to replace anything, and replacing must be named explicitly.

**Interfaces:**
- Consumes: nothing
- Produces: `copy_realisations(source_dir: Path, events_dir: Path, overwrite_existing: bool = False) -> tuple[int, list[str], list[str]]` — returns the number copied, the names of files skipped for want of an integer id, and the rupture ids **refused** because a `realisation.json` already exists and `overwrite_existing` was False.
- Produces: CLI flag `--overwrite-existing`, off by default. With refusals and the flag off, the command exits **1**.

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

    copied, skipped, refused = cre.copy_realisations(source, events)

    assert copied == 2
    assert skipped == []
    assert refused == []
    assert (events / "100932" / "realisation.json").read_text() == '{"a": 1}'
    assert (events / "71220" / "realisation.json").read_text() == '{"b": 2}'


def test_copy_realisations_skips_files_without_an_integer_id(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text("{}", encoding="utf-8")
    (source / "notes.json").write_text("{}", encoding="utf-8")
    events = tmp_path / "events"

    copied, skipped, _ = cre.copy_realisations(source, events)

    assert copied == 1
    assert skipped == ["notes.json"]


def test_copy_realisations_refuses_to_replace_without_the_flag(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"new": 1}', encoding="utf-8")
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    copied, _, refused = cre.copy_realisations(source, events)

    assert copied == 0
    assert refused == ["100932"]
    # The existing file is untouched -- this is the whole point of the gate.
    assert (events / "100932" / "realisation.json").read_text() == '{"old": 1}'


def test_copy_realisations_replaces_when_the_flag_is_given(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"new": 1}', encoding="utf-8")
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    copied, _, refused = cre.copy_realisations(source, events, overwrite_existing=True)

    assert copied == 1
    assert refused == []
    assert (events / "100932" / "realisation.json").read_text() == '{"new": 1}'


def test_copy_realisations_still_creates_new_events_alongside_refusals(
    tmp_path: Path,
) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"new": 1}', encoding="utf-8")
    (source / "realisation_71220.json").write_text('{"fresh": 1}', encoding="utf-8")
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    copied, _, refused = cre.copy_realisations(source, events)

    assert copied == 1
    assert refused == ["100932"]
    assert (events / "71220" / "realisation.json").read_text() == '{"fresh": 1}'
    assert (events / "100932" / "realisation.json").read_text() == '{"old": 1}'
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


def copy_realisations(
    source_dir: Path, events_dir: Path, overwrite_existing: bool = False
) -> tuple[int, list[str], list[str]]:
    """Copy each realisation into ``events_dir/<rupture id>/realisation.json``.

    New event directories are always created. An **existing** realisation is
    replaced only when ``overwrite_existing`` is True; otherwise it is left
    untouched and its rupture id is returned as refused.

    Parameters
    ----------
    source_dir : Path
        Directory of ``realisation_<id>.json`` files.
    events_dir : Path
        Directory to create per-rupture event directories under.
    overwrite_existing : bool
        Whether to replace realisations that already exist.

    Returns
    -------
    tuple of (int, list of str, list of str)
        The number of realisations copied, the names of any files skipped
        because no integer id could be read from the filename, and the rupture
        ids refused because a realisation already existed.
    """
    copied = 0
    skipped: list[str] = []
    refused: list[str] = []

    for realisation_ffp in sorted(source_dir.glob("*.json")):
        match = re.search(r"\d+", realisation_ffp.name)
        if match is None:
            skipped.append(realisation_ffp.name)
            continue

        rupture_id = match.group()
        event_dir = events_dir / rupture_id
        target = event_dir / "realisation.json"
        if target.exists() and not overwrite_existing:
            refused.append(rupture_id)
            continue

        event_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(realisation_ffp, target)
        copied += 1

    return copied, skipped, refused


@app.command()
def copy_realisations_to_event_dirs(
    source_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    events_dir: Annotated[Path, typer.Argument(file_okay=False)],
    overwrite_existing: Annotated[
        bool,
        typer.Option(
            help="Replace realisations that already exist. Without this, existing "
            "files are left untouched and the command exits 1."
        ),
    ] = False,
) -> None:
    """Distribute completed realisations into per-event CyberShake directories.

    Parameters
    ----------
    source_dir : Path
        Directory of ``realisation_<id>.json`` files.
    events_dir : Path
        Directory to create per-rupture event directories under.
    overwrite_existing : bool
        Whether to replace realisations that already exist.
    """
    copied, skipped, refused = copy_realisations(
        source_dir, events_dir, overwrite_existing
    )

    print(f"Copied {copied} realisation(s) into {events_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with no integer id:")
        for name in skipped:
            print(f"  {name}")
    if refused:
        print(
            f"\nRefused to replace {len(refused)} existing realisation(s). "
            f"Re-run with --overwrite-existing to replace them:"
        )
        for rupture_id in refused:
            print(f"  {rupture_id}")
        raise typer.Exit(code=1)


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

Expected: `5 passed`, and ruff reports **`All checks passed!`** — the `D103` from Task 2 Step 4 is now gone.

- [ ] **Step 7: Commit**

```bash
git add workflow/scripts/copy_realisations_to_event_dirs.py tests/test_copy_realisations_to_event_dirs.py pyproject.toml
git commit -m "refactor(scripts): make copy-realisations-to-event-dirs a proper CLI

The script was committed with a hardcoded events path that is doubly stale: the
flow/ layout was replaced by an inner package directory, and the repo was then
renamed cybershake_nshm_2022 -> cs_nshm_2022. Move it in with the other campaign
tools, take both directories as arguments, and give it an entry point so the
whole campaign runs through entry points.

Deploying is destructive -- it replaces the files the content checker compares
against -- so replacing an existing realisation now requires --overwrite-existing.
Without it, existing files are left untouched, the refused rupture ids are
listed, and the command exits 1. New event directories are still created either
way, so a partial deployment can be completed without being forced to authorise
overwrites it does not need."
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

## Task 7a: `reconcile_parameters` — the comparison engine

`pegasus` is under active development, so its scientific defaults keep moving while this campaign's own overrides (`felipe_scripts/`) stay put. When the two disagree, somebody has to decide — and today nobody is asked, so the disagreement resolves silently. That is how the deployed set came to be missing `PGD`.

This task builds the pure comparison layer: given the three sources of a parameter, decide whether they agree, agree *closely enough*, or genuinely conflict. Task 7b puts a decision record and an interactive CLI on top of it.

The crucial distinction is the middle case. The deployed `im.fas_frequencies` differ from `felipe_scripts/frequencies.csv` in 165 of 389 values — by at most 57 ULP, ≤ 6.69e-15 relative. That is the same intended log-spaced grid produced by a different floating-point path, not a scientific difference. If it were treated as a conflict it would demand a decision on every run forever.

**Files:**
- Create: `workflow/scripts/reconcile_parameters.py`
- Test: `tests/test_reconcile_parameters.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `WATCHED_SECTIONS: tuple[str, ...]` — the ten realisation sections that are parameters rather than per-event derivations
  - `DEFAULT_TOLERANCE: float` — `1e-9`
  - `values_equivalent(left: object, right: object, tolerance: float = DEFAULT_TOLERANCE) -> bool`
  - `is_discrete(value: object) -> bool` — whether set-union is well defined for this value
  - `flatten_sections(sections: dict[str, object], watched: tuple[str, ...] = WATCHED_SECTIONS) -> dict[str, object]` — `{"im.ims": [...], ...}`
  - `Conflict` dataclass with fields `path: str`, `candidates: dict[str, object]`, `equivalent: list[tuple[str, str]]`, `discrete: bool`
  - `find_conflicts(by_source: dict[str, dict[str, object]], tolerance: float = DEFAULT_TOLERANCE) -> list[Conflict]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reconcile_parameters.py`:

```python
"""Tests for the parameter reconciliation comparison engine."""

from workflow.scripts import reconcile_parameters as rp


def test_values_equivalent_for_identical_lists() -> None:
    assert rp.values_equivalent([1.0, 2.0], [1.0, 2.0])


def test_values_equivalent_absorbs_float_noise_within_tolerance() -> None:
    # The real felipe-vs-deployed fas_frequencies case: ~1 ULP apart.
    assert rp.values_equivalent(
        [0.013489631494355309, 0.014125378225985656],
        [0.0134896314943553, 0.0141253782259856],
    )


def test_values_not_equivalent_beyond_tolerance() -> None:
    assert not rp.values_equivalent([1.0], [1.0000001])


def test_values_not_equivalent_for_different_lengths() -> None:
    assert not rp.values_equivalent([1.0, 2.0], [1.0, 2.0, 3.0])


def test_values_equivalent_compares_nested_dicts() -> None:
    assert rp.values_equivalent({"a": {"b": 1.0}}, {"a": {"b": 1.0}})
    assert not rp.values_equivalent({"a": {"b": 1.0}}, {"a": {"c": 1.0}})


def test_values_equivalent_does_not_conflate_bools_with_ints() -> None:
    # bool is a subclass of int in Python; True must not equal 1 here.
    assert not rp.values_equivalent(True, 1)


def test_is_discrete_true_for_a_list_of_names() -> None:
    assert rp.is_discrete(["PGA", "PGV", "pSA"])


def test_is_discrete_false_for_a_float_grid() -> None:
    # Union on a float grid manufactures near-duplicate points; never offer it.
    assert not rp.is_discrete([0.1, 0.2, 0.3])


def test_is_discrete_false_for_scalars_and_empty_lists() -> None:
    assert not rp.is_discrete("PGA")
    assert not rp.is_discrete([])


def test_flatten_sections_produces_dotted_paths() -> None:
    flat = rp.flatten_sections({"im": {"ims": ["PGA"], "valid_periods": [0.1]}})

    assert flat == {"im.ims": ["PGA"], "im.valid_periods": [0.1]}


def test_flatten_sections_ignores_unwatched_sections() -> None:
    flat = rp.flatten_sections({"im": {"ims": ["PGA"]}, "sources": {"A": {}}})

    assert flat == {"im.ims": ["PGA"]}


def test_find_conflicts_is_empty_when_every_source_agrees() -> None:
    flat = {"im.ims": ["PGA", "PGV"]}

    assert rp.find_conflicts({"defaults": flat, "deployed": dict(flat)}) == []


def test_find_conflicts_reports_a_set_difference() -> None:
    conflicts = rp.find_conflicts(
        {
            "defaults": {"im.ims": ["PGA", "PGV", "PGD"]},
            "deployed": {"im.ims": ["PGA", "PGV"]},
        }
    )

    assert len(conflicts) == 1
    assert conflicts[0].path == "im.ims"
    assert conflicts[0].candidates["defaults"] == ["PGA", "PGV", "PGD"]
    assert conflicts[0].discrete is True


def test_find_conflicts_treats_tolerance_equal_sources_as_agreeing() -> None:
    conflicts = rp.find_conflicts(
        {
            "felipe": {"im.fas_frequencies": [0.013489631494355309]},
            "deployed": {"im.fas_frequencies": [0.0134896314943553]},
        }
    )

    assert conflicts == []


def test_find_conflicts_records_equivalence_alongside_a_real_conflict() -> None:
    # The real im.fas_frequencies case: defaults hold a different grid, while
    # felipe and deployed differ only by float noise.
    conflicts = rp.find_conflicts(
        {
            "defaults": {"im.fas_frequencies": [0.1, 0.2]},
            "felipe": {"im.fas_frequencies": [0.013489631494355309, 0.2]},
            "deployed": {"im.fas_frequencies": [0.0134896314943553, 0.2]},
        }
    )

    assert len(conflicts) == 1
    assert conflicts[0].equivalent == [("felipe", "deployed")]
    assert conflicts[0].discrete is False


def test_find_conflicts_ignores_paths_only_one_source_has_an_opinion_on() -> None:
    conflicts = rp.find_conflicts(
        {"defaults": {"im.ims": ["PGA"]}, "felipe": {}}
    )

    assert conflicts == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_reconcile_parameters.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.scripts.reconcile_parameters'`.

- [ ] **Step 3: Write the comparison engine**

Create `workflow/scripts/reconcile_parameters.py`:

```python
#!/usr/bin/env python3
"""Reconcile campaign parameters against the scientific defaults.

Description
-----------
Compares the parameter sections of a realisation set against three sources --
the scientific defaults, the campaign's override files, and the values actually
deployed -- and reports where they genuinely disagree.

Only *parameters* are watched. The sections derived per event from the rupture
database and the seeds (``sources``, ``magnitudes``, ``rupture_propagation``,
``rakes``, ``domain``, ``seeds``, ``metadata``) are never compared here: they
differ between events by design.
"""

import dataclasses
import math
from typing import Any

# Realisation sections that hold parameters rather than per-event derivations.
# Verified 2026-07-27: each of these carries exactly one distinct value across
# all 291 deployed events, so one decision necessarily covers the whole set.
WATCHED_SECTIONS: tuple[str, ...] = (
    "im",
    "velocity_model",
    "emod3d",
    "resolution",
    "srf",
    "velocity_model_1d",
    "hf_velocity_model_1d",
    "hf",
    "bb",
    "rupture_velocity",
)

# Relative tolerance below which two numbers are the same parameter written
# differently. The observed felipe-vs-deployed fas_frequencies noise is 6.7e-15.
DEFAULT_TOLERANCE = 1e-9


def values_equivalent(
    left: object, right: object, tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """Return whether two parameter values are the same, allowing float noise.

    Comparison is purely relative -- ``abs_tol`` is zero -- so a value that
    changed from exactly zero is always reported as a difference rather than
    being absorbed.

    Parameters
    ----------
    left : object
        One value.
    right : object
        The other value.
    tolerance : float
        Relative tolerance for numeric comparison.

    Returns
    -------
    bool
        True when the values are equal, or numerically equal within tolerance.
    """
    # bool is a subclass of int; compare it by identity before the numeric case.
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=tolerance, abs_tol=0.0)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            values_equivalent(one, other, tolerance)
            for one, other in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            values_equivalent(left[key], right[key], tolerance) for key in left
        )
    return left == right


def is_discrete(value: object) -> bool:
    """Return whether set-union is a meaningful operation on this value.

    Union is offered only for non-empty lists of strings -- a list of intensity
    measure names, say. It is deliberately refused for numeric grids: unioning
    two float grids that differ by rounding produces near-duplicate points a
    fraction of a percent apart inside a grid whose real spacing is percent-level.

    Parameters
    ----------
    value : object
        The candidate value.

    Returns
    -------
    bool
        True when the value is a non-empty list of strings.
    """
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, str) for item in value)


def flatten_sections(
    sections: dict[str, Any], watched: tuple[str, ...] = WATCHED_SECTIONS
) -> dict[str, Any]:
    """Flatten the watched sections into dotted parameter paths.

    Conflicts are resolved key by key, not section by section, so that adopting
    a new ``im.ims`` does not force a decision about ``im.valid_periods``.

    Parameters
    ----------
    sections : dict
        Realisation sections keyed by section name.
    watched : tuple of str
        Section names to include.

    Returns
    -------
    dict
        Maps ``"<section>.<key>"`` to its value. Sections that are not
        dictionaries are kept whole under their own name.
    """
    flat: dict[str, Any] = {}
    for section in watched:
        if section not in sections:
            continue
        body = sections[section]
        if isinstance(body, dict):
            for key, value in body.items():
                flat[f"{section}.{key}"] = value
        else:
            flat[section] = body
    return flat


@dataclasses.dataclass
class Conflict:
    """A parameter whose sources genuinely disagree.

    Attributes
    ----------
    path : str
        Dotted parameter path, e.g. ``"im.ims"``.
    candidates : dict
        Maps each source name to the value it proposes. Only sources that have
        an opinion on this parameter appear.
    equivalent : list of tuple of (str, str)
        Source pairs that are not identical but agree within tolerance. Recorded
        so the report can say *why* two candidates that look different are not
        the thing being asked about.
    discrete : bool
        Whether every candidate is discrete, and so union may be offered.
    """

    path: str
    candidates: dict[str, Any]
    equivalent: list[tuple[str, str]]
    discrete: bool


def find_conflicts(
    by_source: dict[str, dict[str, Any]], tolerance: float = DEFAULT_TOLERANCE
) -> list[Conflict]:
    """Find every parameter whose sources genuinely disagree.

    A parameter is reported only when at least two sources differ by more than
    ``tolerance``. Paths that only one source has an opinion on are skipped:
    there is nothing to decide.

    Parameters
    ----------
    by_source : dict
        Maps a source name (``"defaults"``, ``"felipe"``, ``"deployed"``) to that
        source's flattened parameters.
    tolerance : float
        Relative tolerance for numeric comparison.

    Returns
    -------
    list of Conflict
        Conflicts, ordered by parameter path.
    """
    conflicts: list[Conflict] = []
    paths = sorted({path for flat in by_source.values() for path in flat})

    for path in paths:
        candidates = {
            source: flat[path] for source, flat in by_source.items() if path in flat
        }
        if len(candidates) < 2:
            continue

        sources = list(candidates)
        equivalent: list[tuple[str, str]] = []
        conflicting = False
        for index, left in enumerate(sources):
            for right in sources[index + 1 :]:
                if candidates[left] == candidates[right]:
                    continue
                if values_equivalent(candidates[left], candidates[right], tolerance):
                    equivalent.append((left, right))
                else:
                    conflicting = True

        if not conflicting:
            continue

        conflicts.append(
            Conflict(
                path=path,
                candidates=candidates,
                equivalent=equivalent,
                discrete=all(is_discrete(value) for value in candidates.values()),
            )
        )

    return conflicts
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_reconcile_parameters.py -q
```

Expected: `16 passed`.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/reconcile_parameters.py tests/test_reconcile_parameters.py
git commit -m "feat(scripts): add the parameter reconciliation comparison engine

Compares a realisation set's parameter sections against the scientific
defaults, the campaign overrides, and the deployed values, and reports where
they genuinely disagree.

Three classifications, not two. Values equal within a relative tolerance are
treated as agreeing, because the deployed im.fas_frequencies differ from
felipe_scripts/frequencies.csv by at most 57 ULP -- the same intended grid via a
different floating-point path. Without that, the campaign would be asked to
decide the same non-question on every run.

Union is offered only for lists of strings. On a float grid it manufactures
near-duplicate points ~1e-16 apart inside percent-level spacing, so it is
refused there by construction rather than by convention."
```

---

## Task 7b: `reconcile_parameters` — the decision record and the CLI

Task 7a decides *whether* sources disagree. This task decides *what to do about it*, records the answer, and makes the answer durable.

The record is what lets this survive active development. Each entry stores the chosen source **and a fingerprint of the value that choice resolved to**. On a later run, a decision whose source still hashes the same stays silent; one whose source moved underneath re-prompts, flagged. So the second and subsequent `pegasus` merges only ask about what actually changed.

**Files:**
- Modify: `workflow/scripts/reconcile_parameters.py`
- Modify: `pyproject.toml` (add a `[project.scripts]` entry)
- Test: `tests/test_reconcile_parameters.py`

**Interfaces:**
- Consumes: `Conflict`, `find_conflicts`, `flatten_sections`, `WATCHED_SECTIONS` from Task 7a; `workflow.defaults.load_defaults`; `workflow.scripts.complete_realisations.load_overrides`
- Produces:
  - `value_fingerprint(value: object) -> str` — sha256 of the value's canonical JSON
  - `Decision` dataclass with fields `source: str`, `reason: str`, `decided: str`, `sha256: str`
  - `resolve_value(decision: Decision, candidates: dict[str, object]) -> object`
  - `decision_is_current(decision: Decision, candidates: dict[str, object]) -> bool`
  - `load_decisions(decisions_ffp: Path) -> dict[str, Decision]` — keyed by dotted path
  - `save_decisions(decisions_ffp: Path, decisions: dict[str, Decision]) -> None`
  - `read_deployed_parameters(events_dir: Path) -> tuple[dict[str, object], dict[str, dict[str, list[str]]]]` — the deployed parameters, plus per-path divergence groups
  - Entry point `reconcile-parameters`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reconcile_parameters.py`. Add `import json` and `from pathlib import Path` to the imports at the top of the file.

```python
def test_value_fingerprint_is_stable_across_key_order() -> None:
    assert rp.value_fingerprint({"a": 1, "b": 2}) == rp.value_fingerprint(
        {"b": 2, "a": 1}
    )


def test_value_fingerprint_distinguishes_different_values() -> None:
    assert rp.value_fingerprint([1, 2]) != rp.value_fingerprint([1, 3])


def test_resolve_value_returns_the_chosen_source() -> None:
    decision = rp.Decision(
        source="defaults", reason="adopt PGD", decided="2026-07-27", sha256=""
    )

    resolved = rp.resolve_value(decision, {"defaults": ["PGA"], "deployed": ["PGV"]})

    assert resolved == ["PGA"]


def test_resolve_value_unions_discrete_candidates_deterministically() -> None:
    decision = rp.Decision(
        source="union", reason="keep both", decided="2026-07-27", sha256=""
    )

    resolved = rp.resolve_value(
        decision, {"defaults": ["PGA", "PGD"], "deployed": ["PGA", "PGV"]}
    )

    # Sources are visited in sorted order, so the result is reproducible.
    assert resolved == ["PGA", "PGD", "PGV"]


def test_resolve_value_rejects_a_source_that_has_no_opinion() -> None:
    decision = rp.Decision(
        source="felipe", reason="", decided="2026-07-27", sha256=""
    )

    try:
        rp.resolve_value(decision, {"defaults": ["PGA"]})
    except ValueError as error:
        assert "felipe" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_decision_is_current_when_the_source_still_hashes_the_same() -> None:
    candidates = {"defaults": ["PGA", "PGD"]}
    decision = rp.Decision(
        source="defaults",
        reason="adopt PGD",
        decided="2026-07-27",
        sha256=rp.value_fingerprint(["PGA", "PGD"]),
    )

    assert rp.decision_is_current(decision, candidates)


def test_decision_is_stale_when_the_source_moved_underneath() -> None:
    decision = rp.Decision(
        source="defaults",
        reason="adopt PGD",
        decided="2026-07-27",
        sha256=rp.value_fingerprint(["PGA", "PGD"]),
    )

    assert not rp.decision_is_current(decision, {"defaults": ["PGA", "PGD", "PGV"]})


def test_decisions_round_trip_through_yaml(tmp_path: Path) -> None:
    decisions_ffp = tmp_path / "campaign_parameters.yaml"
    decisions = {
        "im.ims": rp.Decision(
            source="defaults",
            reason="adopt PGD from pegasus ec2fb25",
            decided="2026-07-27",
            sha256="abc123",
        )
    }

    rp.save_decisions(decisions_ffp, decisions)

    assert rp.load_decisions(decisions_ffp) == decisions


def test_load_decisions_returns_empty_for_a_missing_file(tmp_path: Path) -> None:
    assert rp.load_decisions(tmp_path / "absent.yaml") == {}


def test_read_deployed_parameters_reads_a_consistent_set(tmp_path: Path) -> None:
    events = tmp_path / "events"
    for rupture_id in ("100932", "101084"):
        (events / rupture_id).mkdir(parents=True)
        (events / rupture_id / "realisation.json").write_text(
            json.dumps({"im": {"ims": ["PGA"]}, "sources": {"whatever": 1}}),
            encoding="utf-8",
        )

    parameters, divergence = rp.read_deployed_parameters(events)

    assert parameters == {"im.ims": ["PGA"]}
    assert divergence == {}


def test_read_deployed_parameters_reports_a_partially_diverged_set(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    for rupture_id, ims in (("100932", ["PGA"]), ("101084", ["PGA", "PGD"])):
        (events / rupture_id).mkdir(parents=True)
        (events / rupture_id / "realisation.json").write_text(
            json.dumps({"im": {"ims": ims}}), encoding="utf-8"
        )

    _, divergence = rp.read_deployed_parameters(events)

    assert set(divergence) == {"im.ims"}
    assert sorted(len(ids) for ids in divergence["im.ims"].values()) == [1, 1]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_reconcile_parameters.py -q
```

Expected: FAIL — `AttributeError: module '...' has no attribute 'value_fingerprint'`.

- [ ] **Step 3: Add the decision record**

Extend the imports at the top of `workflow/scripts/reconcile_parameters.py`:

```python
import collections
import dataclasses
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from workflow.defaults import DefaultsVersion, load_defaults
```

**Do not** import `load_overrides` at module level here. Task 7d makes
`complete_realisations` import *this* module, so a module-level import in the
other direction is a circular import that fails on whichever module is loaded
first. It is imported inside the command body in Step 6 instead, where it is the
only thing needed from that module.

Append to the module:

```python
def value_fingerprint(value: object) -> str:
    """Return a stable sha256 fingerprint of a parameter value.

    Parameters
    ----------
    value : object
        Any JSON-serialisable parameter value.

    Returns
    -------
    str
        Hex sha256 of the value's canonical JSON, so dictionary key order
        cannot change the fingerprint.
    """
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class Decision:
    """A recorded resolution of one parameter conflict.

    Attributes
    ----------
    source : str
        The chosen source: ``defaults``, ``felipe``, ``deployed`` or ``union``.
    reason : str
        Why this source was chosen. Required, not decorative: an unexplained
        decision is the failure this campaign exists to eliminate.
    decided : str
        ISO date the decision was made.
    sha256 : str
        Fingerprint of the value the decision resolved to, so a later run can
        tell whether the chosen source has moved underneath it.
    """

    source: str
    reason: str
    decided: str
    sha256: str


def resolve_value(decision: Decision, candidates: dict[str, Any]) -> Any:
    """Return the value a decision selects from the available candidates.

    Parameters
    ----------
    decision : Decision
        The recorded decision.
    candidates : dict
        Maps source name to proposed value.

    Returns
    -------
    object
        The selected value.

    Raises
    ------
    ValueError
        If the decision names a source that has no opinion on this parameter.
    """
    if decision.source == "union":
        merged: list[Any] = []
        for source in sorted(candidates):
            for item in candidates[source]:
                if item not in merged:
                    merged.append(item)
        return merged
    if decision.source not in candidates:
        raise ValueError(
            f"Decision names source {decision.source!r}, which has no value here. "
            f"Available: {sorted(candidates)}."
        )
    return candidates[decision.source]


def decision_is_current(decision: Decision, candidates: dict[str, Any]) -> bool:
    """Return whether a decision still resolves to the value it recorded.

    Parameters
    ----------
    decision : Decision
        The recorded decision.
    candidates : dict
        Maps source name to proposed value, as computed now.

    Returns
    -------
    bool
        True when the chosen source still yields the recorded fingerprint.
    """
    try:
        return value_fingerprint(resolve_value(decision, candidates)) == decision.sha256
    except (ValueError, TypeError):
        return False


def load_decisions(decisions_ffp: Path) -> dict[str, Decision]:
    """Load recorded decisions, keyed by dotted parameter path.

    Parameters
    ----------
    decisions_ffp : Path
        Path to the campaign parameters YAML.

    Returns
    -------
    dict of str to Decision
        Empty when the file does not exist -- the first run has no history.
    """
    if not decisions_ffp.is_file():
        return {}
    with open(decisions_ffp, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    decisions: dict[str, Decision] = {}
    for section, entries in raw.items():
        for key, entry in entries.items():
            decisions[f"{section}.{key}"] = Decision(
                source=str(entry["source"]),
                reason=str(entry["reason"]),
                decided=str(entry["decided"]),
                sha256=str(entry["sha256"]),
            )
    return decisions


def save_decisions(decisions_ffp: Path, decisions: dict[str, Decision]) -> None:
    """Write decisions to YAML, nested by section for readability.

    Parameters
    ----------
    decisions_ffp : Path
        Path to write.
    decisions : dict of str to Decision
        Decisions keyed by dotted parameter path.
    """
    nested: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    for path, decision in sorted(decisions.items()):
        section, _, key = path.partition(".")
        nested[section][key] = dataclasses.asdict(decision)
    decisions_ffp.parent.mkdir(parents=True, exist_ok=True)
    with open(decisions_ffp, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(nested), handle, sort_keys=True, default_flow_style=False)


def read_deployed_parameters(
    events_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
    """Read the watched parameters from every deployed realisation.

    Parameters
    ----------
    events_dir : Path
        Directory holding one ``<rupture id>/realisation.json`` per event.

    Returns
    -------
    tuple of (dict, dict)
        ``(parameters, divergence)``. ``parameters`` maps each dotted path to the
        value carried by the most events. ``divergence`` maps a dotted path to
        ``{fingerprint: [rupture ids]}`` for every path the events disagree on,
        and is empty for a consistent set -- which is what a fully deployed
        campaign looks like.
    """
    by_path: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    values_by_fingerprint: dict[str, Any] = {}

    for realisation_ffp in sorted(events_dir.glob("*/realisation.json")):
        rupture_id = realisation_ffp.parent.name
        with open(realisation_ffp, encoding="utf-8") as handle:
            realisation = json.load(handle)
        for path, value in flatten_sections(realisation).items():
            fingerprint = value_fingerprint(value)
            values_by_fingerprint[fingerprint] = value
            by_path[path][fingerprint].append(rupture_id)

    parameters: dict[str, Any] = {}
    divergence: dict[str, dict[str, list[str]]] = {}
    for path, groups in by_path.items():
        majority = max(groups, key=lambda key: len(groups[key]))
        parameters[path] = values_by_fingerprint[majority]
        if len(groups) > 1:
            divergence[path] = dict(groups)

    return parameters, divergence
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_reconcile_parameters.py -q
```

Expected: `27 passed`.

- [ ] **Step 5: Add the interactive CLI**

Append to `workflow/scripts/reconcile_parameters.py`:

```python
app = typer.Typer()
console = Console()

# Human-readable provenance, shown beside each candidate so the choice is made
# on visible evidence rather than recall.
SOURCE_BLURB = {
    "defaults": "scientific defaults at HEAD (pegasus)",
    "felipe": "campaign override files (felipe_scripts/)",
    "deployed": "values currently in the deployed realisations",
}


def describe(value: Any) -> str:
    """Return a one-line summary of a parameter value.

    Parameters
    ----------
    value : object
        The value to summarise.

    Returns
    -------
    str
        Element count and range for numeric lists, the items for short lists,
        and the repr otherwise.
    """
    if isinstance(value, list) and value:
        if all(isinstance(item, (int, float)) and not isinstance(item, bool)
               for item in value):
            return f"{len(value)} values  {min(value):.6g} .. {max(value):.6g}"
        if len(value) <= 12:
            return f"{len(value)}  " + " ".join(str(item) for item in value)
        return f"{len(value)} items"
    return repr(value)


def prompt_for_decision(conflict: Conflict, stale: bool) -> Decision:
    """Present one conflict and ask which source wins.

    Parameters
    ----------
    conflict : Conflict
        The conflict to resolve.
    stale : bool
        Whether a previous decision existed but its source has since moved.

    Returns
    -------
    Decision
        The chosen resolution, fingerprinted against the value it resolves to.
    """
    heading = f"CONFLICT  {conflict.path}"
    if stale:
        heading += "   [previously resolved, source has changed]"
    console.print(f"\n[bold]{heading}[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("choose")
    table.add_column("source")
    table.add_column("value")
    table.add_column("provenance")
    for source in sorted(conflict.candidates):
        table.add_row(
            source, source, describe(conflict.candidates[source]),
            SOURCE_BLURB.get(source, ""),
        )
    if conflict.discrete:
        table.add_row(
            "union", "union",
            describe(resolve_value(
                Decision("union", "", "", ""), conflict.candidates
            )),
            "every candidate's items combined",
        )
    console.print(table)

    for left, right in conflict.equivalent:
        console.print(
            f"  note: {left} and {right} agree within tolerance "
            f"-- this is not what you are being asked about"
        )
    if not conflict.discrete:
        console.print(
            "  note: union is not offered -- these values are not discrete, and "
            "unioning numeric grids manufactures near-duplicate points"
        )

    choices = sorted(conflict.candidates) + (["union"] if conflict.discrete else [])
    source = typer.prompt("  choose", type=click.Choice(choices))
    reason = ""
    while not reason.strip():
        reason = typer.prompt("  reason")

    return Decision(
        source=source,
        reason=reason.strip(),
        decided=datetime.date.today().isoformat(),
        sha256=value_fingerprint(
            resolve_value(Decision(source, "", "", ""), conflict.candidates)
        ),
    )
```

Add `import click` beside `import typer` — `typer` re-exports `click`, and `click.Choice` is what constrains the prompt to valid sources.

- [ ] **Step 6: Add the command**

Append to `workflow/scripts/reconcile_parameters.py`:

```python
@app.command()
def reconcile_parameters(
    events_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    decisions_ffp: Annotated[Path, typer.Argument(dir_okay=False)],
    defaults_version: Annotated[
        DefaultsVersion, typer.Option()
    ] = DefaultsVersion.v24_2_2_1,
    felipe_scripts_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False)
    ] = Path("felipe_scripts"),
    tolerance: Annotated[float, typer.Option()] = DEFAULT_TOLERANCE,
    non_interactive: Annotated[bool, typer.Option()] = False,
) -> None:
    """Reconcile deployed parameters against the defaults, recording each decision.

    Parameters
    ----------
    events_dir : Path
        Directory of deployed ``<rupture id>/realisation.json`` files.
    decisions_ffp : Path
        Campaign parameters YAML to read and update.
    defaults_version : DefaultsVersion
        Scientific defaults version to compare against.
    felipe_scripts_dir : Path
        Directory containing the campaign override files.
    tolerance : float
        Relative tolerance below which numeric values are treated as agreeing.
    non_interactive : bool
        Exit non-zero on any unresolved conflict instead of prompting.
    """
    # Imported here, not at module level: complete_realisations imports this
    # module (Task 7d), so importing it back at module level would be circular.
    from workflow.scripts.complete_realisations import load_overrides

    defaults = flatten_sections(load_defaults(defaults_version))
    overrides = load_overrides(felipe_scripts_dir)
    felipe = {
        "im.valid_periods": overrides.valid_periods.tolist(),
        "im.fas_frequencies": overrides.fas_frequencies.tolist(),
        "velocity_model.version": overrides.vm_version,
        "velocity_model.rrup_interpolants": overrides.rrup_interpolants.tolist(),
    }
    deployed, divergence = read_deployed_parameters(events_dir)

    if divergence:
        console.print(
            f"[yellow]The deployed set is not consistent: "
            f"{len(divergence)} parameter(s) differ between events.[/yellow]"
        )
        for path, groups in sorted(divergence.items()):
            console.print(f"  {path}: {len(groups)} distinct values")
            for fingerprint, rupture_ids in groups.items():
                console.print(
                    f"    {fingerprint[:12]}  {len(rupture_ids)} event(s), "
                    f"e.g. {', '.join(sorted(rupture_ids)[:3])}"
                )
        console.print(
            "  The majority value is used as the deployed candidate below.\n"
        )

    conflicts = find_conflicts(
        {"defaults": defaults, "felipe": felipe, "deployed": deployed}, tolerance
    )
    decisions = load_decisions(decisions_ffp)

    settled = 0
    unresolved: list[str] = []
    for conflict in conflicts:
        existing = decisions.get(conflict.path)
        if existing is not None and decision_is_current(existing, conflict.candidates):
            settled += 1
            continue
        if non_interactive:
            unresolved.append(conflict.path)
            continue
        decisions[conflict.path] = prompt_for_decision(
            conflict, stale=existing is not None
        )

    if unresolved:
        console.print(
            f"\n[red]{len(unresolved)} unresolved conflict(s); "
            f"re-run without --non-interactive to decide them:[/red]"
        )
        for path in unresolved:
            console.print(f"  {path}")
        raise typer.Exit(code=1)

    save_decisions(decisions_ffp, decisions)
    console.print(
        f"\n{len(conflicts)} conflict(s): {settled} already settled, "
        f"{len(conflicts) - settled} decided now."
    )
    console.print(f"Decisions written to {decisions_ffp}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 7: Register the entry point**

In `pyproject.toml`, under `[project.scripts]`, immediately after the `complete-realisations` line, add:

```toml
reconcile-parameters = "workflow.scripts.reconcile_parameters:app"
```

- [ ] **Step 8: Confirm the whole gate is clean**

```bash
uv sync --all-extras --dev
uv run pytest tests/test_reconcile_parameters.py -q
uv run ruff check
uv run ty check --exclude workflow/schemas.py --exclude setup.py
fdfind . workflow/ -E "__init__.py" --extension py | xargs numpydoc lint
```

Expected: `27 passed`, `All checks passed!`, and no numpydoc output.

- [ ] **Step 9: Commit**

```bash
git add workflow/scripts/reconcile_parameters.py tests/test_reconcile_parameters.py pyproject.toml
git commit -m "feat(scripts): record parameter decisions and add reconcile-parameters

Each conflict is presented with every candidate's provenance, element count and
range, and resolved to defaults, felipe, deployed, or -- for discrete values
only -- their union. The choice and a required reason are written to a YAML
decision record.

Each entry also stores a fingerprint of the value the choice resolved to. That
is what makes this repeatable while pegasus keeps moving: a decision whose
source still hashes the same stays silent, one whose source moved re-prompts
flagged as such, and a newly conflicting parameter prompts fresh. So the second
and subsequent merges only ask about what actually changed.

--non-interactive exits non-zero on any unresolved conflict, so CI and
unattended reruns cannot silently guess."
```

---

## Task 7c: `verify_realisation_content` — prove a regenerated file changed only where it was meant to

Provenance verification (Tasks 5–6) proves each file records a clean commit. This proves the *content* changed only where somebody decided it should.

The bar is no longer "identical except `log_trail`". The campaign now deliberately updates parameters — `im.ims` gains `PGD` — so the checker must accept exactly those changes and nothing else. **The decision file from Task 7b is the allowlist**, which is what stops the allowlist drifting away from the reasoning behind it.

Two failure modes, both hard stops:

- an **unexpected** difference — a source, a magnitude, a domain, an undecided parameter — means regeneration did something nobody chose;
- a **missing** expected difference means a recorded decision silently failed to apply.

Checking both directions is what makes the decision file load-bearing rather than decorative.

**Files:**
- Create: `workflow/scripts/verify_realisation_content.py`
- Modify: `pyproject.toml` (add a `[project.scripts]` entry)
- Test: `tests/test_verify_realisation_content.py` (create)

**Interfaces:**
- Consumes: `reconcile_parameters.load_decisions`, `reconcile_parameters.values_equivalent`, `reconcile_parameters.DEFAULT_TOLERANCE`
- Produces:
  - `diff_content(expected: object, actual: object, path: str = "", tolerance: float = DEFAULT_TOLERANCE) -> list[str]` — dotted paths at which two realisations differ, skipping the top-level `log_trail`; empty means equivalent
  - `classify_differences(differences: list[str], expected_paths: set[str]) -> tuple[list[str], list[str]]` — `(unexpected, satisfied)` split of observed differences against the decided parameter paths
  - `compare_files(expected_ffp: Path, actual_ffp: Path, expected_paths: set[str], tolerance: float = DEFAULT_TOLERANCE) -> tuple[list[str], list[str]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verify_realisation_content.py`:

```python
"""Tests for the realisation content checker."""

import json
from pathlib import Path

from workflow.scripts import verify_realisation_content as vc
from workflow.scripts.reconcile_parameters import Decision, value_fingerprint


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


def test_diff_content_absorbs_float_noise_within_tolerance() -> None:
    # The deployed/felipe fas_frequencies case: same grid, different float path.
    expected = {"im": {"fas_frequencies": [0.0134896314943553]}}
    actual = {"im": {"fas_frequencies": [0.013489631494355309]}}
    assert vc.diff_content(expected, actual) == []


def test_classify_differences_splits_decided_from_undecided() -> None:
    differences = [
        "im.ims: length 8 != 9",
        "magnitudes.A: 7.1 != 7.2",
    ]

    unexpected, satisfied = vc.classify_differences(differences, {"im.ims"})

    assert unexpected == ["magnitudes.A: 7.1 != 7.2"]
    assert satisfied == ["im.ims"]


def test_classify_differences_matches_nested_and_indexed_paths() -> None:
    differences = ["im.ims[8]: only in actual", "im.ims.extra: only in actual"]

    unexpected, satisfied = vc.classify_differences(differences, {"im.ims"})

    assert unexpected == []
    assert satisfied == ["im.ims"]


def test_classify_differences_does_not_match_a_path_prefix_by_accident() -> None:
    # "im.imsy" must not be absorbed by a decision about "im.ims".
    unexpected, _ = vc.classify_differences(["im.imsy: 1 != 2"], {"im.ims"})

    assert unexpected == ["im.imsy: 1 != 2"]


def test_check_decisions_applied_passes_when_the_value_matches() -> None:
    realisation = {"im": {"ims": ["PGA", "PGD"]}}
    decisions = {
        "im.ims": Decision(
            source="defaults",
            reason="adopt PGD",
            decided="2026-07-27",
            sha256=value_fingerprint(["PGA", "PGD"]),
        )
    }

    assert vc.check_decisions_applied(realisation, decisions) == []


def test_check_decisions_applied_reports_a_decision_that_did_not_take() -> None:
    realisation = {"im": {"ims": ["PGA"]}}
    decisions = {
        "im.ims": Decision(
            source="defaults",
            reason="adopt PGD",
            decided="2026-07-27",
            sha256=value_fingerprint(["PGA", "PGD"]),
        )
    }

    unapplied = vc.check_decisions_applied(realisation, decisions)

    assert len(unapplied) == 1
    assert "im.ims" in unapplied[0]


def test_check_decisions_applied_reports_an_absent_parameter() -> None:
    decisions = {
        "im.ims": Decision(
            source="defaults", reason="", decided="2026-07-27", sha256="abc"
        )
    }

    unapplied = vc.check_decisions_applied({"magnitudes": {}}, decisions)

    assert "absent" in unapplied[0]


def test_compare_files_accepts_a_decided_change_and_rejects_an_undecided_one(
    tmp_path: Path,
) -> None:
    expected_ffp = tmp_path / "original.json"
    actual_ffp = tmp_path / "regenerated.json"
    expected_ffp.write_text(
        json.dumps(
            {
                "im": {"ims": ["PGA"]},
                "magnitudes": {"A": 7.1},
                "log_trail": {"log": ["a"]},
            }
        ),
        encoding="utf-8",
    )
    actual_ffp.write_text(
        json.dumps(
            {
                "im": {"ims": ["PGA", "PGD"]},
                "magnitudes": {"A": 7.2},
                "log_trail": {"log": ["b"]},
            }
        ),
        encoding="utf-8",
    )
    decisions = {
        "im.ims": Decision(
            source="defaults",
            reason="adopt PGD",
            decided="2026-07-27",
            sha256=value_fingerprint(["PGA", "PGD"]),
        )
    }

    unexpected, unapplied = vc.compare_files(expected_ffp, actual_ffp, decisions)

    # The im.ims change was decided; the magnitude change was not.
    assert unexpected == ["magnitudes.A: 7.1 != 7.2"]
    assert unapplied == []
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
"""Verify regenerated realisations changed only where a decision said they would.

Description
-----------
Compares each ``realisation_<id>.json`` in a regenerated directory against the
deployed original at ``events/<id>/realisation.json``. Every field is significant
except ``log_trail``, which legitimately changes when the code is re-run, and the
parameters whose change is recorded in the campaign decision file.

Two things fail the check, and both exit non-zero:

* an **unexpected** difference -- something changed that nobody decided;
* an **unapplied** decision -- a recorded decision did not reach the file.

Checking both directions is what makes the decision file load-bearing.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from workflow.scripts.reconcile_parameters import (
    DEFAULT_TOLERANCE,
    Decision,
    load_decisions,
    value_fingerprint,
    values_equivalent,
)

app = typer.Typer()

MISSING = object()


def diff_content(
    expected: object,
    actual: object,
    path: str = "",
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[str]:
    """Return the dotted paths at which two realisations differ, ignoring log_trail.

    Numeric leaves are compared within ``tolerance``, so the float-path noise
    between the deployed and override frequency grids is not reported as a change.

    Parameters
    ----------
    expected : object
        The reference JSON value (initially the whole realisation dict).
    actual : object
        The value to compare against it.
    path : str
        The dotted path to the current value, used for reporting.
    tolerance : float
        Relative tolerance for numeric comparison.

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
                diffs.extend(
                    diff_content(expected[key], actual[key], child, tolerance)
                )
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(expected)} != {len(actual)}")
        else:
            for index, (exp, act) in enumerate(zip(expected, actual, strict=True)):
                diffs.extend(diff_content(exp, act, f"{path}[{index}]", tolerance))
    elif not values_equivalent(expected, actual, tolerance):
        diffs.append(f"{path}: {expected!r} != {actual!r}")
    return diffs


def classify_differences(
    differences: list[str], decided_paths: set[str]
) -> tuple[list[str], list[str]]:
    """Split observed differences into undecided ones and decided ones.

    Parameters
    ----------
    differences : list of str
        Entries from :func:`diff_content`, each starting ``"<path>: "``.
    decided_paths : set of str
        Dotted parameter paths a decision covers.

    Returns
    -------
    tuple of (list of str, list of str)
        ``(unexpected, satisfied)`` -- differences no decision covers, and the
        decided paths that did in fact change.
    """
    unexpected: list[str] = []
    satisfied: set[str] = set()
    for difference in differences:
        location = difference.split(":", 1)[0]
        owner = next(
            (
                decided
                for decided in decided_paths
                if location == decided
                or location.startswith(f"{decided}.")
                or location.startswith(f"{decided}[")
            ),
            None,
        )
        if owner is None:
            unexpected.append(difference)
        else:
            satisfied.add(owner)
    return unexpected, sorted(satisfied)


def value_at(realisation: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted path, or ``MISSING`` when absent.

    Parameters
    ----------
    realisation : dict
        The parsed realisation.
    path : str
        Dotted path such as ``"im.ims"``.

    Returns
    -------
    object
        The value, or the module-level ``MISSING`` sentinel.
    """
    current: Any = realisation
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current


def check_decisions_applied(
    realisation: dict[str, Any], decisions: dict[str, Decision]
) -> list[str]:
    """Return decided paths whose value in the file is not the decided value.

    Each decision records a fingerprint of the value it resolved to, so this
    needs no access to the defaults or the override files.

    Parameters
    ----------
    realisation : dict
        The regenerated realisation.
    decisions : dict of str to Decision
        Recorded decisions, keyed by dotted path.

    Returns
    -------
    list of str
        One entry per decision that did not reach the file.
    """
    unapplied: list[str] = []
    for path, decision in sorted(decisions.items()):
        value = value_at(realisation, path)
        if value is MISSING:
            unapplied.append(f"{path}: absent from the regenerated realisation")
        elif value_fingerprint(value) != decision.sha256:
            unapplied.append(
                f"{path}: not the decided value "
                f"(decided '{decision.source}', sha256 {decision.sha256[:12]})"
            )
    return unapplied


def compare_files(
    expected_ffp: Path,
    actual_ffp: Path,
    decisions: dict[str, Decision],
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[list[str], list[str]]:
    """Compare two realisation files against the recorded decisions.

    Parameters
    ----------
    expected_ffp : Path
        The deployed original.
    actual_ffp : Path
        The regenerated realisation.
    decisions : dict of str to Decision
        Recorded decisions, keyed by dotted path.
    tolerance : float
        Relative tolerance for numeric comparison.

    Returns
    -------
    tuple of (list of str, list of str)
        ``(unexpected, unapplied)``. Both empty means the file changed exactly
        where it was meant to and nowhere else.
    """
    expected = json.loads(expected_ffp.read_text(encoding="utf-8"))
    actual = json.loads(actual_ffp.read_text(encoding="utf-8"))
    differences = diff_content(expected, actual, tolerance=tolerance)
    unexpected, _ = classify_differences(differences, set(decisions))
    return unexpected, check_decisions_applied(actual, decisions)


@app.command()
def main(
    events_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    regenerated_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    parameters: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Campaign decision file. Without it, no parameter change is "
            "permitted and every difference beyond log_trail fails.",
        ),
    ] = None,
    tolerance: Annotated[float, typer.Option()] = DEFAULT_TOLERANCE,
) -> None:
    """Compare a regenerated realisation set against the deployed originals.

    Parameters
    ----------
    events_dir : Path
        Directory of ``<rupture_id>/realisation.json`` originals.
    regenerated_dir : Path
        Directory of ``realisation_<rupture_id>.json`` regenerated files.
    parameters : Path, optional
        Campaign decision file recording which parameter changes are intended.
    tolerance : float
        Relative tolerance for numeric comparison.
    """
    decisions = load_decisions(parameters) if parameters is not None else {}

    failed: dict[str, list[str]] = {}
    regenerated = sorted(regenerated_dir.glob("realisation_*.json"))
    for actual_ffp in regenerated:
        rupture_id = actual_ffp.stem.removeprefix("realisation_")
        expected_ffp = events_dir / rupture_id / "realisation.json"
        if not expected_ffp.is_file():
            failed[rupture_id] = ["no original to compare against"]
            continue
        unexpected, unapplied = compare_files(
            expected_ffp, actual_ffp, decisions, tolerance
        )
        problems = [f"UNEXPECTED  {entry}" for entry in unexpected]
        problems += [f"UNAPPLIED   {entry}" for entry in unapplied]
        if problems:
            failed[rupture_id] = problems

    print(
        f"Compared {len(regenerated)} realisation(s) against "
        f"{len(decisions)} recorded decision(s); {len(failed)} failed"
    )
    for rupture_id, problems in sorted(failed.items()):
        print(f"\n{rupture_id}:")
        for problem in problems[:20]:
            print(f"  {problem}")
    if failed:
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

Expected: `12 passed`.

- [ ] **Step 6: Commit**

```bash
git add workflow/scripts/verify_realisation_content.py tests/test_verify_realisation_content.py pyproject.toml
git commit -m "feat(scripts): add verify-realisation-content

Compares a regenerated realisation set against the deployed originals field by
field and exits non-zero unless the set changed exactly where it was meant to.

The campaign decision file is the allowlist, so the permitted changes and the
reasoning behind them cannot drift apart. Both directions are checked: an
unexpected difference means regeneration did something nobody chose, and an
unapplied decision means a recorded choice silently failed to reach the file.
Decisions are checked by the value fingerprint they already carry, so the
verifier needs neither the defaults nor the override files.

Numeric leaves are compared within a relative tolerance, so the float-path noise
between the deployed and override frequency grids is not reported as a change."
```

---

## Task 7d: `complete-realisations --parameters` — apply the recorded decisions

Task 7b records what should happen. This makes it happen.

The decision file records a *choice of source*, not a value, so this resolves each decision against the same three candidates the reconciler used and then **checks the resolved value against the fingerprint the decision recorded**. A mismatch means the chosen source moved since the decision was made, and it fails loudly rather than quietly materialising something nobody approved.

**Files:**
- Modify: `workflow/scripts/complete_realisations.py`
- Test: `tests/test_complete_realisations.py`

**Interfaces:**
- Consumes: `reconcile_parameters.{Decision, load_decisions, resolve_value, value_fingerprint, flatten_sections}`; `complete_realisations.load_overrides`
- Produces:
  - `resolve_parameters(decisions: dict[str, Decision], defaults_version: DefaultsVersion, felipe_scripts_dir: Path, events_dir: Path | None) -> dict[str, object]` — dotted path to the decided value
  - `apply_parameters(realisation: dict[str, object], resolved: dict[str, object]) -> None` — in-place
  - CLI option `--parameters <decision file>`, and `--deployed-from <events dir>` for decisions that name `deployed`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_complete_realisations.py`:

```python
def test_apply_parameters_overwrites_only_the_decided_keys() -> None:
    realisation = {
        "im": {"ims": ["PGA"], "valid_periods": [0.1]},
        "magnitudes": {"A": 7.1},
    }

    cr.apply_parameters(realisation, {"im.ims": ["PGA", "PGD"]})

    assert realisation["im"]["ims"] == ["PGA", "PGD"]
    assert realisation["im"]["valid_periods"] == [0.1]
    assert realisation["magnitudes"] == {"A": 7.1}


def test_apply_parameters_creates_an_absent_section() -> None:
    realisation: dict[str, object] = {}

    cr.apply_parameters(realisation, {"im.ims": ["PGA"]})

    assert realisation == {"im": {"ims": ["PGA"]}}


def test_resolve_parameters_rejects_a_decision_whose_source_moved(
    tmp_path: Path,
) -> None:
    # The decision recorded a fingerprint for a value the defaults no longer hold.
    decisions = {
        "im.ims": Decision(
            source="defaults",
            reason="adopt PGD",
            decided="2026-07-27",
            sha256=value_fingerprint(["THIS", "IS", "STALE"]),
        )
    }

    with pytest.raises(ValueError, match="moved since"):
        cr.resolve_parameters(
            decisions, DefaultsVersion.v24_2_2_1, Path("felipe_scripts"), None
        )


def test_resolve_parameters_requires_an_events_dir_for_deployed_decisions() -> None:
    decisions = {
        "im.ims": Decision(
            source="deployed", reason="pin", decided="2026-07-27", sha256="x"
        )
    }

    with pytest.raises(ValueError, match="--deployed-from"):
        cr.resolve_parameters(
            decisions, DefaultsVersion.v24_2_2_1, Path("felipe_scripts"), None
        )
```

Add to the imports at the top of `tests/test_complete_realisations.py`:

```python
import pytest

from workflow.defaults import DefaultsVersion
from workflow.scripts import complete_realisations as cr
from workflow.scripts.reconcile_parameters import Decision, value_fingerprint
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_complete_realisations.py -q -k "parameters"
```

Expected: FAIL — `AttributeError: module '...' has no attribute 'apply_parameters'`.

- [ ] **Step 3: Add resolution and application**

Add to the imports of `workflow/scripts/complete_realisations.py`:

```python
from workflow.scripts.reconcile_parameters import (
    Decision,
    flatten_sections,
    load_decisions,
    read_deployed_parameters,
    resolve_value,
    value_fingerprint,
)
```

This direction is safe: after Task 7b's fix, `reconcile_parameters` imports
nothing from this module at import time, so there is no cycle.

Add these two functions above `complete_one`:

```python
def resolve_parameters(
    decisions: dict[str, Decision],
    defaults_version: DefaultsVersion,
    felipe_scripts_dir: Path,
    events_dir: Path | None,
) -> dict[str, Any]:
    """Resolve each recorded decision to the value it selects.

    The decision file records a choice of source rather than a value, so the
    three candidates are rebuilt here exactly as the reconciler built them, and
    every resolved value is checked against the fingerprint its decision carries.

    Parameters
    ----------
    decisions : dict of str to Decision
        Recorded decisions, keyed by dotted path.
    defaults_version : DefaultsVersion
        Scientific defaults version supplying the ``defaults`` candidate.
    felipe_scripts_dir : Path
        Directory supplying the ``felipe`` candidate.
    events_dir : Path or None
        Directory supplying the ``deployed`` candidate. Required only when some
        decision names ``deployed``.

    Returns
    -------
    dict
        Maps dotted path to the decided value.

    Raises
    ------
    ValueError
        If a decision names ``deployed`` without an events directory, or if a
        resolved value no longer matches the fingerprint the decision recorded.
    """
    if any(decision.source == "deployed" for decision in decisions.values()) and (
        events_dir is None
    ):
        raise ValueError(
            "A decision names source 'deployed'; pass --deployed-from <events dir> "
            "so the deployed candidate can be read."
        )

    defaults = flatten_sections(load_defaults(defaults_version))
    overrides = load_overrides(felipe_scripts_dir)
    felipe = {
        "im.valid_periods": overrides.valid_periods.tolist(),
        "im.fas_frequencies": overrides.fas_frequencies.tolist(),
        "velocity_model.version": overrides.vm_version,
        "velocity_model.rrup_interpolants": overrides.rrup_interpolants.tolist(),
    }
    deployed: dict[str, Any] = {}
    if events_dir is not None:
        deployed, _ = read_deployed_parameters(events_dir)

    resolved: dict[str, Any] = {}
    for path, decision in decisions.items():
        candidates = {
            source: flat[path]
            for source, flat in (
                ("defaults", defaults), ("felipe", felipe), ("deployed", deployed)
            )
            if path in flat
        }
        value = resolve_value(decision, candidates)
        if value_fingerprint(value) != decision.sha256:
            raise ValueError(
                f"Decision for {path} names source '{decision.source}', but that "
                f"source has moved since the decision was recorded "
                f"({decision.decided}). Re-run reconcile-parameters."
            )
        resolved[path] = value
    return resolved


def apply_parameters(realisation: dict[str, Any], resolved: dict[str, Any]) -> None:
    """Write decided parameter values into a realisation, in place.

    Only the decided keys are touched; every other key of the same section is
    left as the defaults and overrides produced it.

    Parameters
    ----------
    realisation : dict
        The realisation to modify.
    resolved : dict
        Maps dotted path to the decided value.
    """
    for path, value in resolved.items():
        section, _, key = path.partition(".")
        realisation.setdefault(section, {})[key] = value
```

- [ ] **Step 4: Apply them in `complete_one`**

Give `complete_one` a `resolved` parameter, defaulting to `None`, and apply it in the normalisation step. Change the signature:

```python
def complete_one(
    src: Path,
    dst: Path,
    defaults_version: DefaultsVersion,
    overrides: Overrides,
    resolved: dict[str, Any] | None = None,
) -> None:
```

Add to its docstring's `Parameters` section:

```
    resolved : dict, optional
        Decided parameter values to apply last, overriding the defaults and the
        override files. Omit to keep the pre-existing behaviour.
```

And change step 6 of its body from:

```python
    # 6. Normalise key order for clean diffing against the reference.
    with open(dst, encoding="utf-8") as handle:
        realisation = json.load(handle)
    realisation = normalize_key_order(realisation)
```

to:

```python
    # 6. Apply the campaign's recorded parameter decisions, then normalise key
    #    order for clean diffing against the reference. Decisions are applied
    #    last so they win over both the defaults and the override files.
    with open(dst, encoding="utf-8") as handle:
        realisation = json.load(handle)
    if resolved:
        apply_parameters(realisation, resolved)
    realisation = normalize_key_order(realisation)
```

Thread it through `_complete_worker` by widening its argument tuple to
`(src, dst, defaults_version, overrides, resolved)` and unpacking accordingly:

```python
    src, dst, defaults_version, overrides, resolved = args
    rupture_id = _rupture_id_from_path(src)
    try:
        complete_one(src, dst, defaults_version, overrides, resolved)
```

Update its type annotation to
`args: tuple[Path, Path, DefaultsVersion, Overrides, dict[str, Any] | None]`.

- [ ] **Step 5: Add the CLI options**

Add to `complete_realisations`'s signature, after `vm_version`:

```python
    parameters: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Campaign decision file recording which parameters to adopt.",
        ),
    ] = None,
    deployed_from: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Events directory supplying the 'deployed' candidate, needed "
            "only when a decision names it.",
        ),
    ] = None,
```

Add the matching docstring entries:

```
    parameters : Path, optional
        Campaign decision file recording which parameters to adopt.
    deployed_from : Path, optional
        Events directory supplying the 'deployed' candidate.
```

Resolve once, immediately after `overrides = load_overrides(felipe_scripts_dir, vm_version)`:

```python
    resolved: dict[str, Any] = {}
    if parameters is not None:
        resolved = resolve_parameters(
            load_decisions(parameters), defaults_version, felipe_scripts_dir,
            deployed_from,
        )
        print(f"Applying {len(resolved)} recorded parameter decision(s).")
```

and widen the work tuples:

```python
    work = [
        (src, output_dir / src.name, defaults_version, overrides, resolved)
        for src in valid_files
    ]
```

- [ ] **Step 6: Run the tests and the gates**

```bash
uv run pytest tests/test_complete_realisations.py -q
uv run ruff check
uv run ty check --exclude workflow/schemas.py --exclude setup.py
```

Expected: all pass, `All checks passed!` from both.

- [ ] **Step 7: Commit**

```bash
git add workflow/scripts/complete_realisations.py tests/test_complete_realisations.py
git commit -m "feat(scripts): apply recorded parameter decisions in complete-realisations

--parameters resolves each recorded decision against the same three candidates
the reconciler compared, and applies the result last, so a decision wins over
both the scientific defaults and the override files.

The decision file records a choice of source rather than a value, so every
resolved value is checked against the fingerprint its decision carries. If the
chosen source has moved since the decision was made, completion fails and says
to re-run reconcile-parameters, rather than quietly materialising a value nobody
approved."
```

---

## Task 7e: `complete-realisations --deploy-dir` — deploy behind the same gate

Deploying from the completer saves a step in the common case, but it must not become a way to skip verification. So the gate is identical to Task 4's, sharing the same helper: `--deploy-dir` alone creates new event directories and **refuses** to replace anything.

**Files:**
- Modify: `workflow/scripts/complete_realisations.py`
- Test: `tests/test_complete_realisations.py`

**Interfaces:**
- Consumes: `copy_realisations_to_event_dirs.copy_realisations` from Task 4
- Produces: CLI options `--deploy-dir <events dir>` and `--overwrite-existing`, both off by default

- [ ] **Step 1: Write the failing test**

Append to `tests/test_complete_realisations.py`:

```python
def test_complete_realisations_refuses_to_deploy_over_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Deployment shares Task 4's helper, so the gate is proved once, here, at
    # the level the campaign actually invokes it.
    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "realisation_100932.json").write_text('{"new": 1}', encoding="utf-8")
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    copied, _, refused = copy_realisations(complete, events)

    assert copied == 0
    assert refused == ["100932"]
    assert (events / "100932" / "realisation.json").read_text() == '{"old": 1}'
```

Add to the imports:

```python
from workflow.scripts.copy_realisations_to_event_dirs import copy_realisations
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_complete_realisations.py -q -k deploy
```

Expected: FAIL — `ImportError` until Task 4 has landed. If Task 4 is already done, this test passes immediately; that is fine, it is a regression guard on shared behaviour.

- [ ] **Step 3: Add the deploy options**

Add to `complete_realisations`'s signature, after `deployed_from`:

```python
    deploy_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            help="Events directory to deploy completed realisations into. "
            "Off by default; nothing outside OUTPUT_DIR is written without it.",
        ),
    ] = None,
    overwrite_existing: Annotated[
        bool,
        typer.Option(
            help="Replace realisations that already exist in --deploy-dir. "
            "Without this, existing files are left untouched."
        ),
    ] = False,
```

Add the matching docstring entries:

```
    deploy_dir : Path, optional
        Events directory to deploy completed realisations into.
    overwrite_existing : bool
        Whether to replace realisations that already exist in ``deploy_dir``.
```

Import the helper at the top of the module:

```python
from workflow.scripts.copy_realisations_to_event_dirs import copy_realisations
```

And add the deployment step at the very end of the command body, after the existing summary prints:

```python
    if deploy_dir is not None:
        if failed or broken_ids:
            print(
                "\nRefusing to deploy: the run had failures. Fix them and re-run."
            )
            raise typer.Exit(code=1)
        copied, _, refused = copy_realisations(
            output_dir, deploy_dir, overwrite_existing
        )
        print(f"\nDeployed {copied} realisation(s) into {deploy_dir}")
        if refused:
            print(
                f"Refused to replace {len(refused)} existing realisation(s). "
                f"Re-run with --overwrite-existing to replace them."
            )
            raise typer.Exit(code=1)
```

Refusing to deploy a run that had failures is deliberate: a partial set silently deployed over a complete one is the worst outcome available here.

- [ ] **Step 4: Run the tests and the gates**

```bash
uv run pytest tests/test_complete_realisations.py -q
uv run ruff check
uv run ty check --exclude workflow/schemas.py --exclude setup.py
fdfind . workflow/ -E "__init__.py" --extension py | xargs numpydoc lint
```

Expected: all pass, `All checks passed!`, no numpydoc output.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/complete_realisations.py tests/test_complete_realisations.py
git commit -m "feat(scripts): add guarded --deploy-dir to complete-realisations

Deploying from the completer saves a step, but must not become a way to skip
verification, so it reuses copy-realisations-to-event-dirs' helper and gate
verbatim: --deploy-dir alone creates new event directories and refuses to
replace anything, and --overwrite-existing is required to replace.

A run with failures or broken stubs refuses to deploy at all. Silently laying a
partial set over a complete one is the worst outcome available here."
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

This task tests a claim rather than making one: `nshmdb.db` has no recorded origin, only a reconstruction from the NSHM2022DB reflog. Rebuilding it from the CRU zip and comparing is what turns that reconstruction into evidence.

Two corrections to the paths this task was originally written with:

- The CRU solution zip is **not** in `/home/arr65/data/`, which holds nothing relevant. It is git-tracked at `/home/arr65/src/NSHM2022DB/tests/CRU_fault_system_solution.zip` (69 MB) — better for provenance than a loose file, because the input is version-controlled and identified by a commit. Do not confuse it with `CRU_fault_system_solution_small.zip` (3.6 KB), which is a test fixture.
- `nshmdb.db` went missing from this machine and was restored on 2026-07-27. It is byte-identical to the database the deployed set came from (sha256 `00e2564806…`), so the comparison below is meaningful. **Confirm that checksum first** — Step 1a — because the file is gitignored and nothing else guarantees which database is sitting there.

**Files:**
- Create: `/home/arr65/data/cs_nshm_2022/nshmdb_rebuilt_20260714.db` (create the directory first)
- Possibly modify: `/home/arr65/src/workflow/nshmdb.db`

- [ ] **Step 1a: Confirm the database present is the database of record**

```bash
sha256sum nshmdb.db
```

Expected: `00e256480618cd15e11fbf744037d037bf3fc2d523fb977ee30e0b84a640bc57`.

If it differs, **stop**. A different database means the deployed realisations' rupture ids may not mean what this campaign assumes, and Step 3's comparison would be testing the wrong thing.

- [ ] **Step 1: Confirm NSHM2022DB is where the reflog says it is**

```bash
git -C /home/arr65/src/NSHM2022DB rev-parse HEAD
git -C /home/arr65/src/NSHM2022DB status --porcelain --untracked-files=no
```

Expected: `95a005a...` and empty output. If HEAD is not `95a005a`, the reconstruction in the spec is void — **stop and report**.

- [ ] **Step 2: Rebuild the database**

This takes a couple of minutes and writes about 1 GB.

```bash
mkdir -p /home/arr65/data/cs_nshm_2022
cd /home/arr65/src/NSHM2022DB
uv run nshm_db_generator \
    /home/arr65/src/NSHM2022DB/tests/CRU_fault_system_solution.zip \
    /home/arr65/data/cs_nshm_2022/nshmdb_rebuilt_20260714.db
cd /home/arr65/src/workflow
```

Record the input's provenance while you are here — it is git-tracked, so a commit identifies it exactly:

```bash
git -C /home/arr65/src/NSHM2022DB log -1 --format='%H' -- tests/CRU_fault_system_solution.zip
sha256sum /home/arr65/src/NSHM2022DB/tests/CRU_fault_system_solution.zip
```

Both values go into `PROVENANCE.md` in Task 13.

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

This value goes into `PROVENANCE.md` in Task 13. If Step 4 took the matching branch it will still be `00e256480618cd15e11fbf744037d037bf3fc2d523fb977ee30e0b84a640bc57`, unchanged from Step 1a.

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

## Task 10a: Reconcile parameters against `pegasus` and commit the decisions

This is where a human decides what the regenerated set will actually contain. It must run **before** the pilot and the campaign, and **before** Task 14 overwrites the deployed files — the deployed values are one of the three candidates, so once they are replaced the comparison is gone.

As of 2026-07-27 exactly two conflicts are expected. Anything else is new since this plan was written and deserves scrutiny rather than a reflex answer.

- [ ] **Step 1: Confirm the deployed set is the untouched source of truth**

```bash
git -C /home/arr65/src/cs_nshm_2022 rev-parse --abbrev-ref HEAD    # expect: main
git -C /home/arr65/src/cs_nshm_2022 status --porcelain -- cs_nshm_2022/events    # expect: empty
ls -d /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events/*/ | wc -l  # expect: 291
```

If the events tree is not clean, **stop** — decisions must be taken against the committed originals, not against edited files.

- [ ] **Step 2: See the conflicts without deciding anything**

```bash
uv run reconcile-parameters \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/campaign_parameters.yaml \
    --non-interactive
```

Expected on a first run: exit 1, listing exactly

```
im.ims
im.fas_frequencies
```

`--non-interactive` never writes, so this is a safe look before committing to anything. If the deployed set is reported as inconsistent — a parameter differing *between* events — **stop and understand why** before deciding: that means a previous deployment ran only partway.

- [ ] **Step 3: Decide each conflict**

```bash
uv run reconcile-parameters \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/campaign_parameters.yaml
```

The two expected conflicts, and the reasoning recorded with the 2026-07-27 design:

| conflict | candidates | decision | reason to record |
| --- | --- | --- | --- |
| `im.ims` | defaults 9 (with `PGD`); deployed 8 | **defaults** | `PGD` was added to the scientific defaults by `ec2fb25`; adopting it is the point of this amendment. `union` resolves to the same 9 values, since defaults are a strict superset. |
| `im.fas_frequencies` | defaults 100 pts (0.1–100 Hz); felipe 389 pts (0.0132–100 Hz) | **felipe** | The campaign's grid is the richer one — finer spacing, and it extends a decade lower. The tool will also note that felipe and deployed agree within tolerance; that is float-path noise, not a third option. |

`im.valid_periods` will **not** be raised: defaults, felipe and deployed are identical, `ec2fb25` having absorbed felipe's list. That is worth noticing — the override is now a no-op, which is recorded as deferred cleanup in the design.

Write real reasons. The `reason` field is what makes this file worth more than the diff it produces.

- [ ] **Step 4: Read back what you recorded**

```bash
cat /home/arr65/src/cs_nshm_2022/cs_nshm_2022/campaign_parameters.yaml
uv run reconcile-parameters \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/campaign_parameters.yaml \
    --non-interactive
```

Expected on the second invocation: **exit 0**, reporting `2 conflict(s): 2 already settled, 0 decided now`. That is the fingerprint mechanism working — the decisions are durable, and a later run stays silent unless a source moves.

- [ ] **Step 5: Commit the decisions to `cs_nshm_2022`**

```bash
cd /home/arr65/src/cs_nshm_2022
git add cs_nshm_2022/campaign_parameters.yaml
git commit -m "feat: record the campaign's parameter decisions

Which parameters this realisation set takes from the pegasus scientific
defaults, which from the campaign's own override files, and why. Produced by
reconcile-parameters against the deployed set.

Two decisions: adopt PGD in im.ims from the defaults, and keep the campaign's
389-point fas_frequencies grid over the defaults' 100-point one. Each entry
carries a fingerprint of the value it resolved to, so a later pegasus merge
re-prompts only for parameters whose chosen source actually moved."
cd /home/arr65/src/workflow
```

- [ ] **Step 6: Confirm the `workflow` tree is still clean**

```bash
git status --porcelain --untracked-files=no    # expect: empty
```

Reconciling reads `workflow` and writes into `cs_nshm_2022`; it must not have touched a tracked file here. If this is not empty, **stop and investigate** — a dirty tree changes the version stamp every realisation is about to record.

---

## Task 10b: Pilot — prove the regeneration before the full run

This is the load-bearing gate. Inheriting the seeds reproduces the content only if commit N's code derives the same realisation from them as the ad-hoc code did — and the ad-hoc baker (`bake_realisations.py`) no longer exists, `complete-realisations` is its successor, and the pegasus merge brought magnitude-convention (BoldM) changes. Prove it on a handful of events before committing to 291.

- [ ] **Step 1: Build a small pilot rupture list**

Include at least one multi-fault event. `149379` is multi-fault; the other three round out the sample.

```bash
PILOT=/tmp/claude-1000/-home-arr65-src-workflow/b518f6e9-16db-4ae1-a09d-e4bf7d6e1754/scratchpad/pilot
mkdir -p "$PILOT"
printf 'chosen_nshm_id\n149379\n100932\n101084\n101091\n' > "$PILOT/pilot.csv"
EVENTS=/home/arr65/src/cs_nshm_2022/cs_nshm_2022/events
PARAMS=/home/arr65/src/cs_nshm_2022/cs_nshm_2022/campaign_parameters.yaml
```

- [ ] **Step 2: Regenerate them, inheriting the deployed seeds**

```bash
uv run generate-realisations-from-csv \
    nshmdb.db "$PILOT/pilot.csv" "$PILOT/minimal" 24.2.2.1 \
    --inherit-seeds-from "$EVENTS"
uv run complete-realisations "$PILOT/minimal" "$PILOT/complete" \
    --defaults-version 24.2.2.1 --vm-version 2.09 \
    --parameters "$PARAMS"
```

Expected: `Inherited seeds for 4 of 4 rupture(s); 0 drew fresh seeds.`, then 4 complete realisations, `Applying 2 recorded parameter decision(s).`, and no errors.

**If any rupture drew fresh seeds, stop.** The whole point is that these four replay the seeds already in the deployed files; a fresh draw silently produces a plausible realisation inconsistent with the SRFs built from the original.

- [ ] **Step 3: Verify the pilot changed only where it was meant to**

```bash
uv run verify-realisation-content "$EVENTS" "$PILOT/complete" --parameters "$PARAMS"
```

Expected: `Compared 4 realisation(s) against 2 recorded decision(s); 0 failed`.

- [ ] **Step 4: Gate on the result**

If **0 failed**, commit N reproduces the originals from their inherited seeds and applies the decisions exactly; proceed to Task 11.

If **any** event fails, **stop**. The command distinguishes the two cases and both matter:

- `UNEXPECTED` — regeneration changed something nobody decided. Likely causes, in order: `complete-realisations` differing from the vanished `bake_realisations.py`; the BoldM magnitude-convention changes from the pegasus merge; or the area-weighted fault selection (`9f35c90`). A `sources` or `rupture_propagation` difference would additionally implicate the database, so re-check Task 9 Step 1a's checksum before chasing the code.
- `UNAPPLIED` — a recorded decision did not reach the file. That is a bug in Task 7d's resolution or application, not a scientific finding.

Any code fix moves commit N, so re-run Task 8 (gates + pin) and Task 10 (force the stamp) afterwards. Do **not** run the full campaign against a failing pilot.

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

These are gitignored, so this touches nothing tracked. The old outputs are still preserved in the `cs_nshm_2022` repo's git history, and Task 14 replaces the working copies.

- [ ] **Step 2: Generate the minimal stubs**

```bash
uv run generate-realisations-from-csv \
    nshmdb.db \
    annealed_minimal_ruptures.csv \
    minimal_realisations \
    24.2.2.1 \
    --inherit-seeds-from /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events
```

`--inherit-seeds-from` makes each stub replay the seeds already recorded in its deployed realisation, so the content reproduces the original rather than drawing a fresh set. The two excluded ruptures have no deployed file and fall back to a fresh draw, which is moot — they fail before any seed is used.

Expected: `Done. Processed 293 rupture ID(s).`, with two failures printed — ruptures **59421** and **95011**, both `ValueError: The graph must be connected to find a spanning tree`. **These two failures are a pass condition.**

Then, critically:

```
Inherited seeds for 291 of 293 rupture(s); 2 drew fresh seeds.
```

**291 and 2, exactly.** Any other split means some event silently drew fresh seeds and will not reproduce its original — **stop** and find out which, rather than discovering it in Task 11a.

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
    --vm-version 2.09 \
    --parameters /home/arr65/src/cs_nshm_2022/cs_nshm_2022/campaign_parameters.yaml
```

Expected: `Applying 2 recorded parameter decision(s).`, then `Completed 291 realisation(s)`, no skips (there are no broken stubs left to skip), no failures.

Note that **no `--deploy-dir` is given**. Deployment is Task 14, after verification — this run must not touch `cs_nshm_2022`.

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

## Task 11a: Verify the full set changed only where it was meant to

Task 11 Step 4 proved every file records commit N. This proves every file reproduces the *content* it replaces, save the decided parameters. Run it now, while the deployed files in `cs_nshm_2022` are still the pre-campaign originals — Task 14 overwrites them, and with them the only thing left to compare against.

- [ ] **Step 1: Compare the whole set against the deployed originals**

```bash
uv run verify-realisation-content \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events \
    complete_realisations \
    --parameters /home/arr65/src/cs_nshm_2022/cs_nshm_2022/campaign_parameters.yaml
```

Expected: `Compared 291 realisation(s) against 2 recorded decision(s); 0 failed`.

- [ ] **Step 2: Spot-check that the intended change actually happened**

A clean verification proves nothing changed that should not have. Confirm, independently, that the thing this amendment exists for *did*:

```bash
python3 -c "
import json
old = json.load(open('/home/arr65/src/cs_nshm_2022/cs_nshm_2022/events/149379/realisation.json'))
new = json.load(open('complete_realisations/realisation_149379.json'))
print('deployed ims :', old['im']['ims'])
print('regenerated  :', new['im']['ims'])
print('PGD adopted  :', 'PGD' in new['im']['ims'] and 'PGD' not in old['im']['ims'])
"
```

Expected: `PGD adopted  : True`. If it prints `False`, the decision did not take and `verify-realisation-content` should have caught it as `UNAPPLIED` — investigate both before continuing.

- [ ] **Step 3: Gate on the result**

If **0 failed** and the spot-check passes, the regenerated set differs from the originals only in `log_trail` and the two decided parameters; proceed to tag and distribute.

If **any** event fails, **stop** — do not tag, record, or distribute a set that changed in ways nobody chose. The pilot (Task 10b) should have caught this; a failure surfacing only at full scale points to an event outside the pilot sample. Investigate the printed fields and reconcile exactly as in Task 10b Step 4. A fix moves commit N, so re-run Tasks 8, 10, 10b and 11.

- [ ] **Step 4: Confirm the tree is still clean**

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

- 291 files, at `cs_nshm_2022/cs_nshm_2022/events/<rupture id>/realisation.json`
- Per-file sha256: `manifest.csv`, alongside this file.
- Identical to the 2026-07-09 set except `log_trail` and the parameters recorded in `campaign_parameters.yaml`, verified in both directions by `verify-realisation-content`; seeds inherited from the set it replaces.
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
| `nshmdb.db` | `00e256480618cd15e11fbf744037d037bf3fc2d523fb977ee30e0b84a640bc57` | Built by `nshm_db_generator.py` at NSHM2022DB `«NSHM2022DB sha»` from the CRU zip below. Confirm this checksum still matches before quoting it — the file is gitignored. |
| `CRU_fault_system_solution.zip` | `«sha256»` | The NSHM 2022 crustal fault system solution, from Jake Faulkner. Git-tracked in NSHM2022DB at `tests/`, last touched by `«commit»`. Its own upstream release identifier is not recorded. |
| `annealed_minimal_ruptures.csv` | `«sha256»` | 293 rupture ids, provided by Jake Faulkner as the first sample of ruptures to simulate for this campaign. The selection procedure implied by "annealed" is not documented. |
| `campaign_parameters.yaml` | `«sha256»` | Which parameters this set takes from the `pegasus` scientific defaults and which from the campaign's override files, with a reason and a value fingerprint per decision. Produced by `reconcile-parameters`; committed in `cs_nshm_2022`. |
| the 2026-07-09 realisation set | — | The source of this set's seeds, inherited per event via `--inherit-seeds-from`. Superseded by this set; recoverable from `cs_nshm_2022` history at `«pre-campaign commit»`. |
| `uv.lock` | `«sha256»` | Pins `nshmdb` 2025.12.1, `source_modelling` 2026.6.2, `velocity-modelling` 2026.2.1, `qcore-utils` 2025.12.2, `im-calculation` 2025.12.5, `oq-wrapper` 2025.12.5. |

### The database rebuild

`nshmdb.db` had no recorded origin. It was reconstructed from evidence — the
NSHM2022DB reflog shows a single clone at `95a005a` on 2026-07-08 11:42:46, the
database was written at 11:44:03, its file mtime still reads 2026-07-08 11:44,
and its schema matches the repo's only generator — and then **tested by
rebuilding it**:

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

### Where the parameters came from

Every parameter section is either taken from the scientific defaults at the
commit above, or overridden by this campaign. Where the two disagreed, the
choice was made deliberately and recorded in `campaign_parameters.yaml`:

| parameter | source | why |
| --- | --- | --- |
| `im.ims` | defaults | Adopts `PGD`, added to the defaults by `ec2fb25`. This set previously lacked it. |
| `im.fas_frequencies` | campaign override | 389 points from 0.0132 Hz, against the defaults' 100 from 0.1 Hz — finer, and a decade wider. |
| `im.valid_periods` | defaults | Not a conflict: `ec2fb25` absorbed the campaign's list into the defaults verbatim, so the override is now a no-op. |
| `velocity_model.version`, `velocity_model.rrup_interpolants` | campaign override | Campaign-specific, no competing default. |
| `emod3d`, `resolution`, `srf`, `velocity_model_1d`, `hf_velocity_model_1d`, `hf`, `bb`, `rupture_velocity` | defaults | Verified equal to the defaults across all 291 files; no override, no conflict. |

The campaign's override values themselves come from `felipe_scripts/`
(`Mw_rrup_mod.txt`, `periods.csv`, `frequencies.csv`) in `ucgmsim/workflow` at
the commit above.

One numerical note, recorded because it is real and small rather than hidden
because it is small: the previous set's `im.fas_frequencies` differed from
`felipe_scripts/frequencies.csv` in 165 of 389 values, by at most 57 ULP
(≤ 6.7e-15 relative) — the same intended log-spaced grid produced by a different
floating-point path. This set carries the override file's values exactly.

## Commands

Run from the repository root, on a clean tree at the commit above, after
`uv sync --reinstall-package workflow --all-extras --dev`:

```
uv run verify-realisation-provenance --preflight
uv run reconcile-parameters «cs_nshm_2022»/cs_nshm_2022/events «cs_nshm_2022»/cs_nshm_2022/campaign_parameters.yaml --non-interactive
uv run generate-realisations-from-csv nshmdb.db annealed_minimal_ruptures.csv minimal_realisations 24.2.2.1 --inherit-seeds-from «cs_nshm_2022»/cs_nshm_2022/events
uv run complete-realisations minimal_realisations complete_realisations --defaults-version 24.2.2.1 --vm-version 2.09 --parameters «cs_nshm_2022»/cs_nshm_2022/campaign_parameters.yaml
uv run verify-realisation-provenance complete_realisations
uv run verify-realisation-content «cs_nshm_2022»/cs_nshm_2022/events complete_realisations --parameters «cs_nshm_2022»/cs_nshm_2022/campaign_parameters.yaml
uv run copy-realisations-to-event-dirs complete_realisations «cs_nshm_2022»/cs_nshm_2022/events --overwrite-existing
```

Note that the first four commands read the **pre-campaign** events directory: it supplies both the seeds and the deployed comparison values. Re-running this sequence against the *deployed* set reproduces it exactly, because the seeds are then read back from the files this campaign wrote and the decisions are already settled — but the `verify-realisation-content` step then compares the set against itself and proves nothing. To genuinely re-verify, compare against `«pre-campaign commit»` from `cs_nshm_2022` history.

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
**inherited** from the previous set, not re-drawn. They were originally drawn
from OS entropy by `Seeds.random_seeds()`; they carry no intrinsic meaning, and
were reproduced verbatim so the source geometry of this set is identical to the
one the existing SRFs and downstream results were built from.

There is no separate seed manifest. Each event's seeds were read straight from
the `seeds` block of the file it replaces, via
`generate-realisations-from-csv --inherit-seeds-from`, and only that block was
read — every other field was derived fresh. The seeds therefore live in exactly
one place: the realisations themselves, which are self-describing and cannot
drift out of sync with a second copy.

The consequence for reproducibility is worth stating plainly. Re-running the
pipeline against **this** set reproduces it, because the seeds are read back out
of it. Re-deriving it from the set it replaced requires that set, recoverable
from `cs_nshm_2022` history at `«pre-campaign commit»`. Every file was verified
against that predecessor — differing only in `log_trail` and the parameters in
`campaign_parameters.yaml`, checked in both directions — by
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

## Task 14: Distribute to `cs_nshm_2022`

- [ ] **Step 1: Confirm the destination**

```bash
git -C /home/arr65/src/cs_nshm_2022 rev-parse --abbrev-ref HEAD
git -C /home/arr65/src/cs_nshm_2022 status --porcelain
ls -d /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events/*/ | wc -l
```

Expected: branch `main`, clean, 291 existing event directories. If the tree is not clean, **stop** — do not overwrite uncommitted work.

- [ ] **Step 2: Confirm the gate actually refuses, then distribute**

Run it once **without** the overwrite flag. This is not a formality: it proves the safety gate is live before you disable it, and it is the last cheap moment to notice you are pointed at the wrong directory.

```bash
uv run copy-realisations-to-event-dirs \
    complete_realisations \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events
```

Expected: `Copied 0 realisation(s)`, then `Refused to replace 291 existing realisation(s)`, exit 1. All 291 already exist, so all 291 are refused and **nothing is written**.

If it reports copying anything, the destination is not the deployed set — **stop**.

Now deploy for real:

```bash
uv run copy-realisations-to-event-dirs \
    complete_realisations \
    /home/arr65/src/cs_nshm_2022/cs_nshm_2022/events \
    --overwrite-existing
```

Expected: `Copied 291 realisation(s) into ...`, no skips, no refusals. `completion_summary.csv` and `error_log.txt` are not JSON, so they are not copied.

This is the point of no return: the originals this campaign was verified against are now gone from the working tree. They remain in `cs_nshm_2022` history — record that commit as `«pre-campaign commit»` in `PROVENANCE.md`, since Task 13's reproducibility note depends on it:

```bash
git -C /home/arr65/src/cs_nshm_2022 rev-parse HEAD
```

Take this **before** committing Step 6, so it names the last commit holding the previous set.

- [ ] **Step 3: Verify what landed**

```bash
cd /home/arr65/src/cs_nshm_2022
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
   /home/arr65/src/cs_nshm_2022/cs_nshm_2022/PROVENANCE.md
cp /home/arr65/src/workflow/docs/campaigns/2026-07-14-nshm2022-realisations/manifest.csv \
   /home/arr65/src/cs_nshm_2022/cs_nshm_2022/manifest.csv
```

- [ ] **Step 5: Verify the manifest describes the files that actually landed**

A manifest is worth nothing unless it describes these files.

```bash
uv --directory /home/arr65/src/workflow run python - <<'PY'
import csv
import hashlib
from pathlib import Path

root = Path("/home/arr65/src/cs_nshm_2022/cs_nshm_2022")
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
cd /home/arr65/src/cs_nshm_2022
git add cs_nshm_2022/events cs_nshm_2022/PROVENANCE.md cs_nshm_2022/manifest.csv
git commit -m "feat: regenerate all 291 realisations with verified provenance

Replaces the 2026-07-09 batch, whose log_trail recorded a stale build stamp
naming a commit that did not contain the code that ran, an old script name, and
paths that no longer existed.

Every file now records commit «commit N» of ucgmsim/workflow, asserted clean
before the run and verified across all 291 afterwards. Inputs are pinned by
checksum in PROVENANCE.md; per-file hashes are in manifest.csv.

Seeds were inherited from the files this batch replaces, so every source-derived
field reproduces exactly: verify-realisation-content confirms each file differs
only in log_trail and in the parameters recorded in campaign_parameters.yaml.

Those parameter changes are deliberate. im.ims gains PGD, adopted from the
pegasus scientific defaults (ec2fb25). im is read only by im-calc and has no
bearing on source geometry, so the SRFs and slip animations already built from
the previous batch remain valid."
```

- [ ] **Step 7: Confirm what stays valid, and what is genuinely separate**

Because the seeds were inherited and every source-derived field is verified
identical (Task 11a), **nothing downstream is invalidated**. The 54 GB of SRFs,
the slip animations, and the derived scratch all correspond to these realisations
exactly as before. No SRF regeneration is required, and none is triggered by this
plan.

This holds *despite* the set no longer being byte-identical to its predecessor,
and the reason is specific rather than general: the only fields that changed are
in `im`, and `IntensityMeasureCalculationParameters` is read solely by
`workflow/scripts/im_calc.py`. Neither `realisation_to_srf` nor `generate_stoch`
touches it. Had a decision changed `srf`, `velocity_model` or either 1D velocity
model, this paragraph would not hold and the affected artefacts would need
rebuilding — so check which parameters actually changed before reusing this
reasoning on a later run.

Intensity measures **not** yet computed from this set will now include `PGD`; any
that were computed under the previous batch predate the decision and should be
recomputed if that measure is wanted.

The SRF version-mislabelling is a **separate** matter, unaffected either way. The
local SRFs under `/home/arr65/data/cs_nshm_2022` were already repaired in place by
`scripts/fix_srf_version.py` (a one-line version-header rewrite, no regeneration);
the BSC and Dropbox copies are still mislabelled and still need that script — keep
it. The audit baselines `srf_version_audit_BEFORE_20260714.csv` and
`mislabelled_multi_fault_srfs_BEFORE_20260714.txt`
(`cs_nshm_2022@a623667`) record the pre-fix state (**219 mislabelled, all
multi-fault; 72 correct, all single-fault** — the signature of `stitch_srf_files`
hardcoding `version="1.0"`) and remain the reference for that independent fix.

When you push `cs-nshm2022-prep` or open a PR, flag the good news explicitly: the
realisation provenance is now sound and the downstream artefacts are preserved,
not invalidated.

---

## Deferred: separating the generic tooling for a pegasus PR

Not part of this campaign, and captured here only so it is not lost. The generic tools this branch adds — `complete-realisations` (with `--parameters` and the guarded `--deploy-dir`), `generate-realisations-from-csv` (with `--inherit-seeds-from`), `reconcile-parameters`, `verify-realisation-content`, `verify-realisation-provenance`, `compare-nshmdb`, `copy-realisations-to-event-dirs`, and the area-weighted fault-selection fix `9f35c90` — are reusable and belong on `pegasus`. Nothing in them hard-codes this campaign: every path, defaults version and decision file is an argument. When upstreaming later:

- Cherry-pick only the generic commits. Leave behind the campaign data and personal inputs: `felipe_scripts/` (Felipe's reference inputs — needs his sign-off), `annealed_minimal_ruptures.csv`, the campaign docs, and `copy_realisations_to_event_dirs.py` / `render_all.sh` if they are not promoted into the package.
- `campaign_parameters.yaml` stays in `cs_nshm_2022` — it is campaign data, not tooling.
- The now-redundant `im.valid_periods` override in `felipe_scripts/periods.csv` can be retired once the decision recording it as `source: defaults` has landed; `ec2fb25` absorbed felipe's list into the defaults verbatim, so the override changes nothing. Removing `felipe_scripts/` needs team sign-off, which is why it is deferred rather than done here.

This does not affect commit N or the campaign: the version stamp is derived from git state, not file inventory, and commit N is pinned and tagged regardless of any later separation.

---

## Done

At completion:

- 291 realisations, every one recording commit N with a stamp asserted clean before the run and verified after.
- Identical to the previous set in every field but `log_trail` and the decided parameters — proved on a pilot before the run and across all 291 after, **in both directions** (`verify-realisation-content`): nothing changed that was not decided, and nothing decided failed to apply.
- The set carries `PGD`, adopted deliberately from the `pegasus` defaults rather than acquired by accident.
- The SRFs and slip animations already built from the previous set stay valid, because every source-derived field reproduces exactly and the only changed parameters live in `im`, which feeds `im-calc` alone.
- `campaign_parameters.yaml` committed in `cs_nshm_2022`, recording every decision with a reason, a date and a value fingerprint — so the next `pegasus` merge asks only about what actually moved.
- Seeds inherited from the files they replace, with the inherited/fresh split asserted at 291/2 rather than assumed.
- `nshmdb.db` with a derivation that was tested, not assumed — rebuilt from a git-tracked input and compared against the database of record.
- Every CI gate green — including the two this branch had broken.
- Committed checkers anyone can re-run to confirm all of the above.
- `PROVENANCE.md` pinning every input by checksum, and stating honestly what is *not* known: the CRU zip's upstream release, and what "annealed" means.
