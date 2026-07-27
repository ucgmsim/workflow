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
