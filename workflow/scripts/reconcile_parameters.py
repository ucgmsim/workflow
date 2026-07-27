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

import collections
import dataclasses
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, cast

import click
import typer
import yaml
from rich.console import Console
from rich.table import Table

from workflow.defaults import DefaultsVersion, load_defaults

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
        # ty materializes a bare isinstance-narrowed dict's key type as `Never`
        # in the contravariant `__getitem__` position; cast to a concrete
        # parameterization so indexing type-checks. Runtime behaviour is
        # unaffected -- cast performs no check or conversion.
        left_map = cast(dict[str, object], left)
        right_map = cast(dict[str, object], right)
        return left_map.keys() == right_map.keys() and all(
            values_equivalent(left_map[key], right_map[key], tolerance)
            for key in left_map
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


def resolve_value(source: str, candidates: dict[str, Any]) -> Any:
    """Return the value a source name selects from the available candidates.

    Takes the source name rather than a ``Decision`` so that callers which have
    only a name -- the interactive prompt, computing a union preview -- need not
    build a throwaway decision to reach it.

    Parameters
    ----------
    source : str
        ``defaults``, ``felipe``, ``deployed`` or ``union``.
    candidates : dict
        Maps source name to proposed value.

    Returns
    -------
    object
        The selected value.

    Raises
    ------
    ValueError
        If the name is not ``union`` and no candidate carries it.
    """
    if source == "union":
        merged: list[Any] = []
        for candidate_source in sorted(candidates):
            for item in candidates[candidate_source]:
                if item not in merged:
                    merged.append(item)
        return merged
    if source not in candidates:
        raise ValueError(
            f"Source {source!r} has no value here. Available: {sorted(candidates)}."
        )
    return candidates[source]


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
        resolved = resolve_value(decision.source, candidates)
        return value_fingerprint(resolved) == decision.sha256
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
            describe(resolve_value("union", conflict.candidates)),
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
        sha256=value_fingerprint(resolve_value(source, conflict.candidates)),
    )


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
