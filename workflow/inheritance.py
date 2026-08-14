#!/usr/bin/env python3
"""Decide, per section, between a carried-over value and a freshly derived one.

Description
-----------
Carrying a section over from a deployed realisation keeps a regenerated set
byte-compatible with the products already built from it. Done blindly it also
hides every change in the code that would have produced that section, which is
the opposite of what a provenance campaign is for.

So the carried value is checked against the one the pipeline derived anyway. If
they agree to within tolerance the difference is float noise and the deployed
value is taken silently -- keeping the set bit-identical without anyone having to
look. If they disagree by more than that, something real changed, and the run
refuses until a human records which value to use and why.

The tolerances are per section because the quantities differ in kind.
``sources``, ``magnitudes`` and ``domain`` are physical values where a relative
tolerance is right. ``rupture_propagation`` holds normalised [0, 1] fault
coordinates whose true value at a fault edge is zero and which floating point
delivers as denormal residue -- across the deployed set they reach 4.3e-216 --
so those need an absolute tolerance as well, or every jump point at a fault edge
reads as a total mismatch.

Usage
-----
Not a command. Used by ``generate-realisations-from-csv`` and
``complete-realisations``; see ``docs/rupture_propagation_reproducibility.md``.
"""

import dataclasses
import datetime
import enum
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from workflow.scripts.reconcile_parameters import DEFAULT_TOLERANCE, values_equivalent

# Per-section (relative, absolute) tolerances below which a difference between
# the carried-over value and the derived one is float noise rather than a change.
#
# The absolute term is non-zero only for rupture_propagation. Its jump points are
# normalised fault coordinates, so 1e-6 is a millionth of a fault dimension --
# three orders below the SRF discretisation, and far too small to move a single
# grid point, while still absorbing the denormal residue that a relative
# comparison blows up into a 100% difference.
SECTION_TOLERANCES: dict[str, tuple[float, float]] = {
    "sources": (DEFAULT_TOLERANCE, 0.0),
    "rupture_propagation": (1e-6, 1e-6),
    "magnitudes": (DEFAULT_TOLERANCE, 0.0),
    "rakes": (DEFAULT_TOLERANCE, 0.0),
    "domain": (DEFAULT_TOLERANCE, 0.0),
}

# seeds is an input, not a derivation: the generator is handed it and replays it,
# so there is no independently derived value to check it against.
UNCHECKABLE_SECTIONS = frozenset({"seeds"})


class InheritanceChoice(str, enum.Enum):
    """Which of the two values to write when they genuinely differ."""

    inherited = "inherited"
    derived = "derived"


@dataclasses.dataclass(frozen=True)
class InheritanceDecision:
    """A recorded resolution of one carried-over section's divergence.

    Attributes
    ----------
    choice : InheritanceChoice
        Whether to keep the deployed value or take the freshly derived one.
    reason : str
        Why. Required, not decorative: an unexplained choice here is exactly the
        untraceable decision this campaign exists to eliminate.
    decided : str
        ISO date the decision was made.
    """

    choice: InheritanceChoice
    reason: str
    decided: str


class UndecidedDivergenceError(Exception):
    """A carried-over section differs from the derived one, with no decision."""


def decision_key(rupture_id: str, section: str) -> str:
    """Return the per-rupture key for a decision about one section.

    Parameters
    ----------
    rupture_id : str
        The rupture the decision applies to.
    section : str
        The realisation section the decision applies to.

    Returns
    -------
    str
        The dotted key, ``<rupture id>.<section>``.
    """
    return f"{rupture_id}.{section}"


def load_decisions(decisions_ffp: Path | None) -> dict[str, InheritanceDecision]:
    """Read recorded inheritance decisions from YAML.

    Two levels of key are accepted. A bare section name (``rupture_propagation``)
    applies to every rupture; a dotted ``<rupture id>.<section>`` applies to one
    and wins over the section-wide entry.

    Parameters
    ----------
    decisions_ffp : Path or None
        Path to the decision file. None gives an empty mapping.

    Returns
    -------
    dict of str to InheritanceDecision
        Decisions keyed as they were written.

    Raises
    ------
    ValueError
        If an entry names a choice that is not ``inherited`` or ``derived``, or
        omits its reason.
    """
    if decisions_ffp is None or not decisions_ffp.is_file():
        return {}
    with open(decisions_ffp, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    decisions: dict[str, InheritanceDecision] = {}
    for key, entry in raw.items():
        try:
            choice = InheritanceChoice(entry["choice"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{decisions_ffp}: entry {key!r} must set choice to "
                f"'inherited' or 'derived'"
            ) from exc
        reason = entry.get("reason")
        if not reason:
            raise ValueError(f"{decisions_ffp}: entry {key!r} must give a reason")
        decisions[key] = InheritanceDecision(
            choice=choice, reason=reason, decided=str(entry.get("decided", ""))
        )
    return decisions


def save_decisions(
    decisions_ffp: Path, decisions: dict[str, InheritanceDecision]
) -> None:
    """Write inheritance decisions to YAML.

    Parameters
    ----------
    decisions_ffp : Path
        Path to write.
    decisions : dict of str to InheritanceDecision
        Decisions keyed by section or ``<rupture id>.<section>``.
    """
    serialisable = {
        key: {
            "choice": decision.choice.value,
            "reason": decision.reason,
            "decided": decision.decided or datetime.date.today().isoformat(),
        }
        for key, decision in sorted(decisions.items())
    }
    decisions_ffp.parent.mkdir(parents=True, exist_ok=True)
    with open(decisions_ffp, "w", encoding="utf-8") as handle:
        yaml.safe_dump(serialisable, handle, sort_keys=True, default_flow_style=False)


def decision_for(
    decisions: dict[str, InheritanceDecision], rupture_id: str, section: str
) -> InheritanceDecision | None:
    """Return the decision governing one rupture's section, if any.

    Parameters
    ----------
    decisions : dict of str to InheritanceDecision
        Recorded decisions.
    rupture_id : str
        The rupture being generated.
    section : str
        The section being carried over.

    Returns
    -------
    InheritanceDecision or None
        The per-rupture decision if one exists, otherwise the section-wide one,
        otherwise None.
    """
    return decisions.get(decision_key(rupture_id, section)) or decisions.get(section)


def _walk_differences(
    left: Any, right: Any, path: str
) -> Iterator[tuple[str, str | None, float]]:
    """Yield each differing leaf as ``(path, structural message, relative)``.

    A structural message is present when the two values do not have the same
    shape, or differ in something that is not a number; otherwise the relative
    difference carries the comparison.

    Parameters
    ----------
    left : Any
        One value.
    right : Any
        The other value.
    path : str
        Dotted path to the values being compared, for the messages.

    Yields
    ------
    tuple of (str, str or None, float)
        The path, a structural description or None, and the relative difference.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            yield (
                path,
                f"{path or '<root>'} keys differ: "
                f"+{sorted(right.keys() - left.keys())} "
                f"-{sorted(left.keys() - right.keys())}",
                0.0,
            )
            return
        for key in left:
            yield from _walk_differences(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield path, f"{path} length {len(left)} != {len(right)}", 0.0
            return
        for index, (one, other) in enumerate(zip(left, right, strict=True)):
            yield from _walk_differences(one, other, f"{path}[{index}]")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(left), abs(right))
        yield path, None, abs(left - right) / scale if scale else 0.0
    elif left != right:
        yield path, f"{path}: {left!r} != {right!r}", 0.0


def describe_divergence(inherited: Any, derived: Any) -> str:
    """Return a short human-readable account of how far apart two values are.

    Parameters
    ----------
    inherited : Any
        The deployed value.
    derived : Any
        The freshly derived value.

    Returns
    -------
    str
        A one-line summary naming the worst leaf and its relative difference,
        or a structural description when the two do not have the same shape.
    """
    differences = list(_walk_differences(inherited, derived, ""))
    structural = [message for _, message, _ in differences if message]
    if structural:
        extra = f" (+{len(structural) - 1} more)" if len(structural) > 1 else ""
        return structural[0] + extra

    worst_path, worst_relative = "", 0.0
    for path, _, relative in differences:
        if relative > worst_relative:
            worst_path, worst_relative = path, relative
    if worst_relative:
        return f"worst leaf {worst_path} differs by {worst_relative:.2e} relative"
    return "values differ"


def resolve_section(
    section: str,
    rupture_id: str,
    inherited: Any,
    derived: Any,
    decisions: dict[str, InheritanceDecision],
) -> Any:
    """Return the value to write for one carried-over section.

    Parameters
    ----------
    section : str
        The section being carried over.
    rupture_id : str
        The rupture being generated.
    inherited : Any
        The value read from the deployed realisation.
    derived : Any
        The value the pipeline derived for this run.
    decisions : dict of str to InheritanceDecision
        Recorded resolutions for sections that genuinely diverge.

    Returns
    -------
    Any
        The deployed value when the two agree to within the section's tolerance,
        or when a decision says to keep it; otherwise the derived value.

    Raises
    ------
    UndecidedDivergenceError
        If the two differ by more than the section's tolerance and no decision
        covers them. The caller must not write a partly-inherited file.
    """
    relative, absolute = SECTION_TOLERANCES.get(section, (DEFAULT_TOLERANCE, 0.0))
    if values_equivalent(inherited, derived, relative, absolute):
        return inherited

    decision = decision_for(decisions, rupture_id, section)
    if decision is None:
        raise UndecidedDivergenceError(
            f"rupture {rupture_id}: carried-over {section} differs from the value "
            f"derived this run by more than the section tolerance "
            f"(relative {relative:g}, absolute {absolute:g}) -- "
            f"{describe_divergence(inherited, derived)}. Record a choice for "
            f"'{decision_key(rupture_id, section)}' or '{section}' and re-run."
        )
    return inherited if decision.choice is InheritanceChoice.inherited else derived
