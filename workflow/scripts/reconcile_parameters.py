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
from typing import Any, cast

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
