"""Tests for the carried-over-versus-derived decision layer."""

from pathlib import Path

import pytest

from workflow import inheritance as ih


def decision(choice: str, reason: str = "test") -> ih.InheritanceDecision:
    return ih.InheritanceDecision(
        choice=ih.InheritanceChoice(choice), reason=reason, decided="2026-08-14"
    )


def test_resolve_section_takes_the_inherited_value_when_they_agree_exactly() -> None:
    assert ih.resolve_section("magnitudes", "1", {"a": 7.1}, {"a": 7.1}, {}) == {"a": 7.1}


def test_resolve_section_takes_the_inherited_value_through_float_noise() -> None:
    # Byte-exact adoption: the derived value differs in its last bits, so
    # equality proves which of the two was returned.
    inherited = {"a": 7.1831075072864765}
    derived = {"a": 7.183107507286478}

    resolved = ih.resolve_section("magnitudes", "1", inherited, derived, {})

    assert resolved == inherited
    assert resolved != derived


def test_resolve_section_refuses_a_real_difference() -> None:
    with pytest.raises(ih.UndecidedDivergenceError, match="magnitudes"):
        ih.resolve_section("magnitudes", "1", {"a": 7.1}, {"a": 7.9}, {})


def test_the_refusal_says_how_to_resolve_it() -> None:
    with pytest.raises(ih.UndecidedDivergenceError) as exc_info:
        ih.resolve_section("magnitudes", "100932", {"a": 7.1}, {"a": 7.9}, {})

    message = str(exc_info.value)
    assert "100932.magnitudes" in message
    assert "relative" in message


def test_rupture_propagation_absorbs_denormal_jump_points() -> None:
    # Both s coordinates are a fault edge, i.e. zero, delivered as different
    # denormal residue. Relative comparison calls that a 100% difference, which
    # is what made 8 of 9 multi-fault events fail verification.
    inherited = {"jump_points": {"F": {"from_point": {"s": 1.4087856255707695e-34}}}}
    derived = {"jump_points": {"F": {"from_point": {"s": 3.2e-07}}}}

    assert ih.resolve_section("rupture_propagation", "1", inherited, derived, {}) == (
        inherited
    )


def test_rupture_propagation_still_catches_a_moved_jump_point() -> None:
    # 0.71 is the shift observed on event 242445 when the causality tree changed.
    inherited = {"jump_points": {"F": {"from_point": {"s": 0.12}}}}
    derived = {"jump_points": {"F": {"from_point": {"s": 0.83}}}}

    with pytest.raises(ih.UndecidedDivergenceError):
        ih.resolve_section("rupture_propagation", "1", inherited, derived, {})


def test_a_section_wide_decision_covers_every_rupture() -> None:
    decisions = {"rupture_propagation": decision("inherited")}

    for rupture_id in ("1", "2", "999999"):
        assert (
            ih.resolve_section(
                "rupture_propagation", rupture_id, {"a": 1.0}, {"a": 2.0}, decisions
            )
            == {"a": 1.0}
        )


def test_a_per_rupture_decision_wins_over_the_section_wide_one() -> None:
    decisions = {
        "rupture_propagation": decision("inherited"),
        "42.rupture_propagation": decision("derived"),
    }

    assert ih.resolve_section(
        "rupture_propagation", "42", {"a": 1.0}, {"a": 2.0}, decisions
    ) == {"a": 2.0}
    assert ih.resolve_section(
        "rupture_propagation", "43", {"a": 1.0}, {"a": 2.0}, decisions
    ) == {"a": 1.0}


def test_a_derived_decision_takes_the_freshly_computed_value() -> None:
    decisions = {"magnitudes": decision("derived")}

    assert ih.resolve_section("magnitudes", "1", {"a": 7.1}, {"a": 7.9}, decisions) == (
        {"a": 7.9}
    )


def test_decisions_round_trip_through_yaml(tmp_path: Path) -> None:
    decisions = {
        "rupture_propagation": decision("inherited", "unseeded draw"),
        "42.domain": decision("derived", "grid resized deliberately"),
    }
    path = tmp_path / "decisions.yaml"

    ih.save_decisions(path, decisions)

    assert ih.load_decisions(path) == decisions


def test_load_decisions_is_empty_without_a_file(tmp_path: Path) -> None:
    assert ih.load_decisions(None) == {}
    assert ih.load_decisions(tmp_path / "absent.yaml") == {}


def test_load_decisions_rejects_an_entry_with_no_reason(tmp_path: Path) -> None:
    # An unexplained choice is the untraceable decision this campaign exists to
    # eliminate, so it is refused at read time rather than silently honoured.
    path = tmp_path / "decisions.yaml"
    path.write_text("rupture_propagation:\n  choice: inherited\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reason"):
        ih.load_decisions(path)


def test_load_decisions_rejects_an_unknown_choice(tmp_path: Path) -> None:
    path = tmp_path / "decisions.yaml"
    path.write_text(
        "rupture_propagation:\n  choice: whichever\n  reason: x\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="inherited"):
        ih.load_decisions(path)


def test_describe_divergence_names_the_worst_leaf() -> None:
    described = ih.describe_divergence(
        {"a": {"b": 1.0, "c": 1.0}}, {"a": {"b": 1.0000000001, "c": 2.0}}
    )

    assert ".a.c" in described


def test_describe_divergence_reports_a_structural_difference() -> None:
    described = ih.describe_divergence(
        {"tree": {"A": None, "B": "A"}}, {"tree": {"A": None}}
    )

    assert "keys differ" in described


def test_every_inheritable_section_has_a_tolerance() -> None:
    # A section that can be carried over but has no entry would silently fall
    # back to the relative-only default, which is wrong for rupture_propagation.
    from workflow.scripts.generate_realisations_from_csv import SECTION_KEYS

    checkable = set(SECTION_KEYS) - ih.UNCHECKABLE_SECTIONS
    assert checkable <= set(ih.SECTION_TOLERANCES)
    assert "domain" in ih.SECTION_TOLERANCES
