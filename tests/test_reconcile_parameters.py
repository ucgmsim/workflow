"""Tests for the parameter reconciliation comparison engine."""

import json
from pathlib import Path

import pytest

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


def test_values_equivalent_is_relative_only_by_default() -> None:
    # Two denormal residues of the same "zero". A parameter grid must still call
    # these different -- 0.0132 Hz and 0.0 Hz are different parameters.
    assert not rp.values_equivalent(1.4087856255707695e-34, 3.2e-07)


def test_abs_tolerance_absorbs_near_zero_differences() -> None:
    assert rp.values_equivalent(1.4087856255707695e-34, 3.2e-07, 1e-9, 1e-6)


def test_abs_tolerance_reaches_inside_lists() -> None:
    # Threading it through the container branches is easy to leave out, and the
    # scalar case passing says nothing about the list one.
    assert rp.values_equivalent([1.4087856255707695e-34], [3.2e-07], 1e-9, 1e-6)
    assert not rp.values_equivalent([1.4087856255707695e-34], [3.2e-07], 1e-9, 0.0)


def test_abs_tolerance_reaches_inside_dicts() -> None:
    assert rp.values_equivalent(
        {"s": 1.4087856255707695e-34}, {"s": 3.2e-07}, 1e-9, 1e-6
    )
    assert not rp.values_equivalent(
        {"s": 1.4087856255707695e-34}, {"s": 3.2e-07}, 1e-9, 0.0
    )


def test_abs_tolerance_still_catches_a_real_difference() -> None:
    assert not rp.values_equivalent([0.12], [0.83], 1e-9, 1e-6)


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


def test_value_fingerprint_is_stable_across_key_order() -> None:
    assert rp.value_fingerprint({"a": 1, "b": 2}) == rp.value_fingerprint(
        {"b": 2, "a": 1}
    )


def test_value_fingerprint_distinguishes_different_values() -> None:
    assert rp.value_fingerprint([1, 2]) != rp.value_fingerprint([1, 3])


def test_value_fingerprint_rejects_a_non_json_native_type() -> None:
    # A set (unlike dict/list/str/int/float/bool/None) has no stable JSON
    # encoding: refuse it loudly rather than silently coercing it.
    with pytest.raises(TypeError, match="set"):
        rp.value_fingerprint({"a": {1, 2}})


def test_resolve_value_returns_the_chosen_source() -> None:
    resolved = rp.resolve_value("defaults", {"defaults": ["PGA"], "deployed": ["PGV"]})

    assert resolved == ["PGA"]


def test_resolve_value_unions_discrete_candidates_deterministically() -> None:
    resolved = rp.resolve_value(
        "union", {"defaults": ["PGA", "PGD"], "deployed": ["PGA", "PGV"]}
    )

    # Sources are visited in sorted order, so the result is reproducible.
    assert resolved == ["PGA", "PGD", "PGV"]


def test_resolve_value_rejects_a_source_that_has_no_opinion() -> None:
    with pytest.raises(ValueError, match="felipe"):
        rp.resolve_value("felipe", {"defaults": ["PGA"]})


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
