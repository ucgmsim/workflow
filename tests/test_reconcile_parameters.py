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
