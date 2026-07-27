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
