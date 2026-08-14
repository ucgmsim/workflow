"""Tests for the realisation content checker."""

import json
from pathlib import Path

import pytest
import typer

from workflow.scripts import verify_realisation_content as vc
from workflow.scripts.reconcile_parameters import (
    Decision,
    save_decisions,
    value_fingerprint,
)


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


# -- CLI-level tests: main()'s exit-code contract ---------------------------
#
# The tests above stop at the library layer. Nothing else in the suite proves
# that a non-empty (unexpected, unapplied) pair actually reaches `main`'s
# `raise typer.Exit(code=1)`, or that a regenerated file with no counterpart
# under events_dir is reported rather than silently dropped by the glob loop.
# This is the campaign's last gate before 291 originals are overwritten and
# the comparison target is gone, so that contract is a regression test here
# rather than a one-off manual check.


def test_main_exits_cleanly_when_every_change_is_decided(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events_dir = tmp_path / "events"
    (events_dir / "EVT1").mkdir(parents=True)
    (events_dir / "EVT1" / "realisation.json").write_text(
        json.dumps(
            {
                "im": {"ims": ["PGA"]},
                "magnitudes": {"A": 7.1},
                "log_trail": {"log": ["a"]},
            }
        ),
        encoding="utf-8",
    )
    regenerated_dir = tmp_path / "regenerated"
    regenerated_dir.mkdir()
    (regenerated_dir / "realisation_EVT1.json").write_text(
        json.dumps(
            {
                "im": {"ims": ["PGA", "PGD"]},
                "magnitudes": {"A": 7.1},
                "log_trail": {"log": ["b"]},
            }
        ),
        encoding="utf-8",
    )
    parameters = tmp_path / "decisions.yaml"
    save_decisions(
        parameters,
        {
            "im.ims": Decision(
                source="defaults",
                reason="adopt PGD",
                decided="2026-07-27",
                sha256=value_fingerprint(["PGA", "PGD"]),
            )
        },
    )

    vc.main(events_dir, regenerated_dir, parameters=parameters)

    assert "0 failed" in capsys.readouterr().out


def test_main_exits_1_when_an_undecided_difference_is_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events_dir = tmp_path / "events"
    (events_dir / "EVT1").mkdir(parents=True)
    (events_dir / "EVT1" / "realisation.json").write_text(
        json.dumps(
            {
                "im": {"ims": ["PGA"]},
                "magnitudes": {"A": 7.1},
                "log_trail": {"log": ["a"]},
            }
        ),
        encoding="utf-8",
    )
    regenerated_dir = tmp_path / "regenerated"
    regenerated_dir.mkdir()
    (regenerated_dir / "realisation_EVT1.json").write_text(
        json.dumps(
            {
                # im.ims changed as decided; magnitudes.A changed with no decision.
                "im": {"ims": ["PGA", "PGD"]},
                "magnitudes": {"A": 7.2},
                "log_trail": {"log": ["b"]},
            }
        ),
        encoding="utf-8",
    )
    parameters = tmp_path / "decisions.yaml"
    save_decisions(
        parameters,
        {
            "im.ims": Decision(
                source="defaults",
                reason="adopt PGD",
                decided="2026-07-27",
                sha256=value_fingerprint(["PGA", "PGD"]),
            )
        },
    )

    with pytest.raises(typer.Exit) as exit_info:
        vc.main(events_dir, regenerated_dir, parameters=parameters)

    assert exit_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "UNEXPECTED" in output
    assert "magnitudes.A" in output


def test_main_exits_1_when_a_decision_did_not_take(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events_dir = tmp_path / "events"
    (events_dir / "EVT1").mkdir(parents=True)
    (events_dir / "EVT1" / "realisation.json").write_text(
        json.dumps(
            {
                "im": {"ims": ["PGA"]},
                "magnitudes": {"A": 7.1},
                "log_trail": {"log": ["a"]},
            }
        ),
        encoding="utf-8",
    )
    regenerated_dir = tmp_path / "regenerated"
    regenerated_dir.mkdir()
    (regenerated_dir / "realisation_EVT1.json").write_text(
        json.dumps(
            {
                # The im.ims decision was recorded but never reached this file.
                "im": {"ims": ["PGA"]},
                "magnitudes": {"A": 7.1},
                "log_trail": {"log": ["b"]},
            }
        ),
        encoding="utf-8",
    )
    parameters = tmp_path / "decisions.yaml"
    save_decisions(
        parameters,
        {
            "im.ims": Decision(
                source="defaults",
                reason="adopt PGD",
                decided="2026-07-27",
                sha256=value_fingerprint(["PGA", "PGD"]),
            )
        },
    )

    with pytest.raises(typer.Exit) as exit_info:
        vc.main(events_dir, regenerated_dir, parameters=parameters)

    assert exit_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "UNAPPLIED" in output
    assert "im.ims" in output


def test_main_reports_a_regenerated_file_with_no_original_instead_of_skipping_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    regenerated_dir = tmp_path / "regenerated"
    regenerated_dir.mkdir()
    (regenerated_dir / "realisation_EVT2.json").write_text(
        json.dumps({"magnitudes": {"A": 7.1}}),
        encoding="utf-8",
    )

    with pytest.raises(typer.Exit) as exit_info:
        vc.main(events_dir, regenerated_dir)

    assert exit_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "EVT2" in output
    assert "no original to compare against" in output


def write_pair(
    tmp_path: Path, rupture_id: str, magnitude: float, original: bool = True
) -> tuple[Path, Path]:
    """Write a regenerated file, and optionally the original it matches."""
    events_dir = tmp_path / "events"
    regenerated_dir = tmp_path / "regenerated"
    regenerated_dir.mkdir(exist_ok=True)
    events_dir.mkdir(exist_ok=True)
    if original:
        (events_dir / rupture_id).mkdir(parents=True, exist_ok=True)
        (events_dir / rupture_id / "realisation.json").write_text(
            json.dumps({"magnitudes": {"A": 7.1}, "log_trail": {"log": ["a"]}}),
            encoding="utf-8",
        )
    (regenerated_dir / f"realisation_{rupture_id}.json").write_text(
        json.dumps({"magnitudes": {"A": magnitude}, "log_trail": {"log": ["b"]}}),
        encoding="utf-8",
    )
    return events_dir, regenerated_dir


def test_main_accepts_an_event_with_no_original_when_allow_new_is_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A rupture the deployed set never held is a supported case, not an error:
    # it draws fresh seeds and derives its own rupture propagation.
    events_dir, regenerated_dir = write_pair(tmp_path, "EVT1", 7.1)
    write_pair(tmp_path, "EVT2", 7.5, original=False)

    vc.main(events_dir, regenerated_dir, allow_new=True)

    output = capsys.readouterr().out
    # Reported, not silently ignored -- the operator has to be able to see which
    # ruptures were treated as new.
    assert "EVT2" in output
    assert "1 new" in output


def test_main_still_fails_a_bad_event_when_allow_new_is_set(
    tmp_path: Path,
) -> None:
    events_dir, regenerated_dir = write_pair(tmp_path, "EVT1", 7.2)
    write_pair(tmp_path, "EVT2", 7.5, original=False)

    with pytest.raises(typer.Exit) as exit_info:
        vc.main(events_dir, regenerated_dir, allow_new=True)

    assert exit_info.value.exit_code == 1


def test_main_exits_1_on_an_empty_regenerated_directory(tmp_path: Path) -> None:
    # "Compared 0 realisation(s); 0 failed" must not read as success. Every
    # per-file check passes vacuously on a set that holds no files.
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    regenerated_dir = tmp_path / "regenerated"
    regenerated_dir.mkdir()

    with pytest.raises(typer.Exit) as exit_info:
        vc.main(events_dir, regenerated_dir)

    assert exit_info.value.exit_code == 1


def test_main_exits_1_when_the_count_does_not_match_expect_count(
    tmp_path: Path,
) -> None:
    # With --allow-new, pointing at the wrong events directory would otherwise
    # pass silently as "every rupture is new". The count is what catches that.
    events_dir, regenerated_dir = write_pair(tmp_path, "EVT1", 7.1)

    with pytest.raises(typer.Exit) as exit_info:
        vc.main(events_dir, regenerated_dir, expect_count=291)

    assert exit_info.value.exit_code == 1


def test_main_accepts_a_matching_expect_count(tmp_path: Path) -> None:
    events_dir, regenerated_dir = write_pair(tmp_path, "EVT1", 7.1)

    vc.main(events_dir, regenerated_dir, expect_count=1)
