"""Tests for the realisation provenance verifier."""

import json
from pathlib import Path

import pytest
import typer

from workflow.scripts import verify_realisation_provenance as vp
from workflow.scripts.complete_realisations import FELIPE_SECTION_ORDER

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


# The CLI's exit code is the campaign's abort condition -- Task 11's audit is the
# step that proves all 291 files record commit N -- so it is asserted directly.


def test_main_exits_0_when_every_file_is_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_realisation(tmp_path / "realisation_100932.json")
    write_realisation(tmp_path / "realisation_100933.json")
    monkeypatch.setattr(vp.metadata, "version", lambda name: CLEAN_VERSION)

    vp.verify_realisation_provenance(tmp_path)


def test_main_exits_1_when_a_file_records_the_wrong_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_realisation(tmp_path / "realisation_100932.json")
    write_realisation(tmp_path / "realisation_100933.json", version=DIRTY_VERSION)
    monkeypatch.setattr(vp.metadata, "version", lambda name: CLEAN_VERSION)

    with pytest.raises(typer.Exit) as exit_info:
        vp.verify_realisation_provenance(tmp_path)

    assert exit_info.value.exit_code == 1


def test_main_exits_1_when_preflight_finds_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vp, "git_is_clean", lambda repo_root: False)
    monkeypatch.setattr(vp, "git_head_sha", lambda repo_root: CLEAN_SHA)
    monkeypatch.setattr(vp.metadata, "version", lambda name: CLEAN_VERSION)

    with pytest.raises(typer.Exit) as exit_info:
        vp.verify_realisation_provenance(None, preflight=True, repo_root=tmp_path)

    assert exit_info.value.exit_code == 1


def test_main_exits_0_when_preflight_is_satisfied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vp, "git_is_clean", lambda repo_root: True)
    monkeypatch.setattr(vp, "git_head_sha", lambda repo_root: CLEAN_SHA)
    monkeypatch.setattr(vp.metadata, "version", lambda name: CLEAN_VERSION)

    vp.verify_realisation_provenance(None, preflight=True, repo_root=tmp_path)


def test_main_exits_1_when_given_neither_a_directory_nor_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vp.metadata, "version", lambda name: CLEAN_VERSION)

    with pytest.raises(typer.Exit) as exit_info:
        vp.verify_realisation_provenance(None)

    assert exit_info.value.exit_code == 1


def test_main_exits_1_on_an_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "Compared 0 realisations, 0 failed" must not read as success: an empty
    # directory means the generation step produced nothing, not that it passed.
    monkeypatch.setattr(vp.metadata, "version", lambda name: CLEAN_VERSION)

    with pytest.raises(typer.Exit) as exit_info:
        vp.verify_realisation_provenance(tmp_path)

    assert exit_info.value.exit_code == 1


def test_main_exits_1_when_the_count_does_not_match_expect_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_realisation(tmp_path / "realisation_100932.json")
    monkeypatch.setattr(vp.metadata, "version", lambda name: CLEAN_VERSION)

    with pytest.raises(typer.Exit) as exit_info:
        vp.verify_realisation_provenance(tmp_path, expect_count=291)

    assert exit_info.value.exit_code == 1
