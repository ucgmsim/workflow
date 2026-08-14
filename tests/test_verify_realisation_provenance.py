"""Tests for the realisation provenance verifier."""

from pathlib import Path

import pytest

from workflow.scripts import verify_realisation_provenance as vp

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
