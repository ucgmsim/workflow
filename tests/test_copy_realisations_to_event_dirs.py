"""Tests for the copy_realisations_to_event_dirs campaign tool."""

from pathlib import Path

from workflow.scripts import copy_realisations_to_event_dirs as cre


def test_copy_realisations_creates_one_dir_per_rupture(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"a": 1}', encoding="utf-8")
    (source / "realisation_71220.json").write_text('{"b": 2}', encoding="utf-8")
    events = tmp_path / "events"

    copied, skipped, refused = cre.copy_realisations(source, events)

    assert copied == 2
    assert skipped == []
    assert refused == []
    assert (events / "100932" / "realisation.json").read_text() == '{"a": 1}'
    assert (events / "71220" / "realisation.json").read_text() == '{"b": 2}'


def test_copy_realisations_skips_files_without_an_integer_id(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text("{}", encoding="utf-8")
    (source / "notes.json").write_text("{}", encoding="utf-8")
    events = tmp_path / "events"

    copied, skipped, _ = cre.copy_realisations(source, events)

    assert copied == 1
    assert skipped == ["notes.json"]


def test_copy_realisations_refuses_to_replace_without_the_flag(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"new": 1}', encoding="utf-8")
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    copied, _, refused = cre.copy_realisations(source, events)

    assert copied == 0
    assert refused == ["100932"]
    # The existing file is untouched -- this is the whole point of the gate.
    assert (events / "100932" / "realisation.json").read_text() == '{"old": 1}'


def test_copy_realisations_replaces_when_the_flag_is_given(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"new": 1}', encoding="utf-8")
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    copied, _, refused = cre.copy_realisations(source, events, overwrite_existing=True)

    assert copied == 1
    assert refused == []
    assert (events / "100932" / "realisation.json").read_text() == '{"new": 1}'


def test_copy_realisations_still_creates_new_events_alongside_refusals(
    tmp_path: Path,
) -> None:
    source = tmp_path / "complete"
    source.mkdir()
    (source / "realisation_100932.json").write_text('{"new": 1}', encoding="utf-8")
    (source / "realisation_71220.json").write_text('{"fresh": 1}', encoding="utf-8")
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    copied, _, refused = cre.copy_realisations(source, events)

    assert copied == 1
    assert refused == ["100932"]
    assert (events / "71220" / "realisation.json").read_text() == '{"fresh": 1}'
    assert (events / "100932" / "realisation.json").read_text() == '{"old": 1}'
