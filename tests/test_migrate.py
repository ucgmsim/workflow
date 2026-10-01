import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import schema
from rich.console import RenderableType
from typer.testing import CliRunner, Result

from workflow.defaults import DefaultsVersion
from workflow.scripts import migrate
from workflow.scripts.migrate import Action

OLD = DefaultsVersion.v24_2_2_4
NEW = DefaultsVersion.v26_7_1Hz

runner = CliRunner()


def old_realisation() -> dict[str, Any]:
    """A realisation written with every 24.2.2.4 default filled in."""
    return {
        "metadata": {"name": "event", "version": "1", "defaults_version": str(OLD)},
        **copy.deepcopy(migrate._default_sections(OLD)),
    }


def write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))
    return path


def read(path: Path) -> Any:
    return json.loads(path.read_text())


class Answers(migrate._Prompter):
    """Prompter answering each action from a fixed table, recording what was asked."""

    def __init__(self, **answers: bool) -> None:
        super().__init__()
        self.answers = {
            Action[name.upper()]: answer for name, answer in answers.items()
        }
        self.asked: list[tuple[Action, str | None]] = []

    def ask(
        self,
        question: str,
        action: Action,
        key: str | None = None,
        details: list[RenderableType] | None = None,
    ) -> bool:
        self.asked.append((action, key))
        return self.answers.get(action, False)

    def choose(
        self,
        question: str,
        action: Action,
        key: str,
        options: list[tuple[migrate._KeyPath, str]],
        details: list[RenderableType],
    ) -> list[migrate._KeyPath]:
        paths = [path for path, _ in options]
        return paths if self.ask(question, action, key) else []


def run_migrate(prompter: migrate._Prompter, path: Path) -> migrate._MigrationResult:
    return migrate._migrate(path, NEW, migrate._realisation_configurations(), prompter)


def run_cli(path: Path, *flags: str, **kwargs: Any) -> Result:
    return runner.invoke(migrate.app, [str(path), str(NEW), *flags], **kwargs)


def script_input(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[str], str]
) -> list[str]:
    """Answer prompts with ``answer``, returning the list of prompts asked."""
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return answer(prompt)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


@pytest.fixture
def realisation(tmp_path: Path) -> Path:
    return write(tmp_path / "event" / "realisation.json", old_realisation())


def test_differing_values_are_updated_and_new_sections_added(
    realisation: Path,
) -> None:
    result = run_migrate(Answers(fill=True, update=True), realisation)

    assert not result.errors
    migrated = result.migrated
    assert migrated["metadata"]["defaults_version"] == str(NEW)
    assert migrated["bb"]["flo"] == 1.0
    assert migrated["velocity_model"]["fault_buffer"] == 14.0
    assert migrated["sw4"] == migrate._default_sections(NEW)["sw4"]
    # Sections without new defaults are kept as they are.
    assert migrated["resolution"] == old_realisation()["resolution"]


def test_assume_yes_keeps_differing_values_unless_overwrite() -> None:
    prompter = migrate._Prompter(assume_yes=True)
    assert prompter.ask("?", Action.FILL, "bb")
    assert prompter.ask("?", Action.WRITE)
    options: list[tuple[migrate._KeyPath, str]] = [(("flo",), "flo")]
    assert prompter.choose("?", Action.UPDATE, "bb", options, []) == []

    prompter = migrate._Prompter(assume_yes=True, overwrite=True)
    assert prompter.choose("?", Action.UPDATE, "bb", options, []) == [("flo",)]


def test_missing_keys_are_filled_without_touching_other_values(
    realisation: Path,
) -> None:
    data = read(realisation)
    del data["hf"]["source"]["stress_drop_bars"]
    write(realisation, data)

    result = run_migrate(Answers(fill=True), realisation)

    expected = migrate._default_sections(NEW)["hf"]["source"]["stress_drop_bars"]
    assert result.migrated["hf"]["source"]["stress_drop_bars"] == expected
    # Updates were not accepted, so differing values are kept.
    assert result.migrated["bb"]["flo"] == 0.25


def test_migrate_does_not_write(realisation: Path) -> None:
    before = realisation.read_text()
    result = run_migrate(Answers(fill=True, update=True, trim=True), realisation)
    assert result.changed
    assert realisation.read_text() == before


def test_nested_unknown_keys_are_trimmed(realisation: Path) -> None:
    data = read(realisation)
    data["bb"]["unknown"] = 1
    data["hf"]["source"]["also_unknown"] = 2
    write(realisation, data)

    prompter = Answers(trim=True)
    result = run_migrate(prompter, realisation)

    assert not result.errors
    assert "unknown" not in result.migrated["bb"]
    assert "also_unknown" not in result.migrated["hf"]["source"]
    assert (Action.TRIM, "bb") in prompter.asked
    assert (Action.TRIM, "hf") in prompter.asked


def test_declining_trim_reports_error(realisation: Path) -> None:
    data = read(realisation)
    data["bb"]["unknown"] = 1
    write(realisation, data)

    result = run_migrate(Answers(trim=False), realisation)

    assert result.migrated["bb"]["unknown"] == 1
    assert any("bb" in error for error in result.errors)


def test_sections_without_defaults_are_validated(realisation: Path) -> None:
    data = read(realisation)
    data["domain"] = {"not_a_domain_key": 1}
    write(realisation, data)

    result = run_migrate(Answers(), realisation)

    assert any("domain" in error for error in result.errors)


def test_non_realisation_json_is_skipped(tmp_path: Path) -> None:
    path = write(tmp_path / "stations.json", {"stations": []})
    result = run_migrate(Answers(), path)
    assert result.status is migrate._Status.SKIPPED
    assert not result.errors


def test_invalid_json_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{")
    result = run_migrate(Answers(), path)
    assert result.errors


@pytest.mark.parametrize(
    ("section_schema", "section", "expected"),
    [
        ({"a": int, "b": {"c": int}}, {"a": 1, "b": {"c": 1, "z": 1}}, [("b", "z")]),
        ({"a": int}, {"a": 1, "x": 1, "y": 2}, [("x",), ("y",)]),
    ],
)
def test_extract_error_finds_unknown_key_paths(
    section_schema: dict, section: dict, expected: list[tuple[str, ...]]
) -> None:
    with pytest.raises(schema.SchemaError) as error:
        schema.Schema(section_schema).validate(section)
    _, unknown = migrate._extract_error("section", error.value)
    assert sorted(unknown) == expected


def test_find_realisations_accepts_files_and_directories(tmp_path: Path) -> None:
    a = write(tmp_path / "a" / "realisation.json", {})
    b = write(tmp_path / "b" / "realisation.json", {})
    single = write(tmp_path / "single.json", {})

    found = migrate._find_realisations(
        [single, tmp_path / "a", a, tmp_path / "b"], "*.json"
    )

    assert found == [single, a, b]


# Command line


def test_single_file_with_yes_keeps_differing_values(realisation: Path) -> None:
    result = run_cli(realisation, "--yes")

    assert result.exit_code == 0, result.output
    migrated = read(realisation)
    assert migrated["metadata"]["defaults_version"] == str(NEW)
    assert "sw4" in migrated
    assert migrated["bb"]["flo"] == 0.25


def test_yes_overwrite_updates_differing_values(realisation: Path) -> None:
    result = run_cli(realisation, "--yes", "--overwrite")

    assert result.exit_code == 0, result.output
    assert read(realisation)["bb"]["flo"] == 1.0


def test_overwrite_needs_yes_or_check(realisation: Path) -> None:
    result = run_cli(realisation, "--overwrite")
    assert result.exit_code != 0


def test_directory_and_bad_files_do_not_stop_the_run(tmp_path: Path) -> None:
    realisation = write(tmp_path / "event" / "realisation.json", old_realisation())
    write(tmp_path / "event" / "stations.json", {"stations": []})
    (tmp_path / "event" / "broken.json").write_text("{")

    result = run_cli(tmp_path, "--yes")

    # The broken file is reported as an error, but the realisation is still migrated.
    assert result.exit_code == 1, result.output
    assert read(realisation)["metadata"]["defaults_version"] == str(NEW)


def test_check_does_not_write_and_fails_when_migration_needed(
    realisation: Path,
) -> None:
    before = realisation.read_text()

    result = run_cli(realisation, "--check")
    assert result.exit_code == 1, result.output
    assert realisation.read_text() == before

    run_cli(realisation, "--yes")
    result = run_cli(realisation, "--check")
    assert result.exit_code == 0, result.output


def test_dry_run_does_not_write(realisation: Path) -> None:
    before = realisation.read_text()
    result = run_cli(realisation, "--yes", "--dry-run")
    assert result.exit_code == 0, result.output
    assert realisation.read_text() == before


def test_yes_and_check_conflict(realisation: Path) -> None:
    result = run_cli(realisation, "--yes", "--check")
    assert result.exit_code != 0


def test_no_input_exits_without_writing(realisation: Path) -> None:
    before = realisation.read_text()
    result = run_cli(realisation, input="")
    assert result.exit_code == 2, result.output
    assert "--yes" in result.output
    assert realisation.read_text() == before


def test_declining_write_leaves_file_untouched(
    realisation: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = realisation.read_text()
    # Accept every section change with "!", then decline the write.
    prompts = script_input(
        monkeypatch, lambda prompt: "n" if "Write changes" in prompt else "!"
    )

    result = run_cli(realisation)

    assert result.exit_code == 0, result.output
    assert any("Write changes" in prompt for prompt in prompts)
    assert realisation.read_text() == before


def test_backup_only_made_when_writing(realisation: Path) -> None:
    backup = realisation.with_suffix(".json.bak")
    original = realisation.read_text()

    run_cli(realisation, "--yes", "--backup", ".bak")
    assert backup.read_text() == original

    # Nothing left to migrate, so the backup of the original is kept.
    run_cli(realisation, "--yes", "--backup", ".bak")
    assert backup.read_text() == original


def test_copy_only_copies_selected_configs(tmp_path: Path) -> None:
    template = write(
        tmp_path / "template.json",
        {"resolution": {"resolution": 0.5}, "bb": {"flo": 1.0}},
    )
    realisation = write(
        tmp_path / "realisations" / "r.json",
        {"resolution": {"resolution": 0.1}, "bb": {"flo": 9.0}},
    )

    result = runner.invoke(
        migrate.copy_app,
        [str(template), str(realisation.parent), "--configs", "resolution"],
    )

    assert result.exit_code == 0, result.output
    assert read(realisation) == {"resolution": {"resolution": 0.5}, "bb": {"flo": 9.0}}


def test_copy_rejects_configs_missing_from_template(tmp_path: Path) -> None:
    template = write(tmp_path / "template.json", {"bb": {"flo": 1.0}})
    realisation = write(tmp_path / "realisations" / "r.json", {"bb": {"flo": 9.0}})

    result = runner.invoke(
        migrate.copy_app,
        [str(template), str(realisation.parent), "--configs", "resolution"],
    )

    assert result.exit_code != 0
    assert read(realisation) == {"bb": {"flo": 9.0}}


def test_details_are_printed_only_the_first_time(tmp_path: Path) -> None:
    for name in ["a", "b", "c"]:
        write(tmp_path / name / "realisation.json", old_realisation())

    result = run_cli(tmp_path, "--yes", "--overwrite")

    assert result.exit_code == 0, result.output
    assert result.output.count('-    "fault_buffer": 2.0') == 1
    assert result.output.count('+    "fault_buffer": 14.0') == 1
    assert result.output.count("velocity_model: updated fault_buffer") == 1
    assert result.output.count("changes listed above") == 2


def test_always_answers_are_not_asked_or_explained_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ["a", "b"]:
        write(tmp_path / name / "realisation.json", old_realisation())
    prompts = script_input(monkeypatch, lambda prompt: "!")

    result = run_cli(tmp_path)

    assert result.exit_code == 0, result.output
    # Every question was answered on the first realisation.
    second = result.output.split(str(tmp_path / "b" / "realisation.json"))[1]
    assert "fault_buffer" not in second
    assert "changes listed above" in second
    assert len(prompts) == len(set(prompts))
    for name in ["a", "b"]:
        migrated = read(tmp_path / name / "realisation.json")
        assert migrated["metadata"]["defaults_version"] == str(NEW)


# Choosing keys


def with_flo_and_fmin(flo: float = 0.25) -> dict[str, Any]:
    """An old realisation where bb and velocity_model differ from the new defaults."""
    data = old_realisation()
    data["bb"]["flo"] = flo
    data["bb"]["fmin"] = 0.3
    return data


class Picker:
    """Scripted stand-in for the terminal: typed answers, ticks and confirmations."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        typed: list[str],
        ticks: list[list[str]],
        confirmations: list[migrate._Response],
    ) -> None:
        self.typed, self.ticks, self.confirmations = typed, ticks, confirmations
        self.prompts: list[str] = []
        self.shown: list[tuple[list[str], list[bool]]] = []
        monkeypatch.setattr("builtins.input", self.input)
        monkeypatch.setattr(migrate, "_terminal_available", lambda: True)
        monkeypatch.setattr(migrate, "_select_keys", self.select_keys)
        monkeypatch.setattr(migrate, "_confirm_selection", self.confirm_selection)

    def input(self, prompt: str) -> str:
        self.prompts.append(prompt)
        # Only the bb update question is scripted; decline everything else.
        return self.typed.pop(0) if "values in bb" in prompt else "n"

    def select_keys(
        self,
        question: str,
        options: list[tuple[migrate._KeyPath, str]],
        ticks: dict[migrate._KeyPath, bool],
    ) -> dict[migrate._KeyPath, bool]:
        self.shown.append(([label for _, label in options], list(ticks.values())))
        ticked = self.ticks.pop(0)
        return {path: ".".join(path) in ticked for path, _ in options}

    def confirm_selection(self) -> migrate._Response:
        return self.confirmations.pop(0)


def test_selected_keys_are_updated_and_the_rest_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path / "realisation.json", with_flo_and_fmin())
    picker = Picker(monkeypatch, ["s"], [["flo"]], [migrate._Response.YES])

    result = run_migrate(migrate._Prompter(), path)

    assert result.migrated["bb"]["flo"] == 1.0
    assert result.migrated["bb"]["fmin"] == 0.3
    labels, checked = picker.shown[0]
    assert labels == ["flo: 0.25 -> 1.0", "fmin: 0.3 -> 0.2"]
    # Every key starts ticked to migrate.
    assert checked == [True, True]


def test_declining_selection_returns_to_the_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path / "realisation.json", with_flo_and_fmin())
    picker = Picker(
        monkeypatch,
        ["s", "s", "n"],
        [["flo"], ["flo"]],
        [migrate._Response.NO, migrate._Response.NO],
    )

    result = run_migrate(migrate._Prompter(), path)

    assert result.migrated["bb"] == read(path)["bb"]
    # Reopening the list keeps the previous ticks.
    assert picker.shown[1][1] == [True, False]
    assert sum("values in bb" in prompt for prompt in picker.prompts) == 3


def test_yes_to_selection_asks_again_next_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [
        write(tmp_path / name / "realisation.json", with_flo_and_fmin())
        for name in ["a", "b"]
    ]
    picker = Picker(monkeypatch, ["s", "y"], [["fmin"]], [migrate._Response.YES])
    prompter = migrate._Prompter()

    first, second = (run_migrate(prompter, path) for path in paths)

    assert (first.migrated["bb"]["flo"], first.migrated["bb"]["fmin"]) == (0.25, 0.2)
    assert (second.migrated["bb"]["flo"], second.migrated["bb"]["fmin"]) == (1.0, 0.2)
    assert not picker.typed


def test_always_remembers_the_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [
        write(tmp_path / name / "realisation.json", with_flo_and_fmin(flo))
        for name, flo in [("a", 0.25), ("b", 0.5)]
    ]
    picker = Picker(monkeypatch, ["s"], [["fmin"]], [migrate._Response.AUTO])
    prompter = migrate._Prompter()

    first, second = (run_migrate(prompter, path) for path in paths)

    # The same keys are migrated in the second realisation, without asking.
    assert (first.migrated["bb"]["flo"], first.migrated["bb"]["fmin"]) == (0.25, 0.2)
    assert (second.migrated["bb"]["flo"], second.migrated["bb"]["fmin"]) == (0.5, 0.2)
    assert sum("values in bb" in prompt for prompt in picker.prompts) == 1


def test_remembered_selection_asks_again_for_new_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_data = with_flo_and_fmin()
    second_data = with_flo_and_fmin()
    second_data["bb"]["fmax"] = 50.0
    first_path = write(tmp_path / "a" / "realisation.json", first_data)
    second_path = write(tmp_path / "b" / "realisation.json", second_data)
    picker = Picker(monkeypatch, ["s", "n"], [["fmin"]], [migrate._Response.AUTO])
    prompter = migrate._Prompter()

    run_migrate(prompter, first_path)
    second = run_migrate(prompter, second_path)

    # fmax was not in the remembered selection, so the question is asked again.
    assert sum("values in bb" in prompt for prompt in picker.prompts) == 2
    assert second.migrated["bb"]["fmax"] == 50.0
