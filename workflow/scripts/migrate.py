"""Migrate realisations to a new defaults version.

Each realisation is migrated in memory and written at most once, after
the user has approved the changes. Missing keys are filled in, values
that differ from the new defaults can be updated key by key, unknown
keys can be trimmed, and every section present in the realisation is
checked against its schema.
"""

import copy
import dataclasses
import functools
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Annotated, Any, TypeGuard

import parse
import schema
import typer
from rich.console import Console
from rich.markup import escape

from qcore import cli
from workflow import realisations, utils
from workflow.defaults import DefaultsVersion
from workflow.realisations import Seeds

app = typer.Typer()
copy_app = typer.Typer()
clone_app = typer.Typer()
console = Console()


# Every use site refers to a RealisationConfiguration *subclass* (the classes
# returned by realisation_configurations), not an instance of one.
type ConfigType = type[realisations.RealisationConfiguration]
type KeyPath = tuple[str, ...]


def is_realisation_configuration(cls: object) -> TypeGuard[ConfigType]:
    """Returns True if the class is a subclass of realisation configuration.

    Parameters
    ----------
    cls : object
        Object to check.

    Returns
    -------
    bool
        True if class is a realisation configuration.
    """
    return (
        cls != realisations.RealisationConfiguration
        and inspect.isclass(cls)
        and issubclass(cls, realisations.RealisationConfiguration)
    )


def realisation_configurations() -> list[ConfigType]:
    """Return a list of all realisation configurations.

    Returns
    -------
    list[ConfigType]
        A list of all realisation configuration types.
    """
    return [
        cls
        for name, cls in inspect.getmembers(realisations)
        if is_realisation_configuration(cls)
    ]


def loadable_defaults(
    configurations: list[ConfigType], defaults: DefaultsVersion
) -> dict[ConfigType, realisations.RealisationConfiguration]:
    """Filter a list of realisation configurations for those with loadable defaults.

    Parameters
    ----------
    configurations : list[ConfigType]
        Configurations to filter.
    defaults : defaults.DefaultsVersion
        Defaults to try and load.

    Returns
    -------
    dict[ConfigType, realisations.RealisationConfiguration]
        A mapping from realisation configuration types to their
        defaults specified by ``defaults``.

    Raises
    ------
    TypeError
        If ``configurations`` contains a type that is not a
        realisation configuration.
    """
    config_defaults: dict[ConfigType, realisations.RealisationConfiguration] = {}
    for config in configurations:
        if not is_realisation_configuration(config):
            raise TypeError(
                f"{config=} should be a subclass of realisations.RealisationConfiguration"
            )
        else:
            try:
                default_config = config.read_from_defaults(defaults)
                config_defaults[config] = default_config
            except realisations.RealisationParseError:
                continue
    return config_defaults


@functools.cache
def default_sections(defaults_version: DefaultsVersion) -> dict[str, Any]:
    """Return the defaults for a version as they would be written to a realisation.

    Parameters
    ----------
    defaults_version : DefaultsVersion
        Defaults version to load.

    Returns
    -------
    dict[str, Any]
        Mapping from configuration key to the JSON form of its defaults.
    """
    defaults = loadable_defaults(realisation_configurations(), defaults_version)
    return {
        config._config_key: json.loads(
            json.dumps(default.to_dict(), default=realisations.path_serialiser)
        )
        for config, default in defaults.items()
    }


class Response(Enum):
    """Enum for response to prompts asked of user."""

    YES = auto()
    NO = auto()
    AUTO = auto()  # Always (!)
    NEVER = auto()  # Never (N)
    SELECT = auto()  # Choose which keys (s)


class Action(Enum):
    """Migration actions that can be taken on realisation configuration."""

    FILL = auto()
    """Add keys that are missing from the realisation."""
    UPDATE = auto()
    """Update values that differ from the new defaults."""
    TRIM = auto()
    """Remove keys the schema does not recognise."""
    WRITE = auto()
    """Write the migrated realisation to disk."""


class PromptUnavailableError(Exception):
    """Raised when a question needs an answer but there is no terminal to ask."""


def yes_no_always_prompt(raw_prompt: str, allow_select: bool = False) -> Response:
    """Prompt user for a decision, handling y, n, !, N and optionally s.

    Parameters
    ----------
    raw_prompt : str
        Prompt to prepend to options.
    allow_select : bool
        If True, also offer ``s`` to choose which keys the answer applies to.

    Returns
    -------
    Response
        Response from user.

    Raises
    ------
    PromptUnavailableError
        If standard input is closed, e.g. when run in a batch job.
    """
    prompt = f"{raw_prompt} ({'y/n/!/N/s' if allow_select else 'y/n/!/N'}): "
    response_map = {
        "N": Response.NEVER,
        "!": Response.AUTO,
        "A": Response.AUTO,
        "y": Response.YES,
        "n": Response.NO,
    }
    help_text = "y = yes, n = no, ! = yes to this question from now on, N = no to this question from now on"
    if allow_select:
        response_map["s"] = Response.SELECT
        help_text += ", s = choose which keys"
    while True:
        try:
            raw_response = input(prompt).strip()
        except EOFError as e:
            raise PromptUnavailableError(
                "No answer to prompt; rerun with --yes or --check to migrate without prompts."
            ) from e
        if raw_response in response_map:
            return response_map[raw_response]
        console.print(help_text)


def terminal_available() -> bool:
    """Whether standard input is a terminal, so keys can be picked from a list.

    Returns
    -------
    bool
        True if standard input is a terminal.
    """
    return sys.stdin.isatty()


def select_keys(
    question: str, options: list[tuple[KeyPath, str]], ticks: dict[KeyPath, bool]
) -> dict[KeyPath, bool]:
    """Let the user tick which keys to change, with the arrow keys and space.

    Parameters
    ----------
    question : str
        Question to show above the list.
    options : list[tuple[KeyPath, str]]
        Each key, with a label describing its change.
    ticks : dict[KeyPath, bool]
        Whether each key starts ticked.

    Returns
    -------
    dict[KeyPath, bool]
        Whether each key is ticked.
    """
    # Imported here as prompt_toolkit is only needed if the user picks keys.
    import questionary

    choices = [
        questionary.Choice(label, value=path, checked=ticks[path])
        for path, label in options
    ]
    chosen = questionary.checkbox(
        question,
        choices=choices,
        instruction="(● migrate, ○ keep; space: toggle, a: toggle all, enter: done)",
    ).unsafe_ask()
    return {path: path in chosen for path, _ in options}


def confirm_selection() -> Response:
    """Ask whether to apply the keys just selected.

    Returns
    -------
    Response
        `Response.YES` to apply them this time, `Response.AUTO` to apply
        the same selection from now on, or `Response.NO` to go back to
        the question.
    """
    import questionary

    choices = [
        questionary.Choice("Yes (ask again next time)", value=Response.YES),
        questionary.Choice("No (back to the question)", value=Response.NO),
        questionary.Choice(
            "Always (apply this selection from now on)", value=Response.AUTO
        ),
    ]
    return questionary.select("Apply changes?", choices=choices).unsafe_ask()


class Prompter:
    """Ask migration questions, remembering "always" and "never" answers.

    Each question comes with details explaining it, which are printed
    only when the question is not answered from an earlier answer.

    Parameters
    ----------
    assume_yes : bool
        If True, answer every question without prompting. Every action
        is accepted except `Action.UPDATE`, which is accepted only if
        ``overwrite`` is set, so existing values are kept by default.
    overwrite : bool
        With ``assume_yes``, update values that differ from the new
        defaults.
    """

    def __init__(self, assume_yes: bool = False, overwrite: bool = False) -> None:  # noqa: D107
        self.assume_yes = assume_yes
        self.overwrite = overwrite
        self.remembered: dict[tuple[str | None, Action], Response] = {}
        self.selections: dict[tuple[str, Action], dict[KeyPath, bool]] = {}
        """Remembered selections: whether to change each key."""

    def _preset(self, key: str | None, action: Action) -> Response | None:
        response = self.remembered.get((key, action))
        if response is None and self.assume_yes:
            keep = action is Action.UPDATE and not self.overwrite
            response = Response.NEVER if keep else Response.AUTO
        return response

    def _settle(self, key: str | None, action: Action, response: Response) -> bool:
        if response in (Response.AUTO, Response.NEVER):
            self.remembered[(key, action)] = response
        return response in (Response.YES, Response.AUTO)

    def _explain(self, key: str | None, action: Action, details: list[str]) -> None:
        if (key, action) not in self.remembered:
            for line in details:
                console.print(line)

    def ask(
        self,
        question: str,
        action: Action,
        key: str | None = None,
        details: list[str] | None = None,
    ) -> bool:
        """Ask a question, or answer it from a remembered response.

        Parameters
        ----------
        question : str
            Question to ask.
        action : Action
            Action the question asks about.
        key : str | None
            Configuration key the question is about. Remembered
            answers apply to the same action on the same key.
        details : list[str] | None
            Lines explaining the question.

        Returns
        -------
        bool
            True if the action should be taken.
        """
        self._explain(key, action, details or [])
        response = self._preset(key, action) or yes_no_always_prompt(question)
        return self._settle(key, action, response)

    def choose(
        self,
        question: str,
        action: Action,
        key: str,
        options: list[tuple[KeyPath, str]],
        details: list[str],
    ) -> list[KeyPath]:
        """Ask which of several keys an action applies to.

        As well as the answers `ask` accepts, the user can pick keys
        from a list. A remembered selection is reused only when it
        covers every key; otherwise the question is asked again.

        Parameters
        ----------
        question : str
            Question to ask.
        action : Action
            Action the question asks about.
        key : str
            Configuration key the question is about.
        options : list[tuple[KeyPath, str]]
            Keys the action could apply to, each with a plain text
            description of its change.
        details : list[str]
            Lines explaining the question.

        Returns
        -------
        list[KeyPath]
            The keys to apply the action to.
        """
        paths = [path for path, _ in options]
        preset = self._preset(key, action)
        selection = self.selections.get((key, action))
        if preset is None and selection is not None and selection.keys() >= set(paths):
            return [path for path in paths if selection[path]]

        self._explain(key, action, details)
        if preset is not None:
            return paths if self._settle(key, action, preset) else []

        ticks = dict.fromkeys(paths, True)
        while True:
            response = yes_no_always_prompt(question, allow_select=terminal_available())
            if response is not Response.SELECT:
                return paths if self._settle(key, action, response) else []
            ticks = select_keys(question, options, ticks)
            confirmation = confirm_selection()
            if confirmation is Response.NO:
                continue
            if confirmation is Response.AUTO:
                self.selections[(key, action)] = ticks
            return [path for path in paths if ticks[path]]


_MISSING = object()


def get_path(data: Any, path: KeyPath) -> Any:
    """Look up a nested key, returning a sentinel if it is not present.

    Parameters
    ----------
    data : Any
        Nested dictionaries to look in.
    path : KeyPath
        Keys to follow.

    Returns
    -------
    Any
        The value at ``path``, or ``_MISSING``.
    """
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return _MISSING
        data = data[key]
    return data


def set_path(data: dict[str, Any], path: KeyPath, value: Any) -> None:
    """Set a nested key, creating (or replacing non-dict) parents as needed.

    Parameters
    ----------
    data : dict[str, Any]
        Nested dictionaries to modify.
    path : KeyPath
        Keys to follow.
    value : Any
        Value to set.
    """
    for key in path[:-1]:
        if not isinstance(data.get(key), dict):
            data[key] = {}
        data = data[key]
    data[path[-1]] = copy.deepcopy(value)


def remove_path(data: Any, path: KeyPath) -> bool:
    """Remove a nested key.

    Parameters
    ----------
    data : Any
        Nested dictionaries to modify.
    path : KeyPath
        Keys to follow.

    Returns
    -------
    bool
        True if the key was present and has been removed.
    """
    parent = get_path(data, path[:-1])
    if not isinstance(parent, dict) or path[-1] not in parent:
        return False
    del parent[path[-1]]
    return True


def leaf_paths(data: dict[str, Any], prefix: KeyPath = ()) -> list[KeyPath]:
    """List the paths to every non-dictionary value in nested dictionaries.

    Parameters
    ----------
    data : dict[str, Any]
        Nested dictionaries.
    prefix : KeyPath
        Path to prepend to every result.

    Returns
    -------
    list[KeyPath]
        Paths to every leaf value. Empty dictionaries count as leaves.
    """
    paths = []
    for key, value in data.items():
        if isinstance(value, dict) and value:
            paths.extend(leaf_paths(value, (*prefix, key)))
        else:
            paths.append((*prefix, key))
    return paths


def dotted(paths: list[KeyPath]) -> str:
    """Format key paths for display.

    Parameters
    ----------
    paths : list[KeyPath]
        Paths to format.

    Returns
    -------
    str
        Comma separated, dot joined paths.
    """
    return ", ".join(".".join(path) for path in paths)


def compare_section(
    current: Any, new_defaults: dict[str, Any]
) -> tuple[list[KeyPath], list[KeyPath]]:
    """Find how a section differs from the new defaults.

    Parameters
    ----------
    current : Any
        The section in the realisation.
    new_defaults : dict[str, Any]
        The section in the defaults being migrated to.

    Returns
    -------
    list[KeyPath]
        Keys present in the new defaults but not in the realisation.
    list[KeyPath]
        Keys whose value differs from the new defaults.
    """
    missing, different = [], []
    for path in leaf_paths(new_defaults):
        value = get_path(current, path)
        if value is _MISSING:
            missing.append(path)
        elif value != get_path(new_defaults, path):
            different.append(path)
    return missing, different


def describe_change(
    action: Action, path: KeyPath, current: Any, new_defaults: dict[str, Any]
) -> str:
    """Describe a proposed change to one key.

    Parameters
    ----------
    action : Action
        The action that would make the change.
    path : KeyPath
        Path to the key within the section.
    current : Any
        The section in the realisation.
    new_defaults : dict[str, Any]
        The section in the defaults being migrated to.

    Returns
    -------
    str
        The change, as plain text.
    """
    name = ".".join(path)
    new = get_path(new_defaults, path)
    if action is Action.FILL:
        return f"+ {name} = {new!r}"
    return f"{name}: {get_path(current, path)!r} -> {new!r}"


def extract_error(name: str, error: schema.SchemaError) -> tuple[str, list[KeyPath]]:
    """Returns the formatted error string and the paths of any unknown keys.

    Parameters
    ----------
    name : str
        Name of configuration to parse.
    error : schema.SchemaError
        Schema error encountered.

    Returns
    -------
    str
        Human readable error message.
    list[KeyPath]
        Paths to unknown keys identified in the error, relative to
        the configuration section.
    """
    autos = [auto for auto in error.autos if isinstance(auto, str)]
    parents = tuple(
        match.group(1)
        for auto in autos
        if (match := re.match(r"^Key '(.*?)' error", auto))
    )
    location = ".".join((name, *parents))
    last_error = autos[-1] if autos else str(error)

    if match := re.match(r"^Wrong keys? (.*?) in \{", last_error):
        unknown_keys = re.findall(r"'(.*?)'", match.group(1))
        return (
            f"Unknown keys in {location}: [red]{', '.join(unknown_keys)}[/red]",
            [(*parents, key) for key in unknown_keys],
        )

    return f"Error in {location}: {last_error}", []


def validate_section(
    config: ConfigType, section: Any, prompter: Prompter
) -> tuple[list[str], list[str]]:
    """Check a section loads, offering to remove unknown keys until it does.

    Parameters
    ----------
    config : ConfigType
        Configuration the section should load as.
    section : Any
        The section to check. Unknown keys are removed in place.
    prompter : Prompter
        Asks whether to remove unknown keys.

    Returns
    -------
    list[str]
        Descriptions of the changes made.
    list[str]
        Errors that stop the section from loading.
    """
    key = config._config_key
    changes = []
    while True:
        try:
            config.from_dict(copy.deepcopy(section))
            return changes, []
        except schema.SchemaError as schema_error:
            message, unknown_keys = extract_error(key, schema_error)
            if not unknown_keys or not prompter.ask(
                f"  Remove {dotted(unknown_keys)} from {key}?",
                Action.TRIM,
                key,
                [f"  {message}"],
            ):
                return changes, [message]
            # Every removal must succeed, or the loop would not make progress.
            if not all(remove_path(section, path) for path in unknown_keys):
                return changes, [message]
            changes.append(f"{key}: removed {dotted(unknown_keys)}")
        except Exception as error:  # noqa: BLE001
            return changes, [f"Error in {key}: {error}"]


# The questions asked for keys missing from a section, and for keys
# that differ from the new defaults, in the order they are asked.
SECTION_QUESTIONS = [
    (Action.FILL, "Add missing keys to {key}?", "added", "green"),
    (Action.UPDATE, "Update values in {key} to the defaults?", "updated", "yellow"),
]


def fill_section(
    key: str, data: dict[str, Any], new_section: dict[str, Any], prompter: Prompter
) -> list[str]:
    """Offer to bring a section of a realisation in line with the new defaults.

    Parameters
    ----------
    key : str
        Configuration key of the section.
    data : dict[str, Any]
        The realisation, modified in place.
    new_section : dict[str, Any]
        The section in the defaults being migrated to.
    prompter : Prompter
        Asks which keys to add and update.

    Returns
    -------
    list[str]
        Descriptions of the changes made.
    """
    current = data.get(key, _MISSING)
    changes = []
    for paths, (action, question, verb, colour) in zip(
        compare_section(current, new_section), SECTION_QUESTIONS, strict=True
    ):
        if not paths:
            continue
        options = [
            (path, describe_change(action, path, current, new_section))
            for path in paths
        ]
        labels = (
            [f"+ {key} (whole section)"]
            if current is _MISSING
            else [label for _, label in options]
        )
        details = [f"    [{colour}]{escape(label)}[/{colour}]" for label in labels]
        chosen = prompter.choose(
            f"  {question.format(key=key)}", action, key, options, details
        )
        if not chosen:
            continue
        if current is _MISSING and len(chosen) == len(paths):
            changes.append(f"{key}: added from defaults")
        else:
            changes.append(f"{key}: {verb} {dotted(chosen)}")
        if not isinstance(current, dict):
            current = data[key] = {}
        for path in chosen:
            set_path(current, path, get_path(new_section, path))
    return changes


class Status(Enum):
    """Outcome of migrating a realisation."""

    UNCHANGED = auto()
    WRITTEN = auto()
    NOT_WRITTEN = auto()
    """There were changes, but they were not written (declined or dry run)."""
    SKIPPED = auto()
    """The file is not a realisation."""


@dataclasses.dataclass
class MigrationResult:
    """The result of migrating one realisation in memory."""

    path: Path
    original: Any = None
    migrated: Any = None
    changes: list[str] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)
    status: Status = Status.UNCHANGED

    @property
    def changed(self) -> bool:
        """Whether the migrated realisation differs from the original."""
        return self.migrated != self.original


def migrate(
    realisation: Path,
    defaults_version: DefaultsVersion,
    check_configs: list[ConfigType],
    prompter: Prompter,
) -> MigrationResult:
    """Migrate a realisation to a new defaults version in memory.

    Nothing is written to disk; see `write_json`.

    Parameters
    ----------
    realisation : Path
        Path to realisation.
    defaults_version : DefaultsVersion
        Defaults to update to.
    check_configs : list[ConfigType]
        Configurations to check.
    prompter : Prompter
        Asks the user which changes to make.

    Returns
    -------
    MigrationResult
        The original and migrated realisation, with the changes made
        and any errors that remain.
    """
    result = MigrationResult(realisation)
    try:
        with open(realisation, encoding="utf-8") as f:
            result.original = json.load(f)
    except json.JSONDecodeError as error:
        result.errors.append(f"Invalid JSON: {error}")
        return result

    metadata = (
        result.original.get("metadata") if isinstance(result.original, dict) else None
    )
    if not isinstance(metadata, dict):
        result.status = Status.SKIPPED
        return result

    data = copy.deepcopy(result.original)
    result.migrated = data

    old_version = metadata.get("defaults_version")
    new_defaults = default_sections(defaults_version)

    if old_version != defaults_version:
        data["metadata"]["defaults_version"] = str(defaults_version)
        result.changes.append(f"defaults_version: {old_version} -> {defaults_version}")

    for config in check_configs:
        key = config._config_key
        if key in new_defaults:
            result.changes.extend(fill_section(key, data, new_defaults[key], prompter))
        if key in data:
            changes, errors = validate_section(config, data[key], prompter)
            result.changes.extend(changes)
            result.errors.extend(errors)

    return result


def write_json(path: Path, data: Any, backup: str | None) -> None:
    """Write JSON over a file in one step, optionally backing it up first.

    Parameters
    ----------
    path : Path
        File to replace.
    data : Any
        JSON data to write.
    backup : str | None
        If given, copy the original to a file with this suffix first.
    """
    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + backup))
    # Write next to the original and rename, so an interrupted write
    # never leaves a truncated file behind.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as f:
        json.dump(data, f, indent=4)
    shutil.copymode(path, f.name)
    os.replace(f.name, path)


def find_realisations(paths: list[Path], glob: str) -> list[Path]:
    """Expand files and directories into a list of realisation files.

    Parameters
    ----------
    paths : list[Path]
        Files, used as given, and directories, searched recursively.
    glob : str
        Glob pattern for realisations in directories.

    Returns
    -------
    list[Path]
        Realisation files, without duplicates, in the order given.
    """
    found: dict[Path, None] = {}
    for path in paths:
        if path.is_dir():
            found.update(dict.fromkeys(sorted(path.rglob(glob))))
        else:
            found[path] = None
    return list(found)


@cli.from_docstring(app, name="migrate")
def migrate_all(
    paths: Annotated[list[Path], typer.Argument(exists=True)],
    defaults_version: DefaultsVersion,
    glob: str = "*.json",
    backup: str | None = None,
    dry_run: bool = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    check: bool = False,
    overwrite: bool = False,
) -> None:
    """Migrate realisations to a new defaults version.

    Missing keys are filled in, values that differ from the new
    defaults are updated, and unknown keys are removed, after asking.
    Answer s to choose which values to update. Each realisation is
    written once, after you approve its changes.
    Exits with status 1 if any realisation has errors, or with
    --check, if any realisation needs migrating.

    Parameters
    ----------
    paths : list[Path]
        Realisation files, or directories to search for realisations.
    defaults_version : DefaultsVersion
        Defaults version to migrate to.
    glob : str
        Glob pattern to look for realisations in directories.
    backup : str | None
        If given, backup the realisation file with named suffix before
        writing it. Equivalent to the ``-iext`` flag used in sed. Has
        no effect when combined with dry run.
    dry_run : bool
        If given, print instead of writing. Useful to check what would
        be migrated.
    yes : bool
        Accept every change without asking, except updating values
        that differ from the new defaults, which are kept unless
        --overwrite is given.
    check : bool
        Report what would change without asking or writing, and exit
        with status 1 if anything would. Useful in CI or batch jobs.
    overwrite : bool
        With --yes or --check, update values that differ from the new
        defaults instead of keeping them.

    Raises
    ------
    typer.BadParameter
        If both --yes and --check are given, or --overwrite is given
        without either.
    typer.Exit
        If any realisation has errors or, with --check, needs migrating.
    """
    if yes and check:
        raise typer.BadParameter("--yes and --check cannot be used together.")
    if overwrite and not (yes or check):
        raise typer.BadParameter("--overwrite needs --yes or --check.")
    dry_run = dry_run or check
    prompter = Prompter(assume_yes=yes or check, overwrite=overwrite)
    configs = realisation_configurations()
    statuses: Counter[Status] = Counter()
    with_errors = 0
    listed: set[str] = set()

    try:
        for realisation in find_realisations(paths, glob):
            console.print(f"[bold blue]{realisation}[/bold blue]")
            result = migrate(realisation, defaults_version, configs, prompter)
            with_errors += bool(result.errors)
            for error in result.errors:
                console.print(f"  [bold red]{error}[/bold red]")
            if result.status is Status.SKIPPED:
                console.print("  Not a realisation (no metadata), skipping.")
            elif result.changed:
                new_changes = [c for c in result.changes if c not in listed]
                for change in new_changes:
                    console.print(f"  {change}")
                listed.update(new_changes)
                if already_listed := len(result.changes) - len(new_changes):
                    console.print(f"  {already_listed} changes listed above")
                if dry_run:
                    console.print("  DRY RUN: not writing changes.")
                write = not dry_run and prompter.ask(
                    f"  Write changes to {realisation}?", Action.WRITE
                )
                if write:
                    write_json(realisation, result.migrated, backup)
                result.status = Status.WRITTEN if write else Status.NOT_WRITTEN
            statuses[result.status] += 1
    except PromptUnavailableError as error:
        console.print(f"[bold red]{error}[/bold red]")
        raise typer.Exit(code=2) from error

    console.print(
        f"{statuses[Status.WRITTEN]} written, "
        f"{statuses[Status.NOT_WRITTEN]} with changes not written, "
        f"{statuses[Status.UNCHANGED]} unchanged, "
        f"{statuses[Status.SKIPPED]} skipped, "
        f"{with_errors} with errors."
    )
    if with_errors or (check and statuses[Status.NOT_WRITTEN]):
        raise typer.Exit(code=1)


@cli.from_docstring(copy_app)
def copy_configs(
    realisation_template: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    realisation_directory: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    configs: list[str] | None = None,
    backup: str | None = None,
    glob: str = "*.json",
) -> None:
    """Utility to copy blocks of configurations between a template and a directory of realisations.

    Realisation configurations can be partially specified, so that
    some values can be replaced without replacing all of the others.

    Parameters
    ----------
    realisation_template : Path
        Template realisation to copy from.
    realisation_directory : Path
        Directory containing realisation files.
    configs : list[str]
        Configurations to copy. If None, will copy all configurations
        in the template.
    backup : str | None
        If given, backup the realisation file with named suffix before
        copying. Equivalent to the ``-iext`` flag used in sed.
    glob : str
        Glob pattern to look for realisations.

    Raises
    ------
    typer.BadParameter
        If a configuration in ``configs`` is not in the template.
    """
    with open(realisation_template) as f:
        template = json.load(f)

    configs = configs or list(template)
    if unknown := [config for config in configs if config not in template]:
        raise typer.BadParameter(f"Not in {realisation_template}: {', '.join(unknown)}")
    selected = {config: template[config] for config in configs}

    for realisation_path in realisation_directory.rglob(glob):
        with open(realisation_path, encoding="utf-8") as f:
            realisation = json.load(f)

        utils.merge_dictionaries(realisation, selected)
        write_json(realisation_path, realisation, backup)


@cli.from_docstring(clone_app)
def clone(
    realisation_directory: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    num_realisations: int,
    realisation_template: str = "{event}_R{realisation:d}",
    regenerate_seeds: bool = True,
) -> None:
    """Utility to clone realisations with updated seeds.

    Parameters
    ----------
    realisation_directory : Path
        Directory containing realisation files.
    num_realisations : int
        Number of realisations to copy.
    realisation_template : str, optional
        Template structure for realisation names
    regenerate_seeds : bool, optional
        If set, re-roll seeds configuration.
    """

    realisations = defaultdict(set)
    for realisation in realisation_directory.iterdir():
        realisation_path = realisation / "realisation.json"
        parsed_content = parse.parse(realisation_template, realisation.name)
        if not (realisation.is_dir() and realisation_path.exists() and parsed_content):
            continue
        assert isinstance(parsed_content, parse.Result)
        event = parsed_content["event"]
        realisation_number = int(parsed_content["realisation"])
        realisations[event].add(realisation_number)

    for event, existing_realisations in realisations.items():
        base_realisation = min(existing_realisations)
        base_realisation_path = realisation_directory / realisation_template.format(
            event=event, realisation=base_realisation
        )
        for i in range(base_realisation + 1, num_realisations + 1):
            # Handles cases like clarence_R1, clarence_R3 existing already.
            if i in existing_realisations:
                continue
            realisation_path = realisation_directory / realisation_template.format(
                event=event, realisation=i
            )
            shutil.copytree(base_realisation_path, realisation_path)
            if regenerate_seeds:
                realisation_json = realisation_path / "realisation.json"
                seeds = Seeds.random_seeds()
                seeds.write_to_realisation(realisation_json)


if __name__ == "__main__":
    app()
