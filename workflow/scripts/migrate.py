"""Check that realisation can be loaded, if it can't automatically trim extraneous tags and offer to fill in default values."""

import difflib
import inspect
import json
import re
import shutil
from collections import defaultdict
from collections.abc import MutableMapping
from enum import Enum, auto
from pathlib import Path
from typing import Annotated, TypeGuard

import parse
import schema
import typer
from rich.console import Console

from qcore import cli
from workflow import realisations, utils
from workflow.defaults import DefaultsVersion
from workflow.realisations import Seeds

app = typer.Typer()
console = Console()


# Every use site refers to a RealisationConfiguration *subclass* (the classes
# returned by realisation_configurations), not an instance of one.
type ConfigType = type[realisations.RealisationConfiguration]


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


class Response(Enum):
    """Enum for response to prompts asked of user."""

    YES = auto()
    NO = auto()
    AUTO = auto()  # Always (!)
    NEVER = auto()  # Never (N)


class Action(Enum):
    """Migration actions that can be taken on realisation configuration."""

    MIGRATE = auto()
    TRIM = auto()
    FILL = auto()
    UPDATE = auto()


def yes_no_always_prompt(raw_prompt: str) -> Response:
    """Prompt user for a decision, handling y, n, !, and N.


    Parameters
    ----------
    raw_prompt : str
        Prompt to prepend to options.


    Returns
    -------
    Response
        Response from user.
    """

    prompt = f"{raw_prompt} (y/n/!/N): "
    response_map = {
        "N": Response.NEVER,
        "!": Response.AUTO,
        "A": Response.AUTO,
        "y": Response.YES,
        "n": Response.NO,
    }
    while True:
        raw_response = input(prompt).strip()
        if raw_response in response_map:
            return response_map[raw_response]


def autofill(
    realisation: Path,
    config: realisations.RealisationConfiguration,
    dry_run: bool,
) -> None:
    """Autofill realisation with defaults from config.

    Parameters
    ----------
    realisation : Path
        Realisation to write to.
    config : realisations.RealisationConfiguration
        Config to write.
    dry_run : bool
        If True, print to console instead of writing.
    """
    if dry_run:
        console.print(
            f"DRY RUN: Would merge with {config.__class__.__name__} defaults in {realisation}"
        )
    else:
        config.write_to_realisation(realisation)


def extract_error(
    name: str, schema: schema.Schema, e: schema.SchemaError
) -> tuple[str, list[str]]:
    """Returns the formatted error string and a list of extraneous keys found.


    Parameters
    ----------
    name : str
        Name of configuration to parse.
    schema : schema.Schema
        Schema to read.
    e : schema.SchemaError
        Schema error encountered.


    Returns
    -------
    str
        Human readable error message.
    list[str]
        Unknown keys identified in error.
    """

    path_segments = [str(a) for a in e.autos if isinstance(a, str)]
    keys = []
    for segment in path_segments:
        if match := re.match(r"^Key '(.*?)'", segment):
            keys.append(match.group(1))

    last_error = e.autos[-1] if e.autos else str(e)
    extraneous_keys = []
    assert isinstance(last_error, str)
    if "Wrong keys" in last_error:
        extraneous_keys = re.findall(r"'(.*?)'", last_error.split(" in {")[0])
        error_msg = f"Extraneous keys found: [red]{', '.join(extraneous_keys)}[/red]"
        return f"Error in {name}: {error_msg}", extraneous_keys

    if match := re.match(r"^Wrong key '(.*?)'", last_error):
        unknown_key = match.group(1)
        return f"Error in {name}: Unknown key '{unknown_key}'", [unknown_key]

    return f"Error in {name}: {last_error}", []


def should_trim_keys(config: ConfigType, extra_keys: list[str]) -> Response:
    """Prompts user if they want to trim extra keys.

    Parameters
    ----------
    config : ConfigType
        Config to trim keys from.
    extra_keys : list[str]
        Extra keys to trim.

    Returns
    -------
    Response
        Response from user to prompt.
    """
    return yes_no_always_prompt(
        f"Remove extraneous keys {extra_keys} from {config._config_key}?"
    )


def should_update(config: ConfigType) -> Response:
    """Prompt user to merge config with default values.

    Parameters
    ----------
    config : ConfigType
        Config to merge with.

    Returns
    -------
    Response
        Response from user to prompt.
    """
    return yes_no_always_prompt(f"Merge with defaults for {config._config_key}?")


RENAMED_KEYS: dict[str, dict[str, str]] = {
    realisations.Seeds._config_key: {"genslip_seed": "rupture_seed"},
}
"""Keys that changed name, by config key, as ``{old: new}``.

A renamed key cannot be handled by trimming and refilling like every other change:
:class:`~workflow.realisations.Seeds` has no defaults to refill *from*, and generating a
fresh block would change every other seed in it -- including `hf_seed`, which nothing
about this rename touches. So the value is carried over to its new name instead.
"""


def rename_keys(realisation: Path, dry_run: bool) -> None:
    """Carry values in `RENAMED_KEYS` over to the names they are read under now.

    Parameters
    ----------
    realisation : Path
        Path to realisation.
    dry_run : bool
        If True, print instead of renaming.
    """
    with open(realisation, "r") as f:
        data = json.load(f)

    renamed = []
    for config_key, renames in RENAMED_KEYS.items():
        config_data = data.get(config_key)
        if not isinstance(config_data, dict):
            continue
        for old, new in renames.items():
            if old in config_data and new not in config_data:
                config_data[new] = config_data.pop(old)
                renamed.append(f"{config_key}.{old} -> {config_key}.{new}")

    if not renamed:
        return

    if dry_run:
        console.print(f"DRY RUN: Would rename {', '.join(renamed)} in {realisation}")
        return

    console.print(f"Renaming {', '.join(renamed)} in {realisation}")
    with open(realisation, "w") as f:
        json.dump(data, f, indent=4)


def trim_keys(
    realisation: Path,
    config: ConfigType,
    extra_keys: list[str],
    dry_run: bool,
) -> None:
    """Trim extra keys from realisation.

    Parameters
    ----------
    realisation : Path
        Path to realisation.
    config : ConfigType
        Config to trim from.
    extra_keys : list[str]
        Keys to trim.
    dry_run : bool
        If True, print instead of trimming.
    """
    if dry_run:
        console.print(f"DRY RUN: Would remove {extra_keys} from {realisation}")
    else:
        with open(realisation, "r") as f:
            data = json.load(f)

        config_data = data[config._config_key]
        for k in extra_keys:
            config_data.pop(k, None)

        with open(realisation, "w") as f:
            json.dump(data, f, indent=4)


def print_diff(config_a: dict, config_b: dict) -> None:
    """Pretty print diff between two dictionaries.

    Parameters
    ----------
    config_a : dict
        Dictionary a.
    config_b : dict
        Dictionary b.
    """
    config_a_str = json.dumps(config_a, indent=4, default=realisations.path_serialiser)
    config_b_str = json.dumps(config_b, indent=4, default=realisations.path_serialiser)

    diff = difflib.context_diff(
        config_a_str.splitlines(keepends=True),
        config_b_str.splitlines(keepends=True),
        fromfile="Current",
        tofile="Defaults",
    )

    for line in diff:
        if line.startswith("+ "):
            console.print(f"[green]{line}[/green]", end="")
        elif line.startswith("- "):
            console.print(f"[red]{line}[/red]", end="")
        elif line.startswith("!"):
            console.print(f"[yellow]{line}[/yellow]", end="")
        else:
            console.print(line, end="")


def migrate(
    realisation: Path,
    defaults_version: DefaultsVersion,
    check_configs: list[ConfigType],
    defaults: dict[ConfigType, realisations.RealisationConfiguration],
    auto_response: MutableMapping[tuple[ConfigType, Action], Response],
    dry_run: bool,
) -> None:
    """Attempt to migrate realisation to new defaults set.

    Parameters
    ----------
    realisation : Path
        Path to realisation.
    defaults_version : DefaultsVersion
        Defaults to update to.
    check_configs : list[ConfigType]
        Configurations to check.
    defaults : dict[ConfigType, realisations.RealisationConfiguration]
        Defaults to use.
    auto_response : MutableMapping[tuple[ConfigType, Action], Response]
        Auto response map recording user's always and never requests.
    dry_run : bool
        If True, print instead of writing to realisations.
    """
    metadata = realisations.RealisationMetadata.read_from_realisation(realisation)
    if metadata.defaults_version != defaults_version:
        console.print(
            f"Updating defaults in {realisation} from {metadata.defaults_version} to {defaults_version}"
        )
        if not dry_run:
            metadata.defaults_version = defaults_version
            metadata.write_to_realisation(realisation)
    try:
        with open(realisation, "r") as f:
            json_data = json.load(f)
    except json.JSONDecodeError:
        console.print(
            f"[bold red]Invalid JSON in {realisation}, skipping...[/bold red]"
        )
        return

    rename_keys(realisation, dry_run)
    if not dry_run:
        with open(realisation, "r") as f:
            json_data = json.load(f)

    for config in check_configs:
        default_config = defaults.get(config)
        # A config with no defaults has nothing to diff against, but it can still carry
        # keys the schema no longer knows, so it goes on to the read below.
        default_config_dict = default_config.to_dict() if default_config else None
        current_config = json_data.get(config._config_key, {})
        if default_config_dict is not None and current_config != default_config_dict:
            print_diff(current_config, default_config_dict)
            print()
            response = auto_response.get((config, Action.UPDATE)) or should_update(
                config
            )

            if response in (Response.AUTO, Response.NEVER):
                auto_response[(config, Action.UPDATE)] = response

            if response in (response.AUTO, response.YES):
                autofill(
                    realisation,
                    default_config,
                    dry_run=dry_run,
                )

        try:
            _ = config.read_from_realisation(realisation)
        except realisations.RealisationParseError:
            if config not in defaults and config != realisations.Seeds:
                console.print(
                    f"[bold red]Missing required configuration {config.__class__.__name__}[/bold red]"
                )
        except schema.SchemaError as error:
            console.print(f"[red]Schema error for {realisation}[/red]")

            default_config = defaults.get(config)
            error, extra_keys = extract_error(config._config_key, config._schema, error)
            console.print(error)
            if extra_keys:
                response = auto_response.get((config, Action.TRIM)) or should_trim_keys(
                    config, extra_keys
                )

                if response in (Response.AUTO, Response.NEVER):
                    auto_response[(config, Action.TRIM)] = response

                if response in (response.AUTO, response.YES):
                    trim_keys(realisation, config, extra_keys, dry_run)
                    # Try to read one more time
                    try:
                        _ = config.read_from_realisation(realisation)
                    except schema.SchemaError as error:
                        error, _ = extract_error(
                            config._config_key, config._schema, error
                        )
                        console.print(
                            f"[bold red]Unrecoverable schema error for {realisation}[/bold red]"
                        )
                        console.print(error)

        except Exception as e:  # noqa: BLE001
            console.print(
                f"[bold red]Could not load realisation {realisation} for unrecoverable reason:[/bold red]"
            )
            console.print(str(e))


@cli.from_docstring(app, name="migrate")
def migrate_all(
    realisation_directory: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    defaults_version: DefaultsVersion,
    glob: str = "*.json",
    backup: str | None = None,
    dry_run: bool = False,
) -> None:
    """Migrate all realisations in a directory to the current workflow version.

    Parameters
    ----------
    realisation_directory : Path
        Path containing realisations.
    defaults_version : DefaultsVersion
        Defaults version to migrate to.
    glob : str
        Glob pattern to look for realisations.
    backup : str | None
        If given, backup the realisation file with named suffix before
        running migration. Equivalent to the ``-iext`` flag used in
        sed. Has no effect when combined with dry run.
    dry_run : bool
        If given, print instead of writing. Useful to check what would
        be migrated.
    """
    auto_response = {}
    configs = realisation_configurations()
    defaults = loadable_defaults(configs, defaults_version)

    for realisation in realisation_directory.rglob(glob):
        if backup and not dry_run:  # only make a copy if we actually modify the file.
            shutil.copy(
                realisation, realisation.with_suffix(realisation.suffix + backup)
            )

        migrate(
            realisation,
            defaults_version,
            configs,
            defaults,
            auto_response,
            dry_run,
        )


@cli.from_docstring(app)
def copy(
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
        in realisation file.
    backup : str | None
        If given, backup the realisation file with named suffix before
        running migration. Equivalent to the ``-iext`` flag used in
        sed. Has no effect when combined with dry run.
    glob : str
        Glob pattern to look for realisations.
    """
    with open(realisation_template) as f:
        template = json.load(f)

    configs = configs or list(template)

    for realisation_path in realisation_directory.rglob(glob):
        if backup:
            shutil.copy(
                realisation_path,
                realisation_path.with_suffix(realisation_path.suffix + backup),
            )

        with open(realisation_path) as f:
            realisation = json.load(f)

        utils.merge_dictionaries(realisation, template)

        with open(realisation_path, "w") as f:
            json.dump(realisation, f, indent=4)


@cli.from_docstring(app)
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
        if not (realisation.is_dir and realisation_path.exists() and parsed_content):
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
