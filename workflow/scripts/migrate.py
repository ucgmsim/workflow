"""Check that realisation can be loaded, if it can't automatically trim extraneous tags and offer to fill in default values."""

import difflib
import inspect
import json
import re
from collections.abc import MutableMapping
from enum import Enum, auto
from pathlib import Path

import schema
import typer
from rich.console import Console

from workflow import realisations
from workflow.defaults import DefaultsVersion

app = typer.Typer()
console = Console()


class Response(Enum):
    YES = auto()
    NO = auto()
    AUTO = auto()  # Always (!)
    NEVER = auto()  # Never (N)


class Action(Enum):
    MIGRATE = auto()
    TRIM = auto()
    FILL = auto()
    UPDATE = auto()


def is_realisation_configuration(cls: type) -> bool:
    return (
        cls != realisations.RealisationConfiguration
        and inspect.isclass(cls)
        and issubclass(cls, realisations.RealisationConfiguration)
    )


def realisation_configurations() -> list[type]:
    return [
        cls
        for name, cls in inspect.getmembers(realisations)
        if is_realisation_configuration(cls)
    ]


def loadable_defaults(
    configurations: list[type], defaults: DefaultsVersion
) -> dict[type, realisations.RealisationConfiguration]:
    config_defaults = {}
    for config in configurations:
        if not issubclass(config, realisations.RealisationConfiguration):
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


def yes_no_always_prompt(raw_prompt: str) -> Response:
    """Prompt user for a decision, handling y, n, !, and N."""
    prompt = f"{raw_prompt} (y/n/!/N): "
    response_map = {
        "N": Response.NEVER,
        "!": Response.AUTO,
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
    if dry_run:
        console.print(
            f"DRY RUN: Would merge with {config.__class__.__name__} defaults in {realisation}"
        )
    else:
        config.write_to_realisation(realisation)


def extract_error(
    name: str, schema: schema.Schema, e: schema.SchemaError
) -> tuple[str, list[str]]:
    """Returns the formatted error string and a list of extraneous keys found."""
    path_segments = [str(a) for a in e.autos if isinstance(a, str)]
    keys = []
    for segment in path_segments:
        if match := re.match(r"^Key '(.*?)'", segment):
            keys.append(match.group(1))

    last_error = e.autos[-1] if e.autos else str(e)
    extraneous_keys = []

    # Handle multiple wrong keys: "Wrong keys 'dt', 'resolution' in..."
    if "Wrong keys" in last_error:
        # Extract everything between single quotes
        extraneous_keys = re.findall(r"'(.*?)'", last_error.split(" in {")[0])
        error_msg = f"Extraneous keys found: [red]{', '.join(extraneous_keys)}[/red]"
        return f"Error in {name}: {error_msg}", extraneous_keys

    # Fallback to existing logic for "Wrong key" (singular/typo)
    if match := re.match(r"^Wrong key '(.*?)'", last_error):
        unknown_key = match.group(1)
        # ... (keep your existing fuzzy matching logic here) ...
        return f"Error in {name}: Unknown key '{unknown_key}'", [unknown_key]

    return f"Error in {name}: {last_error}", []


def should_trim_keys(config: type, extra_keys: list[str]) -> Response:
    return yes_no_always_prompt(f"Remove extraneous keys {extra_keys}?")


def should_update(config: type) -> Response:
    return yes_no_always_prompt(f"Merge with defaults for {config._config_key}?")


def trim_keys(
    realisation: Path,
    config: type,
    extra_keys: list[str],
    dry_run: bool,
) -> None:

    if dry_run:
        console.print(f"DRY RUN: Would remove {extra_keys} from {realisation}")
    else:
        # Load raw data, delete keys, save back
        with open(realisation, "r") as f:
            data = json.load(f)

        config_data = data[config._config_key]
        for k in extra_keys:
            config_data.pop(k, None)

        with open(realisation, "w") as f:
            json.dump(data, f, indent=4)


def print_diff(config_a: dict, config_b: dict) -> None:
    config_a_str = json.dumps(config_a, indent=4)
    config_b_str = json.dumps(config_b, indent=4)

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
    check_configs: list[type],
    defaults: dict[type, realisations.RealisationConfiguration],
    auto_response: MutableMapping[tuple[type, Action], Response],
    dry_run: bool,
) -> None:
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

    for config in check_configs:
        if not issubclass(config, realisations.RealisationConfiguration):
            raise TypeError(
                f"{config=} should be a subclass of realisations.RealisationConfiguration"
            )
        elif default_config := defaults.get(config):
            default_config_dict = default_config.to_dict()
            current_config = json_data.get(config._config_key, dict())
            if current_config != default_config_dict:
                print_diff(current_config, default_config_dict)
                print("")
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

        # Basic validation complete, now try to resolve schema errors
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


@app.command()
def migrate_all(
    realiasation_directory: Path,
    defaults_version: DefaultsVersion,
    glob: str = "*.json",
    dry_run: bool = False,
) -> None:

    auto_response = dict()
    configs = realisation_configurations()
    defaults = loadable_defaults(configs, defaults_version)

    for realisation in realiasation_directory.rglob(glob):
        migrate(
            realisation,
            defaults_version,
            configs,
            defaults,
            auto_response,
            dry_run,
        )


if __name__ == "__main__":
    app()
