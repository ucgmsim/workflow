"""Check that realisation can be loaded, if it can't automatically trim extraneous tags and offer to fill in default values."""

import difflib
import inspect
import json
import re
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
    prompt = f"[bold]{raw_prompt}[/bold] (y/n/!/N): "
    while True:
        # Use console.input to support rich markup in the prompt
        raw_response = console.input(prompt).strip()

        # Exact match for case-sensitive 'N' (Never)
        if raw_response == "N":
            return Response.NEVER
        elif raw_response == "!":
            return Response.AUTO
        elif raw_response.lower() == "y":
            return Response.YES
        elif raw_response.lower() == "n":
            return Response.NO


def prompt_autofill(
    realisation: Path,
    config: realisations.RealisationConfiguration,
    auto_state: Response | None,
    dry_run: bool,
) -> Response:
    response = auto_state

    if response not in (Response.AUTO, Response.NEVER):
        response = yes_no_always_prompt(
            f"Defaults are available for {config.__class__.__name__}, autofill?"
        )

    if response in (Response.YES, Response.AUTO):
        if dry_run:
            console.print(f"[magenta]DRY RUN: Would autofill {realisation}[/magenta]")
        else:
            config.write_to_realisation(realisation)

    return response


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
    table_path = f"[bold cyan]{name}[/bold cyan]"
    extraneous_keys = []

    # Handle multiple wrong keys: "Wrong keys 'dt', 'resolution' in..."
    if "Wrong keys" in last_error:
        # Extract everything between single quotes
        extraneous_keys = re.findall(r"'(.*?)'", last_error.split(" in {")[0])
        error_msg = (
            f"Extraneous keys found: [bold red]{', '.join(extraneous_keys)}[/bold red]"
        )
        return f"Error in {table_path}: {error_msg}", extraneous_keys

    # Fallback to existing logic for "Wrong key" (singular/typo)
    if match := re.match(r"^Wrong key '(.*?)'", last_error):
        unknown_key = match.group(1)
        # ... (keep your existing fuzzy matching logic here) ...
        return f"Error in {table_path}: Unknown key '{unknown_key}'", [unknown_key]

    return f"Error in {table_path}: {last_error}", []


def prompt_migrate(
    realisation: Path,
    config: type,
    error: schema.SchemaError,
    defaults: realisations.RealisationConfiguration | None,
    auto_state: Response | None,
    dry_run: bool,
) -> Response:
    assert issubclass(config, realisations.RealisationConfiguration)
    name = config._config_key

    console.print(f"[red]Error loading {name}:[/red]")
    error_msg, extraneous_keys = extract_error(
        config._config_key, config._schema, error
    )
    console.print(error_msg)

    response = auto_state
    if extraneous_keys:
        if response not in (Response.AUTO, Response.NEVER):
            response = yes_no_always_prompt(
                f"Remove extraneous keys {extraneous_keys}?"
            )

        if response in (Response.YES, Response.AUTO):
            if dry_run:
                console.print(
                    f"[magenta]DRY RUN: Would remove {extraneous_keys} from {realisation}[/magenta]"
                )
            else:
                # Load raw data, delete keys, save back
                with open(realisation, "r") as f:
                    data = json.load(f)

                # Note: This logic assumes keys are at the root or you'd need
                # to traverse based on the 'keys' list from pprint_error
                config_data = data[config._config_key]
                for k in extraneous_keys:
                    config_data.pop(k, None)

                with open(realisation, "w") as f:
                    json.dump(data, f, indent=4)
                console.print(f"[green]Successfully trimmed {realisation}[/green]")
            return response
    if defaults:
        if response not in (Response.AUTO, Response.NEVER):
            response = yes_no_always_prompt(
                "Defaults are available, replace with defaults?"
            )

        if response in (Response.YES, Response.AUTO):
            if dry_run:
                console.print(
                    f"[magenta]DRY RUN: Would migrate defaults to {realisation}[/magenta]"
                )
            else:
                defaults.write_to_realisation(realisation)
    else:
        return response or Response.NO

    return response


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


def prompt_update(
    realisation: Path,
    loaded_config: realisations.RealisationConfiguration,
    default_config: realisations.RealisationConfiguration,
    auto_state: Response | None,
    dry_run: bool,
) -> Response:
    loaded_conf_dict = loaded_config.to_dict()
    default_dict = default_config.to_dict()
    response = auto_state

    if loaded_conf_dict != default_dict:
        console.print("[yellow]Defaults differ from saved value:[/yellow]")
        print_diff(loaded_conf_dict, default_dict)

        if response not in (Response.AUTO, Response.NEVER):
            response = yes_no_always_prompt("Accept defaults?")

        if response in (Response.YES, Response.AUTO):
            if dry_run:
                console.print(
                    f"[magenta]DRY RUN: Would update {realisation} with defaults[/magenta]"
                )
            else:
                default_config.write_to_realisation(realisation)
    else:
        return response or Response.NO

    return response


def migrate(
    realisation: Path,
    defaults_version: DefaultsVersion,
    check_configs: list[type],
    defaults: dict[type, realisations.RealisationConfiguration],
    auto_fill: dict[type, Response],
    auto_migrate: dict[type, Response],
    auto_update: dict[type, Response],
    dry_run: bool,
) -> None:
    metadata = realisations.RealisationMetadata.read_from_realisation(realisation)
    if metadata.defaults_version != defaults_version:
        console.print(
            f"[magenta]Updating defaults in {realisation} from {metadata.defaults_version} to {defaults}[/magenta]"
        )
        if not dry_run:
            metadata.defaults_version = defaults_version
            metadata.write_to_realisation(realisation)

    for config in check_configs:
        if not issubclass(config, realisations.RealisationConfiguration):
            raise TypeError(
                f"{config=} should be a subclass of realisations.RealisationConfiguration"
            )
        else:
            try:
                loaded_config = config.read_from_realisation(realisation)
                if default_config := defaults.get(config):
                    response = prompt_update(
                        realisation,
                        loaded_config,
                        default_config,
                        auto_update.get(config),
                        dry_run,
                    )
                    if response in (Response.AUTO, Response.NEVER):
                        auto_update[config] = response

            except realisations.RealisationParseError:
                if default_config := defaults.get(config):
                    response = prompt_autofill(
                        realisation,
                        default_config,
                        auto_state=auto_fill.get(config),
                        dry_run=dry_run,
                    )
                    if response in (Response.AUTO, Response.NEVER):
                        auto_fill[config] = response

            except schema.SchemaError as error:
                default_config = defaults.get(config)
                response = prompt_migrate(
                    realisation,
                    config,
                    error,
                    default_config,
                    auto_state=auto_migrate.get(config),
                    dry_run=dry_run,
                )
                if response in (Response.AUTO, Response.NEVER):
                    auto_migrate[config] = response

            except Exception as e:  # noqa: BLE001
                console.print(
                    f"[bold red]Could not load realisation {realisation} for unrecoverable reason:[/bold red]"
                )
                console.print(str(e))
                console.print("[yellow]Skipping[/yellow]")


@app.command()
def migrate_all(
    realiasation_directory: Path,
    defaults_version: DefaultsVersion,
    glob: str = "*.json",
    dry_run: bool = False,
) -> None:
    if dry_run:
        console.print(
            "[bold magenta]*** RUNNING IN DRY RUN MODE - NO FILES WILL BE MODIFIED ***[/bold magenta]"
        )

    auto_fill: dict[type, Response] = {}
    auto_migrate: dict[type, Response] = {}
    auto_update: dict[type, Response] = {}

    configs = realisation_configurations()
    defaults = loadable_defaults(configs, defaults_version)

    for realisation in realiasation_directory.rglob(glob):
        migrate(
            realisation,
            defaults_version,
            configs,
            defaults,
            auto_fill,
            auto_migrate,
            auto_update,
            dry_run,
        )


if __name__ == "__main__":
    app()
