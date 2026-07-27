#!/usr/bin/env python3
"""Generate Realisations From CSV.

Description
-----------
Reads a CSV file containing NSHM rupture IDs (with a ``chosen_nshm_id`` column)
and calls ``nshm2022-to-realisation`` for each one, writing the resulting
realisation files into an output directory.

Inputs
------
1. A copy of the NSHM 2022 database (``nshmdb.db``).
2. A CSV file with a ``chosen_nshm_id`` column listing rupture IDs.
3. An output directory where realisation files will be written.
4. The version of the scientific defaults to use.

Outputs
-------
One realisation YAML file per rupture ID, written to the output directory.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer
using the ``generate-realisations-from-csv`` command which is installed after
running ``pip install workflow@git+https://github.com/ucgmsim/workflow``.

Usage
-----
``generate-realisations-from-csv NSHM_DB_FILE CSV_FILE OUTPUT_DIR DEFAULTS_VERSION``

For More Help
-------------
See the output of ``generate-realisations-from-csv --help``.
"""

import json
import subprocess
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from tqdm import tqdm

from workflow.defaults import DefaultsVersion

app = typer.Typer()


def generate_one(
    nshmdb_path: Path,
    rupture_id: int,
    realisation_ffp: Path,
    defaults_version: DefaultsVersion,
    seeds: dict[str, int] | None = None,
) -> str | None:
    """Generate one minimal realisation stub for a rupture.

    Parameters
    ----------
    nshmdb_path : Path
        Path to the NSHM 2022 database file.
    rupture_id : int
        The NSHM rupture id to generate.
    realisation_ffp : Path
        Path the realisation stub is written to.
    defaults_version : DefaultsVersion
        Scientific default parameters version to use.
    seeds : dict of str to int, optional
        When given, written into the stub before generation so
        nshm2022-to-realisation replays these seeds via
        ``Seeds.read_from_realisation_or_random`` instead of drawing fresh ones.

    Returns
    -------
    str or None
        An error message if generation failed, otherwise None.
    """
    if seeds is not None:
        realisation_ffp.write_text(
            json.dumps({"metadata": {}, "seeds": seeds}), encoding="utf-8"
        )
    cmd = [
        "nshm2022-to-realisation",
        str(nshmdb_path),
        str(rupture_id),
        str(realisation_ffp),
        str(defaults_version),
        "--dip-delta",
        "20",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # metadata and seeds are written before the point at which a disconnected
        # rupture graph fails, so a crash leaves a source-less file behind.
        realisation_ffp.unlink(missing_ok=True)
        return (
            f"Failed to generate realisation for rupture {rupture_id}, skipping:\n"
            f"--- stdout ---\n{exc.stdout}\n"
            f"--- stderr ---\n{exc.stderr}\n"
            f"--- return code: {exc.returncode} ---\n"
        )
    return None


def read_inherited_seeds(events_dir: Path, rupture_id: int) -> dict[str, int] | None:
    """Read the seed block of an already-deployed realisation.

    Only the ``seeds`` key is taken. Every other field in the deployed file is
    ignored, so the regenerated realisation is derived fresh from the database
    and the current code rather than inherited.

    Parameters
    ----------
    events_dir : Path
        Directory holding one ``<rupture id>/realisation.json`` per event.
    rupture_id : int
        The NSHM rupture id to look up.

    Returns
    -------
    dict of str to int, or None
        The deployed seed block, or None when the event has no deployed
        realisation or that realisation carries no seeds. None makes the caller
        fall back to a fresh random draw, which is correct for a brand-new event.
    """
    realisation_ffp = events_dir / str(rupture_id) / "realisation.json"
    if not realisation_ffp.is_file():
        return None
    with open(realisation_ffp, encoding="utf-8") as handle:
        realisation = json.load(handle)
    return realisation.get("seeds") or None


@app.command()
def generate_realisations_from_csv(
    nshmdb_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    csv_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument(writable=True)],
    defaults_version: Annotated[DefaultsVersion, typer.Argument()],
    inherit_seeds_from: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help=(
                "Events directory to inherit seeds from. Each rupture reuses the "
                "seeds recorded in <events dir>/<rupture id>/realisation.json; "
                "ruptures with no deployed file get a fresh random draw."
            ),
        ),
    ] = None,
) -> None:
    """Generate realisation stub files for every rupture ID listed in a CSV file.

    Parameters
    ----------
    nshmdb_path : Path
        Path to the NSHM 2022 database file (``nshmdb.db``).
    csv_file : Path
        Path to a CSV file containing a ``chosen_nshm_id`` column with rupture IDs.
    output_dir : Path
        Directory where the generated realisation files will be written.
    defaults_version : DefaultsVersion
        Scientific default parameters version to use.
    inherit_seeds_from : Path, optional
        Events directory to inherit seeds from. Omit to draw fresh seeds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / "error_log.txt"
    error_log_handle = error_log.open("w", encoding="utf-8")

    df = pd.read_csv(csv_file)
    if "chosen_nshm_id" not in df.columns:
        print(f"CSV file {csv_file} must contain a 'chosen_nshm_id' column.")
        raise typer.Exit(code=1)

    rupture_ids = df["chosen_nshm_id"].dropna().astype(int).tolist()

    inherited = 0
    for rupture_id in tqdm(rupture_ids, desc="Generating realisations"):
        realisation_ffp = output_dir / f"realisation_{rupture_id}.json"
        seeds = (
            read_inherited_seeds(inherit_seeds_from, rupture_id)
            if inherit_seeds_from is not None
            else None
        )
        if seeds is not None:
            inherited += 1
        try:
            error_msg = generate_one(
                nshmdb_path, rupture_id, realisation_ffp, defaults_version, seeds=seeds
            )
        except FileNotFoundError:
            print(
                "\n'nshm2022-to-realisation' command not found. "
                "Is the workflow package installed?"
            )
            raise typer.Exit(code=1)
        if error_msg is not None:
            print(f"\n{error_msg}")
            error_log_handle.write(error_msg + "\n")
            error_log_handle.flush()

    error_log_handle.close()
    print(f"\nDone. Processed {len(rupture_ids)} rupture ID(s).")
    if inherit_seeds_from is not None:
        print(
            f"Inherited seeds for {inherited} of {len(rupture_ids)} rupture(s); "
            f"{len(rupture_ids) - inherited} drew fresh seeds."
        )
    print(f"Error log written to {error_log}")


if __name__ == "__main__":
    app()