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

    Returns
    -------
    str or None
        An error message if generation failed, otherwise None.
    """
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


@app.command()
def generate_realisations_from_csv(
    nshmdb_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    csv_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument(writable=True)],
    defaults_version: Annotated[DefaultsVersion, typer.Argument()],
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / "error_log.txt"
    error_log_handle = error_log.open("w", encoding="utf-8")

    df = pd.read_csv(csv_file)
    if "chosen_nshm_id" not in df.columns:
        print(f"CSV file {csv_file} must contain a 'chosen_nshm_id' column.")
        raise typer.Exit(code=1)

    rupture_ids = df["chosen_nshm_id"].dropna().astype(int).tolist()

    for rupture_id in tqdm(rupture_ids, desc="Generating realisations"):
        realisation_ffp = output_dir / f"realisation_{rupture_id}.json"
        try:
            error_msg = generate_one(
                nshmdb_path, rupture_id, realisation_ffp, defaults_version
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
    print(f"Error log written to {error_log}")


if __name__ == "__main__":
    app()