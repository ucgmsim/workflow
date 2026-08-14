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
from typing import Annotated, cast

import pandas as pd
import typer
from tqdm import tqdm

from workflow.defaults import DefaultsVersion

app = typer.Typer()


# The keys schemas.SEED_SCHEMA and schemas.RUPTURE_PROPAGATION_SCHEMA require. A
# block carrying anything else is rejected here rather than inside the
# subprocess, where a schema failure is one line in captured stderr.
SEED_KEYS = frozenset(
    {
        "nshm_to_realisation_seed",
        "rupture_propagation_seed",
        "genslip_seed",
        "srfgen_seed",
        "hf_seed",
    }
)
RUPTURE_PROPAGATION_KEYS = frozenset(
    {"rupture_causality_tree", "jump_points", "hypocentre"}
)


def generate_one(
    nshmdb_path: Path,
    rupture_id: int,
    realisation_ffp: Path,
    defaults_version: DefaultsVersion,
    seeds: dict[str, int] | None = None,
    rupture_propagation: dict[str, object] | None = None,
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
    rupture_propagation : dict, optional
        When given, written over the generated ``rupture_propagation`` section
        after generation succeeds. The generator has no option to accept one, so
        it derives a section and this replaces it. See
        ``docs/rupture_propagation_reproducibility.md`` for why the derived
        section cannot be reproduced and this carry-over is a stopgap.

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

    if rupture_propagation is not None:
        with open(realisation_ffp, encoding="utf-8") as handle:
            realisation = json.load(handle)
        if "rupture_propagation" not in realisation:
            # Adding the key rather than replacing one would invent a section the
            # generator never derived, so refuse and leave nothing behind.
            realisation_ffp.unlink(missing_ok=True)
            return (
                f"Failed to inherit rupture propagation for rupture {rupture_id}, "
                f"skipping: the generated realisation has no rupture_propagation "
                f"section to replace.\n"
            )
        # Assigning to an existing key preserves its position, so the section
        # order nshm2022-to-realisation wrote is left untouched.
        realisation["rupture_propagation"] = rupture_propagation
        realisation_ffp.write_text(json.dumps(realisation), encoding="utf-8")

    return None


def _read_inherited_section(
    events_dir: Path, rupture_id: int, section: str, required_keys: frozenset[str]
) -> dict[str, object] | None:
    """Read one section of an already-deployed realisation.

    Parameters
    ----------
    events_dir : Path
        Directory holding one ``<rupture id>/realisation.json`` per event.
    rupture_id : int
        The NSHM rupture id to look up.
    section : str
        Top-level key to read.
    required_keys : frozenset of str
        The keys the section must carry exactly.

    Returns
    -------
    dict or None
        The deployed section, or None when the event has no deployed realisation
        or that realisation carries no such section.

    Raises
    ------
    ValueError
        If the section is present but does not carry exactly ``required_keys``.
    """
    realisation_ffp = events_dir / str(rupture_id) / "realisation.json"
    if not realisation_ffp.is_file():
        return None
    with open(realisation_ffp, encoding="utf-8") as handle:
        realisation = json.load(handle)
    inherited = realisation.get(section)
    if not inherited:
        return None
    if inherited.keys() != required_keys:
        missing = sorted(required_keys - inherited.keys())
        unexpected = sorted(inherited.keys() - required_keys)
        raise ValueError(
            f"{realisation_ffp} has a malformed {section} block: "
            f"missing {missing}, unexpected {unexpected}"
        )
    return inherited


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

    Raises
    ------
    ValueError
        If the seed block is present but incomplete. Passing a partial block on
        would count as inherited here and then fail schema validation inside the
        subprocess, where the failure is easy to miss.
    """
    return cast(
        dict[str, int] | None,
        _read_inherited_section(events_dir, rupture_id, "seeds", SEED_KEYS),
    )


def read_inherited_rupture_propagation(
    events_dir: Path, rupture_id: int
) -> dict[str, object] | None:
    """Read the rupture propagation block of an already-deployed realisation.

    Unlike seeds, this section is *derived* content rather than an input, so
    carrying it over means the regenerated file's causality tree, jump points
    and hypocentre are copied rather than reproduced. That is deliberate and
    temporary: the tree is drawn from an unseeded RNG and cannot be reproduced
    at all. See ``docs/rupture_propagation_reproducibility.md``.

    Parameters
    ----------
    events_dir : Path
        Directory holding one ``<rupture id>/realisation.json`` per event.
    rupture_id : int
        The NSHM rupture id to look up.

    Returns
    -------
    dict or None
        The deployed rupture propagation block, or None when the event has no
        deployed realisation or that realisation carries no such section.

    Raises
    ------
    ValueError
        If the section is present but incomplete.
    """
    return _read_inherited_section(
        events_dir, rupture_id, "rupture_propagation", RUPTURE_PROPAGATION_KEYS
    )


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
    inherit_rupture_propagation_from: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help=(
                "Events directory to carry the rupture_propagation section over "
                "from, verbatim. Unlike seeds this is derived content, so the "
                "regenerated causality tree, jump points and hypocentre are "
                "copied rather than reproduced -- a stopgap for an unseeded RNG. "
                "See docs/rupture_propagation_reproducibility.md."
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
    inherit_rupture_propagation_from : Path, optional
        Events directory to carry the ``rupture_propagation`` section over from.
        Omit to keep the section the generator derives.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / "error_log.txt"
    error_log_handle = error_log.open("w", encoding="utf-8")

    df = pd.read_csv(csv_file)
    if "chosen_nshm_id" not in df.columns:
        print(f"CSV file {csv_file} must contain a 'chosen_nshm_id' column.")
        raise typer.Exit(code=1)

    rupture_ids = df["chosen_nshm_id"].dropna().astype(int).tolist()

    if inherit_seeds_from is not None and inherit_rupture_propagation_from is None:
        # Inheriting seeds alone does not reproduce a multi-fault event: around 30
        # of 219 draw a different causality tree every run regardless of seed.
        print(
            "WARNING: inheriting seeds without --inherit-rupture-propagation-from. "
            "Multi-fault causality trees are drawn from an unseeded RNG, so roughly "
            "30 of them will differ from the deployed set on every run. See "
            "docs/rupture_propagation_reproducibility.md."
        )

    inherited_seeds = 0
    inherited_propagation = 0
    for rupture_id in tqdm(rupture_ids, desc="Generating realisations"):
        realisation_ffp = output_dir / f"realisation_{rupture_id}.json"
        try:
            seeds = (
                read_inherited_seeds(inherit_seeds_from, rupture_id)
                if inherit_seeds_from is not None
                else None
            )
            rupture_propagation = (
                read_inherited_rupture_propagation(
                    inherit_rupture_propagation_from, rupture_id
                )
                if inherit_rupture_propagation_from is not None
                else None
            )
        except ValueError as exc:
            error_msg = (
                f"Failed to read inherited content for rupture {rupture_id}, "
                f"skipping: {exc}\n"
            )
            print(f"\n{error_msg}")
            error_log_handle.write(error_msg + "\n")
            error_log_handle.flush()
            continue

        if seeds is not None:
            inherited_seeds += 1
        if rupture_propagation is not None:
            inherited_propagation += 1
        try:
            error_msg = generate_one(
                nshmdb_path,
                rupture_id,
                realisation_ffp,
                defaults_version,
                seeds=seeds,
                rupture_propagation=rupture_propagation,
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
            f"Inherited seeds for {inherited_seeds} of {len(rupture_ids)} rupture(s); "
            f"{len(rupture_ids) - inherited_seeds} drew fresh seeds."
        )
    if inherit_rupture_propagation_from is not None:
        print(
            f"Inherited rupture propagation for {inherited_propagation} of "
            f"{len(rupture_ids)} rupture(s); "
            f"{len(rupture_ids) - inherited_propagation} kept the derived section."
        )
    print(f"Error log written to {error_log}")


if __name__ == "__main__":
    app()