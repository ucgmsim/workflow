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

import enum
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from tqdm import tqdm

from workflow.defaults import DefaultsVersion

app = typer.Typer()


# Top-level keys each inheritable section must carry, per workflow.schemas. A
# block carrying anything else is rejected here rather than inside the
# subprocess, where a schema failure is one line in captured stderr.
SECTION_KEYS: dict[str, frozenset[str]] = {
    "seeds": frozenset(
        {
            "nshm_to_realisation_seed",
            "rupture_propagation_seed",
            "genslip_seed",
            "srfgen_seed",
            "hf_seed",
        }
    ),
    "sources": frozenset({"source_geometries"}),
    "rupture_propagation": frozenset(
        {"rupture_causality_tree", "jump_points", "hypocentre"}
    ),
    "magnitudes": frozenset({"magnitudes"}),
    "rakes": frozenset({"rakes"}),
}


class InheritableSection(str, enum.Enum):
    """A realisation section ``generate-realisations-from-csv`` can carry over."""

    seeds = "seeds"
    sources = "sources"
    rupture_propagation = "rupture_propagation"
    magnitudes = "magnitudes"
    rakes = "rakes"


def generate_one(
    nshmdb_path: Path,
    rupture_id: int,
    realisation_ffp: Path,
    defaults_version: DefaultsVersion,
    seeds: dict[str, int] | None = None,
    inherited: dict[str, Any] | None = None,
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
        When given, written into the stub *before* generation so
        nshm2022-to-realisation replays these seeds via
        ``Seeds.read_from_realisation_or_random`` instead of drawing fresh ones.
        Seeds are an input: they change what gets derived.
    inherited : dict, optional
        Sections written over the generated ones *after* generation succeeds.
        These are outputs being replaced, not inputs -- the generator has no
        option to accept them, so it derives each section and this overwrites it.
        See ``docs/rupture_propagation_reproducibility.md``.

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

    if inherited:
        with open(realisation_ffp, encoding="utf-8") as handle:
            realisation = json.load(handle)
        absent = sorted(section for section in inherited if section not in realisation)
        if absent:
            # Adding a key rather than replacing one would invent a section the
            # generator never derived, so refuse and leave nothing behind.
            realisation_ffp.unlink(missing_ok=True)
            return (
                f"Failed to inherit content for rupture {rupture_id}, skipping: the "
                f"generated realisation has no {absent} section(s) to replace.\n"
            )
        # Assigning to an existing key preserves its position, so the section
        # order nshm2022-to-realisation wrote is left untouched.
        realisation.update(inherited)
        realisation_ffp.write_text(json.dumps(realisation), encoding="utf-8")

    return None


def read_inherited_sections(
    events_dir: Path, rupture_id: int, sections: Iterable[str]
) -> dict[str, Any]:
    """Read the named sections of an already-deployed realisation.

    Only the named sections are taken; every other field in the deployed file is
    ignored, so the rest of the regenerated realisation is derived fresh from the
    database and the current code.

    Note what carrying a section over costs. ``seeds`` is an *input*, so
    inheriting it makes the regenerated file reproduce the original. Every other
    section here is *derived*, so inheriting it means the value is copied rather
    than reproduced, and any change in the code that would have produced it
    becomes invisible. See ``docs/rupture_propagation_reproducibility.md``.

    Parameters
    ----------
    events_dir : Path
        Directory holding one ``<rupture id>/realisation.json`` per event.
    rupture_id : int
        The NSHM rupture id to look up.
    sections : iterable of str
        Section names to read. Each must be a key of :data:`SECTION_KEYS`.

    Returns
    -------
    dict
        The deployed sections, keyed by name. Sections the deployed file does not
        carry are omitted, and an event with no deployed realisation at all gives
        an empty dict -- which makes the caller derive everything, correct for a
        rupture the deployed set never held.

    Raises
    ------
    ValueError
        If a section is present but does not carry exactly the keys its schema
        requires. Passing a partial block on would count as inherited here and
        then fail validation inside the subprocess, where it is easy to miss.
    """
    realisation_ffp = events_dir / str(rupture_id) / "realisation.json"
    if not realisation_ffp.is_file():
        return {}
    with open(realisation_ffp, encoding="utf-8") as handle:
        realisation = json.load(handle)

    inherited: dict[str, Any] = {}
    for section in sections:
        block = realisation.get(section)
        if not block:
            continue
        required = SECTION_KEYS[section]
        if block.keys() != required:
            missing = sorted(required - block.keys())
            unexpected = sorted(block.keys() - required)
            raise ValueError(
                f"{realisation_ffp} has a malformed {section} block: "
                f"missing {missing}, unexpected {unexpected}"
            )
        inherited[section] = block
    return inherited


@app.command()
def generate_realisations_from_csv(
    nshmdb_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    csv_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument(writable=True)],
    defaults_version: Annotated[DefaultsVersion, typer.Argument()],
    inherit_from: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help=(
                "Events directory to carry sections over from, holding one "
                "<rupture id>/realisation.json per event. Requires --inherit."
            ),
        ),
    ] = None,
    inherit: Annotated[
        list[InheritableSection] | None,
        typer.Option(
            help=(
                "Section to carry over verbatim, repeatable. Ruptures with no "
                "deployed file derive everything, which is correct for a new "
                "event. Requires --inherit-from."
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
    inherit_from : Path, optional
        Events directory to carry sections over from.
    inherit : list of InheritableSection, optional
        Sections to carry over verbatim.
    """
    sections = [section.value for section in inherit or []]

    # The two options are meaningless apart, and silently doing nothing is the
    # failure mode that matters: the run looks like it inherited and did not.
    if (inherit_from is None) != (not sections):
        print("--inherit-from and --inherit must be given together.")
        raise typer.Exit(code=1)

    if sections and "rupture_propagation" not in sections:
        # Inheriting anything else without this does not reproduce a multi-fault
        # event: around 30 of 219 draw a different causality tree every run.
        print(
            "WARNING: inheriting without 'rupture_propagation'. Multi-fault "
            "causality trees are drawn from an unseeded RNG, so roughly 30 of them "
            "will differ from the deployed set on every run. See "
            "docs/rupture_propagation_reproducibility.md."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / "error_log.txt"
    error_log_handle = error_log.open("w", encoding="utf-8")

    df = pd.read_csv(csv_file)
    if "chosen_nshm_id" not in df.columns:
        print(f"CSV file {csv_file} must contain a 'chosen_nshm_id' column.")
        raise typer.Exit(code=1)

    rupture_ids = df["chosen_nshm_id"].dropna().astype(int).tolist()

    inherited_counts: dict[str, int] = dict.fromkeys(sections, 0)
    for rupture_id in tqdm(rupture_ids, desc="Generating realisations"):
        realisation_ffp = output_dir / f"realisation_{rupture_id}.json"
        try:
            inherited = (
                read_inherited_sections(inherit_from, rupture_id, sections)
                if inherit_from is not None
                else {}
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

        for section in inherited:
            inherited_counts[section] += 1

        # Seeds are an input -- written before generation so the generator
        # replays them. Everything else is an output, overwritten afterwards.
        seeds = inherited.pop("seeds", None)
        try:
            error_msg = generate_one(
                nshmdb_path,
                rupture_id,
                realisation_ffp,
                defaults_version,
                seeds=seeds,
                inherited=inherited,
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
    for section in sections:
        count = inherited_counts[section]
        print(
            f"Inherited {section} for {count} of {len(rupture_ids)} rupture(s); "
            f"{len(rupture_ids) - count} derived their own."
        )
    print(f"Error log written to {error_log}")


if __name__ == "__main__":
    app()