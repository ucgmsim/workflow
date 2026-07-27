#!/usr/bin/env python3
"""Distribute completed realisations into per-event CyberShake directories.

Description
-----------
For every ``realisation_<id>.json`` in the source directory, create a directory
named after the rupture id under the events directory and copy the realisation
into it as ``realisation.json``.

Usage
-----
``copy-realisations-to-event-dirs SOURCE_DIR EVENTS_DIR``
"""

import re
import shutil
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


def copy_realisations(
    source_dir: Path, events_dir: Path, overwrite_existing: bool = False
) -> tuple[int, list[str], list[str]]:
    """Copy each realisation into ``events_dir/<rupture id>/realisation.json``.

    New event directories are always created. An **existing** realisation is
    replaced only when ``overwrite_existing`` is True; otherwise it is left
    untouched and its rupture id is returned as refused.

    Parameters
    ----------
    source_dir : Path
        Directory of ``realisation_<id>.json`` files.
    events_dir : Path
        Directory to create per-rupture event directories under.
    overwrite_existing : bool
        Whether to replace realisations that already exist.

    Returns
    -------
    tuple of (int, list of str, list of str)
        The number of realisations copied, the names of any files skipped
        because no integer id could be read from the filename, and the rupture
        ids refused because a realisation already existed.
    """
    copied = 0
    skipped: list[str] = []
    refused: list[str] = []

    for realisation_ffp in sorted(source_dir.glob("*.json")):
        match = re.search(r"\d+", realisation_ffp.name)
        if match is None:
            skipped.append(realisation_ffp.name)
            continue

        rupture_id = match.group()
        event_dir = events_dir / rupture_id
        target = event_dir / "realisation.json"
        if target.exists() and not overwrite_existing:
            refused.append(rupture_id)
            continue

        event_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(realisation_ffp, target)
        copied += 1

    return copied, skipped, refused


@app.command()
def copy_realisations_to_event_dirs(
    source_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    events_dir: Annotated[Path, typer.Argument(file_okay=False)],
    overwrite_existing: Annotated[
        bool,
        typer.Option(
            help="Replace realisations that already exist. Without this, existing "
            "files are left untouched and the command exits 1."
        ),
    ] = False,
) -> None:
    """Distribute completed realisations into per-event CyberShake directories.

    Parameters
    ----------
    source_dir : Path
        Directory of ``realisation_<id>.json`` files.
    events_dir : Path
        Directory to create per-rupture event directories under.
    overwrite_existing : bool
        Whether to replace realisations that already exist.
    """
    copied, skipped, refused = copy_realisations(
        source_dir, events_dir, overwrite_existing
    )

    print(f"Copied {copied} realisation(s) into {events_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with no integer id:")
        for name in skipped:
            print(f"  {name}")
    if refused:
        print(
            f"\nRefused to replace {len(refused)} existing realisation(s). "
            f"Re-run with --overwrite-existing to replace them:"
        )
        for rupture_id in refused:
            print(f"  {rupture_id}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
