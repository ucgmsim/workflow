"""Station Subset Extraction.

Description
-----------
Extract a subset of stations from waveform and intensity measure files.

Inputs
------
Any of the LF, HF and BB waveform files (from `lf-to-xarray`, `hf-sim` and
`bb-sim`) and the intensity measure file (from `im-calc`).

Outputs
-------
For each input, a copy holding only the stations whose names match a pattern,
written alongside the input with an `_subset` suffix (e.g. `realisation.bb`
becomes `realisation_subset.bb`). By default, the subset is the observation
stations (three or four character names), which leaves out the simulated
grid stations.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `extract-subset` command which is installed after running `pip install ucgmsim-workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`extract-subset [OPTIONS] INPUT_FFPS...`

For More Help
-------------
See the output of `extract-subset --help`.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
import xarray as xr

from qcore import cli
from workflow import log_utils

app = typer.Typer()

OBSERVATION_STATION_PATTERN = r"^\w{3,4}$"
"""Matches observation station names, which are three or four characters long."""

SUBSET_SUFFIX = "_subset"
"""Suffix added to the stem of each input file to name its subset."""


def subset_path(input_ffp: Path) -> Path:
    """Name the subset of a file, alongside it.

    Parameters
    ----------
    input_ffp : Path
        The file to subset.

    Returns
    -------
    Path
        The input path with `_subset` appended to its stem.
    """
    return input_ffp.with_stem(input_ffp.stem + SUBSET_SUFFIX)


def matches_any(station_names: Sequence[str], patterns: Sequence[str]) -> list[bool]:
    """Test station names against a set of regular expressions.

    Parameters
    ----------
    station_names : Sequence[str]
        The station names to test.
    patterns : Sequence[str]
        Regular expressions, searched for (not fully matched) in each name, so
        anchor them with `^` and `$` to match whole names.

    Returns
    -------
    list[bool]
        For each station, True if any pattern matches its name.
    """
    compiled = [re.compile(pattern) for pattern in patterns]
    return [
        any(pattern.search(str(name)) for pattern in compiled) for name in station_names
    ]


def subset_stations(tree: xr.DataTree, patterns: Sequence[str]) -> xr.DataTree:
    """Select the matching stations from every node with a station dimension.

    Parameters
    ----------
    tree : xr.DataTree
        A waveform dataset (as a single node tree) or intensity measure tree.
    patterns : Sequence[str]
        Regular expressions; a station is kept if any of them match its name.

    Returns
    -------
    xr.DataTree
        The tree with only the matching stations. Nodes without a station
        dimension are unchanged.
    """

    def _subset(dataset: xr.Dataset) -> xr.Dataset:
        if "station" not in dataset.dims:
            return dataset
        return dataset.isel(
            station=matches_any(dataset["station"].values.tolist(), patterns)
        )

    return tree.map_over_datasets(_subset)


@cli.from_docstring(app)
@log_utils.log_call()
def extract_subset(
    input_ffps: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
    patterns: Annotated[list[str] | None, typer.Option("--pattern", "-p")] = None,
) -> list[Path]:
    """Extract the stations matching a pattern from waveform and IM files.

    Parameters
    ----------
    input_ffps : list[Path]
        LF, HF, BB waveform or intensity measure files to subset.
    patterns : list[str], optional
        Regular expressions for the station names to keep. A station is kept if
        any pattern is found in its name. Defaults to observation stations
        (names of three or four word characters).

    Returns
    -------
    list[Path]
        The subset files written, one per input.
    """
    patterns = patterns or [OBSERVATION_STATION_PATTERN]
    logger = log_utils.get_logger(__name__)
    output_ffps = []
    for input_ffp in input_ffps:
        output_ffp = subset_path(input_ffp)
        # Open lazily, without dask, so only the selected stations are read.
        with xr.open_datatree(input_ffp, engine="h5netcdf") as tree:
            subset = subset_stations(tree, patterns)
            n_stations = {
                node.path: node.dataset.sizes["station"]
                for node in subset.subtree
                if "station" in node.dataset.dims
            }
            if not n_stations:
                raise ValueError(f"{input_ffp} has no station dimension to subset.")
            if not any(n_stations.values()):
                logger.warning(
                    "no stations matched", input_ffp=input_ffp, patterns=patterns
                )
            subset.to_netcdf(output_ffp, engine="h5netcdf")
        logger.info(
            "wrote subset",
            input_ffp=input_ffp,
            output_ffp=output_ffp,
            n_stations=max(n_stations.values()),
        )
        output_ffps.append(output_ffp)
    return output_ffps


if __name__ == "__main__":
    app()
