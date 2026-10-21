"""IMDB Creation.

Description
-----------
Collect the simulated and empirical intensity measures of many realisations
into one intensity measure database (IMDB).

Inputs
------
Intensity measure files from [Intensity Measure Calculation](#intensity-measure-calculation),
one per realisation, each identified by the realisation's name. Realisations
named like clones of one event (`<event>_R<n>`, see `clone-realisations`) are
grouped into that event; any other realisation is the only one of its event.

Outputs
-------
A DuckDB database in the [IMDB](https://github.com/ucgmsim/imdb) schema,
holding a `simulated` record per station and component, and a `gmm` record
per station, ground motion model and component for the empirical estimates.
Simulated records are left out for stations inside the solver's absorbing
layer, where the waveforms are not ground-motion predictions.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `create-imdb` command which is installed after running `pip install ucgmsim-workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`create-imdb [OPTIONS] OUTPUT_FFP INTENSITY_MEASURE_FFPS...`

For More Help
-------------
See the output of `create-imdb --help`.
"""

from collections import Counter, defaultdict
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import parse
import typer
import xarray as xr

from imdb import IMDB, schema
from qcore import cli
from workflow import log_utils, sw4
from workflow.sites import OBSERVATION_STATION_PATTERN

app = typer.Typer()

REALISATION_NAME_TEMPLATE = "{event}_R{realisation:d}"
"""How realisations cloned from one event are named, as a `parse` template
(the `clone-realisations` default)."""

SITE_COLUMNS = {
    "latitude": "lat",
    "longitude": "lon",
    "vs30": "vs30",
    "z1pt0": "z1p0",
    "z2pt5": "z2p5",
}
"""Map from the station coordinates `im-calc` writes to IMDB site columns."""

SUPERGRID_DEPTH = sw4.SUPERGRID_DEPTH_COORDINATES["SGDEPTH"]
"""The station coordinate holding penetration into the absorbing layer."""


def event_name(realisation_name: str) -> str:
    """Identify the event a realisation belongs to.

    Parameters
    ----------
    realisation_name : str
        The realisation's name, which `im-calc` records as `event`.

    Returns
    -------
    str
        The event the realisation was cloned from, if its name follows
        `REALISATION_NAME_TEMPLATE`, otherwise the realisation's own name.
    """
    parsed = parse.parse(REALISATION_NAME_TEMPLATE, realisation_name)
    return parsed["event"] if isinstance(parsed, parse.Result) else realisation_name


def survey(
    intensity_measure_ffps: list[Path],
) -> tuple[
    list[tuple[str, str, Path]], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Identify the realisation in each intensity measure file, and the spectral grid covering them all.

    Parameters
    ----------
    intensity_measure_ffps : list[Path]
        The intensity measure files.

    Returns
    -------
    realisations : list of (str, str, Path)
        The event, realisation and file of each realisation, ordered by event.
    periods : np.ndarray
        The sorted union of every file's pSA periods.
    frequencies : np.ndarray
        The sorted union of every file's FAS frequencies. These differ
        between realisations with different timesteps, because `im-calc`
        stops at the Nyquist frequency.

    Raises
    ------
    ValueError
        If two files hold realisations of the same name.
    """
    realisations = []
    periods = frequencies = np.array([], dtype=np.float64)
    for intensity_measure_ffp in intensity_measure_ffps:
        with xr.open_datatree(intensity_measure_ffp) as dtree:
            rel_id = dtree.attrs["event"]
            realisations.append((event_name(rel_id), rel_id, intensity_measure_ffp))
            if "pSA" in dtree:
                periods = np.union1d(periods, dtree["pSA"].period.values)
            if "FAS" in dtree:
                frequencies = np.union1d(frequencies, dtree["FAS"].frequency.values)

    counts = Counter(rel_id for _, rel_id, _ in realisations)
    if duplicates := [rel_id for rel_id, count in counts.items() if count > 1]:
        raise ValueError(
            "Intensity measure files share a realisation name (give each "
            f"realisation its own `metadata.name`): {duplicates}"
        )
    return sorted(realisations), periods, frequencies


def read_stations(dtree: xr.DataTree) -> pd.DataFrame:
    """Read the per-station parameters `im-calc` attaches to every IM.

    Parameters
    ----------
    dtree : xr.DataTree
        The intensity measure tree.

    Returns
    -------
    pd.DataFrame
        The station location, site parameters, distances and absorbing layer
        penetration, indexed by station name.
    """
    dataset = next(node.dataset for node in dtree.subtree if node.has_data)
    return pd.DataFrame(
        {
            name: coordinate.values
            for name, coordinate in dataset.coords.items()
            if coordinate.dims == ("station",) and name != "station"
        },
        index=pd.Index(dataset.station.values, name="station"),
    )


def event_row(dtree: xr.DataTree, event_id: str) -> dict[str, Any]:
    """Describe an event from the source summary `im-calc` records.

    Parameters
    ----------
    dtree : xr.DataTree
        The intensity measure tree of any realisation of the event.
    event_id : str
        The event.

    Returns
    -------
    dict
        A row of the IMDB `events` table. Dip direction and length are left
        empty because `im-calc` reduces multi-fault ruptures to a single
        moment-averaged source, for which neither is defined. The magnitude
        is BoldM, as `im-calc` records it.
    """
    attrs = dtree.attrs
    tect_type = attrs.get("tect_type")
    return {
        "event_id": event_id,
        "magnitude": attrs["magnitude"],
        "tect_type": tect_type.upper() if tect_type else None,
        "dip": attrs["dip"],
        "dtop": attrs["ztor"],
        "dbottom": attrs["zbot"],
        "source_wkt": attrs["source"],
        "trace_wkt": attrs["trace"],
        "domain_wkt": attrs["domain"],
    }


def realisation_row(dtree: xr.DataTree, rel_id: str, event_id: str) -> dict[str, Any]:
    """Describe a realisation from the source summary `im-calc` records.

    Parameters
    ----------
    dtree : xr.DataTree
        The intensity measure tree of the realisation.
    rel_id : str
        The realisation.
    event_id : str
        The event the realisation belongs to.

    Returns
    -------
    dict
        A row of the IMDB `realisations` table. The magnitude is BoldM, as
        `im-calc` records it.
    """
    attrs = dtree.attrs
    return {
        "rel_id": rel_id,
        "event_id": event_id,
        "magnitude": attrs["magnitude"],
        "rake": attrs["rake"],
        "hypo_lat": attrs["hypo_lat"],
        "hypo_lon": attrs["hypo_lon"],
        "hypo_depth": attrs["hypo_depth"],
    }


def site_rows(stations: pd.DataFrame) -> pd.DataFrame:
    """Describe stations as IMDB sites.

    Parameters
    ----------
    stations : pd.DataFrame
        The per-station parameters, indexed by station name.

    Returns
    -------
    pd.DataFrame
        Rows of the IMDB `sites` table.
    """
    sites = stations[list(SITE_COLUMNS)].rename(columns=SITE_COLUMNS)
    sites["is_real"] = sites.index.str.fullmatch(OBSERVATION_STATION_PATTERN)
    return sites.rename_axis("site_id").reset_index()


def site_event_rows(stations: pd.DataFrame, event_id: str) -> pd.DataFrame:
    """Describe the distances from stations to an event's rupture.

    Parameters
    ----------
    stations : pd.DataFrame
        The per-station parameters, indexed by station name.
    event_id : str
        The event.

    Returns
    -------
    pd.DataFrame
        Rows of the IMDB `site_event` table. `rx` and `ry` are empty for
        ruptures with non-planar sources, as `im-calc` does not calculate them.
    """
    site_events = stations.reindex(columns=["rrup", "rjb", "rx", "ry"])
    site_events["event_id"] = event_id
    return site_events.rename_axis("site_id").reset_index()


def _spectrum(
    dataset: xr.Dataset, component: str, dim: str
) -> npt.NDArray[np.float32] | None:
    """Extract one component of a spectral IM, if it has that component."""
    if component not in dataset:
        return None
    return (
        dataset[component]
        .transpose("station", dim)
        .values.astype(np.float32, copy=False)
    )


def simulated_records(
    dtree: xr.DataTree,
    rel_id: str,
    stations: pd.Index,
    periods: npt.NDArray[np.float64],
    frequencies: npt.NDArray[np.float64],
) -> Iterator[dict[str, Any]]:
    """Read the simulated intensity measures of a realisation.

    Parameters
    ----------
    dtree : xr.DataTree
        The intensity measure tree.
    rel_id : str
        The realisation.
    stations : pd.Index
        The stations to read.
    periods : np.ndarray
        The database's periods, which pSA is placed on.
    frequencies : np.ndarray
        The database's frequencies, which FAS is placed on.

    Yields
    ------
    dict
        The `IMDB.add_records` arguments for the records of each component.
    """

    def read(im: str) -> xr.Dataset:
        return dtree[im].dataset.sel(station=stations) if im in dtree else xr.Dataset()

    scalars = {im: read(im) for im in schema.SCALAR_IMS}
    psa = read("pSA").reindex(period=periods)
    fas = read("FAS").reindex(frequency=frequencies)
    for component in schema.COMPONENTS:
        scalar_values = {
            im: dataset[component].values
            for im, dataset in scalars.items()
            if component in dataset
        }
        spectra = {
            "pSA": _spectrum(psa, component, "period"),
            "FAS": _spectrum(fas, component, "frequency"),
        }
        if scalar_values or any(value is not None for value in spectra.values()):
            records = pd.DataFrame(
                {
                    "rel_id": rel_id,
                    "site_id": stations,
                    "component": component,
                    "kind": "simulated",
                }
                | scalar_values
            )
            yield {"df": records, **spectra}


def empirical_records(
    dtree: xr.DataTree,
    rel_id: str,
    stations: pd.Index,
    periods: npt.NDArray[np.float64],
) -> Iterator[dict[str, Any]]:
    """Read the empirical intensity measure estimates of a realisation.

    `im-calc` stores the log-mean and log-standard deviation of each
    estimate; the IMDB holds the median and the log-standard deviation.

    Parameters
    ----------
    dtree : xr.DataTree
        The intensity measure tree.
    rel_id : str
        The realisation.
    stations : pd.Index
        The stations to read.
    periods : np.ndarray
        The database's periods, which pSA is placed on.

    Yields
    ------
    dict
        The `IMDB.add_records` arguments for the records of each ground
        motion model and component.
    """
    estimates: dict[tuple[str, str], dict[str, xr.Dataset]] = defaultdict(dict)
    for node in dtree.subtree:
        if node.parent is not None and node.parent.name == "empirical":
            attrs = node.attrs
            estimates[attrs["model"], attrs["component"]][
                attrs["intensity_measure"]
            ] = node.dataset.sel(station=stations)

    for (model, component), ims in estimates.items():
        records = pd.DataFrame(
            {
                "rel_id": rel_id,
                "site_id": stations,
                "component": component,
                "kind": "gmm",
                "gmm_key": model,
            }
        )
        spectra = {}
        for im, estimate in ims.items():
            if im == "pSA":
                estimate = estimate.reindex(period=periods).transpose(
                    "station", "period"
                )
                spectra["pSA"] = np.exp(estimate["mean"].values).astype(np.float32)
                spectra["pSA_sigma"] = estimate["std_Total"].values.astype(np.float32)
            else:
                records[im] = np.exp(estimate["mean"].values)
                records[f"{im}_sigma"] = estimate["std_Total"].values
        yield {"df": records, **spectra}


@cli.from_docstring(app)
@log_utils.log_call()
def create_imdb(
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    intensity_measure_ffps: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
) -> None:
    """Collect the intensity measures of many realisations into one IMDB.

    Parameters
    ----------
    output_ffp : Path
        The database to create. Replaced if it exists.
    intensity_measure_ffps : list[Path]
        Intensity measure files from `im-calc`, one per realisation.
    """
    logger = log_utils.get_logger(__name__)
    realisations, periods, frequencies = survey(intensity_measure_ffps)

    # Build beside the output and move it into place once complete, so an
    # interrupted run never leaves a partial database at the output path.
    partial_ffp = output_ffp.with_name(output_ffp.name + ".partial")
    partial_ffp.unlink(missing_ok=True)
    db = IMDB.create(
        partial_ffp,
        periods=periods.tolist(),
        frequencies=frequencies.tolist(),
        db_meta={
            "source": "ucgmsim workflow simulations",
            "creator": f"create-imdb (ucgmsim-workflow {metadata.version('ucgmsim-workflow')})",
            "sort_order": "event_id",
        },
    )
    with db:
        sites: set[str] = set()
        event_sites: dict[str, set[str]] = {}
        for event_id, rel_id, intensity_measure_ffp in realisations:
            with xr.open_datatree(intensity_measure_ffp) as dtree:
                stations = read_stations(dtree)

                new_sites = ~stations.index.isin(sites)
                if new_sites.any():
                    db.add_sites(site_rows(stations[new_sites]))
                    sites.update(stations.index[new_sites])

                # Realisations of an event share its rupture, so the event and
                # its site distances are written once, from the first of them.
                if event_id not in event_sites:
                    db.add_events(pd.DataFrame([event_row(dtree, event_id)]))
                    event_sites[event_id] = set()
                db.add_realisations(
                    pd.DataFrame([realisation_row(dtree, rel_id, event_id)])
                )

                new_event_sites = ~stations.index.isin(event_sites[event_id])
                if new_event_sites.any():
                    db.add_site_event(
                        site_event_rows(stations[new_event_sites], event_id)
                    )
                    event_sites[event_id].update(stations.index[new_event_sites])

                simulated = stations.index[~(stations[SUPERGRID_DEPTH] > 0)]
                for records in simulated_records(
                    dtree, rel_id, simulated, periods, frequencies
                ):
                    db.add_records(**records)
                for records in empirical_records(
                    dtree, rel_id, stations.index, periods
                ):
                    db.add_records(**records)

            logger.info(
                "added realisation",
                intensity_measure_ffp=intensity_measure_ffp,
                rel_id=rel_id,
                event_id=event_id,
                n_simulated_stations=len(simulated),
            )

        if problems := db.validate():
            raise ValueError(f"Database failed validation: {problems}")
    partial_ffp.replace(output_ffp)


if __name__ == "__main__":
    app()
