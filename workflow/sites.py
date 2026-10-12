"""Per-station site parameters.

Every stage that needs to know about a site reads it from one parquet file,
indexed by station name, with one row per station. `generate-station-coordinates`
reads the full site table and writes the in-domain stations to
`stations.parquet`, which `hf-sim` and `bb-sim` read in turn.

Columns
-------
latitude, longitude : float
    Station location (degrees). In the filtered copy written for EMOD3D, this is
    the location of the grid point the station was snapped to.
vs30 : float
    Time-averaged shear-wave velocity in the top 30 m (m/s).
vs30_simulation : float
    The Vs30 the simulation resolves at the site (m/s). Not yet read by any stage.
z1pt0 : float
    Depth to the 1.0 km/s shear-wave velocity horizon (km).
z2pt5 : float
    Depth to the 2.5 km/s shear-wave velocity horizon (km).
kappa_s : float
    Near-surface attenuation (s).
q_frequency_exponent : float
    The x in Q(f) = Q0 f^x.
"""

from pathlib import Path

import pandas as pd

INDEX = "station"
"""The name of the index column: the station name."""

COLUMNS = (
    "latitude",
    "longitude",
    "vs30",
    "vs30_simulation",
    "z1pt0",
    "z2pt5",
    "kappa_s",
    "q_frequency_exponent",
)
"""The columns every site table carries, besides its index."""


def validate(sites: pd.DataFrame) -> pd.DataFrame:
    """Check a site table carries every site parameter for every station.

    Parameters
    ----------
    sites : pd.DataFrame
        The site table, indexed by station name.

    Returns
    -------
    pd.DataFrame
        The site table, restricted to the site parameter columns.

    Raises
    ------
    ValueError
        If the index is not the station name, station names repeat, a
        column is missing, or a station is missing a value.
    """
    if sites.index.name != INDEX:
        raise ValueError(
            f"Site table must be indexed by {INDEX!r}, not {sites.index.name!r}."
        )
    if duplicated := sorted(set(sites.index[sites.index.duplicated()])):
        raise ValueError(f"Site table repeats stations: {duplicated}.")
    if missing := [column for column in COLUMNS if column not in sites.columns]:
        raise ValueError(f"Site table is missing columns: {missing}.")
    sites = sites[list(COLUMNS)]
    if sites.isna().any(axis=None):
        missing_values = {
            column: sites.index[sites[column].isna()].tolist()
            for column in COLUMNS
            if sites[column].isna().any()
        }
        raise ValueError(f"Site table has stations missing values: {missing_values}.")
    return sites


def read_sites(sites_ffp: Path) -> pd.DataFrame:
    """Read and validate a site table.

    Parameters
    ----------
    sites_ffp : Path
        The parquet file to read.

    Returns
    -------
    pd.DataFrame
        The site table, indexed by station name.

    Raises
    ------
    ValueError
        If the table does not carry every site parameter for every station.
    """
    return validate(pd.read_parquet(sites_ffp))


def write_sites(sites: pd.DataFrame, sites_ffp: Path) -> None:
    """Validate and write a site table.

    Parameters
    ----------
    sites : pd.DataFrame
        The site table, indexed by station name.
    sites_ffp : Path
        The parquet file to write.

    Raises
    ------
    ValueError
        If the table does not carry every site parameter for every station.
    """
    validate(sites).to_parquet(sites_ffp)
