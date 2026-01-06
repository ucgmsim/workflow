"""Miscellaneous workflow utilities that couldn't go anywhere else."""

import os
import tempfile
import urllib.request
from collections.abc import Mapping
from typing import Any, TypeVar, overload

import geopandas as gpd
import numpy as np
import psutil
import shapely
from shapely import Geometry, Polygon, geometry

from qcore import coordinates

NZ_COASTLINE_URL = "https://www.dropbox.com/scl/fi/zkohh794y0s2189t7b1hi/NZ.gmt?rlkey=02011f4morc4toutt9nzojrw1&st=vpz2ri8x&dl=1"


def read_nz_coastline() -> gpd.GeoDataFrame:
    """Read the New Zealand coastline from NZ.gmt file.

    Returns
    -------
    gpd.GeoDataFrame
        The geodataframe representing the NZ coastline.
    """
    with (
        tempfile.NamedTemporaryFile(mode="wb", suffix=".gmt") as f,
        urllib.request.urlopen(NZ_COASTLINE_URL) as source,
    ):
        f.write(source.read())
        return gpd.read_file(f.name)


def get_nz_outline_polygon() -> Geometry:
    """Get the outline polygon of New Zealand.

    Returns
    -------
    Polygon
        The outline polygon of New Zealand.
    """

    gpd_df = read_nz_coastline()
    island_polygons = [
        Polygon(
            coordinates.wgs_depth_to_nztm(
                np.array(geometry.mapping(island)["coordinates"])[:, ::-1]
            )
        )
        for island in gpd_df.geometry
    ]
    south_island, north_island = sorted(
        island_polygons, key=lambda island: island.area, reverse=True
    )[:2]
    south_island = south_island.simplify(100)
    north_island = north_island.simplify(100)
    return shapely.union(south_island, north_island)


def get_available_cores() -> int:
    """Get the available number of cores for a job.

    Returns
    -------
    int
        Either the reported number of cores from the multiprocessing
        module, or the number of allocated cores if running in a slurm
        environment.

    Raises
    ------
    RuntimeError
        If the number of CPUs cannot be determined on the system.
    """
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])

    if "SLURM_NPROCS" in os.environ:
        return int(os.environ["SLURM_NPROCS"])

    # A process's CPU affinity is the set of CPU cores that the
    # current process is allowed to use. On a typical setup, the CPU
    # affinity contains every core on the system so that
    #
    # len(psutil.Process().cpu_affinity()) == multiprocessing.cpu_count().
    #
    # On HPC clusters, slurm and PBS set the CPU affinity to schedule
    # processes onto certain cores. This allows the scheduler to share
    # nodes between jobs. Hence
    #
    # len(psutil.Process().cpu_affinity()) == cores allocated for job on node.
    #
    # CPU affinity is a kernel-level feature, and exposed to the
    # process. This is the most reliable way to set CPU cores. It also
    # means that using `taskset(1)` on any other system will be
    # respected by workflow jobs.
    process = psutil.Process()
    if hasattr(process, "cpu_affinity"):
        return len(process.cpu_affinity())
    elif cpu_count := psutil.cpu_count():
        return cpu_count
    else:
        raise RuntimeError("Cannot determine CPU count.")


K = TypeVar("K")
V1 = TypeVar("V1")
V2 = TypeVar("V2")
V3 = TypeVar("V3")


# These overloads provide better type inference in the common case
@overload
def dict_zip(
    __d1: Mapping[K, V1], *, strict: bool = ...
) -> dict[K, tuple[V1]]: ...  # numpydoc ignore=GL08


@overload
def dict_zip(
    __d1: Mapping[K, V1], __d2: Mapping[K, V2], *, strict: bool = ...
) -> dict[K, tuple[V1, V2]]: ...  # numpydoc ignore=GL08


@overload
def dict_zip(
    __d1: Mapping[K, V1],
    __d2: Mapping[K, V2],
    __d3: Mapping[K, V3],
    *,
    strict: bool = True,
) -> dict[K, tuple[V1, V2, V3]]: ...  # numpydoc ignore=GL08


@overload
def dict_zip(
    *dicts: Mapping[K, Any], strict: bool = ...
) -> dict[K, tuple[Any, ...]]: ...  # numpydoc ignore=GL08


def dict_zip(*dicts: Mapping[K, Any], strict: bool = True) -> dict[K, tuple[Any, ...]]:
    """
    Takes the product of one or more dictionaries.

    Parameters
    ----------
    *dicts : list of dict
        Variable number of dictionaries.
    strict : bool, default False
        If True, raise an error if the keys in `dicts` are not all the same.

    Returns
    -------
    dict
        A dictionary where each value is a tuple of the corresponding values from the input dictionaries.

    Raises
    ------
    ValueError
        If strict is True and the keys in the dictionaries are not all the same.
    """
    if not dicts:
        return {}

    keys: set[K] = set(dicts[0].keys())

    if strict and any(set(d) != keys for d in dicts[1:]):
        raise ValueError("Keys in dictionaries are not all the same.")
    else:
        for dict in dicts[1:]:
            keys = keys.intersection(dict.keys())

    result = {key: tuple(d[key] for d in dicts) for key in list(keys)}
    return result
