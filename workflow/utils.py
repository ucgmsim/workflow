"""Miscellaneous workflow utilities that couldn't go anywhere else."""

import os
import tempfile
import urllib.request

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
    else:
        return psutil.cpu_count()
