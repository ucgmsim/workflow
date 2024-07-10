#!usr/bin/env python
"""
SRF file generation.

This script facilitates the generation of SRF files from realisation
specifications. It generates fault geometries, generates SRF
files from fault geometries, and stitches together these files in the order
of rupture propagation.

Usage
-----
To generate a SRF file from a realisation specification:

```
$ python srf_generation.py path/to/realisation.yaml output_directory
```
"""

import collections
import functools
import multiprocessing
import re
import subprocess
from pathlib import Path
from typing import Annotated, Generator

import numpy as np
import pandas as pd
import typer
from qcore import binary_version, coordinates, grid, gsf, srf_new

from source_modelling.sources import IsSource
from workflow import realisations
from workflow.realisations import (
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
    SRFConfig,
)


def normalise_name(name: str) -> str:
    """Normalise a name (fault name, realisation name) as a filename.

    Parameters
    ----------
    name : str
        The name to normalise

    Returns
    -------
    str
        The normalised equivalent of this name. Normalised names are entirely
        lower case, and all non-alphanumeric characters are replaced with "_".
    """
    return re.sub(r"[^A-z0-9]", "_", name.lower())


def generate_fault_gsf(
    name: str,
    geometry: IsSource,
    rake: float,
    gsf_output_directory: Path,
    subdivision_resolution: float,
):
    """Write the fault geometry of a fault to a GSF file.

    Parameters
    ----------
    fault : RealisationFault
        The fault to write.
    gsf_output_directory : Path
        The directory to output the GSF file to.
    subdivision_resolution : float
        The geometry resolution.
    """
    gsf_output_filepath = gsf_output_directory / f"{name}.gsf"
    gsf_df = pd.DataFrame(
        [
            {
                "strike": plane.strike,
                "dip": plane.dip,
                "length": plane.length,
                "width": plane.width,
                "rake": rake,
                "meshgrid": grid.coordinate_meshgrid(
                    plane.corners[0],
                    plane.corners[1],
                    plane.corners[-1],
                    subdivision_resolution * 1000,
                ),
            }
            for plane in geometry.planes
        ]
    )
    gsf.write_fault_to_gsf_file(
        gsf_output_filepath, gsf_df, subdivision_resolution * 1000
    )

    return gsf_output_filepath


def generate_fault_srf(
    name: str,
    fault: IsSource,
    rake: float,
    magnitude: float,
    hypocentre_local_coordinates: np.ndarray,
    output_directory: Path,
    subdivision_resolution: float,
    srf_config: SRFConfig,
):
    """Generate an SRF file for a given fault.

    Parameters
    ----------
    fault : RealisationFault
        The fault to generate the SRF file for.
    realisation : Realisation
        The realisation the fault belongs to.
    output_directory : Path
        The output directory.
    subdivision_resolution : float
        The geometry resolution.
    """
    gsf_output_directory = output_directory / "gsf"
    gsf_file_path = generate_fault_gsf(
        name,
        fault,
        rake,
        gsf_output_directory,
        subdivision_resolution,
    )

    genslip_bin = binary_version.get_genslip_bin(srf_config.genslip_version)

    nx = sum(
        grid.gridpoint_count_in_length(plane.length_m, subdivision_resolution * 1000)
        for plane in fault.planes
    )
    ny = grid.gridpoint_count_in_length(
        fault.planes[0].width_m, subdivision_resolution * 1000
    )
    genslip_hypocentre_coords = np.array([fault.length, fault.width]) * (
        hypocentre_local_coordinates - np.array([-1 / 2, 0])
    )
    genslip_cmd = [
        genslip_bin,
        "read_erf=0",
        "write_srf=1",
        "read_gsf=1",
        "write_gsf=0",
        f"infile={gsf_file_path}",
        f"mag={magnitude}",
        f"nstk={nx}",
        f"ndip={ny}",
        "ns=1",
        "nh=1",
        f"seed={srf_config.genslip_seed}",
        f"velfile={srf_config.genslip_velocity_model}",
        f"shypo={genslip_hypocentre_coords[0]}",
        f"dhypo={genslip_hypocentre_coords[1]}",
        f"dt={srf_config.genslip_dt}",
        "plane_header=1",
        "srf_version=1.0",
        "seg_delay={0}",
        "rvfac_seg=-1",
        "gwid=-1",
        "side_taper=0.02",
        "bot_taper=0.02",
        "top_taper=0.0",
        "rup_delay=0",
        "alpha_rough=0.0",
    ]

    srf_file_path = output_directory / "srf" / (name + ".srf")
    with open(srf_file_path, "w", encoding="utf-8") as srf_file_handle:
        subprocess.run(
            genslip_cmd, stdout=srf_file_handle, stderr=subprocess.PIPE, check=True
        )


def tree_nodes_in_order(
    tree: dict[str, str],
) -> Generator[str, None, None]:
    """Generate faults in topologically sorted order.

    Parameters
    ----------
    faults : list[RealisationFault]
        List of RealisationFault objects.

    Yields
    ------
    RealisationFault
        The next fault in the topologically sorted order.
    """
    tree_child_map = collections.defaultdict(list)
    for cur, parent in tree.items():
        if parent:
            tree_child_map[parent].append(cur)

    def in_order_traversal(
        node: str,
    ) -> Generator[str, None, None]:
        yield node
        for child in tree_child_map[node]:
            yield from in_order_traversal(child)

    initial_fault = next(cur for cur, parent in tree.items() if not parent)
    yield from in_order_traversal(initial_fault)


def stitch_srf_files(
    faults: dict[str, IsSource],
    rupture_propogation: RupturePropagationConfig,
    output_directory: Path,
    output_name: str,
) -> Path:
    """Stitch SRF files together in the order of rupture propagation.

    Parameters
    ----------
    realisation_obj : Realisation
        The realisation containing the faults and rupture propagation order.
    output_directory : Path
        The output directory containing fault SRF files.

    Returns
    -------
    Path
        The path to the stitched together SRF file.
    """
    srf_output_filepath = output_directory / f"{output_name}.srf"
    order = list(tree_nodes_in_order(rupture_propogation.rupture_causality_tree))
    with open(srf_output_filepath, "w", encoding="utf-8") as srf_file_output:
        fault_points = {}
        header = []

        srf_new.write_version(srf_file_output)

        for fault_name in order:
            fault = faults[fault_name]
            parent = rupture_propogation.rupture_causality_tree[fault_name]
            with open(
                output_directory / "srf" / (normalise_name(fault_name) + ".srf"),
                "r",
                encoding="utf-8",
            ) as fault_srf_file:
                srf_new.read_version(fault_srf_file)
                fault_header = srf_new.read_srf_headers(fault_srf_file)
                if parent:
                    for plane in fault_header:
                        # The value of -999, -999 is used in the SRF spec to say
                        # "no hypocentre for this segment".
                        plane.shyp = -999
                        plane.dhyp = -999
                header.extend(fault_header)
                point_count = srf_new.read_points_count(fault_srf_file)
                fault_points[fault_name] = srf_new.read_srf_n_points(
                    point_count, fault_srf_file
                )

        srf_new.write_srf_header(srf_file_output, header)
        srf_new.write_point_count(
            srf_file_output, sum(len(points) for points in fault_points.values())
        )
        for fault_name in order:
            t_delay = 0
            parent = rupture_propogation.rupture_causality_tree[fault_name]
            if parent:
                # find closest grid point to the jump location
                # compute the time delay as equal to the tinit of this point (for now)
                fault = faults[fault_name]
                jump_pair = fault.fault_coordinates_to_wgs_depth_coordinates(
                    rupture_propogation.jump_points[fault_name]
                )
                parent_coords = fault.fault_coordinates_to_wgs_depth_coordinates(
                    coordinates.wgs_depth_to_nztm(jump_pair.from_point)
                )
                parent_fault_points = fault_points[parent]
                grid_points = coordinates.wgs_depth_to_nztm(
                    np.array(
                        [
                            [point.lat, point.lon, point.dep * 1000]
                            for point in parent_fault_points
                        ]
                    )
                )
                jump_index = int(
                    np.argmin(
                        coordinates.distance_between_wgs_depth_coordinates(
                            parent_coords, grid_points
                        )
                    )
                )
                t_delay = parent_fault_points[jump_index].tinit
            cur_fault_points = fault_points[fault_name]
            for point in cur_fault_points:
                point.tinit += t_delay
                srf_new.write_srf_point(srf_file_output, point)

        return srf_output_filepath


def generate_fault_srfs_parallel(
    faults: dict[str, IsSource],
    rupture_propagation_config: RupturePropagationConfig,
    output_directory: Path,
    subdivision_resolution: float,
    srf_config: SRFConfig,
):
    """Generate fault SRF files in parallel.

    Parameters
    ----------
    realisation : Realisation
        The realisation to generate fault SRF files for.
    output_directory : Path
        The directory to output the fault SRF files.
    subdivision_resolution : float
        The geometry resolution.
    """
    # need to do this before multiprocessing because of race conditions
    gsf_directory = output_directory / "gsf"
    gsf_directory.mkdir(exist_ok=True)
    srf_directory = output_directory / "srf"
    srf_directory.mkdir(exist_ok=True)
    fault_names = [normalise_name(fault_name) for fault_name in faults]
    magnitudes = rupture_propagation_config.magnitudes
    rakes = rupture_propagation_config.rakes
    hypocentres = {
        fault_name: jump_point.to_point
        for fault_name, jump_point in rupture_propagation_config.jump_points
    }
    hypocentres[rupture_propagation_config.initial_fault] = (
        rupture_propagation_config.hypocentre
    )

    srf_generation_parameters = [
        (
            normalise_name(fault_name),
            faults[fault_name],
            rakes[fault_name],
            magnitudes[fault_name],
            hypocentres[fault_name],
        )
        for fault_name in faults
    ]

    with multiprocessing.Pool() as worker_pool:
        worker_pool.starmap(
            functools.partial(
                generate_fault_srf,
                output_directory=output_directory,
                subdivision_resolution=subdivision_resolution,
                srf_config=srf_config,
            ),
            srf_generation_parameters,
        )


def main(
    realisation_filepath: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="The filepath of the YAML file containing the realisation data.",
            dir_okay=False,
        ),
    ],
    output_directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            writable=True,
            file_okay=False,
            help="The output directory path for SRF files.",
        ),
    ],
    subdivision_resolution: Annotated[
        float, typer.Option(help="Geometry resolution (in km)", min=0)
    ] = 0.1,
):
    """Generate a type-5 SRF file from a given realisation specification."""
    srf_config: SRFConfig = realisations.read_config_from_realisation(
        SRFConfig, realisation_filepath
    )
    rupture_propagation: RupturePropagationConfig = (
        realisations.read_config_from_realisation(
            RupturePropagationConfig, realisation_filepath
        )
    )
    source_config: SourceConfig = realisations.read_config_from_realisation(
        SourceConfig, realisation_filepath
    )
    metadata: RealisationMetadata = realisations.read_config_from_realisation(
        RealisationMetadata, realisation_filepath
    )
    generate_fault_srfs_parallel(
        source_config.source_geometries,
        rupture_propagation,
        output_directory,
        subdivision_resolution,
        srf_config,
    )
    stitch_srf_files(
        source_config.source_geometries,
        rupture_propagation,
        output_directory,
        normalise_name(metadata.name),
    )


if __name__ == "__main__":
    typer.run(main)
