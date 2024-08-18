#!usr/bin/env python
"""
SRF file generation.

This script facilitates the generation of SRF files from realisation
specifications. It generates fault geometries, generates SRF
files from fault geometries, and stitches together these files in the order
of rupture propagation.
"""

import collections
import functools
import multiprocessing
import re
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Generator, Iterable, Optional

import numpy as np
import pandas as pd
import scipy as sp
import typer
from qcore import coordinates, grid, gsf
from scipy.sparse import csr_array

from source_modelling import srf
from source_modelling.sources import IsSource
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
                "meshgrid": grid.coordinate_patchgrid(
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
    srf_config: SRFConfig,
    velocity_model_path: Path,
    genslip_path: Path,
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
        srf_config.resolution,
    )

    nx = sum(
        grid.gridpoint_count_in_length(plane.length_m, srf_config.resolution * 1000) - 1
        for plane in fault.planes
    )
    ny = (
        grid.gridpoint_count_in_length(
            fault.planes[0].width_m, srf_config.resolution * 1000
        )
        - 1
    )
    genslip_hypocentre_coords = np.array([fault.length, fault.width]) * (
        hypocentre_local_coordinates - np.array([1 / 2, 0])
    )
    genslip_cmd = [
        str(genslip_path),
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
        f"velfile={velocity_model_path}",
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
        print(" ".join(genslip_cmd))
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


def fast_concat(csr_arrays: list[csr_array]):
    max_columns = max(arr.shape[1] for arr in csr_arrays)
    padded_arrays = [
        sp.sparse.hstack(
            [arr, sp.sparse.csr_array((arr.shape[0], max_columns - arr.shape[1]))]
        )
        if arr.shape[1] < max_columns
        else arr
        for arr in csr_arrays
    ]
    return sp.sparse.vstack(padded_arrays)


def concatenate_slip_values(
    slip_values: Iterable[csr_array],
) -> Optional[csr_array]:
    slip_values = list(slip_values)
    if sum(slip.count_nonzero() for slip in slip_values) == 0:
        return None
    concatenated_slip = fast_concat(slip_values)
    return concatenated_slip


def stitch_srf_files(
    faults: dict[str, IsSource],
    rupture_propagation: RupturePropagationConfig,
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
    order = list(tree_nodes_in_order(rupture_propagation.rupture_causality_tree))
    srf_files: dict[str, srf.SrfFile] = {}

    for fault_name in order:
        fault = faults[fault_name]
        parent = rupture_propagation.rupture_causality_tree[fault_name]
        srf_ffp = output_directory / "srf" / (normalise_name(fault_name) + ".srf")
        srf_file = srf.read_srf(srf_ffp)
        if parent:
            # The value of -999, -999 is used in the SRF spec to say
            # "no hypocentre for this segment."
            srf_file.header["shyp"] = -999
            srf_file.header["dhyp"] = -999
            parent_srf = srf_files[parent]
            parent_coords = fault.fault_coordinates_to_wgs_depth_coordinates(
                rupture_propagation.jump_points[fault_name].from_point
            )
            jump_index = int(
                np.argmin(
                    coordinates.distance_between_wgs_depth_coordinates(
                        parent_srf.points[["lat", "lon", "dep"]].to_numpy()
                        * np.array([1, 1, 1000]),
                        parent_coords,
                    )
                )
            )
            t_delay = parent_srf.points["tinit"].iloc[jump_index]
            print(f"Delaying {fault_name} by {t_delay:.2f}s")
            print(
                f"Point {parent_coords}, grid point {parent_srf.points.iloc[jump_index]}"
            )
            srf_file.points["tinit"] += t_delay
        srf_files[fault_name] = srf_file
    output_srf_file = srf.SrfFile(
        "1.0",
        pd.concat(srf_file.header for srf_file in srf_files.values()),
        pd.concat(srf_file.points for srf_file in srf_files.values()),
        concatenate_slip_values(
            srf_file.slipt1_array
            if srf_file.slipt1_array is not None
            else csr_array((len(srf_file.points), 1))
            for srf_file in srf_files.values()
        ),
        concatenate_slip_values(
            srf_file.slipt2_array
            if srf_file.slipt2_array is not None
            else csr_array((len(srf_file.points), 1))
            for srf_file in srf_files.values()
        ),
        concatenate_slip_values(
            srf_file.slipt3_array
            if srf_file.slipt3_array is not None
            else csr_array((len(srf_file.points), 1))
            for srf_file in srf_files.values()
        ),
    )
    srf.write_srf(srf_output_filepath, output_srf_file)
    return srf_output_filepath


def generate_fault_srfs_parallel(
    faults: dict[str, IsSource],
    rupture_propagation_config: RupturePropagationConfig,
    output_directory: Path,
    srf_config: SRFConfig,
    velocity_model_path: Path,
    genslip_path: Path,
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
    magnitudes = rupture_propagation_config.magnitudes
    rakes = rupture_propagation_config.rakes
    hypocentres = {
        fault_name: jump_point.to_point
        for fault_name, jump_point in rupture_propagation_config.jump_points.items()
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
                srf_config=srf_config,
                velocity_model_path=velocity_model_path,
                genslip_path=genslip_path,
            ),
            srf_generation_parameters,
        )


def generate_srf(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="The filepath of the YAML file containing the realisation data.",
            dir_okay=False,
        ),
    ],
    output_srf_filepath: Annotated[
        Path,
        typer.Argument(
            writable=True, help="The filepath for the final SRF file.", dir_okay=False
        ),
    ],
    work_directory: Annotated[
        Path,
        typer.Option(
            help="Path to output intermediate geometry and SRF files",
            exists=True,
            file_okay=False,
        ),
    ] = Path("/out"),
    velocity_model: Annotated[
        Path,
        typer.Option(
            help="Path to the genslip velocity model.", readable=True, dir_okay=False
        ),
    ] = Path("/genslip_velocity_model.vmod"),
    genslip_path: Annotated[
        Path,
        typer.Option(help="Path to genslip binary.", readable=True, dir_okay=False),
    ] = Path("/EMOD3D/tools/genslip_v5.4.2"),
):
    """Generate a type-5 SRF file from a given realisation specification."""
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    srf_config = SRFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    rupture_propagation = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    source_config = SourceConfig.read_from_realisation(realisation_ffp)

    generate_fault_srfs_parallel(
        source_config.source_geometries,
        rupture_propagation,
        work_directory,
        srf_config,
        velocity_model,
        genslip_path,
    )
    srf_name = normalise_name(metadata.name)
    stitch_srf_files(
        source_config.source_geometries,
        rupture_propagation,
        work_directory,
        srf_name,
    )
    srf_config.write_to_realisation(realisation_ffp)

    shutil.copyfile(work_directory / (srf_name + ".srf"), output_srf_filepath)


def main():
    typer.run(generate_srf)


if __name__ == "__main__":
    main()
