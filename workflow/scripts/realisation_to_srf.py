#!/usr/bin/env python
"""SRF Generation.

Description
-----------
Produce an SRF from a realisation.

Inputs
------
A realisation file containing:
1. A source configuration,
2. A rupture propagation configuration,
3. A metadata configuration.

Typically, this information comes from a stage like [NSHM To Realisation](#nshm-to-realisation).

Outputs
-------
1. An [SRF](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-SRFFormat) file containing the source slip definition for the realisation,
2. An updated realisation file containing the parameters used for SRF generation copied from the scientific defaults.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `realisation-to-srf` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you are executing on your own computer you also need to specify the work directory (with the `--work-directory` flag), a 1D velocity model (`--velocity-model-ffp`), and the path to a genslip binary (`--genslip-path`).

Usage
-----
`realisation-to-srf [OPTIONS] REALISATION_FFP OUTPUT_SRF_FILEPATH`

For More Help
-------------
See the output of `realisation-to-srf --help`.

Visualisation
-------------
You can visualise the output of this stage using the SRF plotting tools in the [source modelling](https://github.com/ucgmsim/visualization/blob/plots/wiki/Plotting-Tools.md) repository. Many of the tools take realisations as optional arguments to enhance the plot output.
"""

import dataclasses
import functools
import multiprocessing
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import scipy as sp
import typer
from scipy.sparse import csr_array

from qcore import cli, coordinates
from source_modelling import gsf, moment, rupture_propagation, srf
from source_modelling.sources import Fault, IsSource, Point
from workflow import log_utils, realisations, utils
from workflow.log_utils import log_call
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RealisationMetadata,
    Resolution,
    RupturePropagationConfig,
    Seeds,
    SourceConfig,
    SRFConfig,
    VelocityModel1D,
)

app = typer.Typer()


def normalise_name(name: str) -> str:
    """
    Normalise a name (fault name, realisation name) as a filename.

    Parameters
    ----------
    name : str
        The name to normalise.

    Returns
    -------
    str
        The normalised equivalent of this name. Normalised names are entirely
        lower case, and all non-alphanumeric characters are replaced with "_".
    """

    return re.sub(r"\W", "_", name.lower())


def generate_fault_gsf(
    name: str,
    geometry: IsSource,
    rake: float,
    gsf_output_directory: Path,
    subdivision_resolution: float,
    slip: float | None = None,
    init_time: float | None = None,
) -> Path:
    """Write the fault geometry of a fault to a GSF file.

    Parameters
    ----------
    name : str
        The name of the fault.
    geometry : IsSource
        The geometry of the fault.
    rake : float
        The rake of the fault.
    gsf_output_directory : Path
        The directory to output the GSF file to.
    subdivision_resolution : float
        The geometry resolution.
    slip : float, optional
        The slip value to write to the GSF file. If None, the slip will not be written.
    init_time : float, optional
        The initial time (INIT_TIME) to write to the GSF file.
        If None, a default value of -1 will be set by `gsf.write_gsf()`.

    Returns
    -------
    Path
        The path to the generated GSF file.
    """
    gsf_output_filepath = gsf_output_directory / f"{normalise_name(name)}.gsf"
    gsf_df = gsf.source_to_gsf_dataframe(geometry, subdivision_resolution)
    gsf_df["loc_rake"] = rake
    if slip is not None:
        gsf_df["slip"] = slip
    if init_time is not None:
        gsf_df["init_time"] = init_time
    gsf.write_gsf(gsf_df, gsf_output_filepath)
    return gsf_output_filepath


def concatenate_csr_arrays(csr_arrays: list[csr_array]) -> csr_array:
    """Concatenate a list of sparse arrays.

    Concatenates a list of sparse arrays with varying column
    widths. It will do this by padding the arrays and assuming that
    all returned arrays should be left-aligned.

    Parameters
    ----------
    csr_arrays : list[csr_array]
        The sparse arrays to concatenate.

    Returns
    -------
    csr_array
        The left-aligned and concatenated sparse array. If the arrays have
        dimensions (r_i, c_i), then the output array will have dimensions
        (∑ r_i, max {c_i}).
    """
    max_columns = max(arr.shape[1] for arr in csr_arrays)
    padded_arrays = [
        (
            sp.sparse.hstack(
                [
                    arr,
                    sp.sparse.csr_array(
                        (arr.shape[0], max_columns - arr.shape[1]), dtype=np.float32
                    ),
                ]
            )
            if arr.shape[1] < max_columns
            else arr
        )
        for arr in csr_arrays
    ]
    return sp.sparse.vstack(padded_arrays)


def concatenate_slip_values(
    slip_values: Iterable[csr_array],
) -> csr_array | None:
    """Concatenate a list of slip arrays.

    Parameters
    ----------
    slip_values : Iterable[csr_array]
        The slip arrays to concatenate.

    Returns
    -------
    csr_array or None
        The concatenated slip array, or None if the slip array
        contains no non-zero values.
    """
    slip_values = list(slip_values)
    if sum(slip.count_nonzero() for slip in slip_values) == 0:
        return None
    concatenated_slip = concatenate_csr_arrays(slip_values)
    return concatenated_slip


def stitch_srf_files(
    faults: dict[str, IsSource],
    rupture_propagation_config: RupturePropagationConfig,
    output_directory: Path,
    output_name: str,
) -> Path:
    """Stitch SRF files together in the order of rupture propagation.

    Parameters
    ----------
    faults : dict[str, IsSource]
        The faults and their geometries.
    rupture_propagation_config : RupturePropagationConfig
        The rupture propagation configuration.
    output_directory : Path
        The output directory containing fault SRF files.
    output_name : str
        The name of the output SRF file.

    Returns
    -------
    Path
        The path to the stitched together SRF file.
    """
    output_srf_path = output_directory / f"{output_name}.srf"

    # Determine the order of faults from the rupture causality tree
    srf_stitching_order = rupture_propagation.tree_nodes_in_order(
        rupture_propagation_config.rupture_causality_tree
    )

    srf_file_map: dict[str, srf.SrfFile] = {}

    def compute_rupture_delay(
        tinit: float, parent_coordinates: np.ndarray, child_coordinates: np.ndarray
    ) -> float:
        """Compute the rupture jump delay between the parent SRF and child SRF.

        delay = T_p + d / s
        where T_p is the initial rupture time of the closest point on
        the parent SRF, d is the jumping distance, and s is the average
        rupture speed between the two jump points.

        Parameters
        ----------
        tinit : float
            The initiation time for the jumping point on the parent SRF.
        parent_coordinates : np.ndarray
            The coordinates of the parent jumping location, WGS84+depth (jumping from here).
        child_coordinates : np.ndarray
            The coordinates of the child jumping location, WGS84+depth (jumping to here).

        Returns
        -------
        float
            The delay for the child SRF.
        """
        jump_distance_km = (
            coordinates.distance_between_wgs_depth_coordinates(
                parent_coordinates, child_coordinates
            )
            / 1000
        )
        # TODO: Add exponential random variable here to perturb the jump delay.
        return tinit + jump_distance_km / 3.5

    def process_fault(fault_name: str) -> None:
        """Delay fault SRF according to rupture delay from parents + propagation speed across the gap between faults.

        Parameters
        ----------
        fault_name : str
            The fault to process.
        """
        fault = faults[fault_name]
        parent_fault_name = rupture_propagation_config.rupture_causality_tree.get(
            fault_name
        )
        fault_srf_path = output_directory / "srf" / f"{normalise_name(fault_name)}.srf"
        fault_srf = srf.read_srf(fault_srf_path)

        if parent_fault_name:
            jump_point = rupture_propagation_config.jump_points[fault_name]
            # For all SRFs except the initial SRF, there is no hypocentre,
            # so we set shyp, dhyp to -999. This means "no hypocentre in
            # this segment".
            fault_srf.header["shyp"] = -999
            fault_srf.header["dhyp"] = -999

            parent_srf = srf_file_map[parent_fault_name]
            parent_fault = faults[parent_fault_name]

            parent_coordinates = (
                parent_fault.fault_coordinates_to_wgs_depth_coordinates(
                    jump_point.from_point
                )
            )
            child_coordinates = fault.fault_coordinates_to_wgs_depth_coordinates(
                jump_point.to_point
            )

            jump_index = int(
                np.argmin(
                    coordinates.distance_between_wgs_depth_coordinates(
                        parent_srf.points[["lat", "lon", "dep"]].to_numpy()
                        * np.array([1, 1, 1000]),
                        parent_coordinates,
                    )
                )
            )
            tinit = parent_srf.points["tinit"].iloc[jump_index]
            time_delay = compute_rupture_delay(
                tinit, parent_coordinates, child_coordinates
            )

            logger = log_utils.get_logger(__name__)
            fault_srf.points["tinit"] += time_delay
            logger.info(
                "computed delay",
                fault_name=fault_name,
                delay=time_delay,
                srf_min=fault_srf.points["tinit"].min(),
            )

        srf_file_map[fault_name] = fault_srf

    # Process each fault in the determined order
    for fault_name in srf_stitching_order:
        process_fault(fault_name)

    # Combine SRF file components
    combined_slipt_array = concatenate_slip_values(
        (
            fault_srf.slipt1_array
            if fault_srf.slipt1_array is not None
            else csr_array((len(fault_srf.points), 1))
        )
        for fault_srf in srf_file_map.values()
    )
    assert combined_slipt_array is not None
    combined_srf = srf.SrfFile(
        version="1.0",
        header=pd.concat([fault_srf.header for fault_srf in srf_file_map.values()]),
        points=pd.concat([fault_srf.points for fault_srf in srf_file_map.values()]),
        slipt1_array=combined_slipt_array,
    )

    # Write the combined SRF file
    srf.write_srf(output_srf_path, combined_srf)

    return output_srf_path


@dataclasses.dataclass
class SRFRealisationContext:
    """Realisation configuration for the entire SRF generation process."""

    resolution: Resolution
    """The spatial/temporal resolution"""
    source_config: SourceConfig
    """The sources to generate"""
    rupture_propagation_config: RupturePropagationConfig
    """The rupture propagation for the SRF to follow"""
    magnitudes: Magnitudes
    """The magnitudes of the component faults for individual SRF generation"""
    rakes: Rakes
    """The rakes of the component faults for individual SRF generation"""
    velocity_model_1d: VelocityModel1D
    """The 1D velocity model to use for SRF generation"""
    srf_config: SRFConfig
    """SRF configuration options to apply"""


@dataclasses.dataclass
class SRFEnvironmentContext:
    """Environment context for working directory, output files, and binary paths."""

    genslip_path: Path
    """Path to genslip binary"""
    generic_slip2srf_path: Path
    """Path to generic_slip2srf binary"""
    work_directory: Path
    """Directory to output component SRF files, geometry files"""
    seeds: Seeds
    """Random seeds to apply for reproducibility"""

    @property
    def srf_directory(self) -> Path:  # numpydoc ignore=RT01
        """Path: the directory to write component SRF files."""
        return self.work_directory / "srf"

    @property
    def gsf_directory(self) -> Path:  # numpydoc ignore=RT01
        """Path: the directory to write geometry definition files."""
        return self.work_directory / "gsf"

    @property
    def velocity_model_path(self) -> Path:  # numpydoc ignore=RT01
        """Path: the directory to write the 1D velocity model to."""
        return self.work_directory / "velocity_model"


def generate_fault_srf(
    name: str, params: SRFRealisationContext, environment: SRFEnvironmentContext
) -> None:
    """Generate an SRF file for a given fault.

    Parameters
    ----------
    name : str
        The name of the fault.
    params : SRFRealisationContext
        The SRF realisation context to use.
    environment : SRFEnvironmentContext
        The SRF environment context to use.
    """
    fault = params.source_config.source_geometries[name]

    if isinstance(fault, Point):
        generate_point_source_srf(name, params, environment)
        # Return None here as no other code in this function should be run if generating a point source SRF.
        return None

    resolution = params.srf_config.resolution

    if isinstance(fault, Fault):
        nx = sum(round(plane.length / resolution) for plane in fault.planes)
    else:
        nx = round(fault.length / resolution)
    ny = round(fault.width / resolution)

    gsf_file_path = generate_fault_gsf(
        name,
        fault,
        params.rakes.rakes[name],
        environment.gsf_directory,
        resolution,
    )

    genslip_hypocentre_coords = np.array([fault.length, fault.width]) * (
        params.rupture_propagation_config.hypocentres[name] - np.array([1 / 2, 0])
    )
    gwid = (
        ",".join([str(x) for x in params.srf_config.gwid])
        if params.srf_config.gwid
        else -1
    )
    rvfac_seg = (
        ",".join([str(x) for x in params.srf_config.rvfac_seg])
        if params.srf_config.rvfac_seg
        else -1
    )

    genslip_cmd = [
        str(environment.genslip_path),
        "plane_header=1",
        "srf_version=1.0",
        "read_erf=0",
        "write_srf=1",
        "read_gsf=1",
        "write_gsf=0",
        "ns=1",
        "nh=1",
        f"infile={gsf_file_path}",
        f"mag={params.magnitudes.magnitudes[name]}",
        f"nstk={nx}",
        f"ndip={ny}",
        f"seed={environment.seeds.genslip_seed}",
        f"velfile={environment.velocity_model_path}",
        f"shypo={genslip_hypocentre_coords[0]}",
        f"dhypo={genslip_hypocentre_coords[1]}",
        f"dt={params.resolution.dt}",
        f"side_taper={params.srf_config.side_taper}",
        f"bot_taper={params.srf_config.bot_taper}",
        f"top_taper={params.srf_config.top_taper}",
        f"alpha_rough={params.srf_config.alpha_rough}",
        f"gwid={gwid}",
        f"rvfac_seg={rvfac_seg}",
        f"seg_delay={int(params.srf_config.seg_delay)}",
    ]

    genslip_cmd.extend(
        f"{key}={getattr(params.srf_config, key)}"
        for key in ["ymag_exp", "xmag_exp", "kx_corner", "ky_corner"]
        if getattr(params.srf_config, key) is not None
    )

    srf_file_path = environment.srf_directory / (normalise_name(name) + ".srf")
    with open(srf_file_path, "w", encoding="utf-8") as srf_file_handle:
        logger = log_utils.get_logger(__name__)
        logger.info("executing command", cmd=" ".join(genslip_cmd))
        try:
            proc = subprocess.run(
                genslip_cmd, stdout=srf_file_handle, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                "failed",
                exception=e.output.decode("utf-8"),
                code=e.returncode,
                stderr=e.stderr.decode("utf-8"),
            )
            raise
        logger.info("command completed", stderr=proc.stderr.decode("utf-8"))


def generate_fault_srfs_multi(
    faults: dict[str, IsSource],
    params: SRFRealisationContext,
    environment: SRFEnvironmentContext,
    single_threaded: bool = False,
) -> None:
    """Generate fault SRF files in parallel.

    Parameters
    ----------
    faults : dict[str, IsSource]
        The faults and their geometries.
    params : SRFRealisationContext
        The SRF realisation context to use.
    environment : SRFEnvironmentContext
        The SRF environment context to use.
    single_threaded : bool, optional
        If True, generate each segment SRF on a single thread.
        Defaults to False, which uses multiprocessing to generate SRFs in parallel.
    """
    # need to do this before multiprocessing because of race conditions
    environment.srf_directory.mkdir(exist_ok=True)
    environment.gsf_directory.mkdir(exist_ok=True)
    params.velocity_model_1d.write_velocity_model(environment.velocity_model_path)

    single_threaded = single_threaded or len(faults) == 1
    if single_threaded:
        for name in faults:
            generate_fault_srf(name, params, environment)
    else:
        with multiprocessing.Pool(utils.get_available_cores()) as worker_pool:
            worker_pool.map(
                functools.partial(
                    generate_fault_srf, params=params, environment=environment
                ),
                list(faults),
            )


def generate_point_source_srf(
    name: str,
    params: SRFRealisationContext,
    environment: SRFEnvironmentContext,
) -> None:
    """Generate an SRF file for a given fault.

    Parameters
    ----------
    name : str
        The name of the fault.
    params : SRFRealisationContext
        The SRF realisation context to use.
    environment : SRFEnvironmentContext
        The SRF environment context to use.

    Returns
    -------
    None
        This function does not return a value; it writes the SRF file to disk.
    """

    if params.srf_config.point_source_params is None:
        raise ValueError(
            f"Point source parameters are required for generating SRF for fault '{name}'"
        )

    fault = params.source_config.source_geometries[name]

    resolution = params.resolution.resolution

    # Get magnitude and convert to seismic moment
    magnitude = params.magnitudes.magnitudes[name]
    moment_newton_metre = moment.magnitude_to_moment(magnitude)

    velocity_model_df = params.velocity_model_1d.model
    velocity_model_df["depth_km"] = velocity_model_df["thickness"].cumsum()

    # Get the source depth
    # divide by 1000 to convert depth from meters to kilometers
    source_depth_km = params.source_config.source_geometries[name].centroid[2] / 1000

    fault_area_km2 = (
        params.source_config.source_geometries[name].length
        * params.source_config.source_geometries[name].width
    )

    slip = moment.point_source_slip(
        moment_newton_metre, fault_area_km2, velocity_model_df, source_depth_km
    )

    gsf_file_path = generate_fault_gsf(
        name,
        fault,
        params.rakes.rakes[name],
        environment.gsf_directory,
        resolution,
        slip,
        init_time=params.srf_config.point_source_params.inittime,
    )

    generic_slip2srf_cmd = [
        str(environment.generic_slip2srf_path),
        f"infile={gsf_file_path}",
        f"outfile={environment.srf_directory / (normalise_name(name) + '.srf')}",
        "outbin=0",
        f"stype={params.srf_config.point_source_params.stype}",
        f"dt={params.resolution.dt}",
        "plane_header=1",
        f"risetime={params.srf_config.point_source_params.risetime}",
        f"risetimefac={params.srf_config.point_source_params.risetimefac}",
        f"risetimedep={params.srf_config.point_source_params.risetimedep}",
    ]

    logger = log_utils.get_logger(__name__)
    logger.info("executing command", cmd=" ".join(generic_slip2srf_cmd))

    # generic_slip2srf writes the file directly so there is no need to open a file handle
    try:
        proc = subprocess.run(
            generic_slip2srf_cmd,
            stderr=subprocess.PIPE,
            check=True,
        )

    except subprocess.CalledProcessError as e:
        logger.error(
            "failed",
            exception=e.output.decode("utf-8"),
            code=e.returncode,
            stderr=e.stderr.decode("utf-8"),
        )
        raise
    logger.info("command completed", stderr=proc.stderr.decode("utf-8"))


@cli.from_docstring(app)
@log_call()
def generate_srf(
    realisation_ffp: Annotated[
        Path, typer.Argument(exists=True, readable=True, dir_okay=False)
    ],
    output_srf_filepath: Annotated[Path, typer.Argument(writable=True, dir_okay=False)],
    work_directory: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "/out"
    ),
    genslip_path: Annotated[Path, typer.Option(readable=True, dir_okay=False)] = Path(
        "/EMOD3D/tools/genslip_v5.4.2"
    ),
    generic_slip2srf_path: Annotated[
        Path, typer.Option(readable=True, dir_okay=False)
    ] = Path("/EMOD3D/tools/generic_slip2srf"),
    single_threaded: Annotated[bool, typer.Option()] = False,
) -> None:
    """Generate an SRF file from a given realisation specification.

    This function reads the realisation metadata and configurations from the
    specified JSON file. It then generates fault SRF files using the genslip
    tool and stitches these files into a final SRF file. The SRF configuration
    is updated and written back to the realisation file. Finally, the resulting
    SRF file is copied to the specified output path.

    Parameters
    ----------
    realisation_ffp : Path
        The filepath of the JSON file containing the realisation data.
    output_srf_filepath : Path
        The filepath where the final SRF file will be saved.
    work_directory : Path, optional
        Path to output intermediate geometry and SRF files.
    genslip_path : Path, optional
        Path to the genslip binary.
    generic_slip2srf_path : Path, optional
        Path to the generic_slip2srf binary.

    single_threaded : bool, optional
        If True, generate each segment SRF on a single thread.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    srf_config = SRFConfig.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    seeds = Seeds.read_from_realisation_or_random(realisation_ffp)
    rupture_propagation = RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    velocity_model_1d = VelocityModel1D.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    rakes = Rakes.read_from_realisation(realisation_ffp)
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)
    source_config = SourceConfig.read_from_realisation(realisation_ffp)
    resolution = Resolution.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    params = SRFRealisationContext(
        resolution=resolution,
        source_config=source_config,
        rupture_propagation_config=rupture_propagation,
        magnitudes=magnitudes,
        rakes=rakes,
        velocity_model_1d=velocity_model_1d,
        srf_config=srf_config,
    )

    environment = SRFEnvironmentContext(
        work_directory=work_directory,
        genslip_path=genslip_path,
        generic_slip2srf_path=generic_slip2srf_path,
        seeds=seeds,
    )

    generate_fault_srfs_multi(
        source_config.source_geometries,
        params,
        environment,
        single_threaded=single_threaded,
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
    realisations.append_log_entry(realisation_ffp)
