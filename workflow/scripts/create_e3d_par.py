"""Create EMOD3D Parameters.

Description
-----------
Write parameters for EMOD3D simulation.

Inputs
------
1. A realisation file containing domain parameters, velocity model parameters, and realisation metadata,
2. An SRF file,
3. A generated velocity model,
4. Station coordinates.

Outputs
-------
An EMOD3D parameter file containing a mixture of simulations parameters. Parameters source values from the defaults specified the realisation defaults version. The `emod3d` section of the realisation file overrides default values.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `create-e3d-par` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`create-e3d-par [OPTIONS] REALISATION_FFP SRF_FILE_FFP VELOCITY_MODEL_FFP STATIONS_FFP GRID_FFP OUTPUT_FFP`

For More Help
-------------
See the output of `create-e3d-par --help`. See our description of the [EMOD3D Parameters](https://wiki.canterbury.ac.nz/pages/viewpage.action?pageId=100794983) for documentation on the EMOD3D parameter file format.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from qcore import cli
from workflow import realisations
from workflow.realisations import (
    DomainParameters,
    EMOD3DParameters,
    RealisationMetadata,
    Resolution,
    VelocityModelParameters,
)

app = typer.Typer()


def padded_nz(domain_parameters: DomainParameters, resolution: Resolution) -> int:
    """The number of z gridpoints EMOD3D reads, including the padding row.

    `velocity_modelling` generates one more layer than the domain implies,
    because EMOD3D shifts the model down a gridpoint for the free surface and
    so never reads the last layer (`genmodel.c`). EMOD3D has to be told the
    padded count, and the velocity model files are sized by it.

    Both the `e3d.par` `nz` and `check_domain_against_velocity_model` come
    through here. They must not compute it separately: a check that drifts
    from the value it is checking is worse than no check at all.

    Parameters
    ----------
    domain_parameters : DomainParameters
        The realisation domain parameters.
    resolution : Resolution
        The simulation resolution.

    Returns
    -------
    int
        The z gridpoint count including the free-surface padding row.
    """
    return domain_parameters.nz(resolution.resolution) + 1


def emod3d_domain_parameters(
    resolution: Resolution,
    domain_parameters: DomainParameters,
) -> dict[str, int | float]:
    """Create a dictionary of the EMOD3D domain parameters.

    Parameters
    ----------
    resolution : Resolution
        The realisation resolution parameters.
    domain_parameters : DomainParameters
        The realisation domain parameters.

    Returns
    -------
    dict[str, int | float]
        A dictionary containing the EMOD3D domain parameters.
    """

    nx = domain_parameters.nx(resolution.resolution)
    ny = domain_parameters.ny(resolution.resolution)
    nz = padded_nz(domain_parameters, resolution)
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "h": resolution.resolution,
        "modellat": float(domain_parameters.domain.origin[0]),
        "modellon": float(domain_parameters.domain.origin[1]),
        "modelrot": float(domain_parameters.domain.great_circle_bearing),
    }


def emod3d_duration_parameters(
    resolution: Resolution,
    domain_parameters: DomainParameters,
    min_vs: float,
    dtts: float,
) -> dict[str, int | float]:
    """Create a dictionary of the EMOD3D duration parameters.

    Parameters
    ----------
    resolution : Resolution
        The realisation resolution parameters.
    domain_parameters : DomainParameters
        The domain parameters.
    min_vs : float
        The minimum velocity.
    dtts : float
        The number of dt-increments per timestep.

    Returns
    -------
    dict[str, int | float]
        A dictionary containing the EMOD3D duration parameters.
    """
    flo = min_vs / (5 * resolution.resolution)
    extended_simulation_duration = domain_parameters.duration + 3 / flo
    nt = int(np.round(extended_simulation_duration / resolution.dt))
    return {
        "nt": nt,
        "dump_itinc": nt,
        "flo": flo,
        "dt": resolution.dt,
        "ts_total": int(extended_simulation_duration / (resolution.dt * dtts)),
        "restart_itinc": round(nt / 3),
    }


def emod3d_input_directories(
    srf_file_ffp: Path,
    velocity_model_ffp: Path,
    stations_ffp: Path,
) -> dict[str, Path]:
    """Create a dictionary of the input directories and files for EMOD3D.

    Parameters
    ----------
    srf_file_ffp : Path
        The path to the SRF file.
    velocity_model_ffp : Path
        The path to the velocity model directory.
    stations_ffp : Path
        The path containing the station files.

    Raises
    ------
    ValueError
        If any of the specified files or directories do not exist.

    Returns
    -------
    dict[str, Path]
        A dictionary of all the configured input directories.
    """
    input_paths = {
        "faultfile": srf_file_ffp,
        "seiscords": stations_ffp / "stations.statcords",
        "vmoddir": velocity_model_ffp,
    }
    for key, path in input_paths.items():
        if not path.exists():
            raise ValueError(
                f"The {key} path does not exist. The path given was {path}"
            )
    return input_paths


def emod3d_outputs(metadata: RealisationMetadata, output_ffp: Path) -> dict[str, Path]:
    """Create a dictionary of the output directories for EMOD3D.

    This function also creates all the directories if they do not already exist.

    Parameters
    ----------
    metadata : RealisationMetadata
        The realisation metadata.
    output_ffp : Path
        The root directory of all output files for the run.

    Returns
    -------
    dict[str, Path]
        A dictionary of all the configured output paths.
    """
    outputs = {
        "main_dump_dir": output_ffp / "OutBin",
        "seisdir": output_ffp / "SeismoBin",
        "restartdir": output_ffp / "Restart",
        "logdir": output_ffp / "Log",
        "ts_out_dir": output_ffp / "TSFiles",
        "slipout": output_ffp / "SlipOut",
    }
    for directory in outputs.values():
        directory.mkdir(exist_ok=True)

    outputs["ts_file"] = output_ffp / "OutBin" / f"{metadata.name}_xyts.e3d"
    return outputs


def emod3d_metadata(
    metadata: RealisationMetadata, emod3d_version: str
) -> dict[str, str | Path]:
    """Return a dictionary of the EMOD3D metadata parameters.

    Parameters
    ----------
    metadata : RealisationMetadata
        The realisation metadata.
    emod3d_version : str
        The version of EMOD3D to use.

    Returns
    -------
    dict[str, str | Path]
        A dictionary containing the metadata parameters for EMOD3D.
    """
    return {
        "version": f"{emod3d_version}-mpi",
        "name": metadata.name,
        "restartname": metadata.name,
    }


def format_as_emod3d_value(value: float | str | Path) -> str:
    """Format a value in a format valid for an e3d.par file.

    Parameters
    ----------
    value : int | float | str | Path
        The value to format.

    Returns
    -------
    str
        The value formatted as a string.
    """
    if isinstance(value, (Path, str)):
        return f'"{value}"'
    else:
        return str(value)


def check_domain_against_velocity_model(
    domain_parameters: DomainParameters,
    resolution: Resolution,
    parameters: EMOD3DParameters,
    velocity_model_ffp: Path,
) -> None:
    """Validate that generated velocity model files match the expected domain size.

    Computes the expected file size for the pmodfile, smodfile, and dmodfile
    from the domain's nx, ny, nz (with the EMOD3D free-surface padding row added
    to nz), and compares it against the actual size of each file on disk. A
    mismatch is an error, but a missing file or unreadable file is not
    considered an error because this code is often run in a container with the
    paths only used for templating and hence it would fail on many workflows.

    Parameters
    ----------
    domain_parameters : DomainParameters
        Domain parameters used to compute the expected nx, ny, nz grid
        dimensions at the given resolution.
    resolution : Resolution
        Resolution at which to evaluate the domain's grid dimensions.
    parameters : EMOD3DParameters
        EMOD3D parameters providing the pmodfile, smodfile, and dmodfile
        filenames to check.
    velocity_model_ffp : Path
        Directory containing the velocity model files to validate.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a velocity model file's size does not match the expected size
        computed from the domain parameters.

    """
    nx = domain_parameters.nx(resolution.resolution)
    ny = domain_parameters.ny(resolution.resolution)
    nz = padded_nz(domain_parameters, resolution)

    expected_file_size = nx * ny * nz * np.float32().nbytes
    for filename in [parameters.pmodfile, parameters.smodfile, parameters.dmodfile]:
        velocity_model_file = velocity_model_ffp / filename
        try:
            file_size = velocity_model_file.stat().st_size
        except OSError as e:
            # Unreadable is not a mismatch. This stage is routinely run in a
            # container where the velocity model paths are only being
            # templated and nothing is on disk yet. Handled per file rather
            # than around the loop, so one absent file does not stop the
            # others from being checked.
            print(
                f"WARNING: could not validate domain parameters against velocity model supplied:\n{e}"
            )
            continue
        if file_size != expected_file_size:
            raise RuntimeError(
                f"Velocity model file {velocity_model_file} does not have the expected size (expected: {expected_file_size}, found: {file_size})"
            )


@cli.from_docstring(app)
def create_e3d_par(
    realisation_ffp: Path,
    srf_file_ffp: Path,
    velocity_model_ffp: Path,
    stations_ffp: Path,
    output_ffp: Path,
    emod3d_version: Annotated[str, typer.Option()] = "3.0.13",
) -> None:
    """Create EMOD3D parameter file from provided inputs.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the JSON file containing realisation data.
    srf_file_ffp : Path
        Path to the SRF file used in the simulation.
    velocity_model_ffp : Path
        Path to the velocity model file.
    stations_ffp : Path
        Path to the station files used in the simulation.
    output_ffp : Path
        Path to the directory for output files when running EMOD3D.
    emod3d_version : str, optional
        Version of the EMOD3D binary to use.
    """
    output_ffp.mkdir(exist_ok=True)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    velocity_model_parameters = VelocityModelParameters.read_from_realisation(
        realisation_ffp
    )
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    resolution = Resolution.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    emod3d_parameters = EMOD3DParameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    check_domain_against_velocity_model(
        domain_parameters, resolution, emod3d_parameters, velocity_model_ffp
    )

    e3d_par_values = (
        emod3d_parameters.to_dict()
        | emod3d_domain_parameters(resolution, domain_parameters)
        | emod3d_duration_parameters(
            resolution,
            domain_parameters,
            min_vs=velocity_model_parameters.min_vs,
            dtts=emod3d_parameters.dtts,
        )
        | emod3d_input_directories(srf_file_ffp, velocity_model_ffp, stations_ffp)
        | emod3d_outputs(metadata, output_ffp)
        | emod3d_metadata(metadata, emod3d_version)
    )
    e3d_par_ffp = output_ffp / "e3d.par"

    e3d_par_ffp.write_text(
        "\n".join(
            f"{key}={format_as_emod3d_value(value)}"
            for key, value in e3d_par_values.items()
        )
    )

    realisations.append_log_entry(realisation_ffp)
