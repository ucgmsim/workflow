"""Create EMOD3D parameters sourced from a realisation file."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from workflow import defaults, realisations
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    VelocityModelParameters,
)


def emod3d_domain_parameters(
    domain_parameters: DomainParameters,
) -> dict[str, int | float]:
    """Create a dictionary of the EMOD3D domain parameters.

    Parameters
    ----------
    domain_parameters : DomainParameters
        The realisation domain parameters.

    Returns
    -------
    dict[str, int | float]
        A dictionary containing the EMOD3D domain parameters.
    """
    nx = int(np.ceil(domain_parameters.domain.extent_x / domain_parameters.resolution))
    ny = int(np.ceil(domain_parameters.domain.extent_y / domain_parameters.resolution))
    nz = int(np.ceil(domain_parameters.depth / domain_parameters.resolution))
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "h": domain_parameters.resolution,
        "modellat": domain_parameters.domain.origin[0],
        "modellon": domain_parameters.domain.origin[1],
        "modelrot": domain_parameters.domain.bearing,
    }


def emod3d_duration_parameters(
    domain_parameters: DomainParameters, min_vs: float, dtts: float
) -> dict[str, int | float]:
    """Create a dictionary of the EMOD3D duration parameters.

    Parameters
    ----------
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
    flo = min_vs / (5 * domain_parameters.resolution)
    extended_simulation_duration = domain_parameters.duration + 3 / flo
    nt = int(np.round(extended_simulation_duration / domain_parameters.dt))
    return {
        "nt": nt,
        "dump_itinc": nt,
        "flo": flo,
        "dt": domain_parameters.dt,
        "ts_total": int(extended_simulation_duration / (domain_parameters.dt * dtts)),
    }


def emod3d_input_directories(
    srf_file_ffp: Path, velocity_model_ffp: Path, stations_ffp: Path, grid_ffp: Path
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
    grid_ffp : Path
        The path to the grid and model parameter files.

    Raises
    ------
    ValueError
        If any of the specified files or directories do not exist.

    Returns
    -------
    dict[str, Path]
        A dictionary of all the configured input directories.
    """
    # GRIDFILE & model_params: generated after VM gen,
    input_paths = {
        "faultfile": srf_file_ffp,
        "seiscords": stations_ffp / "stations.statcords",
        "stat_file": stations_ffp / "stations.ll",
        "grid_file": grid_ffp / "grid_file",
        "model_params": grid_ffp / "model_params",
        "vel_mod_params_dir": velocity_model_ffp,
        "vmoddir": velocity_model_ffp,
    }
    for key, path in input_paths.items():
        if not path.exists():
            raise ValueError(
                f"The {key} path does not exist. The path given was {path}"
            )
    return input_paths


def emod3d_output_directories(scratch_ffp: Path) -> dict[str, Path]:
    """Create a dictionary of the output directories for EMOD3D.

    This function also creates all the directories if they do not already exist.

    Parameters
    ----------
    scratch_ffp : Path
        The root directory of all output files for the run.

    Returns
    -------
    dict[str, Path]
        A dictionary of all the configured output paths.
    """
    directories = {
        "main_dump_dir": scratch_ffp / "OutBin",
        "user_scratch": scratch_ffp,
        "sim_dir": scratch_ffp,
        "seisdir": scratch_ffp / "SeismoBin",
        "restartdir": scratch_ffp / "Restart",
        "logdir": scratch_ffp / "Log",
        "ts_file": scratch_ffp / "OutBin" / "waveform_xyts.e3d",
        "ts_out_dir": scratch_ffp / "TSFiles",
        "slipout": scratch_ffp / "SlipOut",
    }
    for directory in directories.values():
        directory.mkdir(exist_ok=True)
    return directories


def emod3d_metadata(
    metadata: RealisationMetadata, emod3d_program: Path, emod3d_version: str
) -> dict[str, str | Path]:
    """Return a dictionary of the EMOD3D metadata parameters.

    Parameters
    ----------
    metadata : RealisationMetadata
        The realisation metadata.
    emod3d_program : Path
        The path to the EMOD3D program.
    emod3d_version : str
        The version of EMOD3D to use.

    Returns
    -------
    dict[str, str | Path]
        A dictionary containing the metadata parameters for EMOD3D.
    """
    return {
        "wcc_prog_dir": emod3d_program,
        "version": f"{emod3d_version}-mpi",
        "name": metadata.name,
        "restartname": metadata.name,
    }


def format_as_emod3d_value(value: int | float | str | Path) -> str:
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


def create_e3d_par(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to realisation JSON", exists=True, readable=True, dir_okay=False
        ),
    ],
    srf_file_ffp: Annotated[
        Path,
        typer.Argument(help="SRF filepath", exists=True, readable=True, dir_okay=False),
    ],
    velocity_model_ffp: Annotated[
        Path,
        typer.Argument(
            help="Velocity model filepath", exists=True, readable=True, file_okay=False
        ),
    ],
    stations_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to station files", exists=True, readable=True, file_okay=False
        ),
    ],
    grid_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to grid coordinate directory",
            exists=True,
            readable=True,
            file_okay=False,
        ),
    ],
    output_ffp: Annotated[
        Path,
        typer.Argument(help="Output path", writable=True, file_okay=False),
    ],
    scratch_ffp: Annotated[
        Path,
        typer.Option(
            help="Scratch filepath for intermediate output.",
            writable=True,
            file_okay=False,
        ),
    ] = Path("/out"),
    defaults_version: Annotated[
        str, typer.Option(help="The version of the EMOD3D defaults to use.")
    ] = "22.2.2.1",
    emod3d_path: Annotated[
        Path,
        typer.Option(
            help="The path to the EMOD3D binary.",
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ] = Path("/EMOD3D/tools/emod3d-mpi_v3.0.8"),
    emod3d_version: Annotated[
        str, typer.Option(help="The version of the EMOD3D binary to use.")
    ] = "3.0.8",
):
    """Create EMOD3D parameters sourced from a realisation file."""
    output_ffp.mkdir(exist_ok=True)
    scratch_ffp.mkdir(exist_ok=True)
    domain_parameters: DomainParameters = realisations.read_config_from_realisation(
        DomainParameters, realisation_ffp
    )
    velocity_model_parameters: VelocityModelParameters = (
        realisations.read_config_from_realisation(
            VelocityModelParameters, realisation_ffp
        )
    )
    metadata: RealisationMetadata = realisations.read_config_from_realisation(
        RealisationMetadata, realisation_ffp
    )
    emod3d_defaults = defaults.load_emod3d_defaults(defaults_version)
    emod3d_parameters = (
        emod3d_defaults
        | emod3d_domain_parameters(domain_parameters)
        | emod3d_duration_parameters(
            domain_parameters,
            min_vs=velocity_model_parameters.min_vs,
            dtts=emod3d_defaults["dtts"],
        )
        | emod3d_input_directories(
            srf_file_ffp, velocity_model_ffp, stations_ffp, grid_ffp
        )
        | emod3d_output_directories(scratch_ffp)
        | emod3d_metadata(metadata, emod3d_path, emod3d_version)
    )
    e3d_par_ffp = scratch_ffp / "e3d.par"

    e3d_par_ffp.write_text(
        "\n".join(
            f"{key}={format_as_emod3d_value(value)}"
            for key, value in emod3d_parameters.items()
        )
    )


def main():
    typer.run(create_e3d_par)


if __name__ == "__main__":
    main()
