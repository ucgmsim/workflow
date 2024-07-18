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
    flo = min_vs / (5 * domain_parameters.resolution)
    extended_simulation_duration = domain_parameters.duration + 3 / flo
    nt = np.round(extended_simulation_duration / domain_parameters.dt)
    return {"nt": nt, "dump_itinc": nt, "flo": flo, "dt": domain_parameters.dt}


def emod3d_input_directories(
    srf_file_ffp: Path, velocity_model_ffp: Path, stations_ffp: Path
) -> dict[str, Path]:
    # GRIDFILE & model_params: generated after VM gen,
    return {
        "faultfile": srf_file_ffp,
        "seiscords": stations_ffp / "stations.statcords",
        "stat_file": stations_ffp / "stations.ll",
        "vel_mod_params_dir": velocity_model_ffp,
        "vmoddir": velocity_model_ffp,
    }


def emod3d_output_directories(scratch_ffp: Path) -> dict[str, Path]:
    directories = {
        "main_dump_dir": scratch_ffp / "OutBin",
        "user_scratch": scratch_ffp,
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
) -> dict[str, Path]:
    return {
        "wcc_prog_dir": emod3d_program,
        "version": f"{emod3d_version}-mpi",
        "name": metadata.name,
        "restartname": metadata.name,
        "n_proc": 512,
    }


def format_as_emod3d_value(value: int | float | str | Path) -> str:
    if isinstance(value, Path):
        return repr(str(value))
    else:
        return repr(value)


def write_emod3d_parameters(
    e3d_par_ffp: Path, emod3d_parameters: dict[str, int | float | str | Path]
) -> None:
    e3d_par_ffp.write_text(
        "\n".join(
            f"{key}={format_as_emod3d_value(value)}"
            for key, value in emod3d_parameters.items()
        )
    )


def run_emod3d(
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
    output_ffp: Annotated[
        Path,
        typer.Argument(help="Output path", exists=True, writable=True, file_okay=False),
    ],
    scratch_ffp: Annotated[
        Path,
        typer.Argument(
            help="Scratch filepath for intermediate output.",
            exists=True,
            writable=True,
            file_okay=False,
        ),
    ] = "/out",
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
    ] = "/EMOD3D/tools/emod3d-3.0.8-mpi",
    emod3d_version: Annotated[
        str, typer.Option(help="The version of the EMOD3D binary to use.")
    ] = "3.0.8",
):
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
            dtts=emod3d_defaults["ddts"],
        )
        | emod3d_input_directories(srf_file_ffp, velocity_model_ffp, stations_ffp)
        | emod3d_output_directories(scratch_ffp)
        | emod3d_metadata(metadata, emod3d_version)
    )
    write_emod3d_parameters(scratch_ffp / "e3d.par", emod3d_parameters)
