#!/usr/bin/env python
"""
Simulates high frequency seismograms for stations.
"""

import subprocess
from pathlib import Path
from typing import Annotated, Any, Iterable, Optional

import numpy as np
import typer

HEAD_STAT = 0x18
FLOAT_SIZE = 0x4

# never changed / unknown function (line 6)
nbu = 4
ift = 0
flo = 0.02
fhi = 19.9
# for line 15
nl_skip = -99
vp_sig = 0.0
vsh_sig = 0.0
rho_sig = 0.0
qs_sig = 0.0
ic_flag = True
# seems to store details in {velocity_name}_{station_name}.1d if not '-1'
velocity_name = "-1"


def format_hf_input(input_lines: Iterable[Any]) -> str:
    return "\n".join(str(line) for line in input_lines)


def run_hf(
    stoch_ffp: Annotated[
        Path,
        typer.Argument(help="Input stoch file.", exists=True),
    ],
    station_file: Annotated[
        Path, typer.Argument(help="Location of station file.", exists=True)
    ],
    out_file: Annotated[
        Path, typer.Argument(help="Filepath for HF output.", file_okay=False)
    ],
    hf_sim_path: Annotated[Path, typer.Option(help="Path to HF sim binary")] = Path(
        "/EMOD3D/tools/hf"
    ),
    t_sec: Annotated[float, typer.Option(help="High frequency output start time.")] = 0,
    sdrop: Annotated[float, typer.Option(help="Stress drop average (bars)")] = 50,
    rayset: Annotated[list[int], typer.Option(help="ray types 1:direct 2:moho")] = [
        1,
        2,
    ],
    no_siteamp: Annotated[
        bool, typer.Option(help="Disable BJ97 site amplification factors.")
    ] = True,
    seed: Annotated[
        int, typer.Option(help="random seed (0: randomised reproducible)")
    ] = 0,
    duration: Annotated[float, typer.Option(help="Output length (s).")] = 100,
    dt: Annotated[float, typer.Option(help="Timestep (s).")] = 0.005,
    fmax: Annotated[float, typer.Option(help="Max simulation frequency (Hz)")] = 10,
    kappa: Annotated[float, typer.Option()] = 0.045,
    qfexp: Annotated[float, typer.Option(help="Q frequency exponent.")] = 0.6,
    rvfac: Annotated[
        float, typer.Option(help="Rupture velocity factor (rupture : Vs)")
    ] = 0.8,
    rvfac_shal: Annotated[
        float, typer.Option(help="rvfac shallow fault multiplier")
    ] = 0.7,
    rvfac_deep: Annotated[
        float, typer.Option(help="rvfac deep fault multiplier")
    ] = 0.7,
    czero: Annotated[
        Optional[float],
        typer.Option(help="C0 coefficient, if not specified use binary default"),
    ] = 2.1,
    calpha: Annotated[
        Optional[float],
        typer.Option(help="Ca coefficient, if not specified use binary default"),
    ] = None,
    mom: Annotated[
        Optional[float],
        typer.Option(help="Seismic moment. If not specified use binary default."),
    ] = None,
    rupv: Annotated[
        Optional[float],
        typer.Option(help="Rupture Velocity. If not specified use rupture model."),
    ] = None,
    velocity_model: Annotated[
        Path,
        typer.Option(
            help="Path to velocity model (1D). Ignored if --site-specific is set"
        ),
    ] = Path("/Cant1D_v2-midQ_leer.1d"),
    site_specific: Annotated[
        bool, typer.Option(help="Enable site-specific calculation.")
    ] = False,
    site_velocity_model_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory containing site specific velocity models. Requires --site-specific."
        ),
    ] = None,
    vs_moho: Annotated[float, typer.Option(help="vs of moho layer")] = 999.9,
    fa_sig1: Annotated[
        float, typer.Option(help="fourier amplitute uncertainty (1)")
    ] = 0,
    fa_sig2: Annotated[
        float, typer.Option(help="fourier amplitude uncertainty (2)")
    ] = 0,
    rv_sig1: Annotated[float, typer.Option(help="Rupture velocity uncertainty.")] = 0.1,
    path_dur: Annotated[
        float,
        typer.Option(
            help="path duration model. 0: GP2010, 1: WUS modification trail/errol, 2: ENA modificiation trial/error"
            ", 11: WUS formutian of BT2014, 12: ENA formulation of BT2015. Models 11 and 12 overpredict for multiple rays."
        ),
    ] = 1,
    dpath_pert: Annotated[
        float, typer.Option(help="Log of path duration multiplier")
    ] = 0,
    stress_parameter_adjustment_tect_type: Annotated[
        int,
        typer.Option(
            min=0,
            max=2,
            help="Adjustment option 0 = off, 1 = active tectonic, 2 = stable continent.",
        ),
    ] = 1,
    stress_parameter_adjustment_target_magnitude: Annotated[
        Optional[float],
        typer.Option(help="Target magnitude (if not specified, infer this value)."),
    ] = None,
    stress_parameter_adjustment_fault_area: Annotated[
        Optional[float], typer.Option(help="Fault area (if not specified, infer value)")
    ] = None,
):
    with open(station_file, "r") as station_file_handle:
        number_of_stations = len(station_file_handle.readlines())

    hf_sim_input = format_hf_input(
        [
            "",
            sdrop,
            station_file,
            out_file,
            f"{len(rayset):d} {' '.join(str(ray) for ray in rayset)}",
            int(not no_siteamp),
            f"{nbu:d} {ift:d} {flo} {fhi}",
            seed,
            1,
            f"{duration} {dt} {fmax} {kappa} {qfexp}",
            f"{rvfac} {rvfac_shal} {rvfac_deep} {czero} {calpha}",
            f"{mom} {rupv}",
            velocity_model,
            vs_moho,
            f"{nl_skip:d} {vp_sig} {vsh_sig} {rho_sig} {qs_sig} {ic_flag:d}",
            velocity_name,
            f"{fa_sig1} {fa_sig2} {rv_sig1}",
            path_dur,
            f"{stress_parameter_adjustment_fault_area} {stress_parameter_adjustment_target_magnitude} {stress_parameter_adjustment_tect_type}",
            0,
            "",
        ]
    )
    with open(velocity_model, "r") as f:
        f.readline()
        vs = np.float32(float(f.readline().split()[2]) * 1000.0)

    # We run binary with subprocess.run rather than subprocess.check_call
    # because subprocess.run accepts the input argument.
    output = subprocess.run(
        str(hf_sim_path),
        input=hf_sim_input,
        check=True,
        text=True,
        stderr=subprocess.PIPE,
    )

    # for some reason, e_dist values are not part of the stdout
    e_dist = np.fromstring(output.stderr, dtype="f4", sep="\n")
    if e_dist.size != number_of_stations:
        print(f"Expected {number_of_stations} e_dist values, got {e_dist.size}")
        print(output.stderr)
        exit(1)

    # write e_dist and vs to file
    with open(out_file, "r+b") as out:
        for i in range(number_of_stations):
            out.seek(HEAD_STAT - 2 * FLOAT_SIZE, 1)
            e_dist[i].tofile(out)
            vs.tofile(out)
