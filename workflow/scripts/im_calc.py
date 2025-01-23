"""Intensity Measure Calculation.

Description
-----------
Calculate intensity measures from broadband waveform files.

Inputs
------
A realisation file containing metadata configuration.

Typically, this information comes from a stage like [NSHM To Realisation](#nshm-to-realisation).

Outputs
-------
A CSV containing intensity measure summary statistics.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `im-calc` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`im-calc [OPTIONS] REALISATION_FFP BROADBAND_SIMULATION_FFP OUTPUT_PATH`

For More Help
-------------
See the output of `im-calc --help`.
"""

import functools
from pathlib import Path
from typing import Annotated, Optional

import h5py
import numexpr as ne
import numpy as np
import pandas as pd
import tqdm
import typer
import xarray as xr

from IM import ims
from IM.im_calculation import IM
from workflow import utils
from workflow.realisations import (
    BroadbandParameters,
    IntensityMeasureCalculationParameters,
    RealisationMetadata,
)

app = typer.Typer()


@app.command(help="Calculate instensity measures for simulation data.")
def calculate_instensity_measures(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(
            help="Realisation filepath", exists=True, dir_okay=False, writable=True
        ),
    ],
    broadband_simulation_ffp: Annotated[
        Path,
        typer.Argument(help="Broadband simulation file.", exists=True, dir_okay=False),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Output path for IM calculation summary statistics.",
            dir_okay=False,
            writable=True,
        ),
    ],
    simulated_stations: Annotated[
        bool, typer.Option(help="If passed, calculate for simulated stations.")
    ] = True,
    psa_rotd_maximum_memory_allocation: Annotated[
        Optional[float],
        typer.Option(
            help="Maximum amount of memory allocated for rotated PSA calculation station buffer, in gigabytes.",
            min=0,
        ),
    ] = None,
):
    """Calculate intensity measures for simulation data.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file.
    broadband_simulation_ffp : Path
        Path to the broadband simulation waveforms.
    output_path : Path
        Output directory for IM calc summary statistics.
    """
    ne.set_num_threads(utils.get_available_cores())

    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_parameters = BroadbandParameters.read_from_realisation(realisation_ffp)
    intensity_measure_parameters = (
        IntensityMeasureCalculationParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )

    with h5py.File(broadband_simulation_ffp, mode="r") as broadband_file:
        waveforms = np.array(broadband_file["waveforms"]).astype(np.float32)

    stations = pd.read_hdf(broadband_simulation_ffp, key="stations")
    if not simulated_stations:
        stations = stations.filter(regex=r"^\w{4}$", axis=0)
        waveforms = waveforms[stations["waveform_index"]]

    intensity_measures = intensity_measure_parameters.ims
    nyquist_frequency = 1 / (2 * broadband_parameters.dt)
    im_function_map = {
        IM.PGA: ims.peak_ground_acceleration,
        IM.PGV: functools.partial(ims.peak_ground_velocity, dt=broadband_parameters.dt),
        IM.CAV: functools.partial(
            ims.cumulative_absolute_velocity, dt=broadband_parameters.dt
        ),
        IM.AI: functools.partial(ims.arias_intensity, dt=broadband_parameters.dt),
        IM.Ds575: functools.partial(
            ims.ds575,
            dt=broadband_parameters.dt,
        ),
        IM.Ds595: functools.partial(
            ims.ds595,
            dt=broadband_parameters.dt,
        ),
        IM.pSA: functools.partial(
            ims.pseudo_spectral_acceleration,
            periods=np.array(
                intensity_measure_parameters.valid_periods, dtype=np.float32
            ),
            dt=broadband_parameters.dt,
            psa_rotd_maximum_memory_allocation=psa_rotd_maximum_memory_allocation * 1e9
            if psa_rotd_maximum_memory_allocation
            else None,
            cores=utils.get_available_cores(),
        ),
        IM.FAS: functools.partial(
            ims.fourier_amplitude_spectra,
            dt=broadband_parameters.dt,
            freqs=intensity_measure_parameters.fas_frequencies[
                intensity_measure_parameters.fas_frequencies <= nyquist_frequency
            ],
            cores=utils.get_available_cores(),
        ),
    }
    for im_name in (pbar := tqdm.tqdm(intensity_measures)):
        pbar.set_description(im_name)
        im_fn = im_function_map[im_name]
        result = im_fn(waveforms)
        if isinstance(result, pd.DataFrame):
            result["station"] = stations.index.values
            result = result.set_index("station")
            result.to_hdf(output_path, key=im_name, mode="a")
        elif isinstance(result, xr.DataArray):
            result = result.assign_coords(station=stations.index.values)
            # NetCDF is a file format that is compatible with HDF5. For
            # legacy reasons, xarray uses `to_netcdf` instead of `to_hdf5`.
            result.to_netcdf(output_path, mode="a", group=im_name, engine="h5netcdf")
