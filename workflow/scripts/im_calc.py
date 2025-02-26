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

import numexpr as ne
import numpy as np
import pandas as pd
import tqdm
import typer
import xarray as xr

from IM import im_reader, ims
from IM.im_calculation import IM
from qcore import cli, coordinates
from workflow import utils
from workflow.realisations import (
    BroadbandParameters,
    IntensityMeasureCalculationParameters,
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
)

app = typer.Typer()


@cli.from_docstring(app)
def calculate_intensity_measures(
    realisation_ffp: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, writable=True)
    ],
    broadband_simulation_ffp: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_path: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    simulated_stations: Annotated[bool, typer.Option()] = True,
    psa_rotd_maximum_memory_allocation: Annotated[
        Optional[float], typer.Option(min=0)
    ] = None,
) -> None:
    """Calculate intensity measures for simulation data.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file.
    broadband_simulation_ffp : Path
        Path to the broadband simulation waveforms.
    output_path : Path
        Output directory for IM calc summary statistics.
    simulated_stations : bool, default True
        If passed, calculate for simulated stations.
    psa_rotd_maximum_memory_allocation : Optional[float]
        Maximum amount of memory allocated for rotated PSA calculation station buffer, in gigabytes.
    """
    ne.set_num_threads(utils.get_available_cores())

    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_parameters = BroadbandParameters.read_from_realisation(realisation_ffp)
    intensity_measure_parameters = (
        IntensityMeasureCalculationParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )
    source_geometries = SourceConfig.read_from_realisation(realisation_ffp)
    rup_prop_config = RupturePropagationConfig.read_from_realisation(realisation_ffp)
    bb_ds = xr.open_dataset(broadband_simulation_ffp, engine="h5netcdf")

    stations = bb_ds[["x", "y", "z", "lf_vs_ref", "epicentre_distance"]].to_dataframe()

    if not simulated_stations:
        stations = stations.filter(regex=r"^\w{4}$", axis=0)

    waveforms = bb_ds.sel(station=stations.index)["waveforms"].values

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
            psa_rotd_maximum_memory_allocation=(
                psa_rotd_maximum_memory_allocation * 1e9
                if psa_rotd_maximum_memory_allocation
                else None
            ),
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

    stations["rrup"] = (
        np.array(
            [
                min(
                    source.rrup_distance(np.append(station, 0))
                    for source in source_geometries.source_geometries.values()
                )
                for station in stations[["latitude", "longitude"]].values
            ]
        )
        / 1000
    )
    stations["rjb"] = (
        np.array(
            [
                min(
                    source.rjb_distance(np.append(station, 0))
                    for source in source_geometries.source_geometries.values()
                )
                for station in stations[["latitude", "longitude"]].values
            ]
        )
        / 1000
    )
    hypocentre = source_geometries.source_geometries[
        rup_prop_config.initial_fault
    ].fault_coordinates_to_wgs_depth_coordinates(rup_prop_config.hypocentre)
    stations["hyp"] = (
        coordinates.distance_between_wgs_depth_coordinates(
            np.hstack(
                (
                    stations[["latitude", "longitude"]].values,
                    np.zeros((len(stations), 1)),
                )
            ),
            hypocentre,
        )
        / 1000
    )
    stations["epi"] = (
        coordinates.distance_between_wgs_depth_coordinates(
            stations[["latitude", "longitude"]].values,
            hypocentre[:2],
        )
        / 1000
    )

    station_metadata = stations[
        list(set(im_reader.IM_METADATA) & set(stations.columns))
    ]

    dataset = xr.Dataset(coords={"station": station_metadata.index}, attrs=bb_ds.attrs)

    # Add each column of the DataFrame as a coordinate
    for column in station_metadata.columns:
        dataset = dataset.assign_coords({column: ("station", station_metadata[column])})

    for im_name in (pbar := tqdm.tqdm(intensity_measures)):
        pbar.set_description(im_name)
        im_fn = im_function_map[im_name]
        result = im_fn(waveforms)

        if isinstance(result, pd.DataFrame):
            result["station"] = stations.index.values
            result = result.set_index("station").to_xarray().to_array(dim="component")
        elif isinstance(result, xr.DataArray):
            result = result.assign_coords(station=stations.index.values)
        dataset[im_name] = result

    im_reader.write_intensity_measures(dataset, output_path)
