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
from typing import Annotated

import numpy as np
import pandas as pd
import tqdm
import typer
import xarray as xr

from IM import im_reader, ims
from IM.im_calculation import IM
from qcore import cli, coordinates
from workflow import realisations, utils
from workflow.realisations import (
    IntensityMeasureCalculationParameters,
    RealisationMetadata,
    Resolution,
    RupturePropagationConfig,
    SourceConfig,
)

PSA_STEP = 10000

app = typer.Typer()


@cli.from_docstring(app)
def calculate_instensity_measures(
    realisation_ffp: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, writable=True)
    ],
    broadband_simulation_ffp: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_path: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
    simulated_stations: Annotated[bool, typer.Option()] = True,
    psa_step: Annotated[int, typer.Option()] = PSA_STEP,
    ko_directory: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    override_ims: Annotated[list[IM] | None, typer.Option("-i", "--im")] = None,
    cores: Annotated[int | None, typer.Option(min=1)] = None,
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
    psa_step : int
        Maximum number stations to read from disk at once for pSA calculation
    ko_directory : Path
        Directory containing the KO matrix files for FAS calculation. Not required for other IMs.
    override_ims : list of str
        Intensity measures to calculate. If not set, reads from the realisation file.
    cores : int or None
        Set the number of cores for parallel processing of IMs. If set
        to `None`, will default to the available cores from
        `utils.get_available_cores`.
    """
    cores = cores or utils.get_available_cores()

    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    resolution = Resolution.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    intensity_measure_parameters = (
        IntensityMeasureCalculationParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )
    source_geometries = SourceConfig.read_from_realisation(realisation_ffp)
    rup_prop_config = RupturePropagationConfig.read_from_realisation(realisation_ffp)

    broadband = xr.open_dataset(broadband_simulation_ffp)

    if not simulated_stations:
        broadband = broadband.where(
            broadband.station.str.match(r"^(\w{4})$"), drop=True
        )

    intensity_measures = override_ims or intensity_measure_parameters.ims

    if IM.FAS in intensity_measures and not ko_directory:
        raise ValueError(
            "FAS calculation requires KO directory. Please provide a valid KO directory."
        )

    nyquist_frequency = 1 / (2 * resolution.dt)

    im_function_map = {
        IM.PGA: functools.partial(ims.peak_ground_acceleration, cores=cores),
        IM.PGV: functools.partial(
            ims.peak_ground_velocity, dt=resolution.dt, cores=cores
        ),
        IM.PGD: functools.partial(
            ims.peak_ground_displacement, dt=resolution.dt, cores=cores
        ),
        IM.CAV: functools.partial(
            ims.cumulative_absolute_velocity, dt=resolution.dt, cores=cores
        ),
        IM.AI: functools.partial(ims.arias_intensity, dt=resolution.dt, cores=cores),
        IM.Ds575: functools.partial(ims.ds575, dt=resolution.dt, cores=cores),
        IM.Ds595: functools.partial(ims.ds595, dt=resolution.dt, cores=cores),
        IM.pSA: functools.partial(
            ims.pseudo_spectral_acceleration,
            periods=np.array(
                intensity_measure_parameters.valid_periods, dtype=np.float64
            ),
            dt=resolution.dt,
            step=psa_step,
            cores=cores,
        ),
        IM.FAS: functools.partial(
            ims.fourier_amplitude_spectra,
            dt=resolution.dt,
            freqs=intensity_measure_parameters.fas_frequencies[
                intensity_measure_parameters.fas_frequencies <= nyquist_frequency
            ],
            ko_directory=ko_directory,
            cores=cores,
        ),
    }
    latitude = broadband.latitude.values
    longitude = broadband.longitude.values

    rrup = (
        np.array(
            [
                min(
                    source.rrup_distance(np.append(station, 0))
                    for source in source_geometries.source_geometries.values()
                )
                for station in np.stack((latitude, longitude), axis=-1)
            ]
        )
        / 1000
    )
    rjb = (
        np.array(
            [
                min(
                    source.rjb_distance(np.append(station, 0))
                    for source in source_geometries.source_geometries.values()
                )
                for station in np.stack((latitude, longitude), axis=-1)
            ]
        )
        / 1000
    )
    hypocentre = source_geometries.source_geometries[
        rup_prop_config.initial_fault
    ].fault_coordinates_to_wgs_depth_coordinates(rup_prop_config.hypocentre)

    hyp = (
        coordinates.distance_between_wgs_depth_coordinates(
            np.stack((latitude, longitude, np.zeros_like(latitude)), axis=-1),
            hypocentre,
        )
        / 1000
    )
    epi = (
        coordinates.distance_between_wgs_depth_coordinates(
            np.stack((latitude, longitude), axis=-1),
            hypocentre[:2],
        )
        / 1000
    )

    dataset = xr.Dataset(
        coords={
            "station": ("station", broadband.station.values),
            "component": (
                "component",
                ["000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100"],
            ),
            "rrup": ("station", rrup),
            "rjb": ("station", rjb),
            "hyp": ("station", hyp),
            "epi": ("station", epi),
        },
        attrs={"hypo_lat": hypocentre[0], "hypo_lon": hypocentre[1]},
    )

    waveform = broadband.waveform.values.astype(np.float64)

    for im_name in (pbar := tqdm.tqdm(intensity_measures)):
        pbar.set_description(im_name)
        im_fn = im_function_map[im_name]

        result = im_fn(waveform)

        if isinstance(result, pd.DataFrame):
            result["station"] = broadband.station.values
            result = result.set_index("station").to_xarray().to_array(dim="component")
        elif isinstance(result, xr.DataArray):
            result = result.assign_coords(station=broadband.station)
        dataset[im_name] = result
        im_reader.write_intensity_measures(dataset, output_path)

    realisations.append_log_entry(realisation_ffp)
