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
from typing import Annotated, Any

import numpy as np
import pandas as pd

# Importing pint_xarray registers pint units with xarray, allowing for
# unit-aware operations. It is not explicitly used, so we ignore the flake8
# F401 error.
import pint_xarray  # noqa: F401
import shapely
import tqdm
import typer
import xarray as xr

from IM import ims
from IM.ims import IM
from qcore import cli, coordinates
from source_modelling import sources
from source_modelling.sources import IsSource
from workflow import realisations, utils
from workflow.realisations import (
    DomainParameters,
    IntensityMeasureCalculationParameters,
    Magnitudes,
    RealisationMetadata,
    Resolution,
    RupturePropagationConfig,
    SourceConfig,
)

PSA_STEP = 10000

app = typer.Typer()


COORDINATE_METADATA = {
    "station": {"description": "Station identifiers"},
    "period": {"description": "Oscillation period", "units": "s"},
    "vs30": {
        "description": "Average shear-wave velocity to 30m depth",
        "units": "m/s",
    },
    "epi": {"description": "Epicentral distance", "units": "km"},
    "hyp": {"description": "Hypocentral distance", "units": "km"},
    "rrup": {"description": "Rupture distance", "units": "km"},
    "rjb": {"description": "Joyner-Boore distance", "units": "km"},
    "rx": {"description": "Generalised strike-parallel distance", "units": "km"},
    "ry": {"description": "Generalised strike-normal distance", "units": "km"},
    "latitude": {"description": "Station latitude", "units": "degrees"},
    "longitude": {"description": "Station longitude", "units": "degrees"},
    "frequency": {"description": "Frequency of motion", "units": "Hz"},
    "period": {"description": "Period of motion", "units": "s"},
}

IM_METADATA = {
    IM.PGA: "Peak ground acceleration",
    IM.PGV: "Peak ground velocity",
    IM.CAV: "Cumulative absolute velocity",
    IM.CAV5: "Cumulative absolute velocity (above 5 cm/s)",
    IM.AI: "Arias intensity",
    IM.Ds575: "Significant duration (5-75%)",
    IM.Ds595: "Significant duration (5-95%)",
    IM.pSA: "Pseudo-spectral acceleration",
    IM.FAS: "Fourier amplitude spectrum",
}


# The 'g0' unit is used for acceleration and is equivalent to 9.81 m/s^2. The
# reason for this is that 'g' is reserved for 'grams'. This is a decision
# made by the `pint` library, which is used to handle the units.
IM_UNITS = {
    IM.PGA: "g0",
    IM.PGV: "cm/s",
    IM.CAV: "m/s",
    IM.CAV5: "m/s",
    IM.AI: "m/s",
    IM.Ds575: "s",
    IM.Ds595: "s",
    IM.FAS: "g0 * s",
    IM.pSA: "g0",
}


def add_distances(
    dtree: xr.DataTree, distances: dict[str, xr.DataArray]
) -> xr.DataTree:
    """Write intensity measures to a file, updating coordinate and variable metadata.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset containing intensity measures to be written.
    """

    def distancify(dataset: xr.Dataset) -> xr.Dataset:
        if "name" not in dataset.attrs:
            return dataset

        dataset = dataset.copy(deep=False)
        dataset.coords.update(distances)
        return dataset

    dtree = dtree.map_over_datasets(distancify)

    return dtree


def add_units(dtree: xr.DataTree) -> xr.DataTree:
    """Write intensity measures to a file, updating coordinate and variable metadata.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset containing intensity measures to be written.
    output_ffp : str or Path
        The file path where the output dataset should be saved.
    """

    def unitify(dataset: xr.Dataset) -> xr.Dataset:
        if "name" not in dataset.attrs:
            return dataset

        dataset = dataset.copy(deep=False)

        for name, description in COORDINATE_METADATA.items():
            if name not in dataset.coords:
                continue
            dataset.coords[name].attrs.update(description)

        name = dataset.attrs["name"]

        for data_var in dataset.data_vars.values():
            data_var.attrs["units"] = IM_UNITS[name]

        description = IM_METADATA[name]
        dataset.attrs["description"] = description
        return dataset

    dtree = dtree.map_over_datasets(unitify)

    return dtree


def _source_polygon(source_geometries: dict[str, IsSource]) -> shapely.Geometry:
    """Extract source polygon in longitude, latitude format.

    Parameters
    ----------
    source_geometries : dict[str, IsSource]
        Realisation faults.

    Returns
    -------
    Geometry
        The union of all fault geometries.
    """
    geometries = []
    for fault in source_geometries.values():
        geometry = fault.geometry

        geometry = shapely.transform(
            geometry, lambda c: coordinates.nztm_to_wgs_depth(c)[:, ::-1]
        )

        geometries.append(geometry)
    return shapely.normalize(shapely.union_all(geometries))


def _trace_polygon(source_geometries: dict[str, IsSource]) -> shapely.Geometry:
    """Extract trace polygon in longitude, latitude format.

    Parameters
    ----------
    source_geometries : dict[str, IsSource]
        Realisation faults.

    Returns
    -------
    Geometry
        The union of all traces of geometries (for geometries that have traces).
    """
    geometries = []
    for fault in source_geometries.values():
        if not hasattr(fault, "trace_geometry"):
            continue
        geometry = fault.trace_geometry
        geometry = shapely.transform(
            geometry, lambda c: coordinates.nztm_to_wgs_depth(c)[:, ::-1]
        )  # ty: ignore[no-matching-overload]

        geometries.append(geometry)

    return shapely.normalize(shapely.union_all(geometries))


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
        Maximum number of stations to read from disk at once for pSA calculation
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
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    rup_prop_config = RupturePropagationConfig.read_from_realisation(realisation_ffp)
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)

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
    station_locations = np.stack((latitude, longitude), axis=-1)

    rrup = (
        np.array(
            [
                min(
                    source.rrup_distance(np.append(station, 0))
                    for source in source_geometries.source_geometries.values()
                )
                for station in station_locations
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
                for station in station_locations
            ]
        )
        / 1000
    )
    hypocentre = source_geometries.source_geometries[
        rup_prop_config.initial_fault
    ].fault_coordinates_to_wgs_depth_coordinates(rup_prop_config.hypocentre)

    hyp = (
        coordinates.distance_between_wgs_depth_coordinates(
            np.c_[station_locations, np.zeros_like(latitude)],
            hypocentre,
        )
        / 1000
    )
    epi = (
        coordinates.distance_between_wgs_depth_coordinates(
            station_locations,
            hypocentre[:2],
        )
        / 1000
    )
    stations = broadband.station.values
    distances: dict[str, Any] = {
        "rrup": ("station", rrup),
        "rjb": ("station", rjb),
        "hyp": ("station", hyp),
        "epi": ("station", epi),
    }
    all_faults_have_rx_ry = all(
        isinstance(source, sources.Plane | sources.Fault)
        for source in source_geometries.source_geometries.values()
    )
    if all_faults_have_rx_ry:
        rx, ry = sources.multi_fault_rx_ry_distance(
            list(source_geometries.source_geometries.values()),  # ty: ignore[invalid-argument-type]
            station_locations,
        )
        rx /= 1000.0
        ry /= 1000.0
        distances["rx"] = xr.DataArray(
            rx, dims="station", coords=dict(station=stations)
        )
        distances["ry"] = xr.DataArray(
            ry, dims="station", coords=dict(station=stations)
        )

    waveform = broadband.waveform.values.astype(np.float64)
    im_results: dict[str, xr.Dataset] = dict()
    for im_name in (pbar := tqdm.tqdm(intensity_measures)):
        pbar.set_description(im_name)
        im_fn = im_function_map[im_name]

        result = im_fn(waveform)

        if isinstance(result, pd.DataFrame):
            result["station"] = broadband.station.values
            result = result.set_index("station").to_xarray()
        elif isinstance(result, xr.DataArray):
            result = result.assign_coords(station=broadband.station).to_dataset(
                "component"
            )
        result.attrs["name"] = im_name
        im_results[im_name] = result

    dtree = xr.DataTree.from_dict(im_results, nested=True)

    dtree.attrs = {
        "hypo_lat": hypocentre[0],
        "hypo_lon": hypocentre[1],
        "source": shapely.to_wkt(_source_polygon(source_geometries.source_geometries)),
        "trace": shapely.to_wkt(_trace_polygon(source_geometries.source_geometries)),
        "domain": shapely.to_wkt(
            shapely.transform(
                domain_parameters.domain.polygon,
                lambda c: coordinates.nztm_to_wgs_depth(c)[:, ::-1],
            )
        ),
        "magnitude": magnitudes.total_magnitude,
        "event": metadata.name,
    }
    dtree = add_distances(dtree, distances)
    dtree = add_units(dtree)
    dtree.to_netcdf(output_path)
    realisations.append_log_entry(realisation_ffp)
