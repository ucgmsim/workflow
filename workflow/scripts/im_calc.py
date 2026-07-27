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

import dataclasses
import functools
import warnings
from pathlib import Path
from typing import Annotated

# NOTE: netCDF4 must be imported before OpenQuake (which oq_wrapper imports).
# OpenQuake pulls in h5py, which is linked against a different build of HDF5
# than netCDF4 is. Whichever of the two loads second fails to open our
# waveform files, so the import order here is load bearing.
import netCDF4  # noqa: F401
import numpy as np
import numpy.typing as npt
import oq_wrapper as oqw
import oq_wrapper.xarray as oqwx
import shapely
import typer
import xarray as xr
from oq_wrapper.estimations import chiou_young_08_calc_z1p0, chiou_young_08_calc_z2p5

from IM import ims
from IM.ims import IM
from qcore import cli, coordinates
from source_modelling import sources
from source_modelling.sources import IsSource
from workflow import realisations
from workflow.realisations import (
    DomainParameters,
    EmpiricalParameters,
    IntensityMeasureCalculationParameters,
    Magnitudes,
    Rakes,
    RealisationMetadata,
    RupturePropagationConfig,
    SourceConfig,
)

app = typer.Typer()


COORDINATE_METADATA = {
    "station": {"description": "Station identifiers"},
    "vs30": {
        "description": "Average shear-wave velocity to 30m depth",
        "units": "m/s",
    },
    "z1pt0": {
        "description": "Depth to the 1.0 km/s shear-wave velocity horizon",
        "units": "km",
    },
    "z2pt5": {
        "description": "Depth to the 2.5 km/s shear-wave velocity horizon",
        "units": "km",
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
    IM.PGD: "Peak ground displacement",
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
    IM.PGD: "cm",
    IM.CAV: "m/s",
    IM.CAV5: "m/s",
    IM.AI: "m/s",
    IM.Ds575: "s",
    IM.Ds595: "s",
    IM.FAS: "g0 * s",
    IM.pSA: "g0",
}


EMPIRICAL_IM_NAMES = {
    IM.PGA: "PGA",
    IM.PGV: "PGV",
    IM.CAV: "CAV",
    IM.AI: "AI",
    IM.Ds575: "Ds575",
    IM.Ds595: "Ds595",
    IM.pSA: "pSA",
}


EMPIRICAL_STATISTIC_METADATA = {
    "mean": "Mean of the natural logarithm of {description}",
    "std_Total": "Total standard deviation of the natural logarithm of {description}",
    "std_Inter": "Between-event standard deviation of the natural logarithm of {description}",
    "std_Intra": "Within-event standard deviation of the natural logarithm of {description}",
}


def add_station_parameters(
    dtree: xr.DataTree, station_parameters: dict[str, xr.DataArray]
) -> xr.DataTree:
    """Attach per-station parameters as coordinates on every leaf of the tree.

    Parameters
    ----------
    dtree : xr.DataTree
        The tree of intensity measure datasets.
    station_parameters : dict
        A map from parameter name (distance and site measures) to the
        per-station values.

    Returns
    -------
    xr.DataTree
        The tree, with the parameters attached to every dataset containing
        data.
    """

    def parameterise(dataset: xr.Dataset) -> xr.Dataset:  # numpydoc ignore=GL08
        if not dataset.data_vars:
            return dataset

        dataset = dataset.copy(deep=False)
        dataset.coords.update(station_parameters)
        return dataset

    dtree = dtree.map_over_datasets(parameterise)

    return dtree


def add_units(dtree: xr.DataTree) -> xr.DataTree:
    """Annotate coordinates and intensity measures with units and descriptions.

    Empirical datasets are left alone, because they are annotated as they
    are calculated (their values are in log-space, so they do not share the
    units of the simulated intensity measures).

    Parameters
    ----------
    dtree : xr.DataTree
        The tree of intensity measure datasets.

    Returns
    -------
    xr.DataTree
        The tree, with unit and description metadata attached.
    """

    def unitify(dataset: xr.Dataset) -> xr.Dataset:  # numpydoc ignore=GL08
        if not dataset.data_vars:
            return dataset

        dataset = dataset.copy(deep=False)

        for name, description in COORDINATE_METADATA.items():
            if name not in dataset.coords:
                continue
            dataset.coords[name].attrs.update(description)

        if "name" not in dataset.attrs:
            return dataset

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


@dataclasses.dataclass
class Distances:
    """Source-to-site distance measures, in kilometres."""

    rrup: xr.DataArray
    """Shortest distance to the rupture plane."""
    rjb: xr.DataArray
    """Shortest distance to the surface projection of the rupture."""
    hyp: xr.DataArray
    """Distance to the hypocentre."""
    epi: xr.DataArray
    """Distance to the epicentre."""
    rx: xr.DataArray | None = None
    """Strike-parallel distance. Only defined for planar sources."""
    ry: xr.DataArray | None = None
    """Strike-normal distance. Only defined for planar sources."""

    def as_dict(self) -> dict[str, xr.DataArray]:
        """Map distance measure name to distances, omitting undefined measures.

        Returns
        -------
        dict
            A map from distance measure name to per-station distances.
        """
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if (value := getattr(self, field.name)) is not None
        }


def calculate_distances(
    source_geometries: SourceConfig,
    hypocentre: np.ndarray,
    broadband: xr.Dataset,
) -> Distances:
    """Calculate source-to-site distances for every station in the broadband.

    Parameters
    ----------
    source_geometries : SourceConfig
        The source geometries of the realisation.
    hypocentre : np.ndarray
        The hypocentre, in latitude, longitude, depth format.
    broadband : xr.Dataset
        The broadband waveform dataset, supplying the station locations.

    Returns
    -------
    Distances
        The distance measures for each station, in kilometres. `rx` and `ry`
        are only calculated if every source in the realisation is planar.
    """
    latitude = broadband.latitude.values
    longitude = broadband.longitude.values
    station_locations = np.stack((latitude, longitude), axis=-1)

    rrup = xr.DataArray(
        np.array(
            [
                min(
                    source.rrup_distance(np.append(station, 0))
                    for source in source_geometries.source_geometries.values()
                )
                for station in station_locations
            ]
        )
        / 1000,
        dims=["station"],
        coords=dict(station=broadband.station),
    )
    rjb = xr.DataArray(
        np.array(
            [
                min(
                    source.rjb_distance(np.append(station, 0))
                    for source in source_geometries.source_geometries.values()
                )
                for station in station_locations
            ]
        )
        / 1000,
        dims=["station"],
        coords=dict(station=broadband.station),
    )

    hyp = xr.DataArray(
        coordinates.distance_between_wgs_depth_coordinates(
            np.c_[station_locations, np.zeros_like(latitude)],
            hypocentre,
        )
        / 1000,
        dims=["station"],
        coords=dict(station=broadband.station),
    )
    epi = xr.DataArray(
        coordinates.distance_between_wgs_depth_coordinates(
            station_locations,
            hypocentre[:2],
        )
        / 1000,
        dims=["station"],
        coords=dict(station=broadband.station),
    )

    distances = Distances(rrup=rrup, rjb=rjb, hyp=hyp, epi=epi)
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
        distances.rx = xr.DataArray(
            rx, dims="station", coords=dict(station=broadband.station)
        )
        distances.ry = xr.DataArray(
            ry, dims="station", coords=dict(station=broadband.station)
        )
    return distances


@dataclasses.dataclass
class SourceParameters:
    """Rupture parameters describing the realisation as a single source."""

    mag: float
    """The total moment magnitude of the rupture."""
    avg_rake: float
    """The moment-averaged rake angle (degrees)."""
    avg_dip: float
    """The moment-averaged dip angle (degrees)."""
    avg_ztor: float
    """The moment-averaged depth to the top of the rupture (km)."""
    avg_zbot: float
    """The moment-averaged depth to the bottom of the rupture (km)."""
    hypo_depth: float
    """The depth of the hypocentre (km)."""


def calculate_source_parameters(
    source_config: SourceConfig,
    magnitudes: Magnitudes,
    rakes: Rakes,
    hypocentre: np.ndarray,
) -> SourceParameters:
    """Reduce a multi-fault realisation to a single set of rupture parameters.

    Ground motion models describe a rupture with a single magnitude, rake,
    dip and depth. Multi-fault realisations are collapsed into these by
    averaging each fault's contribution, weighted by its moment.

    Parameters
    ----------
    source_config : SourceConfig
        The source geometries of the realisation.
    magnitudes : Magnitudes
        The per-fault magnitudes, used for the moment weighting.
    rakes : Rakes
        The per-fault rake angles.
    hypocentre : np.ndarray
        The hypocentre, in latitude, longitude, depth (metres) format.

    Returns
    -------
    SourceParameters
        The rupture parameters of the realisation as a whole.
    """
    mag = magnitudes.total_magnitude

    avg_rake_vector = magnitudes.moment_averaged(rakes.as_vectors())
    avg_rake = np.degrees(np.arctan2(avg_rake_vector[1], avg_rake_vector[0]))

    avg_dip_vector = magnitudes.moment_averaged(
        {
            k: np.array([np.cos(np.radians(f.dip)), np.sin(np.radians(f.dip))])
            for k, f in source_config.source_geometries.items()
        }
    )
    avg_dip = np.degrees(np.arctan2(avg_dip_vector[1], avg_dip_vector[0]))

    if all(
        hasattr(f, "top_m") and hasattr(f, "bottom_m")
        for f in source_config.source_geometries.values()
    ):
        avg_ztor = magnitudes.moment_averaged(
            {k: f.top_m / 1000.0 for k, f in source_config.source_geometries.items()}  # ty: ignore[unresolved-attribute]
        )
        avg_zbot = magnitudes.moment_averaged(
            {k: f.bottom_m / 1000.0 for k, f in source_config.source_geometries.items()}  # ty: ignore[unresolved-attribute]
        )
    else:
        avg_ztor = magnitudes.moment_averaged(
            {
                k: f.centroid[-1] / 1000.0
                for k, f in source_config.source_geometries.items()
            }
        )
        avg_zbot = avg_ztor

    return SourceParameters(
        mag=mag,
        avg_rake=float(avg_rake),
        avg_dip=float(avg_dip),
        avg_ztor=avg_ztor,
        avg_zbot=avg_zbot,
        hypo_depth=float(hypocentre[2]) / 1000.0,
    )


@dataclasses.dataclass
class SiteParameters:
    """Per-station site parameters."""

    vs30: xr.DataArray
    """Average shear-wave velocity to 30m depth (m/s)."""
    z1pt0: xr.DataArray
    """Depth to the 1.0 km/s shear-wave velocity horizon (km)."""
    z2pt5: xr.DataArray
    """Depth to the 2.5 km/s shear-wave velocity horizon (km)."""

    def as_dict(self) -> dict[str, xr.DataArray]:
        """Map site parameter name to per-station values.

        Returns
        -------
        dict
            A map from site parameter name to per-station values.
        """
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


def calculate_site_parameters(vs30: xr.DataArray) -> SiteParameters:
    """Estimate site parameters from vs30.

    Parameters
    ----------
    vs30 : xr.DataArray
        The per-station vs30 values (m/s).

    Returns
    -------
    SiteParameters
        The site parameters, with basin depths estimated using the Chiou
        and Youngs (2008) relations.
    """
    z1pt0 = chiou_young_08_calc_z1p0(vs30)  # ty: ignore[invalid-argument-type]
    z2pt5 = chiou_young_08_calc_z2p5(z1pt0)
    return SiteParameters(vs30=vs30, z1pt0=z1pt0, z2pt5=z2pt5)


def empirical_inputs(
    source_parameters: SourceParameters,
    site_parameters: SiteParameters,
    distances: Distances,
) -> xr.Dataset:
    """Assemble the rupture context ground motion models are evaluated against.

    Parameters
    ----------
    source_parameters : SourceParameters
        The rupture parameters of the realisation.
    site_parameters : SiteParameters
        The per-station site parameters.
    distances : Distances
        The per-station source-to-site distances.

    Returns
    -------
    xr.Dataset
        A dataset of OpenQuake rupture context variables. Site and distance
        variables vary over the station dimension, rupture variables are
        scalars.
    """
    return xr.Dataset(
        dict(
            mag=source_parameters.mag,
            dip=source_parameters.avg_dip,
            rake=source_parameters.avg_rake,
            ztor=source_parameters.avg_ztor,
            zbot=source_parameters.avg_zbot,
            hypo_depth=source_parameters.hypo_depth,
            vs30=site_parameters.vs30,
            z1pt0=site_parameters.z1pt0,
            z2pt5=site_parameters.z2pt5,
            vs30measured=False,
            # TODO: Calculate backarc!
            backarc=False,
        )
        | distances.as_dict()
    )


def annotate_empirical(dataset: xr.Dataset, im_name: IM, model_name: str) -> xr.Dataset:
    """Attach units and descriptions to an empirical intensity measure dataset.

    Ground motion models predict the distribution of the natural logarithm
    of an intensity measure, so the values are dimensionless. The units of
    the intensity measure itself are recorded in the `log_units` attribute.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset of statistics returned by `oq_wrapper`.
    im_name : IM
        The intensity measure the dataset describes.
    model_name : str
        The ground motion model (or logic tree) that produced the dataset.

    Returns
    -------
    xr.Dataset
        The dataset, with metadata attached.
    """
    dataset = dataset.copy(deep=False)
    description = IM_METADATA[im_name]

    for statistic, data_var in dataset.data_vars.items():
        data_var.attrs["units"] = "dimensionless"
        data_var.attrs["log_units"] = IM_UNITS[im_name]
        if statistic in EMPIRICAL_STATISTIC_METADATA:
            data_var.attrs["description"] = EMPIRICAL_STATISTIC_METADATA[
                str(statistic)
            ].format(description=description)

    dataset.attrs["intensity_measure"] = str(im_name)
    dataset.attrs["model"] = model_name
    dataset.attrs["description"] = (
        f"{description} predicted by the {model_name} ground motion model"
    )
    return dataset


def calculate_empirical(
    empirical_config: EmpiricalParameters,
    source_parameters: SourceParameters,
    site_parameters: SiteParameters,
    distances: Distances,
    intensity_measures: list[IM],
    periods: npt.NDArray[np.float64],
) -> dict[str, xr.Dataset]:
    """Calculate empirical intensity measures from ground motion models.

    Each model in the empirical configuration is evaluated for each
    intensity measure a ground motion model can predict. Combinations that
    a model does not support are skipped with a warning.

    Parameters
    ----------
    empirical_config : EmpiricalParameters
        The tectonic type and models to evaluate.
    source_parameters : SourceParameters
        The rupture parameters of the realisation.
    site_parameters : SiteParameters
        The per-station site parameters.
    distances : Distances
        The per-station source-to-site distances.
    intensity_measures : list of IM
        The intensity measures to calculate.
    periods : np.ndarray
        The periods to calculate pSA at.

    Returns
    -------
    dict
        A map from data tree path (`{im}/empirical/{model}`) to the log-mean
        and log-standard deviation of that intensity measure. The paths are
        chosen so this map can be merged with the simulated intensity
        measures before building the output data tree.
    """
    inputs = empirical_inputs(source_parameters, site_parameters, distances)
    tect_type = oqw.constants.TectType(empirical_config.tect_type)

    empirical_results: dict[str, xr.Dataset] = {}
    model_ims = [im for im in intensity_measures if im in EMPIRICAL_IM_NAMES]

    for model_name in empirical_config.models:
        for im_name in model_ims:
            try:
                if model_name in oqw.constants.GMMLogicTree.__members__:
                    dataset = oqwx.run_gmm_logic_tree_xarray(
                        oqw.constants.GMMLogicTree[model_name],
                        tect_type,
                        inputs,
                        EMPIRICAL_IM_NAMES[im_name],
                        periods=periods.tolist(),
                    )
                else:
                    dataset = oqwx.run_gmm_xarray(
                        oqw.constants.GMM[model_name],
                        tect_type,
                        inputs,
                        EMPIRICAL_IM_NAMES[im_name],
                        periods=periods.tolist(),
                    )
            except (ValueError, KeyError, AttributeError) as e:
                warnings.warn(
                    f"Skipping empirical {im_name} for {model_name}: {e}",
                    stacklevel=1,
                )
                continue

            empirical_results[f"{im_name}/empirical/{model_name}"] = annotate_empirical(
                dataset, im_name, model_name
            )

    return empirical_results


ROTD180_LOG_SCALE_FACTOR = np.log1p(0.01) / 2

# Fixed uint16 midpoint offset so negative log-pSA values are representable.
ROTD180_LOG_ADD_OFFSET = -32768 * ROTD180_LOG_SCALE_FACTOR

ROTD180_LOG_FLOOR = 1e-10


def encode_psa_rotd180(rotd180: xr.DataArray) -> xr.DataArray:
    """Delta-encode the full-angle RotD180 pSA curve in log-space.

    Adjacent angles are highly correlated, so log then diff along `angle`
    collapses most of the curve to near-zero values, compressing well.
    Decoding is cumulative, so quantization error grows along the curve.

    Parameters
    ----------
    rotd180 : xr.DataArray
        The full-angle RotD180 pSA curve (g), with an `angle` dimension of
        size 180.

    Returns
    -------
    xr.DataArray
        The log-transformed, delta-encoded curve, with the same shape as
        `rotd180`. `angle=0` holds `log(rotd180)` at that angle, and every
        subsequent angle holds the difference in log-pSA from its
        predecessor. Recovered via `exp(encoded.cumsum("angle"))`.
    """
    log_psa = np.log(np.maximum(rotd180, ROTD180_LOG_FLOOR))
    base = log_psa.isel(angle=slice(0, 1))
    deltas = log_psa.diff("angle")
    return xr.concat([base, deltas], dim="angle").transpose(*rotd180.dims)


def rotd180_netcdf_encoding(encoded_rotd180: xr.DataArray) -> dict:
    """Build the netCDF4 encoding for a delta-encoded `rotd180` variable.

    Parameters
    ----------
    encoded_rotd180 : xr.DataArray
        The output of `encode_psa_rotd180`.

    Returns
    -------
    dict
        A netCDF4 variable encoding that linearly quantizes the log-space
        values to `uint16` and deflates the result.
    """
    return {
        "dtype": "uint16",
        "scale_factor": ROTD180_LOG_SCALE_FACTOR,
        "add_offset": ROTD180_LOG_ADD_OFFSET,
        "_FillValue": None,
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
    }


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
    ko_directory: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    override_ims: Annotated[list[IM] | None, typer.Option("-i", "--im")] = None,
    empirical: Annotated[bool, typer.Option()] = True,
    full_rotd180: Annotated[bool, typer.Option()] = False,
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
    ko_directory : Path
        Directory containing the KO matrix files for FAS calculation. Not required for other IMs.
    override_ims : list of str
        Intensity measures to calculate. If not set, reads from the realisation file.
    empirical : bool, default True
        If passed, additionally estimate intensity measures from the ground
        motion models in the realisation file. Requires the broadband
        waveforms to carry a `vs30` coordinate.
    full_rotd180 : bool, default False
        If passed (and pSA is being calculated), also store pSA at every integer
        rotation angle 0-179 degrees, not just RotD0/50/100. This is roughly
        180x the storage of the summary statistics, so it is quantized and
        delta-encoded before writing, at a ~1% relative-error resolution near
        angle 0 that accumulates (worst case, additively) further along the
        curve.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    intensity_measure_parameters = (
        IntensityMeasureCalculationParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )
    source_geometries = SourceConfig.read_from_realisation(realisation_ffp)
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
    rup_prop_config = RupturePropagationConfig.read_from_realisation(realisation_ffp)
    magnitudes = Magnitudes.read_from_realisation(realisation_ffp)

    # Chunk over stations only: `component` and `time` must each be a single
    # chunk because the IM kernels take them as core dimensions, and the
    # file's own on-disk chunking may split either. Dask sizes the station
    # chunks from its `array.chunk-size` config.
    broadband = xr.open_dataset(broadband_simulation_ffp).chunk(
        {"component": -1, "time": -1, "station": "auto"}
    )
    dt = broadband.attrs["dt"]

    if not simulated_stations:
        broadband = broadband.isel(
            station=broadband.station.str.match(r"^(\w{4})$").values
        )

    intensity_measures = override_ims or intensity_measure_parameters.ims

    if IM.FAS in intensity_measures and not ko_directory:
        raise ValueError(
            "FAS calculation requires KO directory. Please provide a valid KO directory."
        )

    nyquist_frequency = 1 / (2 * dt)

    im_function_map = {
        IM.PGA: (ims.peak_ground_acceleration),
        IM.PGV: functools.partial(ims.peak_ground_velocity, dt=dt),
        IM.PGD: functools.partial(ims.peak_ground_displacement, dt=dt),
        IM.CAV: functools.partial(ims.cumulative_absolute_velocity, dt=dt),
        IM.AI: functools.partial(ims.arias_intensity, dt=dt),
        IM.Ds575: functools.partial(ims.ds575, dt=dt),
        IM.Ds595: functools.partial(ims.ds595, dt=dt),
        IM.pSA: functools.partial(
            ims.pseudo_spectral_acceleration,
            periods=np.array(
                intensity_measure_parameters.valid_periods, dtype=np.float64
            ),
            dt=dt,
            full_rotd180=full_rotd180,
        ),
        IM.FAS: functools.partial(
            ims.fourier_amplitude_spectra,
            dt=dt,
            freqs=intensity_measure_parameters.fas_frequencies[
                intensity_measure_parameters.fas_frequencies <= nyquist_frequency
            ],
            ko_directory=ko_directory,
        ),
    }
    hypocentre = source_geometries.source_geometries[
        rup_prop_config.initial_fault
    ].fault_coordinates_to_wgs_depth_coordinates(rup_prop_config.hypocentre)
    distances = calculate_distances(source_geometries, hypocentre, broadband)
    rakes = Rakes.read_from_realisation(realisation_ffp)
    source_parameters = calculate_source_parameters(
        source_geometries, magnitudes, rakes, hypocentre
    )
    # vs30 is one float per station; load it eagerly rather than letting it
    # propagate as a dask array into the distance/site coordinates below.
    site_parameters = calculate_site_parameters(
        broadband.vs30.astype(np.float64).load()
    )

    # Each IM function is dask-native: it accepts the lazy `waveform` DataArray
    # and returns a lazy Dataset with the same `station` chunking, one data
    # variable per component. Nothing is computed until `dtree.to_netcdf`
    # below, which streams the result chunk by chunk.
    im_results: dict[str, xr.Dataset] = {
        im_name: im_function_map[im_name](broadband.waveform)
        for im_name in intensity_measures
    }

    attributes = {
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
        "rake": source_parameters.avg_rake,
        "dip": source_parameters.avg_dip,
        "ztor": source_parameters.avg_ztor,
        "zbot": source_parameters.avg_zbot,
        "hypo_depth": source_parameters.hypo_depth,
    }
    if empirical:
        empirical_parameters = EmpiricalParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
        im_results |= calculate_empirical(
            empirical_parameters,
            source_parameters,
            site_parameters,
            distances,
            intensity_measures,
            np.array(intensity_measure_parameters.valid_periods, dtype=np.float64),
        )
        attributes["tect_type"] = str(empirical_parameters.tect_type)

    encoding: dict[str, dict[str, dict]] | None = None
    if full_rotd180 and IM.pSA in im_results:
        encoded_rotd180 = encode_psa_rotd180(im_results[IM.pSA]["rotd180"])
        im_results[IM.pSA] = im_results[IM.pSA].assign(rotd180=encoded_rotd180)
        encoding = {"/pSA": {"rotd180": rotd180_netcdf_encoding(encoded_rotd180)}}

    dtree = xr.DataTree.from_dict(im_results, nested=True)

    dtree.attrs = attributes
    dtree = add_station_parameters(
        dtree, distances.as_dict() | site_parameters.as_dict()
    )
    dtree = add_units(dtree)
    dtree.to_netcdf(output_path, encoding=encoding)
    realisations.append_log_entry(realisation_ffp)
