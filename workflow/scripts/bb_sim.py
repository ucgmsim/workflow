"""Broadband Simulation.

Combine high-frequency and low-frequency simulation waveforms for each station into a broadband simulation file.

Description
-----------
Combine high-frequency and low-frequency simulation waveforms for each station into a broadband simulation file.

Inputs
------
1. A realisation file containing:
   - Realisation metadata,
   - Domain parameters.
2. Station list (latitude, longitude, name),
3. Stations VS30 reference values,
4. Low frequency waveform directory,
5. High frequency output file,
6. Velocity model directory.

Outputs
-------
An output [broadband file](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-LF/HF/BBbinaryformat).

Environment
-----------
Can be run in the cybershake container. Can also be run from your own
computer using the `bb-sim` command which is installed after running
`pip install workflow@git+https://github.com/ucgmsim/workflow`. If
running on your own computer, you need to configure a work directory
(`--work-directory`).

Usage
-----
`bb-sim REALISATION_FFP STATION_FFP STATION_VS30_FFP LOW_FREQUENCY_WAVEFORM_DIRECTORY HIGH_FREQUENCY_WAVEFORM_FILE VELOCITY_MODEL_DIRECTORY OUTPUT_FFP`

For More Help
-------------
See the output of `bb-sim --help`.
"""

import functools
import multiprocessing
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import typer
import xarray as xr

from qcore import cli, siteamp_models, timeseries
from workflow import log_utils, utils
from workflow.realisations import (
    BroadbandParameters,
    DomainParameters,
    RealisationMetadata,
)

app = typer.Typer()


@log_utils.log_call(
    exclude_args={
        "lf",
        "hf",
        "hf_padding",
        "lf_padding",
        "broadband_config",
        "n2",
    },
    include_result=False,
)
def bb_simulate_station(
    lf: timeseries.LFSeis,
    hf_path: Path,
    hf_padding: tuple[int, int],
    lf_padding: tuple[int, int],
    broadband_config: BroadbandParameters,
    n2: float,
    station_name: str,
    station: pd.Series,
):
    """Simulate broadband seismic for a single station.

    Combines the low frequency and high frequency waveforms together
    for a single station with appropriate filtering and padding.
    Writes the simulated broadband acceleration data to a file in the
    work directory.

    Parameters
    ----------
    lf : timeseries.LFSeis
        Low-frequency seismic data object.
    hf_path : Path
        Path to the high-frequency seismic data file.
    hf_padding : tuple[int, int]
        Padding for the high-frequency data (start, end).
    lf_padding : tuple[int, int]
        Padding for the low-frequency data (start, end).
    broadband_config : BroadbandParameters
        Configuration parameters for broadband simulation.
    n2 : float
        Site amplification parameter.
    station_name : str
        Name of the seismic station.
    station : pd.Series
        Series containing station metadata including vs and vs30 values.

    Returns
    -------
    np.ndarray
        Simulated broadband acceleration data.
    """
    hf_ds = xr.open_dataset(hf_path, engine="h5netcdf")
    # we expected waveform files to have size n_components (3) * float size (4) * number of padded timesteps.
    station_vs = station["vs"]
    station_vs30 = station["vs30"]
    lf_acc = sp.signal.resample(
        lf["waveforms"].sel(station=station_name).values,
        int(round(lf.attrs["duration"] / broadband_config.dt)),
    )
    hf_acc = sp.signal.resample(
        hf_ds.sel(station=station_name).values,
        int(round(hf_ds.attrs["duration"] / broadband_config.dt)),
    )
    logger = log_utils.get_logger(__name__)

    if np.isnan(lf_acc).any():
        logger.error("Station LF had NaN waveform", station=station)
        raise ValueError(f"Station {station_name} had NaN waveform")
    if np.isnan(hf_acc).any():
        logger.error("Station HF had NaN waveform", station=station)
        raise ValueError(f"Station {station_name} had NaN waveform")

    pga = np.max(np.abs(hf_acc), axis=0) / 981.0
    bb_acc: list[npt.NDArray[np.float32]] = []
    for c in range(3):
        hf_amp_val = siteamp_models.cb_amp(
            broadband_config.dt,
            n2,
            station_vs,
            station_vs30,
            station_vs,
            pga[c],
            fmin=broadband_config.fmin,
            fmidbot=broadband_config.fmidbot,
            version=broadband_config.site_amp_version,
        )

        hf_filtered = timeseries.bwfilter(
            timeseries.ampdeamp(
                hf_acc[:, c],
                hf_amp_val,
                amp=True,
            ),
            broadband_config.dt,
            broadband_config.flo,
            "highpass",
        )
        lf_filtered = timeseries.bwfilter(
            lf_acc[:, c],
            broadband_config.dt,
            broadband_config.flo,
            "lowpass",
        )

        hf_c = np.pad(hf_filtered, hf_padding)
        lf_c = np.pad(lf_filtered, lf_padding)
        bb_comp = (hf_c + lf_c) / 981.0

        bb_acc.append(bb_comp)

    return np.array(bb_acc).T.astype(np.float32)


@cli.from_docstring(app)
@log_utils.log_call()
def combine_hf_and_lf(
    realisation_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    station_vs30_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    low_frequency_waveform_file: Annotated[
        Path, typer.Argument(file_okay=False, exists=True)
    ],
    high_frequency_waveform_file: Annotated[Path, typer.Argument(exists=True)],
    velocity_model_directory: Annotated[
        Path, typer.Argument(file_okay=False, exists=True)
    ],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
):
    """Combine low-frequency and high-frequency seismic waveforms.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file containing parameters for the simulation.
    station_vs30_ffp : Path
        Path to the file containing VS30 reference values for stations.
    low_frequency_waveform_file : Path
        File containing low-frequency waveform data.
    high_frequency_waveform_file : Path
        File containing high-frequency waveform data.
    velocity_model_directory : Path
        Directory containing velocity model files.
    output_ffp : Path
        Path to the output file where the combined broadband waveforms will be saved.
    """
    # load data stores
    lf = xr.open_dataset(low_frequency_waveform_file, engine="h5netcdf")
    hf_ds = xr.open_dataset(high_frequency_waveform_file, engine="h5netcdf")
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_config = BroadbandParameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)

    # As LF has a start time offset it is necessary to pad the start of HF by the same number of timesteps
    # Similar code to account for an end time difference is also present
    # allowing for HF and LF to have separate start times and durations

    bb_start_sec = min(lf.attrs["start_sec"], hf_ds.attrs["t_sec"])
    lf_start_sec_offset = max(lf.attrs["start_sec"] - hf_ds.attrs["t_sec"], 0)
    hf_start_sec_offset = max(hf_ds.attrs["t_sec"] - lf.attrs["start_sec"], 0)
    lf_start_padding = int(round(lf_start_sec_offset / broadband_config.dt))
    hf_start_padding = int(round(hf_start_sec_offset / broadband_config.dt))

    lf_end_padding = int(
        round(
            max(
                hf_ds.attrs["duration"]
                + hf_start_sec_offset
                - (lf.attrs["duration"] + lf_start_sec_offset),
                0,
            )
            / broadband_config.dt
        )
    )
    hf_end_padding = int(
        round(
            max(
                lf.attrs["duration"]
                + lf_start_sec_offset
                - (hf_ds.attrs["duration"] + hf_start_sec_offset),
                0,
            )
            / broadband_config.dt
        )
    )

    if (
        lf_start_padding
        + round(lf.attrs["duration"] / broadband_config.dt)
        + lf_end_padding
        != hf_start_padding
        + round(hf_ds.attrs["duration"] / broadband_config.dt)
        + hf_end_padding
    ):
        raise ValueError("HF and LF padded timesteps do not align.")
    lf_padding = (lf_start_padding, lf_end_padding)
    hf_padding = (hf_start_padding, hf_end_padding)
    bb_nt = int(
        lf_start_padding
        + round(lf.attrs["duration"] / broadband_config.dt)
        + lf_end_padding
    )
    n2 = siteamp_models.nt2n(bb_nt)

    lfvs30refs = (
        np.memmap(
            velocity_model_directory / "vs3dfile.s",
            dtype="<f4",
            shape=(domain_parameters.ny, domain_parameters.nz, domain_parameters.nx),
            mode="r",
        )[lf.coords["y"], 0, lf.coords["x"]]
        * 1000.0
    )

    stations = hf_ds[
        ["longitude", "latitude", "vs", "epicentre_distance"]
    ].to_dataframe()
    stations["waveform_index"] = np.arange(len(stations))
    # ensure that LF and HF agree on station list, sometimes LF can drop a station or two
    stations = stations.loc[lf.coords["station"]]

    station_vs30 = pd.read_csv(
        station_vs30_ffp,
        delimiter=r"\s+",
        header=None,
        names=["name", "vs30"],
    ).set_index("name")
    stations = stations.join(station_vs30, how="inner")

    with multiprocessing.Pool(utils.get_available_cores()) as pool:
        waveforms_raw = np.array(
            list(
                pool.starmap(
                    functools.partial(
                        bb_simulate_station,
                        lf,
                        high_frequency_waveform_file,
                        hf_padding,
                        lf_padding,
                        broadband_config,
                        n2,
                    ),
                    stations.iterrows(),
                )
            ),
            dtype=np.float32,
        )

    ds = xr.Dataset(
        {
            "waveforms": (("station", "time", "component"), waveforms_raw),
            "latitude": ("station", stations["latitude"]),
            "longitude": ("station", stations["longitude"]),
            "vs": ("station", stations["vs"]),
            "hf_epicentre": ("station", stations["epicentre_distance"]),
            "x": ("station", lf.coords["x"]),
            "y": ("station", lf.coords["y"]),
            "lf_vs_ref": ("station", lfvs30refs),
        },
        coords={
            "station": stations.index.values,
            "time": np.arange(bb_nt) * broadband_config.dt,
            "component": ["090", "000", "ver"],
        },
        attrs=hf_ds.attrs
        | {
            "nt": bb_nt,
            "duration": bb_nt * broadband_config.dt,
            "start": bb_start_sec,
        },
    )
    ds.to_netcdf(output_ffp, engine="h5netcdf")
