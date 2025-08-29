"""Broadband Simulation.

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

Outputs
-------
An output broadband file in the HDF5 format.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own
computer using the `bb-sim` command which is installed after running
`pip install workflow@git+https://github.com/ucgmsim/workflow`. If
running on your own computer, you need to configure a work directory
(`--work-directory`).

Usage
-----
`bb-sim REALISATION_FFP STATION_FFP STATION_VS30_FFP LOW_FREQUENCY_WAVEFORM_DIRECTORY HIGH_FREQUENCY_WAVEFORM_FILE OUTPUT_FFP`

For More Help
-------------
See the output of `bb-sim --help`.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import xarray as xr

from qcore import cli, siteamp_models, timeseries
from workflow import log_utils, realisations
from workflow.realisations import (
    BroadbandParameters,
    RealisationMetadata,
)

app = typer.Typer()

G = 1 / 981.0


@cli.from_docstring(app)
@log_utils.log_call()
def combine_hf_and_lf(
    realisation_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    station_vs30_ffp: Annotated[Path, typer.Argument(dir_okay=False, exists=True)],
    low_frequency_waveform_file: Annotated[
        Path, typer.Argument(dir_okay=False, exists=True)
    ],
    high_frequency_waveform_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_ffp: Annotated[Path, typer.Argument(dir_okay=False, writable=True)],
) -> None:
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
    output_ffp : Path
        Path to the output file where the combined broadband waveforms will be saved.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_config = BroadbandParameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )

    # load data stores
    lf = xr.open_dataset(low_frequency_waveform_file)
    hf = xr.open_dataset(high_frequency_waveform_file)
    common_stations = list(
        set(map(str, hf.station.values)) & set(map(str, lf.station.values))
    )
    hf = hf.sel(station=common_stations)
    lf = lf.sel(station=common_stations)

    vs30_df = pd.read_csv(
        station_vs30_ffp, sep=r"\s+", header=None, names=["station", "vsite"]
    ).set_index("station")
    vs30_df = vs30_df.loc[common_stations]
    vs30_df["vref"] = 500.0
    vs30_df["vpga"] = 500.0
    dt = bb_dt = broadband_config.dt

    lf_start_sec = lf.attrs["start_sec"]
    hf_start_sec = hf.attrs["start_sec"]
    dt = lf.attrs["dt"]
    hf_duration = lf_duration = len(lf.time.values) * dt

    bb_start_sec = min(lf_start_sec, hf_start_sec)
    # Calculate time offsets for LF and HF relative to the broadband start time
    lf_start_sec_offset = max(lf_start_sec - bb_start_sec, 0)
    hf_start_sec_offset = max(hf_start_sec - bb_start_sec, 0)

    # Convert time offsets to number of samples (padding at the beginning)
    lf_start_padding_nt = int(np.round(lf_start_sec_offset / bb_dt))
    hf_start_padding_nt = int(np.round(hf_start_sec_offset / bb_dt))

    # Determine the latest end time for the combined broadband waveform
    max_end_sec = max(lf_start_sec + lf_duration, hf_start_sec + hf_duration)

    # Calculate the total duration needed for the combined waveform
    bb_total_duration = max_end_sec - bb_start_sec

    # Calculate the total number of samples for the combined waveform
    bb_nt = int(np.round(bb_total_duration / bb_dt))

    # Calculate padding at the end for LF and HF waveforms
    # This ensures both padded waveforms have the same total length (bb_nt)
    lf_end_padding_nt = bb_nt - (
        lf_start_padding_nt + int(np.round(lf_duration / bb_dt))
    )
    hf_end_padding_nt = bb_nt - (
        hf_start_padding_nt + int(np.round(hf_duration / bb_dt))
    )

    # Ensure padding values are non-negative
    lf_end_padding_nt = max(0, lf_end_padding_nt)
    hf_end_padding_nt = max(0, hf_end_padding_nt)

    bb_waveform = np.empty_like(lf.waveform)
    for i, (lf_component, hf_component) in enumerate(zip(lf.component, hf.component)):
        hf_waveform_raw = hf.sel(component=hf_component).waveform.values
        lf_waveform_raw = lf.sel(component=lf_component).waveform.values

        temp_lf_padded = np.zeros((lf_waveform_raw.shape[0], bb_nt), dtype=np.float32)
        temp_hf_padded = np.zeros((hf_waveform_raw.shape[0], bb_nt), dtype=np.float32)

        temp_lf_padded[
            :, lf_start_padding_nt : lf_start_padding_nt + lf_waveform_raw.shape[1]
        ] = lf_waveform_raw
        temp_hf_padded[
            :,
            hf_start_padding_nt : hf_start_padding_nt + hf_waveform_raw.shape[1],
        ] = hf_waveform_raw

        vs30_df["pga"] = temp_hf_padded.max(axis=1) * G

        hf_amp_val = siteamp_models.cb_amp_multi(
            vs30_df,
        )
        hf_amp_fas_vals = siteamp_models.cb2014_to_fas_amplification_factors(
            hf_amp_val, bb_dt, bb_nt
        )
        hf_waveform_amped = timeseries.ampdeamp(
            temp_hf_padded, hf_amp_fas_vals, amp=True
        )
        hf_filtered = timeseries.bwfilter(
            hf_waveform_amped, bb_dt, 1.0, timeseries.Band.HIGHPASS
        )
        lf_filtered = timeseries.bwfilter(
            temp_lf_padded, bb_dt, 1.0, timeseries.Band.LOWPASS
        )
        bb_waveform[i] = (hf_filtered + lf_filtered) * G

    new_time_coords = np.arange(bb_nt) * bb_dt + bb_start_sec
    xr.Dataset(
        {"waveform": (["component", "station", "time"], bb_waveform)},
        coords={
            "station": common_stations,
            "time": new_time_coords,
            "x": lf.x.values,
            "y": lf.y.values,
            "lat": lf.lat.values,
            "lon": lf.lon.values,
        },
        attrs={
            "units": "g",
        },
    ).to_netcdf(
        output_ffp,
        engine="h5netcdf",
        encoding={
            "waveform": {
                "compression": "zlib",  # Use zlib compression.
                "complevel": 5,  # Compress to level 5 (of 9).
                "fletcher32": True,  # Add Fletcher-32 checksums for long-term storage.
            }
        },
    )
    realisations.append_log_entry(realisation_ffp)
