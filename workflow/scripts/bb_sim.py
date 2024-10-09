"""Broadband Simulation.

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
"""

import functools
import multiprocessing
import os
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import typer

from qcore import siteamp_models, timeseries
from workflow import log_utils
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
        "work_directory",
    }
)
def bb_simulate_station(
    lf: timeseries.LFSeis,
    hf_path: Path,
    hf_padding: tuple[int, int],
    lf_padding: tuple[int, int],
    broadband_config: BroadbandParameters,
    n2: float,
    work_directory: Path,
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
    hf : timeseries.HFSeis
        High-frequency seismic data object.
    hf_padding : tuple[int, int]
        Padding for the high-frequency data (start, end).
    lf_padding : tuple[int, int]
        Padding for the low-frequency data (start, end).
    broadband_config : BroadbandParameters
        Configuration parameters for broadband simulation.
    n2 : float
        Site amplification parameter.
    work_directory : Path
        Directory for temporary files.
    station_name : str
        Name of the seismic station.
    station_vs : float
        vs value of the station site.
    station_vs30 : float
        VS30 value for the station site.
    """
    hf = h5py.File(hf_path)
    output_bb_file = work_directory / f"{station_name}.bb"
    # we expected waveform files to have size n_components (3) * float size (4) * number of padded timesteps.
    expected_bb_size = 12 * (
        lf_padding[0] + round(lf.duration / broadband_config.dt) + lf_padding[1]
    )
    if output_bb_file.exists() and os.stat(output_bb_file).st_size == expected_bb_size:
        return
    station_vs = station["vs"]
    station_vs30 = station["vs30"]
    lf_acc = np.copy(lf.acc(station_name, dt=broadband_config.dt))
    hf_acc = sp.signal.resample(
        np.array(hf["waveforms"][int(station["waveform_index"])]),
        int(round(hf.attrs["duration"] / broadband_config.dt)),
    )
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
        print(
            station_name, ((hf_c + lf_c) / 981.0).max(), ((hf_c + lf_c) / 981.0).min()
        )
        bb_acc.append((hf_c + lf_c) / 981.0)

    bb_acc_numpy = np.array(bb_acc).T.astype(np.float32)
    with open(output_bb_file, "wb") as station_bb_file:
        bb_acc_numpy.tofile(station_bb_file)


@app.command(
    help="Combine low frequency and high frequency waveforms into broadband waveforms"
)
@log_utils.log_call()
def combine_hf_and_lf(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(help="Path to realisation file", dir_okay=False, exists=True),
    ],
    station_vs30_ffp: Annotated[
        Path,
        typer.Argument(
            help="Station VS30 reference values.", dir_okay=False, exists=True
        ),
    ],
    low_frequency_waveform_directory: Annotated[
        Path,
        typer.Argument(
            help="Path to low frequency waveforms",
            file_okay=False,
            exists=True,
        ),
    ],
    high_frequency_waveform_file: Annotated[
        Path,
        typer.Argument(
            help="Path to high frequency station waveform file",
            exists=True,
        ),
    ],
    velocity_model_directory: Annotated[
        Path,
        typer.Argument(
            help="Path to the velocity model directory", file_okay=False, exists=True
        ),
    ],
    output_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to output broadband file.", dir_okay=False, writable=True
        ),
    ],
    work_directory: Annotated[
        Path,
        typer.Option(
            help="Path to work directory", file_okay=False, exists=True, writable=True
        ),
    ],
):
    """Combine low-frequency and high-frequency seismic waveforms.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file containing parameters for the simulation.
    station_ffp : Path
        Path to the station list file containing station metadata.
    station_vs30_ffp : Path
        Path to the file containing VS30 reference values for stations.
    low_frequency_waveform_directory : Path
        Directory containing low-frequency waveform data files.
    high_frequency_waveform_file : Path
        File containing high-frequency waveform data.
    velocity_model_directory : Path
        Directory containing velocity model files.
    output_ffp : Path
        Path to the output file where the combined broadband waveforms will be saved.
    work_directory : Path
        Directory for temporary work files.

    See Also
    --------
    - [Broadband file format](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-LF/HF/BBbinaryformat)
    """
    # load data stores
    lf = timeseries.LFSeis(low_frequency_waveform_directory)
    hf = h5py.File(high_frequency_waveform_file, mode="r")
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_config = BroadbandParameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)

    # As LF has a start time offset it is necessary to pad the start of HF by the same number of timesteps
    # Similar code to account for an end time difference is also present
    # allowing for HF and LF to have separate start times and durations

    bb_start_sec = min(lf.start_sec, hf.attrs["t_sec"])
    lf_start_sec_offset = max(lf.start_sec - hf.attrs["t_sec"], 0)
    hf_start_sec_offset = max(hf.attrs["t_sec"] - lf.start_sec, 0)
    lf_start_padding = int(round(lf_start_sec_offset / broadband_config.dt))
    hf_start_padding = int(round(hf_start_sec_offset / broadband_config.dt))

    lf_end_padding = int(
        round(
            max(
                hf.attrs["duration"]
                + hf_start_sec_offset
                - (lf.duration + lf_start_sec_offset),
                0,
            )
            / broadband_config.dt
        )
    )
    hf_end_padding = int(
        round(
            max(
                lf.duration
                + lf_start_sec_offset
                - (hf.attrs["duration"] + hf_start_sec_offset),
                0,
            )
            / broadband_config.dt
        )
    )

    if (
        lf_start_padding + round(lf.duration / broadband_config.dt) + lf_end_padding
        != hf_start_padding
        + round(hf.attrs["duration"] / broadband_config.dt)
        + hf_end_padding
    ):
        raise ValueError("HF and LF padded timesteps do not align.")
    lf_padding = (lf_start_padding, lf_end_padding)
    hf_padding = (hf_start_padding, hf_end_padding)

    bb_nt = int(
        lf_start_padding + round(lf.duration / broadband_config.dt) + lf_end_padding
    )
    n2 = siteamp_models.nt2n(bb_nt)

    lfvs30refs = (
        np.memmap(
            velocity_model_directory / "vs3dfile.s",
            dtype="<f4",
            shape=(domain_parameters.ny, domain_parameters.nz, domain_parameters.nx),
            mode="r",
        )[lf.stations.y, 0, lf.stations.x]
        * 1000.0
    )

    stations = pd.read_hdf(high_frequency_waveform_file, key="stations").set_index(
        "name"
    )
    stations["waveform_index"] = np.arange(len(stations))
    # ensure that LF and HF agree on station list, sometimes LF can drop a station or two
    stations = stations.loc[lf.stations.index]

    station_vs30 = pd.read_csv(
        station_vs30_ffp,
        delimiter=r"\s+",
        header=None,
        names=["name", "vs30"],
    ).set_index("name")
    stations = stations.join(station_vs30, how="inner")

    with multiprocessing.Pool() as pool:
        pool.starmap(
            functools.partial(
                bb_simulate_station,
                lf,
                high_frequency_waveform_file,
                hf_padding,
                lf_padding,
                broadband_config,
                n2,
                work_directory,
            ),
            stations.iterrows(),
        )

    with h5py.File(output_ffp, "w") as output_h5py:
        header_data = {
            "nt": bb_nt,
            "duration": bb_nt * broadband_config.dt,
            "start": bb_start_sec,
        }
        output_h5py.attrs.update(header_data)
        waveforms_dset = output_h5py.create_dataset(
            "waveforms", shape=(len(stations), bb_nt, 3), dtype=np.float32
        )

        for i, (name, station) in enumerate(stations.iterrows()):
            station_file_path = work_directory / f"{name}.bb"
            with open(station_file_path, mode="rb") as station_file_data:
                waveforms_dset[i] = np.fromfile(
                    station_file_data, dtype=np.float32
                ).reshape((bb_nt, 3))

    stations["name"] = stations.index
    stations["x"] = lf.stations.x
    stations["y"] = lf.stations.y
    stations["z"] = lf.stations.z
    stations["lf_vs_ref"] = lfvs30refs
    stations.to_hdf(output_ffp, key="stations", mode="a")
