"""Broadband Simulation.

Description
-----------
Combine high-frequency and low-frequency simulation waveforms for each station into a broadband simulation file.

Inputs
------
1. A realisation file containing:
   - Realisation metadata,
   - Domain parameters.
2. Stations VS30 reference values,
3. Low frequency waveform file,
4. High frequency waveform file.

Outputs
-------
An output broadband file in the HDF5 format.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own
computer using the `bb-sim` command which is installed after running
`pip install ucgmsim-workflow@git+https://github.com/ucgmsim/workflow`. If
running on your own computer, you need to configure a work directory
(`--work-directory`).

Usage
-----
`bb-sim REALISATION_FFP STATION_VS30_FFP LOW_FREQUENCY_WAVEFORM_FILE HIGH_FREQUENCY_WAVEFORM_FILE OUTPUT_FFP`

For More Help
-------------
See the output of `bb-sim --help`.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyfftw
import scipy as sp
import typer
import xarray as xr

from qcore import cli, timeseries
from site_calculation import amplification
from workflow import log_utils, realisations, sw4
from workflow.realisations import BroadbandParameters, RealisationMetadata
from workflow.schemas import SiteAmpModel

SITE_AMP_MODELS = {
    SiteAmpModel.CB2014: (
        amplification.campbell_bozorgnia_2014,
        amplification.CAMPBELL_BOZORGNIA_2014_FREQUENCIES,
    ),
    SiteAmpModel.BA2018: (
        amplification.bayless_abrahamson_2018,
        amplification.BAYLESS_ABRAHAMSON_2018_FREQUENCIES,
    ),
}

app = typer.Typer()

G = 1 / 981.0
TARGET_CHUNK_BYTES = 256 * 2**20
TAIL_TAPER_FRACTION = 0.05
# Reference Vs30 (m/s) of the high-frequency simulation, i.e. the Vs30
# the waveforms are amplified *from* towards each station's target Vs30.
VS30_SIM = 500.0


def align_datasets(
    lf: xr.Dataset, hf: xr.Dataset, dt: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """Lazily align LF and HF waveforms onto a common time axis.

    Both waveforms are zero-padded to span the same time domain,
    running from the earliest start to the latest end of the two
    simulations, and each keeps the whole time axis in one chunk.

    Parameters
    ----------
    lf : xr.Dataset
        The low-frequency dataset, with a 'start_sec' attribute.
    hf : xr.Dataset
        The high-frequency dataset, with a 'start_sec' attribute.
    dt : float
        The shared timestep of both datasets.

    Returns
    -------
    xr.DataArray
        The aligned low-frequency waveform.
    xr.DataArray
        The aligned high-frequency waveform.
    """
    lf_start = lf.attrs["start_sec"]
    hf_start = hf.attrs["start_sec"]
    start = min(lf_start, hf_start)
    lf_offset = round((lf_start - start) / dt)
    hf_offset = round((hf_start - start) / dt)
    common_nt = max(lf_offset + lf.sizes["time"], hf_offset + hf.sizes["time"])
    common_time = start + np.arange(common_nt) * dt

    def pad_waveform(waveform: xr.DataArray, offset: int) -> xr.DataArray:
        padded = waveform.pad(
            time=(offset, common_nt - offset - waveform.sizes["time"]),
            constant_values=0.0,
        )
        # Padding adds a chunk at either end of the time axis, but the
        # broadband FFTs need the whole axis in each block.
        return padded.assign_coords(time=common_time).chunk(time=-1)

    return (
        pad_waveform(lf["waveform"], lf_offset),
        pad_waveform(hf["waveform"], hf_offset),
    )


def resample_signal(dset: xr.Dataset, dt: float) -> xr.Dataset:
    """Resample waveform dataset to a new time step.

    Parameters
    ----------
    dset : xr.Dataset
        Input dataset with dimensions (component, station, time) and
        attributes 'dt'.
    dt : float
        Desired time step in seconds.

    Returns
    -------
    xr.Dataset
        Resampled dataset with updated time coordinates and dt attribute.
    """
    duration = dset["waveform"].sizes["time"] * dset.attrs["dt"]
    nt = round(duration / dt)

    new_time = np.arange(nt) * dt + dset.attrs["start_sec"]

    resampled_waveform = xr.apply_ufunc(
        sp.signal.resample,
        dset["waveform"],
        # This tells xarray that resample expects an array with all of the time component intact.
        # So it will be passed arrays of shape (n_component, n_stations, n_time) = (i, j, nt)
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        # The old time coordinates no longer apply to the resampled axis.
        exclude_dims={"time"},
        # xarray moves the core dimension to the last axis.
        kwargs={"num": nt, "axis": -1},
        dask="parallelized",
        # dask cannot infer the resampled length.
        dask_gufunc_kwargs={"output_sizes": {"time": nt}},
    )

    resampled_waveform = resampled_waveform.assign_coords(time=new_time)
    # Must drop both waveform variable and time dimension to avoid xarray
    # automatically reindexing the waveform according to the new axes.
    new_dset = dset.drop_vars(["waveform", "time"]).assign(waveform=resampled_waveform)
    new_dset.attrs["dt"] = dt
    return new_dset


class FilterLeg(StrEnum):
    """Which legs the matched Butterworth pair is applied to."""

    BOTH = "both"
    LF = "lf"
    HF = "hf"
    NONE = "none"


def _process_bb_chunk(
    lf_waveform: xr.DataArray,
    hf_waveform: xr.DataArray,
    vs30: xr.DataArray,
    hf_pga: xr.DataArray,
    dt: float,
    config: BroadbandParameters,
    filter_legs: FilterLeg,
) -> xr.DataArray:
    """Compute broadband waveforms for a chunk of stations.

    Applies the selected site amplification model to the high-frequency
    waveforms, then merges them with the low-frequency waveforms using a
    matched pair of high-pass and low-pass Butterworth filters, applied to the
    legs selected by `filter_legs`.

    Parameters
    ----------
    lf_waveform : xr.DataArray
        Low-frequency waveforms (dims component, station, time).
    hf_waveform : xr.DataArray
        High-frequency waveforms on the same axes as `lf_waveform`.
    vs30 : xr.DataArray
        Target Vs30 of each station (dims station).
    hf_pga : xr.DataArray
        Peak absolute HF acceleration (cm/s^2) of each component and station,
        which the site amplification depends on.
    dt : float
        Broadband timestep.
    config : BroadbandParameters
        The merge frequency, amplification band and site amplification
        model to apply.
    filter_legs : FilterLeg
        Which legs to filter at the merge frequency.

    Returns
    -------
    xr.DataArray
        The broadband waveforms in units of g, on the axes of `lf_waveform`.
    """
    lf = lf_waveform.values
    hf = hf_waveform.values
    nt = lf.shape[-1]

    amp_model_fn, amp_model_freqs = SITE_AMP_MODELS[config.site_amp_version]

    # Zero-pad to a length pyfftw can transform efficiently, and
    # pre-compute the FFT output frequencies the amplification is
    # sampled at.
    n_fft = pyfftw.next_fast_len(nt)
    fft_freqs = np.fft.rfftfreq(n_fft, dt)

    # The amplification models require float64 inputs.
    vs30_target = vs30.values.astype(np.float64)
    vs30_sim = np.full_like(vs30_target, VS30_SIM)
    pga = hf_pga.values.astype(np.float64) * G
    filter_lf = filter_legs in (FilterLeg.LF, FilterLeg.BOTH)
    filter_hf = filter_legs in (FilterLeg.HF, FilterLeg.BOTH)

    # Amplify and filter one component at a time to bound the float64
    # intermediates held in memory.
    bb = np.empty_like(lf)
    for i in range(lf.shape[0]):
        amp = amp_model_fn(vs30_target, vs30_sim, pga[i])
        amp = amplification.interpolate_frequencies(amp_model_freqs, fft_freqs, amp)
        # Constrain the amplification to the [fmin, fmax] band, tapering
        # logarithmically at either end.
        amplification.amp_lowpass(fft_freqs, amp, config.fmin, config.fmidbot)
        amplification.amp_highpass(fft_freqs, amp, config.fhightop, config.fmax)
        hf_leg = amplification.amplify_waveform(hf[i], amp, n_fft)
        lf_leg = lf[i]

        if filter_hf:
            hf_leg = timeseries.bwfilter(
                hf_leg, dt, config.flo, timeseries.Band.HIGHPASS
            )
        if filter_lf:
            lf_leg = timeseries.bwfilter(
                lf_leg, dt, config.flo, timeseries.Band.LOWPASS
            )
        bb[i] = (hf_leg + lf_leg) * G
    return lf_waveform.copy(data=bb.astype(np.float32, copy=False))


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
    filter_legs: Annotated[FilterLeg, typer.Option("--filter")] = FilterLeg.BOTH,
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
    filter_legs : FilterLeg
        Which legs to filter at the merge frequency: both (the default), only
        the LF or HF leg, or neither. Skip the LF filter when the solver has
        already low-passed the LF, so it is not filtered twice.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_config = BroadbandParameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    # Open lazily (no dask) and select the common stations *before* chunking.
    # The LF and HF files store stations in different orders, so selecting after
    # chunking is an all-to-all dask shuffle in which every output chunk depends
    # on every input chunk. This will result in materialising the whole array
    # in-memory. Selecting on the lazy backend arrays instead lets each dask
    # chunk read just its own stations from disk.
    lf = xr.open_dataset(low_frequency_waveform_file)
    lf = lf.drop_duplicates("station", keep="first")
    hf = xr.open_dataset(high_frequency_waveform_file)
    hf = hf.drop_duplicates("station", keep="first")

    common_stations = sorted(
        set(map(str, hf.station.values)) & set(map(str, lf.station.values))
    )
    # Chunk over stations only, so every chunk holds complete time
    # series for resampling, alignment and filtering.
    nt = max(len(lf["time"]), len(hf["time"]))
    n_stations = max(1, TARGET_CHUNK_BYTES // (3 * nt * np.float64().itemsize))
    chunking = {"component": -1, "station": n_stations, "time": -1}
    lf = lf.sel(station=common_stations).chunk(chunking)
    hf = hf.sel(station=common_stations).chunk(chunking)

    bb_dt = min(lf.attrs["dt"], hf.attrs["dt"])

    if not np.isclose(lf.attrs["dt"], bb_dt):
        lf = resample_signal(lf, bb_dt)
    if not np.isclose(hf.attrs["dt"], bb_dt):
        hf = resample_signal(hf, bb_dt)

    # Site amplification depends on the untapered HF PGA.
    hf_pga = abs(hf["waveform"]).max("time")
    # Taper the tail of the HF signal itself (before it is zero-padded onto
    # the common time axis) to limit spectral leakage in the amplification.
    tail_taper = np.ones((1, hf.sizes["time"]), dtype=np.float32)
    amplification.taper(tail_taper, TAIL_TAPER_FRACTION)
    hf["waveform"] = hf["waveform"] * xr.DataArray(tail_taper[0], dims="time")

    lf_waveform, hf_waveform = align_datasets(lf, hf, bb_dt)

    vs30_df = pd.read_csv(
        station_vs30_ffp,
        sep=r"\s+",
        header=None,
        names=["station", "vsite"],
    ).set_index("station")
    vs30 = xr.DataArray(
        vs30_df.loc[common_stations, "vsite"].to_numpy(np.float32),
        dims="station",
        coords={"station": common_stations},
    ).chunk(station=n_stations)

    # map_blocks hands each block the matching station slice of every
    # argument. The HF lat/lon coordinates stay on `hf_waveform`, so the
    # output carries only the LF-derived coordinates.
    bb_waveform = xr.map_blocks(
        _process_bb_chunk,
        lf_waveform,
        args=(hf_waveform, vs30, hf_pga),
        kwargs={"dt": bb_dt, "config": broadband_config, "filter_legs": filter_legs},
        template=lf_waveform.astype(np.float32),
    )
    attributes = {
        "units": "g",
        "dt": bb_dt,
        "flo": broadband_config.flo,
        "fmin": broadband_config.fmin,
        "fmidbot": broadband_config.fmidbot,
        "fhightop": broadband_config.fhightop,
        "fmax": broadband_config.fmax,
        "site_amp_model": str(broadband_config.site_amp_version),
        "filter": str(filter_legs),
    }
    # The LF file's supergrid width describes the run that produced the
    # waveforms, and `im-calc` writes it into the IM file's root attributes,
    # so pass it on.
    attributes |= {
        name: lf.attrs[name]
        for name in sw4.SUPERGRID_WIDTH_ATTRIBUTES.values()
        if name in lf.attrs
    }
    bb = xr.Dataset(
        {"waveform": bb_waveform, "vs30": vs30},
        coords={
            "latitude": ("station", lf.lat.values),
            "longitude": ("station", lf.lon.values),
        },
        attrs=attributes,
    )

    bb.to_netcdf(
        output_ffp,
        engine="h5netcdf",
        encoding={
            "waveform": {
                "fletcher32": True,  # Add Fletcher-32 checksums for long-term storage.
            }
        },
    )
    realisations.append_log_entry(realisation_ffp)
