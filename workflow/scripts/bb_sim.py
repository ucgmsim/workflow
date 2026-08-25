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
import pyfftw
import scipy as sp
import typer
import xarray as xr

from qcore import cli, timeseries
from site_calculation import amplification
from workflow import log_utils, realisations
from workflow.realisations import (
    BroadbandParameters,
    RealisationMetadata,
)
from workflow.schemas import SiteAmpModel

# Site amplification model -> (amplification function, model frequencies).
# Both models share the same (vs30, vs30_sim, pga) calling convention.
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


def align_datasets(
    lf: xr.Dataset, hf: xr.Dataset, dt: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """Lazily align LF and HF waveforms onto a common time axis.

    Both waveforms are zero-padded to span the same time domain,
    running from the earliest start to the latest end of the two
    simulations.

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
        return padded.assign_coords(time=common_time)

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

    # NOTE: I am not providing a default start second because we consider it an
    # error not to provide one (no implicit magic behaviour).
    new_time = np.arange(nt) * dt + dset.attrs["start_sec"]

    resampled_waveform = xr.apply_ufunc(
        sp.signal.resample,
        dset["waveform"],
        # This tells xarray that resample expects an array with all of the time component intact.
        # So it will be passed arrays of shape (n_component, n_stations, n_time) = (i, j, nt)
        input_core_dims=[["time"]],
        # This tells xarray that the time dimension is going to be returned in
        # its entirety by scipy resample.
        output_core_dims=[["time"]],
        # This tells xarray that the time coordinates from the dset dataset are no
        # longer any good. They will be dropped from the output array.
        exclude_dims=set(["time"]),
        # Array passed to resample will have time in the inner-most axis and the
        # default axis for resample is 0.
        kwargs=dict(num=nt, axis=-1),
        dask="parallelized",
        # The size of the resampled time dimension cannot be inferred by
        # dask, so it must be given explicitly.
        dask_gufunc_kwargs=dict(output_sizes={"time": nt}),
    ).chunk({"time": -1, "component": -1, "station": dset.chunksizes["station"]})

    resampled_waveform = resampled_waveform.assign_coords(time=new_time)
    # Must drop both waveform variable and time dimension to avoid xarray
    # automatically reindexing the waveform according to the new axes.
    new_dset = dset.drop_vars(["waveform", "time"]).assign(waveform=resampled_waveform)
    new_dset.attrs["dt"] = dt
    return new_dset


# Reference Vs30 (m/s) of the high-frequency simulation, i.e. the Vs30
# the waveforms are amplified *from* towards each station's target Vs30.
VS30_SIM = 500.0


LF_COMPONENTS = ["x", "y", "z"]
"""Component labels every low-frequency output uses.

Written by `lf-to-xarray` for SW4 and by `qcore.timeseries` for EMOD3D:
x = east-west, y = north-south, z = vertical.
"""

HF_COMPONENT_TO_LF = {"090": "x", "000": "y", "ver": "z"}
"""Component relabelling from the high-frequency simulation's convention.

`hf_simulation` names components by azimuth -- 090 for east, 000 for north,
and `ver` for the vertical -- where the low-frequency outputs name the same
three x/y/z. Same components, same order, different labels.
"""


def relabel_hf_components(hf: xr.Dataset) -> xr.Dataset:
    """Put an HF dataset's components on the low-frequency naming.

    Parameters
    ----------
    hf : xr.Dataset
        The high-frequency dataset, as written by `hf-sim`.

    Returns
    -------
    xr.Dataset
        The same dataset with its `component` coordinate relabelled. Labels
        already in the low-frequency convention are left alone, so this is
        idempotent.

    Raises
    ------
    ValueError
        If the components still do not match the low-frequency set. Without
        this check the mismatch is silent: xarray aligns the two datasets on
        `component`, finds no labels in common, and fills every high-frequency
        sample with NaN. The failure then surfaces a long way downstream, as a
        NaN somewhere inside the site amplification model.
    """
    components = [
        HF_COMPONENT_TO_LF.get(str(component), str(component))
        for component in hf.component.values
    ]
    if set(components) != set(LF_COMPONENTS):
        raise ValueError(
            f"High-frequency components {list(hf.component.values)} do not "
            f"correspond to the low-frequency components {LF_COMPONENTS}. "
            f"Add the mapping to HF_COMPONENT_TO_LF."
        )
    return hf.assign_coords(component=components)


def _process_bb_chunk(
    dset: xr.Dataset,
    dt: float,
    flo: float,
    fmin: float,
    fmidbot: float,
    fhightop: float,
    fmax: float,
    site_amp_model: SiteAmpModel,
) -> xr.Dataset:
    """Compute broadband waveforms for a chunk of stations.

    Applies the selected site amplification model to the high-frequency
    waveforms, then merges them with the low-frequency waveforms using a
    matched pair of high-pass and low-pass Butterworth filters.

    Parameters
    ----------
    dset : xr.Dataset
        Dataset with variables ``lf_waveform`` and ``hf_waveform``
        (dims component, station, time) on a common time axis, and
        ``vs30`` (dims station).
    dt : float
        Broadband timestep.
    flo : float
        The frequency (Hz) at which the low-frequency and
        high-frequency waveforms are merged.
    fmin : float
        Frequency (Hz) below which the site amplification is tapered
        out (lowpass end of the amplification band).
    fmidbot : float
        Frequency (Hz) above which the site amplification is applied in
        full at the lowpass end.
    fhightop : float
        Frequency (Hz) below which the site amplification is applied in
        full at the highpass end.
    fmax : float
        Frequency (Hz) above which the site amplification is tapered out
        (highpass end of the amplification band).
    site_amp_model : SiteAmpModel
        The site amplification model to apply.

    Returns
    -------
    xr.Dataset
        Dataset with a single ``waveform`` variable containing the
        broadband waveforms in units of g.
    """
    lf_waveform = dset["lf_waveform"].values
    hf_waveform = dset["hf_waveform"].values
    nt = lf_waveform.shape[-1]

    amp_model_fn, amp_model_freqs = SITE_AMP_MODELS[site_amp_model]

    # Zero-pad to a length pyfftw can transform efficiently, and
    # pre-compute the FFT output frequencies the amplification is
    # sampled at.
    n_fft = pyfftw.next_fast_len(nt)
    fft_freqs = np.fft.rfftfreq(n_fft, dt)

    # The amplification models require float64 inputs.
    vs30 = dset["vs30"].values.astype(np.float64)
    vs30_sim = np.full_like(vs30, VS30_SIM)

    bb_waveform = np.empty(lf_waveform.shape, dtype=np.float32)
    # Site amplification depends on each component's PGA, so amplify
    # component-by-component (vectorised over stations).
    for i in range(bb_waveform.shape[0]):
        pga = np.abs(hf_waveform[i]).max(axis=-1).astype(np.float64) * G

        amp = amp_model_fn(vs30, vs30_sim, pga)
        amp = amplification.interpolate_frequencies(amp_model_freqs, fft_freqs, amp)
        # Constrain the amplification to the [fmin, fmax] band, tapering
        # logarithmically at either end.
        amplification.amp_lowpass(fft_freqs, amp, fmin, fmidbot)
        amplification.amp_highpass(fft_freqs, amp, fhightop, fmax)

        # Taper the tail of the HF waveform (5%) to limit spectral
        # leakage before amplification.
        hf_component = hf_waveform[i].copy()
        amplification.taper(hf_component, 0.05)
        hf_amped = amplification.amplify_waveform(hf_component, amp, n_fft)

        hf_filtered = timeseries.bwfilter(hf_amped, dt, flo, timeseries.Band.HIGHPASS)
        lf_filtered = timeseries.bwfilter(
            lf_waveform[i], dt, flo, timeseries.Band.LOWPASS
        )
        bb_waveform[i] = (hf_filtered + lf_filtered) * G

    return dset.drop_vars(["lf_waveform", "hf_waveform", "vs30"]).assign(
        waveform=(("component", "station", "time"), bb_waveform)
    )


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

    # Open lazily (no dask) and select the common stations *before* chunking.
    # The LF and HF files store stations in different orders, so selecting after
    # chunking is an all-to-all dask shuffle in which every output chunk depends
    # on every input chunk. This will result in materialising the whole array
    # in-memory. Selecting on the lazy backend arrays instead lets each dask
    # chunk read just its own stations from disk.
    lf = xr.open_dataset(low_frequency_waveform_file)
    lf = lf.drop_duplicates("station", keep="first")
    hf = relabel_hf_components(xr.open_dataset(high_frequency_waveform_file))
    hf = hf.drop_duplicates("station", keep="first")

    common_stations = sorted(
        set(map(str, hf.station.values)) & set(map(str, lf.station.values))
    )
    # Chunk over stations only, so every chunk holds complete time
    # series for resampling, alignment and filtering.
    nt = max(len(lf["time"]), len(hf["time"]))
    n_stations = round(TARGET_CHUNK_BYTES / (3 * nt * np.float64().itemsize))
    chunking = {"component": -1, "station": n_stations, "time": -1}
    lf = lf.sel(station=common_stations).chunk(chunking)
    hf = hf.sel(station=common_stations).chunk(chunking)

    bb_dt = min(lf.attrs["dt"], hf.attrs["dt"])

    if not np.isclose(lf.attrs["dt"], bb_dt):
        lf = resample_signal(lf, bb_dt)
    if not np.isclose(hf.attrs["dt"], bb_dt):
        hf = resample_signal(hf, bb_dt)
    common_stations = sorted(
        set(map(str, hf.station.values)) & set(map(str, lf.station.values))
    )
    hf = hf.sel(station=common_stations)
    lf = lf.sel(station=common_stations)
    vs30_df = pd.read_csv(
        station_vs30_ffp,
        sep=r"\s+",
        header=None,
        names=["station", "vsite"],
    ).set_index("station")
    vs30_df["vsite"] = vs30_df["vsite"].astype(np.float32)
    vs30_df = vs30_df.loc[common_stations]

    lf_aligned, hf_aligned = align_datasets(lf, hf, bb_dt)

    # Station-dimension *coordinates* on `lf_aligned` (`supergrid_depth`, and
    # EMOD3D's `x`/`y`) ride from here to the intensity measure file with no
    # help: they survive `map_blocks` below, `_process_bb_chunk` drops data
    # variables only, and `IM.ims` keeps the input's non-dimension
    # coordinates. Anything that must reach the IM file and is *not* a
    # coordinate has to be hand-carried, the way `vs30` is.
    combined = xr.Dataset(
        {
            "lf_waveform": lf_aligned,
            # reset_coords drops the HF lat/lon coordinates, which would
            # otherwise conflict with the LF-derived latitude/longitude.
            "hf_waveform": hf_aligned.reset_coords(drop=True),
            "vs30": vs30_df["vsite"].to_xarray(),
        },
        coords={
            "component": ("component", ["x", "y", "z"]),
            "station": ("station", common_stations),
            "time": lf_aligned.time,
            "latitude": ("station", lf.lat.values),
            "longitude": ("station", lf.lon.values),
        },
        attrs={"units": "g"},
    ).chunk(chunking)

    combined = combined.unify_chunks()
    template = (
        combined["lf_waveform"].astype(np.float32).rename("waveform").to_dataset()
    )
    template.attrs = combined.attrs

    bb = xr.map_blocks(
        _process_bb_chunk,
        combined,
        kwargs=dict(
            dt=bb_dt,
            flo=broadband_config.flo,
            fmin=broadband_config.fmin,
            fmidbot=broadband_config.fmidbot,
            fhightop=broadband_config.fhightop,
            fmax=broadband_config.fmax,
            site_amp_model=broadband_config.site_amp_version,
        ),
        template=template,
    )
    bb["vs30"] = combined["vs30"]
    attributes = dict(
        dt=bb_dt,
        flo=broadband_config.flo,
        fmin=broadband_config.fmin,
        fmidbot=broadband_config.fmidbot,
        fhightop=broadband_config.fhightop,
        fmax=broadband_config.fmax,
        site_amp_model=str(broadband_config.site_amp_version),
    )
    # Attributes, unlike station coordinates, are *not* carried through
    # map_blocks: `template` above only has `combined`'s. The LF file's
    # supergrid width describes the run that produced the waveforms, and
    # `im-calc` writes it into the IM file's root attributes, so pass it on.
    attributes |= {
        name: lf.attrs[name] for name in ("SGWIDTH", "SGWIDTHGP") if name in lf.attrs
    }
    bb.attrs.update(attributes)

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
