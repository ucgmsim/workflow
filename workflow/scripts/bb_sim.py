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

import dataclasses
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
from workflow import log_utils, realisations
from workflow.realisations import (
    BroadbandParameters,
    EMOD3DParameters,
    RealisationMetadata,
    Resolution,
    SW4Parameters,
    VelocityModelParameters,
    find_command,
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

# `qcore.timeseries.bwfilter` is an order-4 Butterworth applied by `sosfiltfilt`,
# i.e. twice. These are the corner shifts it applies internally so that after
# both passes the response is exactly 1/sqrt(2) at the frequency it was handed.
# Mirrored here rather than imported because they are private to qcore;
# `tests/test_bb_filters.py` measures the real filter and fails if they drift.
BB_FILTER_ORDER = 4
BB_FILTER_PASSES = 2
_HIGHPASS_SHIFT = (np.sqrt(2) - 1) ** (1 / 8)
_LOWPASS_SHIFT = 1 / _HIGHPASS_SHIFT

# EMOD3D's `tfilter` takes a `phase` argument and runs a reverse pass when it is
# zero (source.c:1989). Every call site hard-codes `int phase = 0`, so the
# EMOD3D source low-pass is zero-phase, not causal.
EMOD3D_SOURCE_PASSES = 2

# Ceiling on the boost any correction may apply. The two real configurations
# need at most ~2.4x; anything approaching this bound means the source filter
# rolls off faster than the target and the leg is being reconstructed from
# content that is not there.
MAX_BOOST = 10.0
# Below this the high-frequency leg carries too little of the signal for the
# HF-side correction to be meaningful, and dividing by it amplifies nothing but
# numerical noise.
HF_GAIN_FLOOR = 1e-2


class CorrectionLeg(StrEnum):
    """Which leg of the recombination absorbs the source-filter correction."""

    LF = "lf"
    """Restore the low-frequency leg by dividing out its source filter.

    The physically direct choice: it puts the LF leg back to what the solver
    would have produced without the extra filter, and touches nothing else.
    Exact below the matching frequency. Above it the boost is clipped, which
    costs under 0.01 in ln of the total because the LF leg carries almost
    nothing there.
    """
    HF = "hf"
    """Leave the LF alone and fill the missing power from the HF leg.

    Only meaningful near and above the matching frequency. Below it, filling an
    LF deficit from the HF leg needs a boost of ten or more applied to
    stochastic content that has no valid long-period part, so the correction is
    clipped and the power sum is *not* restored. `warn_if_ill_conditioned` says
    so when it happens.
    """
    BOTH = "both"
    """Scale both legs together so the power sum is restored exactly.

    The best-conditioned option -- it needs a boost of under 1.4 anywhere -- but
    it makes up part of the deficit from the high-frequency leg, which is a
    different physical claim from restoring the low-frequency one.
    """


class Solver(StrEnum):
    """The low-frequency solver that produced the LF waveforms."""

    EMOD3D = "emod3d"
    SW4 = "sw4"


@dataclasses.dataclass(frozen=True)
class SourceLowpass:
    """A low-pass a solver has already applied to its source time functions.

    Both solvers filter the source rather than the output. The wave equation is
    linear, so that is equivalent to filtering every trace, which is what lets
    `bb_sim` divide the filter back out here instead of re-running the solver.
    """

    order: int
    """Butterworth order."""
    passes: int
    """1 for a causal filter, 2 for the zero-phase forward-and-back pair."""
    corner: float
    """Corner frequency in Hz."""


def solver_source_lowpass(
    solver: Solver, realisation_ffp: Path, defaults_version: str
) -> SourceLowpass | None:
    """The low-pass `solver` applied to its sources, read from the realisation.

    Parameters
    ----------
    solver : Solver
        The solver that produced the low-frequency waveforms.
    realisation_ffp : Path
        Path to the realisation file.
    defaults_version : str
        The realisation's defaults version.

    Returns
    -------
    SourceLowpass | None
        The filter, or None if the solver applied none.

    Raises
    ------
    ValueError
        If the solver applied a filter this function cannot describe, rather
        than silently correcting for the wrong thing.
    """
    if solver is Solver.SW4:
        sw4_config = SW4Parameters.read_from_realisation_or_defaults(
            realisation_ffp, defaults_version
        )
        prefilter = find_command(sw4_config.commands, "prefilter")
        if prefilter is None:
            return None
        parameters = prefilter.parameters
        if parameters.get("type") != "lowpass":
            raise ValueError(
                f"SW4 prefilter is type={parameters.get('type')!r}; only 'lowpass' "
                "can be corrected for here"
            )
        return SourceLowpass(
            order=int(parameters["order"]),
            passes=int(parameters["passes"]),
            corner=float(parameters["fc2"]),
        )

    emod3d_config = EMOD3DParameters.read_from_realisation_or_defaults(
        realisation_ffp, defaults_version
    )
    if not emod3d_config.bfilt:
        # `tfilter` is guarded by `if(bfilt)`, so zero means no filter at all.
        return None
    if emod3d_config.fhi:
        raise ValueError(
            f"EMOD3D fhi={emod3d_config.fhi} applies a source high-pass as well as "
            "the low-pass; correcting for the low-pass alone would be wrong"
        )
    # EMOD3D's flo is not stored in the realisation or the LF file. It is
    # derived in `create_e3d_par.create_duration_parameters`, and is repeated
    # here from the same two inputs.
    velocity_model = VelocityModelParameters.read_from_realisation_or_defaults(
        realisation_ffp, defaults_version
    )
    resolution = Resolution.read_from_realisation_or_defaults(
        realisation_ffp, defaults_version
    )
    return SourceLowpass(
        order=emod3d_config.bfilt,
        passes=EMOD3D_SOURCE_PASSES,
        corner=velocity_model.min_vs / (5 * resolution.resolution),
    )


def butterworth_gain(
    frequencies: np.ndarray,
    order: int,
    corner: float,
    passes: int,
    dt: float,
    band: str = "lowpass",
) -> np.ndarray:
    """Magnitude response of a Butterworth filter applied `passes` times.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies (Hz) to evaluate the response at.
    order : int
        Butterworth order.
    corner : float
        Corner frequency in Hz.
    passes : int
        Number of times the filter is applied. Two passes square the magnitude.
    dt : float
        Sample interval in seconds.
    band : str
        Either 'lowpass' or 'highpass'.

    Returns
    -------
    np.ndarray
        The magnitude response at `frequencies`.
    """
    sos = sp.signal.butter(
        order, corner, btype=band, output="sos", fs=1.0 / dt
    )
    _, response = sp.signal.sosfreqz(sos, worN=2 * np.pi * frequencies * dt)
    return np.abs(response) ** passes


def warn_if_ill_conditioned(
    dt: float, flo: float, source: SourceLowpass | None, leg: CorrectionLeg
) -> None:
    """Report where the correction is clipped and cannot restore the power sum.

    Called once, rather than per chunk, so the message is not repeated for every
    block dask schedules.

    Parameters
    ----------
    dt : float
        Broadband sample interval.
    flo : float
        The LF/HF matching frequency.
    source : SourceLowpass | None
        The solver's source filter.
    leg : CorrectionLeg
        Which leg absorbs the correction.
    """
    if source is None:
        return
    frequencies = np.logspace(np.log10(flo / 10), np.log10(flo * 5), 400)
    lf_gain, hf_gain = recombination_gains(frequencies, dt, flo, source, leg)
    clipped = np.isclose(lf_gain, MAX_BOOST) | np.isclose(hf_gain, MAX_BOOST)
    if not clipped.any():
        return
    logger = log_utils.get_logger(__name__)
    logger.warning(
        "The source-filter correction is clipped, so the power sum is not fully "
        "restored across the whole band."
        + (
            " Below the matching frequency the high-frequency leg has no valid "
            "content to supply; consider --filter lf or --filter both."
            if leg is CorrectionLeg.HF
            else " This is above the matching frequency, where the "
            "low-frequency leg carries almost nothing, so the effect on the "
            "total is small."
        ),
        correction_leg=str(leg),
        max_boost=MAX_BOOST,
        clipped_from_hz=float(frequencies[clipped].min()),
        clipped_to_hz=float(frequencies[clipped].max()),
    )


def recombination_gains(
    frequencies: np.ndarray,
    dt: float,
    flo: float,
    source: SourceLowpass | None,
    leg: CorrectionLeg,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-leg correction for the source filter already baked into the LF.

    The LF and HF legs are independent realisations, so they add in *power*, and
    `bwfilter`'s shifted pair is designed to be power-complementary:
    ``|H_lp|^2 + |H_hp|^2 = 1``. A source low-pass multiplies the LF leg a second
    time, which breaks that identity and leaves a hole in the transition band.
    These gains put the power sum back where the pair intended.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies (Hz) the gains are sampled at.
    dt : float
        Broadband sample interval.
    flo : float
        The LF/HF matching frequency.
    source : SourceLowpass | None
        The solver's source filter. None applies no correction.
    leg : CorrectionLeg
        Which leg absorbs the correction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Multiplicative gains for the LF and HF legs, both ones if `source` is None.
    """
    ones = np.ones_like(frequencies)
    if source is None:
        return ones, ones

    lowpass = butterworth_gain(
        frequencies, BB_FILTER_ORDER, flo * _LOWPASS_SHIFT, BB_FILTER_PASSES, dt
    )
    highpass = butterworth_gain(
        frequencies,
        BB_FILTER_ORDER,
        flo * _HIGHPASS_SHIFT,
        BB_FILTER_PASSES,
        dt,
        band="highpass",
    )
    source_gain = butterworth_gain(
        frequencies, source.order, source.corner, source.passes, dt
    )

    # What the matched pair is supposed to deliver, and what it delivers once
    # the source filter has been applied to the LF leg a second time.
    target_power = lowpass**2 + highpass**2
    current_power = (source_gain * lowpass) ** 2 + highpass**2

    match leg:
        case CorrectionLeg.LF:
            # Dividing the source filter back out restores the LF leg exactly.
            return np.clip(1.0 / source_gain, None, MAX_BOOST), ones
        case CorrectionLeg.HF:
            # Make the shortfall up from the HF side, leaving the LF as it is.
            shortfall = np.sqrt(
                np.clip(target_power - (source_gain * lowpass) ** 2, 0.0, None)
            )
            hf_gain = np.where(
                highpass > HF_GAIN_FLOOR,
                np.clip(shortfall / np.maximum(highpass, HF_GAIN_FLOOR), None, MAX_BOOST),
                1.0,
            )
            return ones, hf_gain
        case CorrectionLeg.BOTH:
            # One factor on both legs restores the total without either leg
            # carrying the whole boost. Well conditioned everywhere, because
            # `current_power` is bounded below by the high-pass leg.
            shared = np.clip(np.sqrt(target_power / current_power), None, MAX_BOOST)
            return shared, shared


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
    source_lowpass: SourceLowpass | None = None,
    correction_leg: CorrectionLeg = CorrectionLeg.LF,
) -> xr.Dataset:
    """Compute broadband waveforms for a chunk of stations.

    Applies the selected site amplification model to the high-frequency
    waveforms, then merges them with the low-frequency waveforms using a
    matched pair of high-pass and low-pass Butterworth filters.

    Where `source_lowpass` is given, the solver's own source filter is divided
    back out of the recombination first, so the LF leg is filtered once rather
    than twice and the matched pair's power sum is restored.

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
    source_lowpass : SourceLowpass | None
        The low-pass the solver already applied to its source time functions.
        None leaves the recombination exactly as it was.
    correction_leg : CorrectionLeg
        Which leg absorbs the correction for `source_lowpass`.

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
    lf_gain, hf_gain = recombination_gains(
        fft_freqs, dt, flo, source_lowpass, correction_leg
    )
    correcting = source_lowpass is not None
    if correcting:
        # `amplify_waveform` takes one gain curve per station, and these are the
        # same curve for every station in the chunk.
        stations = lf_waveform.shape[1]
        lf_gain = np.tile(lf_gain, (stations, 1))
        hf_gain = np.tile(hf_gain, (stations, 1))

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
        # Applied after the matched pair rather than in place of it, so that
        # with no correction the arithmetic is bit-for-bit what it always was.
        if correcting:
            lf_filtered = amplification.amplify_waveform(lf_filtered, lf_gain, n_fft)
            hf_filtered = amplification.amplify_waveform(hf_filtered, hf_gain, n_fft)
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
    solver: Annotated[
        Solver | None, typer.Option("--solver", case_sensitive=False)
    ] = None,
    filter_leg: Annotated[
        CorrectionLeg, typer.Option("--filter", case_sensitive=False)
    ] = CorrectionLeg.LF,
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
    solver : Solver | None
        The solver that produced the low-frequency waveforms. Given, the source
        low-pass it already applied (SW4's `prefilter` command, or EMOD3D's
        `bfilt` and `flo`) is divided back out of the recombination, so the LF
        leg is filtered once rather than twice. Omitted, the recombination is
        unchanged.
    filter_leg : CorrectionLeg
        Which leg absorbs that correction: `lf` restores the low-frequency leg,
        `hf` fills the missing power from the high-frequency leg instead, and
        `both` scales the two together so neither is boosted hard. Ignored
        without `--solver`.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_config = BroadbandParameters.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    source_lowpass = (
        None
        if solver is None
        else solver_source_lowpass(solver, realisation_ffp, metadata.defaults_version)
    )
    if solver is not None and source_lowpass is None:
        log_utils.get_logger(__name__).warning(
            "The solver applied no source low-pass in this realisation, so the "
            "recombination is unchanged.",
            solver=str(solver),
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
    warn_if_ill_conditioned(
        bb_dt, broadband_config.flo, source_lowpass, filter_leg
    )

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
            source_lowpass=source_lowpass,
            correction_leg=filter_leg,
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
    # Recorded so a broadband file says whether it was corrected and for what.
    # Without this the correction is invisible downstream, and two files that
    # differ by it look identical.
    if source_lowpass is not None:
        attributes |= {
            "source_filter_solver": str(solver),
            "source_filter_order": source_lowpass.order,
            "source_filter_passes": source_lowpass.passes,
            "source_filter_corner": source_lowpass.corner,
            "source_filter_correction": str(filter_leg),
        }
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
