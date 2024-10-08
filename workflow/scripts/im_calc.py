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

import multiprocessing
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Callable

import h5py
import numba
import numexpr as ne
import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm
import typer
from pyfftw.interfaces import numpy_fft as fft

from workflow.realisations import (
    BroadbandParameters,
    IntensityMeasureCalcuationParameters,
    RealisationMetadata,
)

app = typer.Typer()


def response_spectra(
    waveforms: npt.NDArray[np.float32],
    dt: float,
    periods: npt.NDArray[np.float32],
    xi: float = 0.05,
    max_freq_ratio: int = 5,
):
    fourier = fft.rfft(waveforms, axis=1, threads=multiprocessing.cpu_count())
    freq = np.linspace(0, 1 / (2 * dt), num=fourier.shape[1])
    ang_freq = 2 * np.pi * freq  # (nt,)
    oscillator_freq = 2 * np.pi * periods  # (np,)
    h = -np.square(oscillator_freq)[
        :, np.newaxis
    ] / (  # (np x nt) / (nt - np - np * nt) ERROR
        np.square(ang_freq)[np.newaxis, :]
        - np.square(oscillator_freq)[:, np.newaxis]
        - 2j * xi * oscillator_freq[:, np.newaxis] * ang_freq[np.newaxis, :]
    )  # h needs to be a matrix of size (np, nt)
    # fourier = (stat, nt), h = (np, nt) want: (np, stat, nt)
    n = len(fourier)
    m = max(n, int(max_freq_ratio * oscillator_freq.max() / freq[1]))
    scale = m / n
    return scale * fft.irfft(
        fourier[np.newaxis, ...] * h[:, np.newaxis, :],
        axis=2,
        threads=multiprocessing.cpu_count(),
    )
    # now have array of shape (np, nstat, nt)


class ComponentWiseOperation(StrEnum):
    NONE = "cos(theta) * comp_0 + sin(theta) * comp_90"
    ABS = "abs(cos(theta) * comp_0 + sin(theta) * comp_90)"
    SQUARE = "(cos(theta) * comp_0 + sin(theta) * comp_90)*(cos(theta) * comp_0 + sin(theta) * comp_90)"


@numba.njit
def newmark_estimate_psa(
    waveforms: npt.NDArray[np.float32],
    t: npt.NDArray[np.float32],
    dt: float,
    w: npt.NDArray[np.float32],
    xi: float = 0.05,
    gamma: float = 1 / 2,
    beta: float = 1 / 4,
    m: float = 1,
) -> npt.NDArray[np.float32]:
    w = w * np.pi * 2
    c = 2 * xi * w
    a = 1 / (beta * dt) * m + (gamma / beta) * c
    b = 1 / (2 * beta) * m + dt * (gamma / (2 * beta) - 1) * c
    k = np.square(w)
    kbar = k + (gamma / (beta * dt)) * c + 1 / (beta * dt**2) * m
    u = np.zeros(
        shape=(waveforms.shape[0], waveforms.shape[1], w.size), dtype=np.float32
    )
    # calculations for each time step
    dudt = np.zeros_like(w)
    #         ns         np x ns    np x ns
    d2udt2 = np.zeros_like(w)
    for i in range(waveforms.shape[0]):
        for j in range(1, waveforms.shape[1]):
            if j == 1:
                d2udt2 = (-m * waveforms[i, 0] - c * dudt - k * u[i, 0]) / m
            d_pti = -m * (waveforms[i, j] - waveforms[i, j - 1])
            d_pbari = d_pti + a * dudt + b * d2udt2
            d_ui = d_pbari / kbar
            d_dudti = (
                (gamma / (beta * dt)) * d_ui
                - (gamma / beta) * dudt
                + dt * (1 - gamma / (2 * beta)) * d2udt2
            )
            d_d2udt2i = (
                1 / (beta * dt**2) * d_ui
                - 1 / (beta * dt) * dudt
                - 1 / (2 * beta) * d2udt2
            )

            # Convert from incremental formulation and store values in vector
            u[i, j] = u[i, j - 1] + d_ui
            dudt += d_dudti
            d2udt2 += d_d2udt2i

        dudt[:] = 0
        d2udt2[:] = 0
    return u


def rotd_psa_values(
    comp_000: npt.NDArray[np.float32],
    comp_090: npt.NDArray[np.float32],
    w: npt.NDArray[np.float32],
    step: int = 20,
) -> npt.NDArray[np.float32]:
    theta = np.linspace(0, 180, num=180, dtype=np.float32)
    ne.set_num_threads(multiprocessing.cpu_count())
    psa = np.zeros((comp_000.shape[0], comp_000.shape[-1], 2), np.float32)
    out = np.zeros((step, *comp_000.shape[1:], 180), np.float32)
    w = np.square(w * 2 * np.pi)
    for i in tqdm.trange(0, comp_000.shape[0], step):
        step_000 = comp_000[i : i + step]
        step_090 = comp_000[i : i + step]
        psa[i : i + step] = np.transpose(
            w[np.newaxis, np.newaxis, :]
            * np.percentile(
                np.max(
                    ne.evaluate(
                        "abs(comp_000 * cos(theta) + comp_090 * sin(theta))",
                        {
                            "comp_000": step_000[..., np.newaxis],
                            "theta": theta[np.newaxis, ...],
                            "comp_090": step_090[..., np.newaxis],
                        },
                        out=out[: len(step_000)],
                    )[: len(step_000)],
                    axis=1,
                ),
                [50, 100],
                axis=-1,
            ),
            [1, 2, 0],
        )
    return psa


def compute_psa(
    stations: pd.Series,
    waveforms: npt.NDArray[np.float32],
    periods: npt.NDArray[np.float32],
    dt: float,
) -> pd.DataFrame:
    g = 981
    t = np.arange(waveforms.shape[1]) * dt
    comp_0 = newmark_estimate_psa(
        waveforms[:, :, 1],
        t,
        dt,
        periods,
    )

    comp_90 = newmark_estimate_psa(
        waveforms[:, :, 0],
        t,
        dt,
        periods,
    )

    rotd_psa = g * rotd_psa_values(
        comp_0, comp_90, periods, step=multiprocessing.cpu_count()
    )
    conversion_factor = g * np.square(2 * np.pi * periods)[np.newaxis, :]
    comp_0_psa = conversion_factor * np.abs(comp_0).max(axis=1)
    comp_90_psa = conversion_factor * np.abs(comp_90).max(axis=1)
    ver_psa = conversion_factor * np.abs(
        newmark_estimate_psa(waveforms[:, :, 0], t, dt, periods)
    ).max(axis=1)
    geom_psa = np.sqrt(comp_0_psa * comp_90_psa)
    psa_df = pd.DataFrame(
        columns=[
            "station",
            "intensity_measure",
            "000",
            "090",
            "ver",
            "geom",
            "rotd100",
            "rotd50",
        ]
    )
    for i, p in enumerate(periods):
        period_df = pd.DataFrame(
            {
                "station": stations,
                "intensity_measure": f"pSA_{p:.2f}",
                "000": comp_0_psa[:, i],
                "090": comp_90_psa[:, i],
                "ver": ver_psa[:, i],
                "geom": geom_psa[:, i],
                "rotd50": rotd_psa[:, i, 0],
                "rotd100": rotd_psa[:, i, 1],
            }
        )
        psa_df = pd.concat([psa_df, period_df])
    return psa_df


def compute_in_rotations(
    waveforms: npt.NDArray[np.float32],
    function: Callable,
    component_wise_operation: ComponentWiseOperation | str = ComponentWiseOperation.ABS,
) -> pd.DataFrame:
    (stations, nt, _) = waveforms.shape
    values = np.zeros(shape=(stations, 180), dtype=waveforms.dtype)

    comp_0 = waveforms[:, :, 1]
    comp_90 = waveforms[:, :, 0]
    for i in range(180):
        theta = np.deg2rad(i)
        values[:, i] = function(ne.evaluate(component_wise_operation))

    comp_0 = values[:, 0]
    comp_90 = values[:, 90]
    comp_ver = waveforms[:, :, 2]
    match component_wise_operation:
        case ComponentWiseOperation.ABS:
            comp_ver = ne.evaluate("abs(comp_ver)")
        case ComponentWiseOperation.SQUARE:
            comp_ver = ne.evaluate("(comp_ver)**2")
    ver = function(comp_ver)
    rotated_max = np.max(values, axis=1)
    rotated_median = np.median(values, axis=1)
    return pd.DataFrame(
        {
            "000": comp_0,
            "090": comp_90,
            "ver": ver,
            "geom": np.sqrt(comp_0 * comp_90),
            "rotd100": rotated_max,
            "rotd50": rotated_median,
        }
    )


@numba.njit(parallel=True)
def trapz(waveforms: npt.NDArray[np.float32], dt: float) -> npt.NDArray[np.float32]:
    sums = np.zeros((waveforms.shape[0],), np.float32)
    for i in numba.prange(waveforms.shape[0]):
        for j in range(waveforms.shape[1]):
            if j == 0 or j == waveforms.shape[1] - 1:
                sums[i] += waveforms[i, j] / 2
            else:
                sums[i] += waveforms[i, j]
    return sums * dt


def compute_significant_duration(
    waveforms: npt.NDArray[np.float32],
    dt: float,
    percent_low: float,
    percent_high: float,
):
    arias_intensity = np.cumsum(waveforms, axis=1)
    arias_intensity /= arias_intensity[:, -1][:, np.newaxis]
    sum_mask = ne.evaluate(
        "(arias_intensity >= percent_low / 100) & (arias_intensity <= percent_high / 100)"
    )
    threshold_values = np.count_nonzero(sum_mask, axis=1) * dt
    return threshold_values.ravel()


@app.command(help="Calculate instensity measures for simulation data.")
def calculate_instensity_measures(
    realisation_ffp: Annotated[
        Path,
        typer.Argument(
            help="Realisation filepath", exists=True, dir_okay=False, writable=True
        ),
    ],
    broadband_simulation_ffp: Annotated[
        Path,
        typer.Argument(help="Broadband simulation file.", exists=True, dir_okay=False),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Output path for IM calculation summary statistics.",
            dir_okay=False,
            writable=True,
        ),
    ],
    simulated_stations: Annotated[
        bool, typer.Option(help="If passed, calculate for simulated stations.")
    ] = True,
):
    """Calculate intensity measures for simulation data.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file.
    broadband_simulation_ffp : Path
        Path to the broadband simulation waveforms.
    output_path : Path
        Output directory for IM calc summary statistics.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    broadband_parameters = BroadbandParameters.read_from_realisation(realisation_ffp)
    intensity_measure_parameters = (
        IntensityMeasureCalcuationParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )

    with h5py.File(broadband_simulation_ffp, mode="r") as broadband_file:
        waveforms = np.array(broadband_file["waveforms"])

    stations = pd.read_hdf(broadband_simulation_ffp, key="stations")
    intensity_measures = ["psa", "pga", "pgv", "cav", "ai", "ds575", "ds595"]
    intensity_measure_statistics = pd.DataFrame()
    pbar = tqdm.tqdm(intensity_measures)
    g = 981
    for intensity_measure in pbar:
        pbar.set_description(intensity_measure)
        match intensity_measure:
            case "pga":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms, lambda v: v.max(axis=1)
                )  # ~30s
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "pga"
            case "pgv":
                individual_intensity_measure_statistics = compute_in_rotations(
                    np.cumsum(waveforms, axis=1) * g * broadband_parameters.dt,
                    lambda v: v.max(axis=1),
                )
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "pgv"
            case "cav":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms, lambda v: trapz(v, broadband_parameters.dt)
                )  # ~ 30s
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "cav"
            case "ai":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms,
                    lambda v: (np.pi * g) / 2 * trapz(v, broadband_parameters.dt),
                    component_wise_operation=ComponentWiseOperation.SQUARE,
                )  # ~ 30s
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "ai"
            case "ds575":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms,
                    lambda v: compute_significant_duration(
                        v, broadband_parameters.dt, 5, 75
                    ),
                    component_wise_operation=ComponentWiseOperation.SQUARE,
                )  # ~ 45s
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "ds575"
            case "ds595":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms,
                    lambda v: compute_significant_duration(
                        v, broadband_parameters.dt, 5, 95
                    ),
                    component_wise_operation=ComponentWiseOperation.SQUARE,
                )  # ~ 45s
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "ds595"
            case "psa":
                individual_intensity_measure_statistics = compute_psa(
                    stations.index,
                    waveforms,
                    np.array(
                        intensity_measure_parameters.valid_periods, dtype=np.float32
                    ),
                    broadband_parameters.dt,
                )

        intensity_measure_statistics = pd.concat(
            [intensity_measure_statistics, individual_intensity_measure_statistics]
        )

    intensity_measure_statistics.set_index(["station", "intensity_measure"]).to_parquet(
        output_path
    )
