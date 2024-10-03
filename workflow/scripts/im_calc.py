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
    return fft.irfft(
        fourier[np.newaxis, ...] * h[:, np.newaxis, :],
        axis=2,
        threads=multiprocessing.cpu_count(),
    )
    # now have array of shape (np, nstat, nt)


class ComponentWiseOperation(StrEnum):
    NONE = "cos(theta) * comp_0 + sin(theta) * comp_90"
    ABS = "abs(cos(theta) * comp_0 + sin(theta) * comp_90)"
    SQUARE = "(cos(theta) * comp_0 + sin(theta) * comp_90)**2"


def compute_psa(
    waveforms: npt.NDArray[np.float32],
    periods: npt.NDArray[np.float32],
    dt: float,
) -> pd.DataFrame:
    (stations, nt, _) = waveforms.shape
    values = np.zeros(shape=(180, stations, len(periods)), dtype=waveforms.dtype)
    print("Computing comp_0")
    comp_0 = response_spectra(waveforms[:, :, 0], dt, periods)
    print("computing comp_90")
    comp_90 = response_spectra(waveforms[:, :, 1], dt, periods)

    for i in tqdm.trange(180):
        theta = np.deg2rad(i)
        comp = ne.evaluate(ComponentWiseOperation.NONE)
        values[i] = comp.max(axis=-1).T
    return values


def compute_in_rotations(
    waveforms: npt.NDArray[np.float32],
    function: Callable,
    component_wise_operation: ComponentWiseOperation | str = ComponentWiseOperation.ABS,
) -> pd.DataFrame:
    (stations, nt, _) = waveforms.shape
    values = np.zeros(shape=(stations, 180), dtype=waveforms.dtype)

    comp_0 = waveforms[:, :, 0]
    comp_90 = waveforms[:, :, 1]
    for i in range(180):
        theta = np.deg2rad(i)
        values[:, i] = function(ne.evaluate(component_wise_operation))

    comp_0 = values[:, 0]
    comp_90 = values[:, 90]
    rotated_max = np.max(values, axis=1)
    rotated_median = np.median(values, axis=1)
    return pd.DataFrame(
        {
            "000": comp_0,
            "090": comp_90,
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
        "(arias_intensity >= percent_low) & (arias_intensity <= percent_high)"
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
            file_okay=False,
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
    output_path.mkdir(exist_ok=True)
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
    intensity_measures = ["pga", "pgv", "cav", "ai", "ds575"]
    intensity_measure_statistics = pd.DataFrame()
    pbar = tqdm.tqdm(intensity_measures)
    for intensity_measure in pbar:
        pbar.set_description(intensity_measure)
        match intensity_measure:
            case "pga":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms, lambda v: v.max(axis=1)
                )  # ~30s
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "pga"
                individual_intensity_measure_statistics = (
                    individual_intensity_measure_statistics.set_index(
                        ["station", "intensity_measure"]
                    )
                )
            case "pgv":
                individual_intensity_measure_statistics = compute_in_rotations(
                    np.cumsum(waveforms, axis=1) * 981 * broadband_parameters.dt,
                    lambda v: v.max(axis=1),
                )
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "pgv"
                individual_intensity_measure_statistics = (
                    individual_intensity_measure_statistics.set_index(
                        ["station", "intensity_measure"]
                    )
                )
            case "cav":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms, lambda v: trapz(v, broadband_parameters.dt)
                )  # ~ 30s
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "cav"
                individual_intensity_measure_statistics = (
                    individual_intensity_measure_statistics.set_index(
                        ["station", "intensity_measure"]
                    )
                )
            case "ai":
                individual_intensity_measure_statistics = compute_in_rotations(
                    waveforms,
                    lambda v: np.trapz(v, dx=broadband_parameters.dt, axis=1),
                    component_wise_operation=ComponentWiseOperation.SQUARE,
                )  # ~ 30s
                individual_intensity_measure_statistics["station"] = stations.index
                individual_intensity_measure_statistics["intensity_measure"] = "ai"
                individual_intensity_measure_statistics = (
                    individual_intensity_measure_statistics.set_index(
                        ["station", "intensity_measure"]
                    )
                )
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
                individual_intensity_measure_statistics = (
                    individual_intensity_measure_statistics.set_index(
                        ["station", "intensity_measure"]
                    )
                )
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
                individual_intensity_measure_statistics = (
                    individual_intensity_measure_statistics.set_index(
                        ["station", "intensity_measure"]
                    )
                )
        intensity_measure_statistics = pd.concat(
            [intensity_measure_statistics, individual_intensity_measure_statistics]
        )
    print(intensity_measure_statistics)
    # print("Computed DS575")

    # psa = compute_psa(
    #     waveforms,
    #     np.array(
    #         [
    #             0.01,
    #             0.02,
    #             0.03,
    #             0.04,
    #             0.05,
    #             0.075,
    #             0.1,
    #             0.12,
    #             0.15,
    #             0.17,
    #             0.2,
    #             0.25,
    #             0.3,
    #             0.4,
    #             0.5,
    #             0.6,
    #             0.7,
    #             0.75,
    #             0.8,
    #             0.9,
    #             1.0,
    #             1.25,
    #             1.5,
    #             2.0,
    #             2.5,
    #             3.0,
    #             4.0,
    #             5.0,
    #             6.0,
    #             7.5,
    #             10.0,
    #         ]
    #     ),
    #     broadband_parameters.dt,
    # )
    # print("Computed PSA")
    # print(psa)
    # print(sys.getsizeof(pgv))
    # print(stations)
    # print(ds575)
    # print(cav)
    # print(pgv)
    # print(cav)
    # print(ai)
