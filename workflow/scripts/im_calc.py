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
from pathlib import Path
from typing import Annotated

import typer

from IM_calculation.IM import im_calculation
from qcore import qclogging
from workflow.realisations import (
    IntensityMeasureCalcuationParameters,
    RealisationMetadata,
)

app = typer.Typer()


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
    intensity_measure_parameters = (
        IntensityMeasureCalcuationParameters.read_from_realisation_or_defaults(
            realisation_ffp, metadata.defaults_version
        )
    )

    logger = qclogging.get_logger("IM_calc")
    logger.info("IM_Calc started")

    im_options = {
        "pSA": intensity_measure_parameters.valid_periods,
        "SDI": intensity_measure_parameters.valid_periods,
        "FAS": intensity_measure_parameters.fas_frequencies,
    }

    im_calculation.compute_measures_multiprocess(
        broadband_simulation_ffp,
        "binary",  # hard-code for binary file format for broadband waveforms
        wave_type=None,
        station_names=None,  # Passing None -> runs for all stations in the file
        ims=intensity_measure_parameters.ims,
        comp=intensity_measure_parameters.components,
        im_options=im_options,
        output=output_path,
        identifier="realisation",
        rupture="realisation",
        run_type="simulated",
        version="XXpY",  # default value. Not needed?
        process=multiprocessing.cpu_count(),
        simple_output=True,
        units=intensity_measure_parameters.units,
        advanced_im_config=None,
        real_only=False,
        logger=logger,
    )
