"""Generate Rupture Propagation.

Description
-----------
Generate a likely rupture propagation for a realisation.

Inputs
------

1. A realisation containing a source configuration.
2. An initial fault for the rupture to begin on.
3. A list of fault rakes, e.g. Acton=110

Outputs
-------
A realisation file containing:

1. A rupture propagation plan (i.e. how the rupture jumps between faults, and where),
2. The estimated rupture magnitude and apportionment to the involved faults.
3. The definition of the rakes.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `generate-rupture-propagation` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`nshm2022-to-realisation REALISATION_FFP INTIAL_FAULT RAKES`

For More Help
-------------
See the output of `nshm2022-to-realisation --help`.
"""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from qcore.uncertainties import distributions, mag_scaling
from source_modelling import rupture_propagation
from source_modelling.sources import Fault
from workflow import realisations

app = typer.Typer()


def a_to_mw_leonard(area: float, rake: float) -> float:
    """
    Convert fault area and rake to moment magnitude using the Leonard scaling relation.

    Parameters
    ----------
    area : float
        The area of the fault in square kilometres.
    rake : float
        The rake angle of the fault in degrees.

    Returns
    -------
    float
        The estimated moment magnitude of the fault.

    References
    ----------
    Leonard, M. (2010). Earthquake fault scaling: Self-consistent
    relating of rupture length, width, average displacement, and
    moment release. Bulletin of the Seismological Society of America,
    100(5A), 1971-1988.
    """
    return mag_scaling.a_to_mw_leonard(area, 4, 3.99, rake)


def default_magnitude_estimation(
    faults: dict[str, Fault], rakes: dict[str, float]
) -> dict[str, float]:
    """Estimate the magnitudes for a set of faults based on their areas and average rake.

    Parameters
    ----------
    faults : dict
        A dictionary where the keys are fault names and the values are `Fault` objects containing information about each fault.
    rakes : dict
        A dictionary where the keys are fault names and the values are rake angles (in degrees) for each fault.

    Returns
    -------
    dict
        A dictionary where the keys are fault names and the values are the estimated magnitudes for each fault.
    """
    total_area = sum(fault.area() for fault in faults.values())
    avg_rake = np.mean(list(rakes.values()))
    estimated_mw = a_to_mw_leonard(total_area, avg_rake)
    estimated_moment = mag_scaling.mag2mom(estimated_mw)
    return {
        fault_name: mag_scaling.mom2mag((fault.area() / total_area) * estimated_moment)
        for fault_name, fault in faults.items()
    }


def rake_parser(rake_value: str) -> dict[str, float]:
    """Parse a rake key=value pair list.

    Parameters
    ----------
    rake_value : str
        The input key=value list.

    Returns
    -------
    dict[str, float]
        The output rake dictionary.

    Examples
    --------
    >>> rake_parser('Acton=110')
    {"Acton": 110}
    """
    rake_key_values = rake_value.split(",")
    rakes = {}
    for kv_pair in rake_key_values:
        fault_name, fault_rake = kv_pair.split("=")
        rakes[fault_name.replace("_", " ")] = float(fault_rake)
    return rakes


@app.command(help="Generate a like rupture propagation for a given set of sources.")
def generate_rupture_propagation(
    realisation_ffp: Annotated[
        Path, typer.Argument(help="The path to the realisation.")
    ],
    initial_fault: Annotated[str, typer.Argument(help="The initial rupture fault.")],
    rakes: Annotated[
        dict[str, float],
        typer.Argument(
            help="Fault rakes in key-value list format (e.g. Acton=110,Nevis=-110.0). Use '_' instead of spaces.",
            parser=rake_parser,
        ),
    ],
    shypo: Annotated[
        Optional[float], typer.Option(help="Hypocentre s-coordinates", min=0, max=1)
    ] = None,
    dhypo: Annotated[
        Optional[float], typer.Option(help="Hypocentre d-coordinates", min=0, max=1)
    ] = None,
):
    """Generate a likely rupture propagation for a given set of sources.

    Parameters
    ----------
    realisation_ffp : Path
        The realisation with defined sources.
    initial_fault : str
        The initial fault.
    rakes : dict[str, float]
        A dictionary mapping fault names to fault rake values.
    """
    source_config = realisations.SourceConfig.read_from_realisation(realisation_ffp)
    faults = source_config.source_geometries

    if shypo is not None and dhypo is not None:
        expected_hypocentre = np.array([shypo, dhypo])
    else:
        expected_hypocentre = np.array(
            [1 / 2, distributions.truncated_weibull_expected_value(1)]
        )

    rupture_causality_tree = (
        rupture_propagation.estimate_most_likely_rupture_propagation(
            faults, initial_fault
        )
    )

    rupture_propagation_config = realisations.RupturePropagationConfig(
        magnitudes=default_magnitude_estimation(faults, rakes),
        rupture_causality_tree=rupture_causality_tree,
        jump_points=rupture_propagation.jump_points_from_rupture_tree(
            faults, rupture_causality_tree
        ),
        rakes=rakes,
        hypocentre=expected_hypocentre,
    )

    rupture_propagation_config.write_to_realisation(realisation_ffp)


if __name__ == "__main__":
    app()
