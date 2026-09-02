"""Generate Rupture Propagation.

Description
-----------
Generate a likely rupture propagation for a realisation.

Inputs
------

1. A realisation containing a source configuration.
2. An initial fault for the rupture to begin on.


Outputs
-------
A realisation file containing:

1. A rupture propagation plan (i.e. how the rupture jumps between faults, and where),
2. The estimated rupture magnitude and apportionment to the involved faults.


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

import random
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from qcore import cli
from qcore.uncertainties import distributions
from source_modelling import rupture_propagation
from workflow import realisations

app = typer.Typer()


class RuptureStrategy(StrEnum):
    """Rupture propagation strategy."""

    RANDOM = auto()
    MAXIMISING = auto()


@cli.from_docstring(app)
def generate_rupture_propagation(
    realisation_ffp: Annotated[Path, typer.Argument()],
    initial_fault: Annotated[str, typer.Argument()],
    shypo: Annotated[float | None, typer.Option(min=0, max=1)] = None,
    dhypo: Annotated[float | None, typer.Option(min=0, max=1)] = None,
    strategy: Annotated[
        RuptureStrategy,
        typer.Option(case_sensitive=False),
    ] = RuptureStrategy.RANDOM,
    min_connected_depth: Annotated[float, typer.Option(min=0)] = 5.0,
    jump_cutoff: Annotated[
        float,
        typer.Option(min=0),
    ] = 15,
) -> None:
    """Generate a likely rupture propagation for a given set of sources.

    Parameters
    ----------
    realisation_ffp : Path
        The path to the realisation.
    initial_fault : str
        The initial rupture fault.
    shypo : float, optional
        Hypocentre s-coordinates.
    dhypo : float, optional
        Hypocentre d-coordinates.
    strategy : RuptureStrategy
        The rupture propagation strategy to use. Default is `RuptureStrategy.RANDOM`.
    min_connected_depth : float, optional
        The depth to measure the fault distance. Defaults to 5km.
    jump_cutoff : float, optional
        The maximum jump distance between faults in km.
    """
    seeds = realisations.Seeds.read_from_realisation_or_random(realisation_ffp)
    source_config = realisations.SourceConfig.read_from_realisation(realisation_ffp)
    faults = source_config.source_geometries

    random.seed(seeds.rupture_propagation_seed)
    np.random.seed(random.randint(0, 2**32 - 1))
    if shypo is not None and dhypo is not None:
        hypocentre = np.array([shypo, dhypo])
    else:
        hypocentre = np.array(
            [
                distributions.truncated_normal(1 / 2, 1 / 4),
                distributions.truncated_weibull(1),
            ]
        )

    rupture_causality_tree = rupture_propagation.sample_rupture_propagation(
        faults,
        initial_source=initial_fault,
        strategy=strategy,  # type: ignore
        jump_impossibility_limit_distance=round(jump_cutoff * 1000),
    )

    rupture_propagation_config = realisations.RupturePropagationConfig(
        rupture_causality_tree=rupture_causality_tree,
        jump_points=rupture_propagation.jump_points_from_rupture_tree(
            faults, rupture_causality_tree, min_depth=min_connected_depth
        ),
        hypocentre=hypocentre,
    )

    rupture_propagation_config.write_to_realisation(realisation_ffp)
    realisations.append_log_entry(realisation_ffp)


if __name__ == "__main__":
    app()
