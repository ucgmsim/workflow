#!/usr/bin/env python3
"""NSHM To Realisation.

Description
-----------
Construct a realisation from a rupture in the [NSHM 2022](https://nshm.gns.cri.nz/RuptureMap).

Inputs
------

1. A copy of the [NSHM 2022 database](https://www.dropbox.com/scl/fi/50kww45wpsnmtf3pn2okz/nshmdb.db?rlkey=4mjuomuevl1x963fjwfximgldm&st=50ax73gl&dl=0).
2. A rupture id to simulate. You can find a rupture id from the [rupture explorer](https://nshm.gns.cri.nz/RuptureMap). Alternatively, you can use the visualisation tools to find one.
3. The version of the [scientific defaults](https://github.com/ucgmsim/workflow/blob/pegasus/workflow/default_parameters/README.md#L1) to use. If you don't know what version to use, choose the latest version. Versions are specified as `YY.M.D.R`, where `R` is the resolution of the simulation (1 = 100m). For example `24.2.2.1`. The special `develop` version is for testing workflow iterations and not to be used for accurate scientific simulation.

Outputs
-------
A realisation file containing:

1. The definition of all the faults in the the rupture,
2. A rupture propagation plan (i.e. how the rupture jumps between faults, and where),
3. The estimated rupture magnitude and apportionment to the involved faults.
4. The definition of the rakes.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `nshm2022-to-realisation` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`nshm2022-to-realisation [OPTIONS] NSHM_DB_FILE RUPTURE_ID REALISATION_FFP DEFAULTS_VERSION`

For More Help
-------------
See the output of `nshm2022-to-realisation --help`.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from nshmdb import nshmdb
from qcore.uncertainties import distributions, mag_scaling
from source_modelling import rupture_propagation
from source_modelling.sources import Fault
from workflow.defaults import DefaultsVersion
from workflow.log_utils import log_call
from workflow.realisations import (
    RealisationMetadata,
    RupturePropagationConfig,
    Seeds,
    SourceConfig,
)

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


class SamplingStrategy(StrEnum):
    """Rupture propagation strategy to employ."""

    maximising = "maximising"
    random = "random"


@app.command(
    help="Generate realisation stub files from ruptures in the NSHM 2022 database."
)
@log_call()
def generate_realisation(
    nshmdb_path: Annotated[
        Path, typer.Argument(help="Path to NSHMDB.", exists=True, dir_okay=False)
    ],
    rupture_id: Annotated[
        int,
        typer.Argument(
            help="The ID of the rupture to generate the realisation stub for (find this using the NSHM Rupture Explorer)."
        ),
    ],
    realisation_ffp: Annotated[
        Path,
        typer.Argument(help="Location to write out the realisation.", writable=True),
    ],
    defaults_version: Annotated[
        DefaultsVersion,
        typer.Argument(help="Scientific default parameters version to use"),
    ],
    initial_fault: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the fault to use as the initial fault for rupture propagation."
            " If not specified, the initial fault will be drawn proportionally to its likelihood of rupture.",
        ),
    ] = None,
    strategy: Annotated[
        SamplingStrategy,
        typer.Option(
            help="The strategy to use when sampling rupture propagation."
            ' "maximising" will choose the maximally likely rupture propagation tree.'
            ' "random" will choose a random rupture propagation tree.',
        ),
    ] = SamplingStrategy.random,
    jump_cutoff: Annotated[
        float,
        typer.Option(help="The maximum jump distance between faults in km.", min=0),
    ] = 15,
    shypo: Annotated[
        Optional[float],
        typer.Option(
            help="The initial hypocentre strike coordinate (0 - 1)."
            " If not supplied, draw shypo from a truncated normal distribution.",
            min=0,
            max=1,
        ),
    ] = None,
    dhypo: Annotated[
        Optional[float],
        typer.Option(
            help="The initial hypocentre strike coordinate (0 - 1)."
            " If not supplied, draw dhypo from a weibull distribution.",
            min=0,
            max=1,
        ),
    ] = None,
):
    """Generate realisation stub files from ruptures in the NSHM 2022 database.

    This function initializes a connection to the NSHM database, retrieves
    the faults and fault information for the given rupture ID, estimates the
    most likely rupture propagation, and creates configurations and metadata
    for the realisation. The resulting realisation is then written to the
    specified file path.

    Parameters
    ----------
    nshm_db_file : Path
        The NSHM sqlite database containing rupture information and fault geometry.
    rupture_id : int
        The ID of the rupture to generate the realisation stub for. Find
        this using the NSHM Rupture Explorer.
    realisation_ffp : Path
        Location to write out the realisation.
    defaults_version : DefaultsVersion
        Scientific default parameters version to use.
    """

    metadata = RealisationMetadata(
        name=f"Rupture {rupture_id}",
        version="1",
        tag="nshm",
        defaults_version=defaults_version,
    )
    metadata.write_to_realisation(realisation_ffp)
    db = nshmdb.NSHMDB(nshmdb_path)

    faults = db.get_rupture_faults(rupture_id)
    faults_info = db.get_rupture_fault_info(rupture_id)
    seeds = Seeds.read_from_realisation_or_defaults(realisation_ffp)
    np.random.seed(seed=seeds.nshm_to_realisation_seed)
    source_config = SourceConfig(faults)

    rakes = {
        fault_name: fault_info.rake for fault_name, fault_info in faults_info.items()
    }
    magnitudes = default_magnitude_estimation(faults, rakes)
    if not initial_fault:
        mfds_rates = db.most_likely_fault(rupture_id, magnitudes)
        mfds_probabilities = np.array(list(mfds_rates.values()))
        if np.allclose(mfds_probabilities, 0):
            mfds_probabilities = np.ones_like(mfds_probabilities)
        mfds_probabilities /= mfds_probabilities.sum()
        initial_fault = np.random.choice(list(mfds_rates), p=mfds_probabilities)
    elif initial_fault not in faults:
        print(
            f"Initial fault '{initial_fault}' not found in rupture. Options are {', '.join(list(faults))}"
        )
        raise typer.Exit(code=1)

    rupture_causality_tree = rupture_propagation.sample_rupture_propagation(
        faults,
        initial_source=initial_fault,
        strategy=strategy,
        jump_impossibility_limit_distance=jump_cutoff * 1000,
    )

    hypocentre = np.array(
        [
            shypo or distributions.truncated_normal(1 / 2, 1 / 4),
            dhypo or distributions.truncated_weibull(1),
        ]
    )
    rupture_propagation_config = RupturePropagationConfig(
        magnitudes=magnitudes,
        rupture_causality_tree=rupture_causality_tree,
        jump_points=rupture_propagation.jump_points_from_rupture_tree(
            faults, rupture_causality_tree
        ),
        rakes=rakes,
        hypocentre=hypocentre,
    )
    realisation_ffp.parent.mkdir(parents=True, exist_ok=True)
    for section in [source_config, rupture_propagation_config]:
        section.write_to_realisation(realisation_ffp)
