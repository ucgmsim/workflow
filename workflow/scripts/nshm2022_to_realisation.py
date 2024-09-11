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

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from nshmdb import nshmdb
from qcore.uncertainties import distributions, mag_scaling
from source_modelling import rupture_propagation
from source_modelling.sources import Fault
from workflow import realisations
from workflow.defaults import DefaultsVersion

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


@app.command(
    help="Generate realisation stub files from ruptures in the NSHM 2022 database."
)
def generate_realisation(
    nshm_db_file: Annotated[
        Path,
        typer.Argument(
            help="The NSHM sqlite database containing rupture information and fault geometry.",
            readable=True,
            exists=True,
        ),
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
):
    """Generate realisation stub files from ruptures in the NSHM 2022 database.

    This function initializes a connection to the NSHM database, retrieves the faults and fault information for
    the given rupture ID, estimates the most likely rupture propagation, and creates configurations and metadata
    for the realisation. The resulting realisation is then written to the specified file path.

    Parameters
    ----------
    nshm_db_file : Path
        The NSHM sqlite database containing rupture information and fault geometry.
    rupture_id : int
        The ID of the rupture to generate the realisation stub for. Find this using the NSHM Rupture Explorer.
    realisation_ffp : Path
        Location to write out the realisation.
    defaults_version : DefaultsVersion
        Scientific default parameters version to use.
    """
    db = nshmdb.NSHMDB(nshm_db_file)
    faults = db.get_rupture_faults(rupture_id)
    faults_info = db.get_rupture_fault_info(rupture_id)

    initial_fault = list(faults)[0]

    rupture_causality_tree = (
        rupture_propagation.estimate_most_likely_rupture_propagation(
            faults, initial_fault
        )
    )

    source_config = realisations.SourceConfig(faults)

    rakes = {
        fault_name: fault_info.rake for fault_name, fault_info in faults_info.items()
    }
    expected_hypocentre = np.array(
        [1 / 2, distributions.truncated_weibull_expected_value(1)]
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
    metadata = realisations.RealisationMetadata(
        name=f"Rupture {rupture_id}",
        version="1",
        tag="nshm",
        defaults_version=defaults_version,
    )

    for section in [metadata, source_config, rupture_propagation_config]:
        section.write_to_realisation(realisation_ffp)
