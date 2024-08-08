#!/usr/bin/env python3
"""
Generate realisation stub files from ruptures in the NSHM 2022 database.

This script generates YAML realisation stub files from ruptures in the NSHM 2022
database. It extracts fault geometry, computes causality information from the database
and incorporates default parameter values to generate the realisation.
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


def a_to_mw_leonard(area: float, rake: float) -> float:
    return mag_scaling.a_to_mw_leonard(area, 4, 3.99, rake)


def default_magnitude_estimation(
    faults: dict[str, Fault], rakes: dict[str, float]
) -> dict[str, float]:
    total_area = sum(fault.area() for fault in faults.values())
    avg_rake = np.mean(list(rakes.values()))
    estimated_mw = a_to_mw_leonard(total_area, avg_rake)
    estimated_moment = mag_scaling.mag2mom(estimated_mw)
    print(estimated_moment)
    return {
        fault_name: mag_scaling.mom2mag((fault.area() / total_area) * estimated_moment)
        for fault_name, fault in faults.items()
    }


def expected_hypocentre() -> np.ndarray:
    return np.array([1 / 2, distributions.truncated_weibull_expected_value(1)])


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
    dt: Annotated[
        float, typer.Option(help="Time resolution for source modelling.", min=0)
    ] = 0.05,
    genslip_seed: Annotated[
        int,
        typer.Option(
            help="Seed for genslip, used to initialise slip distribution on fault."
        ),
    ] = 1,
    srfgen_seed: Annotated[
        int,
        typer.Option(
            help="Seed for srfgen, used to initialise slip distribution on fault."
        ),
    ] = 1,
):
    "Generate realisation stub files from ruptures in the NSHM 2022 database."
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

    srf_config = realisations.SRFConfig(
        genslip_seed=genslip_seed,
        genslip_dt=dt,
        srfgen_seed=srfgen_seed,
        genslip_version="5.4.2",
    )
    rakes = {
        fault_name: fault_info.rake for fault_name, fault_info in faults_info.items()
    }
    rupture_propagation_config = realisations.RupturePropagationConfig(
        magnitudes=default_magnitude_estimation(faults, rakes),
        rupture_causality_tree=rupture_causality_tree,
        jump_points=rupture_propagation.jump_points_from_rupture_tree(
            faults, rupture_causality_tree
        ),
        rakes=rakes,
        hypocentre=expected_hypocentre(),
    )
    metadata = realisations.RealisationMetadata(
        name=f"Rupture {rupture_id}", version="1", tag="nshm"
    )

    for section in [metadata, source_config, srf_config, rupture_propagation_config]:
        realisations.write_config_to_realisation(section, realisation_ffp)


def main():
    typer.run(generate_realisation)


if __name__ == "__main__":
    main()
