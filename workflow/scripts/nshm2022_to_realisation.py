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

import random
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import typer
from scipy.cluster.hierarchy import DisjointSet

from nshmdb import nshmdb
from nshmdb.nshmdb import FaultSystem
from qcore import cli
from qcore.uncertainties import distributions
from source_modelling import magnitude_scaling, moment, rupture_propagation, sources
from source_modelling.sources import Fault
from workflow import realisations
from workflow.defaults import DefaultsVersion
from workflow.log_utils import log_call
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RealisationMetadata,
    RupturePropagationConfig,
    Seeds,
    SourceConfig,
    SRFConfig,
)

app = typer.Typer()


class SamplingStrategy(StrEnum):
    """Sampling strategy to employ."""

    maximising = "maximising"
    random = "random"


def default_magnitude_estimation(
    faults: dict[str, Fault],
    # NOTE: this must be in quotes because the runtime class DisjointSet is
    # not generic, just the stub implementation.
    components: "DisjointSet[str]",
    avg_rake: float,
    strategy: SamplingStrategy,
) -> dict[str, magnitude_scaling.BoldM]:
    """Estimate the magnitudes for a set of faults based on their areas and average rake.

    Parameters
    ----------
    faults : dict
        A dictionary where the keys are fault names and the values are `Fault` objects containing information about each fault.
    components : DisjointSet
        A disjoint set representing the connected components of the faults.
        Each component corresponds to a group of faults that are connected
        and share a common rupture propagation path.
    avg_rake : float
        The average rake angle of the rupture.
    strategy : SamplingStrategy
        Sampling strategy for magnitude estimation. If maximising choose the
        mean leonard scaling relation magnitude, else sample from the magnitude
        distribution for the total rupture area.

    Returns
    -------
    dict
        A dictionary where the keys are fault names and the values are the
        estimated magnitudes (in the `BoldM` convention) for each fault.
    """
    total_area = sum(fault.area() for fault in faults.values())

    estimated_magnitude = magnitude_scaling.area_to_magnitude(
        magnitude_scaling.ScalingRelation.LEONARD2014,
        total_area,
        avg_rake,
        random=strategy == SamplingStrategy.random,
    )
    estimated_moment = moment.magnitude_to_moment(estimated_magnitude, bold_m=True)
    roots = {components[fault_name] for fault_name in faults}
    component_areas = {
        root: sum(faults[name].area() for name in components.subset(root))
        for root in roots
    }
    total_component_area = sum(
        component_area ** (3 / 2) for component_area in component_areas.values()
    )
    component_moments = {
        root: (component_area ** (3 / 2) / total_component_area) * estimated_moment
        for root, component_area in component_areas.items()
    }
    segment_moments = {}

    for fault_name, fault in faults.items():
        component = components[fault_name]
        component_area = component_areas[component]
        component_moment = component_moments[component]
        segment_moments[fault_name] = (fault.area() / component_area) * component_moment

    return {
        fault_name: moment.moment_to_magnitude(fault_moment, bold_m=True)
        for fault_name, fault_moment in segment_moments.items()
    }


def find_fault_and_hypocentre(
    faults: dict[str, Fault], lat_hypo: float, lon_hypo: float
) -> tuple[str, npt.NDArray[np.float64]]:
    """Find the fault and fault-local hypocentre coordinates corresponding to a hypocentre in lat, lon coordinates.

    Parameters
    ----------
    faults : dict[str, Fault]
        The faults in the rupture.
    lat_hypo : float
        The hypocentre latitude.
    lon_hypo : float
        The hypocentre longitude.

    Returns
    -------
    tuple[str, np.ndarray]
        The name of the fault and the hypocentre coordinates on the fault.

    Raises
    ------
    ValueError
        If the hypocentre is not on any fault.
    """
    hypocentre_wgs = np.array([lat_hypo, lon_hypo])
    for fault_name, fault in faults.items():
        try:
            hypocentre: npt.NDArray[np.float64] = (
                fault.wgs_depth_coordinates_to_fault_coordinates(hypocentre_wgs)
            )
            return fault_name, hypocentre
        except ValueError:
            continue
    raise ValueError("Hypocentre not on any fault.")


@cli.from_docstring(app)
@log_call()
def generate_realisation(
    nshmdb_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    rupture_id: Annotated[int, typer.Argument()],
    realisation_ffp: Annotated[
        Path,
        typer.Argument(writable=True),
    ],
    defaults_version: Annotated[
        DefaultsVersion,
        typer.Argument(),
    ],
    fault_system: Annotated[FaultSystem, typer.Option()] = FaultSystem.Crustal,
    initial_fault: Annotated[
        str | None,
        typer.Option(),
    ] = None,
    strategy: Annotated[
        SamplingStrategy,
        typer.Option(),
    ] = SamplingStrategy.random,
    magnitude_strategy: Annotated[
        SamplingStrategy, typer.Option()
    ] = SamplingStrategy.random,
    jump_cutoff: Annotated[
        float,
        typer.Option(min=0),
    ] = 15,
    shypo: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=1,
        ),
    ] = None,
    dhypo: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=1,
        ),
    ] = None,
    lat_hypo: Annotated[
        float | None,
        typer.Option(
            min=-90,
            max=90,
        ),
    ] = None,
    lon_hypo: Annotated[
        float | None,
        typer.Option(
            min=-180,
            max=180,
        ),
    ] = None,
    separation_distance: Annotated[float, typer.Option()] = 5.0,
    dip_delta: Annotated[float, typer.Option()] = 30.0,
    min_connected_depth: Annotated[float, typer.Option()] = 5.0,
) -> None:
    """Generate realisation stub files from ruptures in the NSHM 2022 database.

    This function initializes a connection to the NSHM database, retrieves
    the faults and fault information for the given rupture ID, estimates the
    most likely rupture propagation, and creates configurations and metadata
    for the realisation. The resulting realisation is then written to the
    specified file path.

    Parameters
    ----------
    nshmdb_path : Path
        Path to NSHMDB.
    rupture_id : int
        The ID of the rupture to generate the realisation stub for (find this using the NSHM Rupture Explorer).
    realisation_ffp : Path
        Location to write out the realisation.
    defaults_version : DefaultsVersion
        Scientific default parameters version to use.
    fault_system : FaultSystem, optional
        The NSHM fault system rupture_id belongs to. Defaults to Crustal.
        rupture_id is an NSHM id (nshmdb.nshmdb.FaultSystem), unique only
        within one fault system, not a raw database primary key.
    initial_fault : str, optional
        The name of the fault to use as the initial fault for rupture
        propagation. If not specified, the initial fault will be drawn
        proportionally to its likelihood of rupture.
    strategy : SamplingStrategy, optional
        The strategy to use when sampling rupture propagation. "maximising" will
        choose the maximally likely rupture propagation tree. "random" will
        choose a random rupture propagation tree.
    magnitude_strategy : SamplingStrategy, optional
        The strategy to use when sampling total rupture magnitude. "maximising"
        will choose the median rupture magnitude. Otherwise, sample from the magnitude
        distribution for the total rupture area.
    jump_cutoff : float, optional
        The maximum jump distance between faults in km.
    shypo : float, optional
        The initial hypocentre strike coordinate (0 - 1). If not supplied, draw
        shypo from a truncated normal distribution.
    dhypo : float, optional
        The initial hypocentre strike coordinate (0 - 1). If not supplied, draw
        dhypo from a weibull distribution.
    lat_hypo : float, optional
        The initial hypocentre latitude (degrees). Will cause an error
        if latitude and longitude are both not supplied, or specify a
        point not on the rupture geometry. Incompatible with shypo, dhypo and initial fault.
    lon_hypo : float, optional
        The initial hypocentre longitude (degrees). Will cause an error
        if latitude and longitude are both not supplied, or specify a
        point not on the rupture geometry. Incompatible with shypo, dhypo and initial fault.
    separation_distance : float, optional
        The maximum distance between faults to consider them connected in km.
        Defaults to 5 km.
    dip_delta : float, optional
        The maximum difference in dip angle between connected faults in degrees.
        Defaults to 30 degrees.
    min_connected_depth : float, optional
        The depth to measure the fault distance. Defaults to 5km.
    """
    # Check a compatible combination of hypocentre parameters is supplied.
    if (lat_hypo is None) != (lon_hypo is None):
        print("Both latitude and longitude must be supplied.")
        raise typer.Exit(code=1)
    if lat_hypo is not None and (
        dhypo is not None or shypo is not None or initial_fault
    ):
        print("Latitude and longitude are incompatible with shypo and dhypo.")
        raise typer.Exit(code=1)

    metadata = RealisationMetadata(
        name=f"Rupture {rupture_id}",
        version="1",
        tag="nshm",
        defaults_version=defaults_version,
    )

    metadata.write_to_realisation(realisation_ffp)
    srf_config = SRFConfig.read_from_realisation_or_defaults(
        realisation_ffp, defaults_version
    )
    db = nshmdb.NSHMDB(nshmdb_path)
    db.connect()
    faults = db.get_rupture_faults(fault_system, rupture_id)
    faults = {
        fault_name: sources.simplify_fault(fault, srf_config.resolution)
        for fault_name, fault in faults.items()
    }

    if initial_fault and initial_fault not in faults:
        print(
            f"Initial fault '{initial_fault}' not found in rupture. Options are {', '.join(list(faults))}"
        )
        raise typer.Exit(code=1)
    faults_info = db.get_rupture_fault_info(fault_system, rupture_id)
    seeds = Seeds.read_from_realisation_or_random(realisation_ffp)
    np.random.seed(seed=seeds.nshm_to_realisation_seed)
    random.seed(seeds.nshm_to_realisation_seed)
    source_config = SourceConfig(faults)

    rakes = {
        fault_name: fault_info.rake for fault_name, fault_info in faults_info.items()
    }
    avg_rake = np.mean(list(rakes.values()))
    components = moment.find_connected_faults(
        faults, separation_distance, dip_delta, min_connected_depth
    )
    magnitudes = default_magnitude_estimation(
        faults, components, float(avg_rake), magnitude_strategy
    )
    if lat_hypo is not None and lon_hypo is not None:
        initial_fault, hypocentre = find_fault_and_hypocentre(
            faults, lat_hypo, lon_hypo
        )
    elif initial_fault:
        hypocentre = np.array(
            [
                shypo
                if shypo is not None
                else distributions.truncated_normal(1 / 2, 1 / 4),
                dhypo if dhypo is not None else distributions.truncated_weibull(1),
            ]
        )
    else:
        # The ty ignore below can be removed once NSHM2022DB merges the change to
        # accept dict[str, BoldM] in most_likely_fault (branch support-BoldM-in-workflow).
        mfds_rates = db.most_likely_fault(fault_system, rupture_id, magnitudes)  # ty: ignore[invalid-argument-type]
        mfds_probabilities = np.array(list(mfds_rates.values()))
        if np.allclose(mfds_probabilities, 0):
            mfds_probabilities = np.ones_like(mfds_probabilities)
        mfds_probabilities /= mfds_probabilities.sum()
        initial_fault = np.random.choice(list(mfds_rates), p=mfds_probabilities)
        hypocentre = np.array(
            [
                shypo
                if shypo is not None
                else distributions.truncated_normal(1 / 2, 1 / 4),
                dhypo if dhypo is not None else distributions.truncated_weibull(1),
            ]
        )

    rupture_causality_tree = rupture_propagation.sample_rupture_propagation(
        faults,
        initial_source=initial_fault,
        strategy=str(strategy),  # type: ignore
        jump_impossibility_limit_distance=round(jump_cutoff * 1000),
    )
    magnitudes = Magnitudes(magnitudes)
    rakes = Rakes(rakes)
    rupture_propagation_config = RupturePropagationConfig(
        rupture_causality_tree=rupture_causality_tree,
        jump_points=rupture_propagation.jump_points_from_rupture_tree(
            faults, rupture_causality_tree, min_depth=min_connected_depth
        ),
        hypocentre=hypocentre,
    )
    realisation_ffp.parent.mkdir(parents=True, exist_ok=True)
    for section in [source_config, rupture_propagation_config, magnitudes, rakes]:
        section.write_to_realisation(realisation_ffp)
    realisations.append_log_entry(realisation_ffp)
