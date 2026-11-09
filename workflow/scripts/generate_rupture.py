"""Rupture generation in the NZCVM velocity model.

Description
-----------
Draw a rupture with the rupture generator, reading shear-wave speed and density from
the realisation's NZCVM velocity model rather than the rupture file's 1D `[medium]`.

    create-rupture-input realisation.json faults.geojson rupture.toml
    generate-rupture realisation.json rupture.toml realisation.srf

The velocity model is the realisation's `nzcvm` layers less Backus averaging, which
averages over a simulation grid's cells and has no meaning at a point on a fault. It is
read at every position the generator asks about: the subfaults, and the straight lines
the shear waves cross between faults.

Inputs
------
1. A realisation with an `nzcvm` section.
2. The rupture file `create-rupture-input` wrote from it.

Outputs
-------
1. The SRF, its header comments carrying the run record.

Environment
-----------
Can be run in the cybershake container, which holds NZCVM's data root.

Usage
-----
`generate-rupture [OPTIONS] REALISATION_FFP RUPTURE_FFP SRF_FFP`

For More Help
-------------
See the output of `generate-rupture --help`.
"""

from pathlib import Path

import numpy as np
import shapely
import typer
from nzcvm.config.layers.backus import BackusAveragedLayerConfig

from qcore import cli
from rupture_generator import Medium, generate, write_rupture
from rupture_generator import config as rupture_config
from rupture_generator.record import run_record
from workflow import log_utils, realisations, velocity_model
from workflow.realisations import NZCVMSettings, RealisationMetadata

app = typer.Typer()

GEOMETRY_BUFFER_M = 1000.0
"""How far beyond the faults' outline the velocity model loads its sub-models."""


def nzcvm_medium(query: velocity_model.PointQuery) -> Medium:
    """The rock the query describes, in the units the generator reads.

    The generator reads shear speed and density at the same positions in turn, so the
    last positions' qualities are kept and the second read is free.

    Parameters
    ----------
    query : PointQuery
        The velocity model, in NZTM metres.

    Returns
    -------
    Medium
        Shear speed (km/s) and density (g/cm^3) at chart positions (NZTM km).
    """
    key: tuple | None = None
    qualities: dict[str, np.ndarray] = {}

    def read(positions_km: np.ndarray, component: str) -> np.ndarray:
        nonlocal key, qualities
        if key != (positions_km.shape, positions_km.tobytes()):
            key = (positions_km.shape, positions_km.tobytes())
            east, north, depth = np.moveaxis(positions_km * 1000.0, -1, 0)
            qualities = query(east, north, depth)
        # NZCVM's m/s and kg/m^3 are a thousand km/s and g/cm^3.
        return qualities[component] / 1000.0

    return Medium(
        shear_speed_km_s=lambda positions_km: read(positions_km, "vs"),
        density_g_cm3=lambda positions_km: read(positions_km, "rho"),
    )


@cli.from_docstring(app)
def generate_rupture(realisation_ffp: Path, rupture_ffp: Path, srf_ffp: Path) -> None:
    """Draw a rupture in the realisation's NZCVM velocity model.

    Parameters
    ----------
    realisation_ffp : Path
        Path to the realisation file, which supplies the velocity model.
    rupture_ffp : Path
        Path to the rupture file `create-rupture-input` wrote.
    srf_ffp : Path
        Path to write the SRF.
    """
    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    settings = NZCVMSettings.read_from_realisation_or_defaults(
        realisation_ffp, metadata.defaults_version
    )
    config = rupture_config.load(rupture_ffp)
    scenario = rupture_config.build(config)
    nodes_m = (
        np.concatenate(
            [
                chart.nodes[..., :2].reshape(-1, 2)
                for chart in scenario.realisation.values()
            ]
        )
        * 1000.0
    )
    # The `per_fault` source `create-rupture-input` writes takes no medium, so only
    # the draw itself reads the rock and swapping it here is the whole change.
    medium = nzcvm_medium(
        velocity_model.point_query(
            settings,
            shapely.MultiPoint(nodes_m).convex_hull.buffer(GEOMETRY_BUFFER_M),
            exclude=(BackusAveragedLayerConfig,),
        )
    )
    drawn = generate(
        scenario.realisation,
        medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )
    write_rupture(
        str(srf_ffp),
        drawn,
        comments=[
            *run_record(rupture_ffp, config).splitlines(),
            "medium = NZCVM (realisation nzcvm layers, less backus), not [medium]",
        ],
    )
    log_utils.get_logger(__name__).info("Wrote SRF", srf=str(srf_ffp))
    realisations.append_log_entry(realisation_ffp)
