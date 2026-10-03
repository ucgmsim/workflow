from pathlib import Path

import pandas as pd
import pytest

from source_modelling import sources
from workflow.defaults import DefaultsVersion
from workflow.realisations import EmpiricalParameters, Rakes, SourceConfig
from workflow.scripts import gcmt_to_realisation

SOLUTIONS = pd.DataFrame(
    {
        "PublicID": ["2103645", "2016p858000", "2017p795065"],
        "Latitude": [-45.1929, -42.6925, -42.3581],
        "Longitude": [166.83, 173.0221, 173.4589],
        "CD": [22.0, 16.0, 11.0],
        "Mo": [5.61e26, 7.04e27, 4.29e23],
        "strike1": [213.0, 219.0, 347.0],
        "dip1": [56.0, 38.0, 76.0],
        "rake1": [98.0, 128.0, 33.0],
        "strike2": [20.0, 354.0, 248.0],
        "dip2": [35.0, 61.0, 58.0],
        "rake2": [79.0, 64.0, 163.0],
    }
)


@pytest.mark.parametrize(
    "event_id, tect_type",
    [
        ("2103645", "subduction_interface"),  # 2003 Fiordland
        ("2016p858000", "active_shallow"),  # 2016 Kaikoura
        ("2017p795065", "subduction_slab"),
    ],
)
def test_gcmt_to_realisation_tectonic_type(
    tmp_path: Path, event_id: str, tect_type: str
) -> None:
    solutions_ffp = tmp_path / "solutions.csv"
    SOLUTIONS.to_csv(solutions_ffp, index=False)
    realisation_ffp = tmp_path / "realisation.json"

    gcmt_to_realisation.gcmt_to_realisation(
        event_id,
        DefaultsVersion.v24_2_2_1,
        realisation_ffp,
        gcmt_to_realisation.SourceType.FINITE_FAULT,
        solution_origin=solutions_ffp,
    )

    empirical = EmpiricalParameters.read_from_realisation(realisation_ffp)
    assert empirical.tect_type == tect_type
    defaults = EmpiricalParameters.read_from_defaults(DefaultsVersion.v24_2_2_1)
    assert empirical.models == defaults.models


def test_gcmt_to_realisation_most_likely_nodal_plane(tmp_path: Path) -> None:
    solutions_ffp = tmp_path / "solutions.csv"
    SOLUTIONS.to_csv(solutions_ffp, index=False)
    realisation_ffp = tmp_path / "realisation.json"

    gcmt_to_realisation.gcmt_to_realisation(
        "2103645",
        DefaultsVersion.v24_2_2_1,
        realisation_ffp,
        gcmt_to_realisation.SourceType.FINITE_FAULT,
        solution_origin=solutions_ffp,
    )

    # The 2003 Fiordland earthquake ruptured the shallow-dipping interface.
    fault = SourceConfig.read_from_realisation(realisation_ffp).source_geometries[
        "2103645"
    ]
    assert isinstance(fault, sources.Fault)
    (plane,) = fault.planes
    assert plane.strike == pytest.approx(20.0, abs=0.5)
    assert plane.dip == pytest.approx(35.0, abs=0.5)
    assert Rakes.read_from_realisation(realisation_ffp).rakes["2103645"] == 79.0
