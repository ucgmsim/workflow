import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from workflow.realisations import RupturePropagationConfig
from workflow.scripts import generate_rupture_propagation

RUPTURE_PROP_TESTS = Path("tests") / "rupture_prop"


@pytest.mark.parametrize(
    "rupture_id,initial_fault,rakes",
    [
        (0, "Acton", {"Acton": 110.0}),
        (1, "Acton", {"Acton": 110.0, "Nevis": 110.0}),
        (599, "Akatarawa", {"Akatarawa": 160.0, "Wellington Hutt Valley: 4": -160.0}),
        (
            1190,
            "Alpine: Caswell",
            {
                "Alpine: Caswell": 180.0,
                "Alpine: Caswell - South George": 180.0,
                "Alpine: George landward": 180.0,
                "Alpine: George to Jacksons": 180.0,
                "Alpine: Jacksons to Kaniere": 160.0,
                "Alpine: Kaniere to Springs Junction": 135.0,
                "Alpine: Springs Junction to Tophouse": 160.0,
                "Browning Pass": 180.0,
            },
        ),
    ],
)
def test_generate_rupture_propagation(
    tmp_path: Path, rupture_id: int, initial_fault: str, rakes: dict[str, float]
):
    shutil.copytree(RUPTURE_PROP_TESTS / str(rupture_id), tmp_path, dirs_exist_ok=True)
    generate_rupture_propagation.generate_rupture_propagation(
        tmp_path / "realisation.json", initial_fault, rakes
    )

    output_rup_prop = RupturePropagationConfig.read_from_realisation(
        tmp_path / "realisation.json"
    )
    expected_rup_prop = RupturePropagationConfig.read_from_realisation(
        tmp_path / "expected.json"
    )
    assert np.allclose(output_rup_prop.hypocentre, expected_rup_prop.hypocentre)
    assert len(output_rup_prop.jump_points) == len(expected_rup_prop.jump_points)
    for fault in output_rup_prop.jump_points:
        expected_jump_point = expected_rup_prop.jump_points[fault]
        output_jump_point = output_rup_prop.jump_points[fault]
        assert np.allclose(
            output_jump_point.from_point, expected_jump_point.from_point, atol=1e-2
        )
        assert np.allclose(
            output_jump_point.to_point, expected_jump_point.to_point, atol=1e-2
        )

    for fault in output_rup_prop.magnitudes:
        assert np.isclose(
            output_rup_prop.magnitudes[fault],
            expected_rup_prop.magnitudes[fault],
            atol=1e-6,
        )


@pytest.mark.parametrize(
    "rakes", [{"Acton": 110.0}, {"Fault A": 120.0, "Fault B": 60.0}]
)
def test_rake_parser(rakes: dict[str, float]):
    rake_format = ",".join(
        f"{key.replace(' ', '_')}={value}" for key, value in rakes.items()
    )
    assert generate_rupture_propagation.rake_parser(rake_format) == rakes
