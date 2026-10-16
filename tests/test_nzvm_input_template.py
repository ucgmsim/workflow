"""Tests for `workflow.scripts.nzvm_input_template`.

These cover how the realisation chooses the grid's terrain decay: the SW4 grid
takes the `nzcvm` section's decay as is, and the EMOD3D grid maps the
velocity model's `topo_type` to the matching NZCVM decay.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nzcvm.config.grids.terrain import Decay, SleveDecay

from workflow import defaults
from workflow.bounding_box import BoundingBox
from workflow.realisations import (
    DomainParameters,
    NZCVMSettings,
    RealisationMetadata,
    VelocityModelParameters,
)
from workflow.scripts import nzvm_input_template
from workflow.scripts.nzvm_input_template import GridFormat

DEFAULTS_VERSIONS = {
    GridFormat.SW4: defaults.DefaultsVersion.v26_7_1Hz,
    GridFormat.EMOD3D: defaults.DefaultsVersion.v24_2_2_4,
}
"""A defaults version for each grid: only the EMOD3D ones carry `resolution`."""


def generate(
    tmp_path: Path,
    format: GridFormat,
    decay: Decay | None = None,
    topo_type: str | None = None,
) -> dict[str, Any]:
    """Run `generate_template` and return the grid it writes.

    Parameters
    ----------
    tmp_path : Path
        Scratch directory for the realisation and the configuration.
    format : GridFormat
        The grid to write.
    decay : Decay | None
        The realisation's NZCVM terrain decay.
    topo_type : str | None
        The velocity model's topography type, or None for the default.

    Returns
    -------
    dict[str, Any]
        The `grid` section of the written configuration.
    """
    defaults_version = DEFAULTS_VERSIONS[format]
    realisation = tmp_path / "realisation.json"
    RealisationMetadata(
        name="test", version="1", defaults_version=defaults_version
    ).write_to_realisation(realisation)
    domain = BoundingBox.from_centroid_bearing_extents(
        centroid=np.array([-43.5, 172.5]), bearing=35.0, extent_x=20.0, extent_y=20.0
    )
    DomainParameters(domain=domain, depth=30.0, duration=10.0).write_to_realisation(
        realisation
    )
    settings = NZCVMSettings.read_from_defaults(defaults_version)
    dataclasses.replace(settings, decay=decay).write_to_realisation(realisation)
    if topo_type is not None:
        velocity_model = VelocityModelParameters.read_from_defaults(defaults_version)
        dataclasses.replace(velocity_model, topo_type=topo_type).write_to_realisation(
            realisation
        )

    output = tmp_path / "config.json"
    nzvm_input_template.generate_template(realisation, output, format=format)
    return json.loads(output.read_text())["grid"]


def test_sw4_grid_keeps_topography_without_a_decay(tmp_path: Path) -> None:
    grid = generate(tmp_path, GridFormat.SW4)

    assert grid.get("decay") is None


def test_sw4_grid_takes_the_realisation_decay(tmp_path: Path) -> None:
    grid = generate(tmp_path, GridFormat.SW4, decay=SleveDecay(scale=1000.0))

    assert grid["decay"] == {"type": "sleve", "scale": 1000.0}


@pytest.mark.parametrize(
    ("topo_type", "decay"),
    [
        ("SQUASHED", {"type": "squashed"}),
        ("SQUASHED_TAPERED", {"type": "tapered", "ratio": 1.0}),
    ],
)
def test_emod3d_grid_maps_the_topography_type(
    tmp_path: Path, topo_type: str, decay: dict[str, Any]
) -> None:
    grid = generate(tmp_path, GridFormat.EMOD3D, topo_type=topo_type)

    assert grid["decay"] == decay


def test_emod3d_grid_rejects_a_topography_type_nzcvm_lacks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BULLDOZED"):
        generate(tmp_path, GridFormat.EMOD3D, topo_type="BULLDOZED")
