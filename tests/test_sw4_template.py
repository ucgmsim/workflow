"""Tests for `workflow.scripts.sw4_template`.

The interesting behaviour here is geometric: the requested domain has to end up
as the *interior* of the SW4 grid, and the bottom refinement has to stay thick
enough to hold the bottom sponge. Both are invariants rather than values, so
they are tested as invariants.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
from nzcvm.formats import sfile

from velocity_modelling.bounding_box import BoundingBox
from workflow import defaults, sw4
from workflow.realisations import DomainParameters, RealisationMetadata, Refinements
from workflow.scripts import sw4_template

SPONGE_KM = 12.0
"""The v26_7_1Hz sponge width, in kilometres."""


@pytest.fixture
def domain() -> BoundingBox:
    """A rotated 20 x 20 km domain, small enough to keep the test sfile cheap.

    Returns
    -------
    BoundingBox
        The domain.
    """
    return BoundingBox.from_centroid_bearing_extents(
        centroid=np.array([-43.5, 172.5]),
        bearing=35.0,
        extent_x=20.0,
        extent_y=20.0,
    )


def write_sfile(
    path: Path,
    shape: tuple[int, int],
    resolution: float,
    topography_height: float = 500.0,
    zmax: float = 1_000_000.0,
) -> None:
    """Write a minimal sfile carrying only what `generate_sw4_input` reads.

    Parameters
    ----------
    path : Path
        Where to write the file.
    shape : tuple[int, int]
        The (north, east) gridpoint counts of the coarsest grid.
    resolution : float
        The coarsest grid's horizontal spacing, in metres.
    topography_height : float
        The highest topography in the model, in metres above sea level.
    zmax : float
        The depth of the bottom of the model, in metres.
    """
    with h5py.File(path, "w") as f:
        f.attrs[sfile.ORIGIN_AZIM_ATTR] = np.array([172.5, -43.5, 35.0])
        f.attrs[sfile.MIN_MAX_DEPTH_ATTR] = np.array([-topography_height, zmax])
        material = f.create_group(sfile.MATERIAL_GROUP)
        # A finer grid over the same footprint, to check the reader picks the
        # coarsest and still gets the same answer.
        for index, factor in enumerate((2, 1)):
            grid = material.create_group(f"grid_{index}")
            grid.attrs[sfile.HORIZONTAL_ATTR] = resolution / factor
            grid.attrs[sfile.NUMBER_OF_COMPONENTS_ATTR] = np.int32(1)
            grid.create_dataset(
                "Cs",
                data=np.zeros(
                    (
                        (shape[0] - 1) * factor + 1,
                        (shape[1] - 1) * factor + 1,
                        3,
                    ),
                    dtype=np.float32,
                ),
            )


def render(
    tmp_path: Path,
    domain: BoundingBox,
    depth_km: float,
    sfile_shape: tuple[int, int] = (121, 121),
    topography_height: float = 500.0,
) -> dict[str, list[dict[str, str]]]:
    """Run `generate_sw4_input` and parse the SW4 file it writes.

    Parameters
    ----------
    tmp_path : Path
        Scratch directory for the realisation, sfile and output.
    domain : BoundingBox
        The requested simulation domain.
    depth_km : float
        The requested simulation depth, in kilometres.
    sfile_shape : tuple[int, int]
        The (north, east) gridpoint counts of the velocity model at 400 m.
    topography_height : float
        The highest topography in the velocity model, in metres.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        Each command's parameters, keyed by command name in file order.
    """
    realisation = tmp_path / "realisation.json"
    RealisationMetadata(
        name="test", version="1", defaults_version=defaults.DefaultsVersion.v26_7_1Hz
    ).write_to_realisation(realisation)
    DomainParameters(domain=domain, depth=depth_km, duration=10.0).write_to_realisation(
        realisation
    )
    velocity_model = tmp_path / "model.sfile"
    write_sfile(velocity_model, sfile_shape, 400.0, topography_height=topography_height)
    output = tmp_path / "sw4.in"
    sw4_template.generate_sw4_input(
        realisation,
        tmp_path / "stations.h5",
        tmp_path / "source.srf",
        velocity_model,
        tmp_path,
        output,
    )

    commands: dict[str, list[dict[str, str]]] = {}
    for line in output.read_text().splitlines():
        if not line.strip():
            continue
        name, *parameters = line.split()
        commands.setdefault(name, []).append(
            dict(parameter.split("=", 1) for parameter in parameters)
        )
    return commands


@pytest.mark.parametrize("depth_km", [10.0, 30.0, 60.0, 120.0, 350.0])
def test_bottom_refinement_holds_the_sponge(
    tmp_path: Path, domain: BoundingBox, depth_km: float
) -> None:
    """SW4's `check_supergrid_thickness` requires `nz[0] >` the sponge thickness.

    Only grid 0 carries a bottom taper, so the requirement is on the bottom
    refinement alone: the layer between the last `refinement zmax` and the
    grid's `z` must be strictly thicker than the sponge, so that at least one
    cell of it is not sponge.
    """
    commands = render(tmp_path, domain, depth_km)
    (grid,) = commands["grid"]
    bottom_top = float(commands["refinement"][-1]["zmax"])

    assert float(grid["z"]) - bottom_top > SPONGE_KM * 1000.0
    # The requested depth is interior and the sponge sits directly below it.
    assert float(grid["z"]) == pytest.approx((depth_km + SPONGE_KM) * 1000.0)


@pytest.mark.parametrize("depth_km", [10.0, 30.0, 60.0, 120.0, 350.0])
def test_grid_spacing_is_the_theoretical_coarsest(
    tmp_path: Path, domain: BoundingBox, depth_km: float
) -> None:
    """The grid spacing can be read from the theoretical refinements.

    `generate_sw4_input` reads the sponge width from the adjusted refinements
    while `create-nzvm-input` reads it from the theoretical ones. That only
    agrees because topography adjustment moves bottoms, never resolutions.
    """
    theoretical = Refinements.read_from_defaults(defaults.DefaultsVersion.v26_7_1Hz)

    (grid,) = render(tmp_path, domain, depth_km)["grid"]

    assert float(grid["h"]) == sw4.coarsest_resolution(theoretical, depth_km)


def test_topography_deepens_a_thin_implicit_layer(
    tmp_path: Path, domain: BoundingBox
) -> None:
    """The worked example in `_adjust_for_topography`.

    1800 m of topography puts the curvilinear bottom at 5400 m, 400 m (two
    200 m cells) below the 5000 m refinement. It is pushed down to give that
    implicit layer 12 cells, while the input refinements stay where they are.
    """
    commands = render(tmp_path, domain, 30.0, topography_height=1800.0)

    (topography,) = commands["topography"]
    assert float(topography["zmax"]) == pytest.approx(7400.0)
    assert [float(r["zmax"]) for r in commands["refinement"]] == [5000.0, 25000.0]


def test_velocity_model_must_cover_the_padded_grid(
    tmp_path: Path, domain: BoundingBox
) -> None:
    """The footprint is `(n - 1) * h` of the coarsest grid.

    The 20 km domain pads to 44 km, so a 48 km (121 points at 400 m) model
    covers it and a 40 km (101 points) model does not.
    """
    render(tmp_path, domain, 30.0, sfile_shape=(121, 121))

    with pytest.raises(ValueError, match="not contained in the velocity model"):
        render(tmp_path, domain, 30.0, sfile_shape=(101, 121))


def test_grid_pads_the_domain_by_one_sponge_per_side(
    tmp_path: Path, domain: BoundingBox
) -> None:
    """The requested domain is the grid's interior, not a slice of the sponge."""
    (grid,) = render(tmp_path, domain, 30.0)["grid"]

    # NOTE: In SW4 x = north, but in the workflow y = north.
    assert float(grid["x"]) == pytest.approx((domain.extent_y + 2 * SPONGE_KM) * 1000)
    assert float(grid["y"]) == pytest.approx((domain.extent_x + 2 * SPONGE_KM) * 1000)
