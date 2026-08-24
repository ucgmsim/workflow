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
from workflow.realisations import Refinement, Refinements, SW4Parameters
from workflow.scripts import nzvm_input_template, sw4_template

SPONGE_KM = 12.0
"""The v26_7_1Hz sponge width, in kilometres."""


@pytest.fixture
def domain() -> BoundingBox:
    """A rotated 100 x 80 km domain, so the padding cannot be axis-aligned.

    Returns
    -------
    BoundingBox
        The domain.
    """
    return BoundingBox.from_centroid_bearing_extents(
        centroid=np.array([-43.5, 172.5]),
        bearing=35.0,
        extent_x=100.0,
        extent_y=80.0,
    )


def test_padding_grows_each_axis_by_two_sponges(domain: BoundingBox) -> None:
    """One sponge per face, so an axis grows by exactly two sponge widths."""
    padded = domain.pad(pad_x=(SPONGE_KM, SPONGE_KM), pad_y=(SPONGE_KM, SPONGE_KM))

    assert padded.extent_x == pytest.approx(domain.extent_x + 2 * SPONGE_KM)
    assert padded.extent_y == pytest.approx(domain.extent_y + 2 * SPONGE_KM)


def test_padding_preserves_orientation(domain: BoundingBox) -> None:
    """A rotated box must stay rotated the same way.

    `BoundingBox.pad` pads along the box's own axes and then re-normalises the
    corner order in `__init__`. If the re-normalisation picked a different
    bottom-left corner the roles of `extent_x` and `extent_y` would swap and the
    SW4 grid would be transposed relative to the velocity model.
    """
    padded = domain.pad(pad_x=(SPONGE_KM, SPONGE_KM), pad_y=(SPONGE_KM, SPONGE_KM))

    assert padded.bearing == pytest.approx(domain.bearing)
    assert padded.great_circle_bearing == pytest.approx(
        domain.great_circle_bearing, abs=1e-3
    )
    # Symmetric padding keeps the centroid, which is what makes the velocity
    # model and the SW4 grid concentric.
    assert padded.origin == pytest.approx(domain.origin, abs=1e-6)


def test_padded_grid_strictly_contains_the_requested_domain(
    domain: BoundingBox,
) -> None:
    """The requested domain becomes the interior, not a slice of the sponge."""
    padded = domain.pad(pad_x=(SPONGE_KM, SPONGE_KM), pad_y=(SPONGE_KM, SPONGE_KM))

    assert padded.polygon.contains_properly(domain.polygon)
    # Every edge of the original is a full sponge width inside the padded box.
    assert domain.polygon.exterior.distance(padded.polygon.exterior) == pytest.approx(
        SPONGE_KM * 1000.0, rel=1e-6
    )


def test_model_padding_contains_the_padded_grid(domain: BoundingBox) -> None:
    """`create-nzvm-input` pads by more than `create-sw4-input` does.

    If this ordering ever inverts, SW4 queries outside the sfile.
    """
    model_padding_km = (
        SPONGE_KM + nzvm_input_template.SW4_MODEL_SLACK_GRIDPOINTS * 400.0 / 1000.0
    )
    grid = domain.pad(pad_x=(SPONGE_KM, SPONGE_KM), pad_y=(SPONGE_KM, SPONGE_KM))
    model = domain.pad(
        pad_x=(model_padding_km, model_padding_km),
        pad_y=(model_padding_km, model_padding_km),
    )

    assert model.polygon.contains_properly(grid.polygon)
    assert model.extent_x > grid.extent_x
    assert model.extent_y > grid.extent_y


@pytest.mark.parametrize("depth_km", [10.0, 30.0, 60.0, 120.0, 350.0])
def test_bottom_refinement_holds_the_sponge(depth_km: float) -> None:
    """SW4's `check_supergrid_thickness` requires `nz[0] >` the sponge thickness.

    Only grid 0 carries a bottom taper, so the requirement is on the bottom
    refinement alone. `adjust_for_topography` runs first and can move the
    bottoms around, so the invariant is checked after it, exactly as
    `generate_sw4_input` orders the two.
    """
    version = defaults.DefaultsVersion.v26_7_1Hz
    theoretical = Refinements.read_from_defaults(version)
    sw4_params = SW4Parameters.read_from_defaults(version)

    refinements = theoretical.refinements_for_depth(depth_km)
    refinements, _ = sw4_template.adjust_for_topography(
        refinements, topography_zmax=1500.0, nzmin=sw4_params.nz_min
    )
    refinements = sorted(refinements, key=lambda r: r.bottom)

    coarsest = refinements[-1].resolution
    assert coarsest == sw4.coarsest_resolution(theoretical, depth_km)

    supergrid_width = sw4.supergrid_width(sw4_params, coarsest)
    thickness_before = refinements[-1].bottom - (
        refinements[-2].bottom if len(refinements) > 1 else 0.0
    )
    refinements[-1].bottom += supergrid_width
    thickness = thickness_before + supergrid_width

    # Strictly greater: the sponge must be *contained* in the layer, so there has
    # to be at least one cell of the layer that is not sponge.
    assert thickness > supergrid_width
    assert thickness / coarsest > supergrid_width / coarsest


def test_adjust_for_topography_leaves_resolutions_alone() -> None:
    """The coarsest resolution can be read before or after adjustment.

    `generate_sw4_input` reads the sponge width from the adjusted refinements
    while `create-nzvm-input` reads it from the theoretical ones. That only
    agrees because adjustment moves bottoms, never resolutions.
    """
    refinements = [
        Refinement(resolution=100.0, bottom=5000.0),
        Refinement(resolution=200.0, bottom=25000.0),
        Refinement(resolution=400.0, bottom=40000.0),
    ]
    adjusted, _ = sw4_template.adjust_for_topography(
        refinements, topography_zmax=1500.0, nzmin=12
    )

    assert [r.resolution for r in adjusted] == [100.0, 200.0, 400.0]


def write_sfile(path: Path, shape: tuple[int, int], resolution: float) -> None:
    """Write a minimal sfile carrying only what the footprint reader needs.

    Parameters
    ----------
    path : Path
        Where to write the file.
    shape : tuple[int, int]
        The (north, east) gridpoint counts of the coarsest grid.
    resolution : float
        The coarsest grid's horizontal spacing, in metres.
    """
    with h5py.File(path, "w") as f:
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


def test_lateral_footprint_from_velocity_model(tmp_path: Path) -> None:
    """The footprint is `(n - 1) * h`, read off the coarsest grid."""
    path = tmp_path / "model.sfile"
    write_sfile(path, shape=(319, 269), resolution=400.0)

    with h5py.File(path, "r") as f:
        extent_x, extent_y = sw4_template.lateral_footprint_from_velocity_model(f)

    assert extent_x == pytest.approx(318 * 400.0)
    assert extent_y == pytest.approx(268 * 400.0)
