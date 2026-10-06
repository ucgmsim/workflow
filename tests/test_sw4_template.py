"""Tests for `workflow.scripts.sw4_template`.

The interesting behaviour here is geometric: the requested domain has to end up
as the *interior* of the SW4 grid, and the bottom refinement has to stay thick
enough to hold the bottom sponge. Both are invariants rather than values, so
they are tested as invariants.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pytest
from nzcvm.formats import sfile

from velocity_modelling.bounding_box import BoundingBox
from workflow import defaults, sw4
from workflow.realisations import (
    DomainParameters,
    RealisationMetadata,
    Refinement,
    SW4Resolution,
)
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


def layered_vs(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """500 m/s to 1 km, 2000 m/s to 8 km, then 4000 m/s."""
    return np.select([z < 1000.0, z < 8000.0], [500.0, 2000.0], 4000.0)


def write_sfile(
    path: Path,
    shape: tuple[int, int],
    resolution: float,
    topography_height: float = 500.0,
    zmax: float = 1_000_000.0,
    interface: float = 10_000.0,
    nk: int = 101,
    vs: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = layered_vs,
    chunk_rows: int | None = 4,
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
        The highest topography in the model, in metres above sea level. The
        surface rises from sea level on its west edge to this on its east edge.
    zmax : float
        The depth of the bottom of the model, in metres.
    interface : float
        The depth of the interface between the two grids, in metres.
    nk : int
        The vertical gridpoint count of each grid.
    vs : callable
        Vs (m/s) as a function of depth (m). Vp is `sqrt(3)` times it.
    chunk_rows : int, optional
        Chunk the material datasets in rows of this many, like NZCVM's (much
        larger) chunks. None stores them contiguously.
    """
    fine_shape = ((shape[0] - 1) * 2 + 1, (shape[1] - 1) * 2 + 1)
    surface = -np.broadcast_to(
        np.linspace(0.0, topography_height, fine_shape[1]), fine_shape
    )
    # A finer grid over the same footprint, to check the readers pick the
    # coarsest and still get the same answer.
    grids = (
        (2, surface, np.full(fine_shape, interface)),
        (1, np.full(shape, interface), np.full(shape, zmax)),
    )
    with h5py.File(path, "w") as f:
        f.attrs[sfile.ORIGIN_AZIM_ATTR] = np.array([172.5, -43.5, 35.0])
        f.attrs[sfile.MIN_MAX_DEPTH_ATTR] = np.array([-topography_height, zmax])
        f.attrs[sfile.NGRIDS_ATTR] = np.int32(len(grids))
        material = f.create_group(sfile.MATERIAL_GROUP)
        interfaces = f.create_group(sfile.SURFACE_GROUP)
        interfaces["z_values_0"] = surface
        for index, (factor, top, bottom) in enumerate(grids):
            interfaces[f"z_values_{index + 1}"] = bottom
            grid = material.create_group(f"grid_{index}")
            grid.attrs[sfile.HORIZONTAL_ATTR] = resolution / factor
            grid.attrs[sfile.NUMBER_OF_COMPONENTS_ATTR] = np.int32(2)
            z = top[..., None] + np.linspace(0.0, 1.0, nk) * (bottom - top)[..., None]
            chunks = (
                None
                if chunk_rows is None
                else (min(chunk_rows, z.shape[0]), *z.shape[1:])
            )
            grid.create_dataset("Cs", data=vs(z).astype(np.float32), chunks=chunks)
            grid.create_dataset(
                "Cp", data=(np.sqrt(3) * vs(z)).astype(np.float32), chunks=chunks
            )


def render(
    tmp_path: Path,
    domain: BoundingBox,
    depth_km: float,
    sfile_shape: tuple[int, int] = (121, 121),
    topography_height: float = 500.0,
    resolution: SW4Resolution | None = None,
    **sfile_kwargs: Any,
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
    resolution : SW4Resolution, optional
        Size SW4's refinements from the velocity model to these targets.
    **sfile_kwargs : Any
        Passed on to `write_sfile`.

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
    if resolution is not None:
        resolution.write_to_realisation(realisation)
    velocity_model = tmp_path / "model.sfile"
    write_sfile(
        velocity_model,
        sfile_shape,
        400.0,
        topography_height=topography_height,
        **sfile_kwargs,
    )
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
    """SW4's `check_supergrid_thickness` requires `nz[0]` to exceed the sponge.

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
def test_grid_spacing_is_never_coarser_than_planned(
    tmp_path: Path, domain: BoundingBox, depth_km: float
) -> None:
    """The grid spacing is within what the model was padded for.

    `create-nzvm-input` pads the model, and `generate-domain` checks the fault
    buffer, for the coarsest spacing SW4 may choose, before the model exists.
    """
    resolution = SW4Resolution.read_from_defaults(defaults.DefaultsVersion.v26_7_1Hz)

    (grid,) = render(tmp_path, domain, depth_km)["grid"]

    assert float(grid["h"]) <= resolution.coarsest_resolution


def test_topography_deepens_a_thin_implicit_layer() -> None:
    """The worked example in `_adjust_for_topography`.

    1800 m of topography puts the curvilinear bottom at 5400 m, 400 m (two
    200 m cells) below the 5000 m refinement. That bottom is pushed down to
    7400 m to give the implicit layer 12 cells, while the input refinements stay
    where they are.
    """
    refinements = [
        Refinement(resolution=100.0, bottom=5000.0),
        Refinement(resolution=200.0, bottom=25000.0),
        Refinement(resolution=400.0, bottom=30000.0),
    ]

    adjusted, topography_zmax = sw4_template._adjust_for_topography(
        refinements, sw4.topography_zmax(0.0, 1800.0), nzmin=12
    )

    assert topography_zmax == pytest.approx(7400.0)
    assert adjusted == refinements


def test_default_refinements_follow_the_model(
    tmp_path: Path, domain: BoundingBox
) -> None:
    """The defaults size SW4's refinements from the model, not its ladder.

    500 m of topography puts `zmax` at 1500 m and stretches the tallest column
    by 4 / 3 above it, which holds 200 m back to 1600 m. That leaves one 100 m
    cell below the curvilinear grid, so `_adjust_for_topography` pushes the
    interface to 2700 m for 12 of them. 400 m needs the 4000 m/s material from
    8 km.
    """
    commands = render(tmp_path, domain, 30.0)

    assert [float(r["zmax"]) for r in commands["refinement"]] == [2700.0, 8000.0]
    (topography,) = commands["topography"]
    assert float(topography["zmax"]) == 1500.0


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


def test_refinements_are_sized_from_the_velocity_model(
    tmp_path: Path, domain: BoundingBox
) -> None:
    """Each layer starts where the material below it is fast enough for it.

    At 8 points per wavelength and 1 Hz, 200 m needs 1600 m/s (from 1 km) and
    400 m needs 3200 m/s (from 8 km). The 100 m layer is then pushed to 1200 m
    to hold `nz_min` = 12 cells. Flat topography keeps the curvilinear stretch
    out of it.
    """
    commands = render(
        tmp_path,
        domain,
        30.0,
        topography_height=0.0,
        resolution=SW4Resolution(
            finest_resolution=100.0,
            coarsest_resolution=400.0,
            minimum_ppw=8.0,
            max_frequency=1.0,
        ),
        zmax=60_000.0,
        interface=10_000.0,
        nk=201,
    )

    assert [float(r["zmax"]) for r in commands["refinement"]] == [1200.0, 8000.0]
    (grid,) = commands["grid"]
    assert float(grid["h"]) == 400.0


def test_curvilinear_stretch_holds_back_coarsening(tmp_path: Path) -> None:
    """Under topography, a curvilinear cell is taller than its nominal spacing.

    1500 m of topography puts `zmax` at 4500 m, so the tallest column's cells
    are stretched by 6000 / 4500. The 2000 m/s material then only resolves
    200 m cells to 1500 m/s, short of the 1600 m/s it needs, until it leaves
    the curvilinear grid.
    """
    velocity_model = tmp_path / "model.sfile"
    write_sfile(
        velocity_model,
        (11, 11),
        400.0,
        topography_height=1500.0,
        zmax=60_000.0,
        interface=10_000.0,
        nk=401,
    )
    with h5py.File(velocity_model) as f:
        elevation_min, elevation_max = (
            sw4_template._elevation_range_from_velocity_model(f)
        )
        topography_zmax = sw4.topography_zmax(elevation_min, elevation_max)
        profile = sw4_template._vs_profile_from_velocity_model(
            f, topography_zmax, bin_size=100.0
        )

    assert (elevation_min, elevation_max) == (0.0, 1500.0)
    assert topography_zmax == 4500.0
    refinements = sw4.size_refinements(
        profile,
        SW4Resolution(
            finest_resolution=100.0,
            coarsest_resolution=200.0,
            minimum_ppw=8.0,
            max_frequency=1.0,
        ),
        depth_m=30_000.0,
        nz_min=12,
    )
    assert refinements[0].bottom == 4600.0


@pytest.mark.parametrize("chunk_rows", [None, 4, 16], ids=["contiguous", "4", "16"])
def test_profile_reads_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_rows: int | None
) -> None:
    """However the model is stored, no read or bin exceeds its budget.

    The budgets here are a few rows, so an unbounded read of a whole grid, or a
    chunk row taller than the budget, would show up. The profile must be the
    same as one read in a single piece.
    """
    velocity_model = tmp_path / "model.sfile"
    write_sfile(
        velocity_model,
        (11, 11),
        400.0,
        topography_height=1500.0,
        zmax=60_000.0,
        nk=41,
        chunk_rows=chunk_rows,
    )
    with h5py.File(velocity_model) as f:
        whole = sw4_template._vs_profile_from_velocity_model(f, 4500.0, 100.0)

    # The fine grid is 21 x 21 x 41 float32: 3 rows of Cs and Cp is 3 * 2 * 21 * 41
    # * 4 bytes. 16-row chunks cannot fit, so they are read partially.
    row_bytes = 2 * 21 * 41 * 4
    monkeypatch.setattr(sw4_template, "PROFILE_READ_BYTES", 3 * row_bytes)
    monkeypatch.setattr(sw4_template, "PROFILE_BLOCK_ELEMENTS", 2 * 21 * 41)
    largest_read = 0
    getitem = h5py.Dataset.__getitem__

    def recording_getitem(self: h5py.Dataset, key: Any) -> Any:
        nonlocal largest_read
        result = getitem(self, key)
        largest_read = max(largest_read, np.asarray(result).nbytes)
        return result

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    with h5py.File(velocity_model) as f:
        pieces = sw4_template._vs_profile_from_velocity_model(f, 4500.0, 100.0)

    assert largest_read <= 3 * row_bytes // 2
    np.testing.assert_array_equal(whole.min_vs, pieces.min_vs)
    np.testing.assert_array_equal(whole.max_wave_speed, pieces.max_wave_speed)


@pytest.mark.parametrize(
    "chunk_rows, budget_rows, expected",
    [
        pytest.param(None, 7, 7, id="contiguous-reads-the-budget"),
        pytest.param(4, 7, 4, id="whole-chunk-rows-that-fit"),
        pytest.param(4, 9, 8, id="several-chunk-rows"),
        pytest.param(16, 7, 7, id="partial-chunks-when-over-budget"),
        pytest.param(None, 1000, 21, id="never-past-the-end"),
        pytest.param(None, 0, 1, id="at-least-one-row"),
    ],
)
def test_profile_read_rows(
    tmp_path: Path, chunk_rows: int | None, budget_rows: int, expected: int
) -> None:
    with h5py.File(tmp_path / "rows.h5", "w") as f:
        chunks = None if chunk_rows is None else (chunk_rows, 5, 3)
        dataset = f.create_dataset(
            "Cs", shape=(21, 5, 3), dtype=np.float32, chunks=chunks
        )
        row_bytes = 2 * 5 * 3 * 4

        assert (
            sw4_template._profile_read_rows(dataset, budget_rows * row_bytes)
            == expected
        )
