from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from schema import SchemaError
from typer.testing import CliRunner

from source_modelling import srf
from source_modelling.srf import SrfFile
from source_modelling.stoch import StochFile
from workflow import defaults, realisations
from workflow.scripts.generate_stoch import (
    _box_average_matrix,
    app,
    convert_srf_to_stoch,
    stoch_resolution,
)

# (nstk, ndip, len, wid) for each plane of the synthetic SRF. The first
# plane divides evenly into the 2km stoch grid used below (dstk = ddip =
# 0.5), the second does not (dstk = ddip = 0.3).
PLANE_SHAPES = [(13, 7, 6.5, 3.5), (9, 5, 2.7, 1.5)]

DT = 0.1


@pytest.fixture
def synthetic_srf() -> SrfFile:
    """Build a small synthetic (version 1.0) SRF with random slip."""
    seed = 1
    rng = np.random.default_rng(seed)
    header = pd.DataFrame(
        [
            {
                "elon": 172.0 + i,
                "elat": -43.5 - i,
                "nstk": nstk,
                "ndip": ndip,
                "len": length,
                "wid": width,
                "stk": 45.0 + 90 * i,
                "dip": 60.0,
                "dtop": 1.0,
                "shyp": 0.5,
                "dhyp": 1.0,
            }
            for i, (nstk, ndip, length, width) in enumerate(PLANE_SHAPES)
        ]
    )
    points_per_plane = (header["nstk"] * header["ndip"]).to_numpy()
    n_points = int(points_per_plane.sum())
    # The rise time of each point is a whole number of timesteps so that
    # the SRF round-trips through disk exactly (on reading, rise = nt * dt).
    nt = rng.integers(1, 6, n_points)
    points = pd.DataFrame(
        {
            "lon": rng.uniform(171, 173, n_points),
            "lat": rng.uniform(-44, -43, n_points),
            "dep": rng.uniform(1, 10, n_points),
            "stk": np.repeat(header["stk"].to_numpy(), points_per_plane),
            "dip": 60.0,
            "area": 1e6,
            "tinit": rng.uniform(0, 10, n_points),
            "dt": DT,
            "rake": rng.uniform(0, 360, n_points),
            "slip": rng.uniform(0, 100, n_points),
            "rise": nt * DT,
        }
    )
    # Slip velocity time series: nt samples per point, integrating to the
    # total slip of that point.
    indptr = np.concatenate([[0], np.cumsum(nt)])
    indices = np.concatenate([np.arange(n) for n in nt])
    data = np.repeat(points["slip"].to_numpy() / (nt * DT), nt)
    slipt1 = sp.csr_array(
        (data, indices, indptr), shape=(n_points, int(nt.max())), dtype=np.float32
    )
    return SrfFile("1.0", header, points, slipt1)


@pytest.fixture
def two_patch_srf(synthetic_srf: SrfFile) -> SrfFile:
    """One 1km x 0.5km plane of two patches, which fits in a single 2km cell."""
    synthetic_srf.header = synthetic_srf.header.iloc[:1].copy()
    synthetic_srf.header.loc[0, ["nstk", "ndip", "len", "wid"]] = [2, 1, 1.0, 0.5]
    synthetic_srf.points = synthetic_srf.points.iloc[:2].copy()
    return synthetic_srf


def covered_fraction(
    n_coarse: int, coarse_dx: float, extent: float, *, centred: bool
) -> np.ndarray:
    """Fraction of each coarse cell that lies over a plane of length `extent`.

    A centred coarse grid splits the overhang evenly between the first and
    last cells, otherwise it all falls in the last cell.
    """
    overhang = (n_coarse * coarse_dx - extent) / 2 if centred else 0.0
    edges = np.arange(n_coarse + 1) * coarse_dx - overhang
    return (np.minimum(edges[1:], extent) - np.maximum(edges[:-1], 0)) / coarse_dx


def fine_moment(srf_file: SrfFile, i: int) -> float:
    """Sum of slip * patch area (in km^2) for plane `i` of an SRF."""
    plane = srf_file.header.iloc[i]
    patch_area = (plane["len"] / plane["nstk"]) * (plane["wid"] / plane["ndip"])
    return float(srf_file.segments[i]["slip"].sum() * patch_area)


def assert_moment_preserved(
    srf_file: SrfFile, stoch_file: StochFile, rel: float = 1e-5
) -> None:
    """Each stoch plane has the same slip x area as its SRF plane."""
    assert len(stoch_file.data) == len(srf_file.header)
    for i, plane in enumerate(stoch_file.data):
        coarse_moment = float(plane.slip.sum()) * plane.header.dx * plane.header.dy
        assert coarse_moment == pytest.approx(fine_moment(srf_file, i), rel=rel)


# --- _box_average_matrix -----------------------------------------------------


def test_box_average_matrix_matches_docstring_example() -> None:
    """Five fine cells of width 3 pooled into three coarse cells of width 5."""
    matrix = _box_average_matrix(5, 3, 3.0, 5.0, centred=True).toarray()
    assert matrix == pytest.approx(
        np.array(
            [
                [3 / 5, 2 / 5, 0, 0, 0],
                [0, 1 / 5, 3 / 5, 1 / 5, 0],
                [0, 0, 0, 2 / 5, 3 / 5],
            ]
        )
    )


@pytest.mark.parametrize("centred", [True, False])
def test_box_average_matrix_is_identity_when_grids_agree(centred: bool) -> None:
    matrix = _box_average_matrix(7, 7, 0.5, 0.5, centred=centred).toarray()
    assert matrix == pytest.approx(np.eye(7))


# (n_fine, fine_dx, coarse_dx), covered by the smallest coarse grid that fits.
GRID_CASES = [
    (100, 0.1, 2.0),
    (13, 0.5, 2.0),
    (9, 0.3, 2.0),
    (37, 0.2, 1.7),
    (5, 3.0, 5.0),
]


def n_coarse_for(n_fine: int, fine_dx: float, coarse_dx: float) -> int:
    return int(np.ceil(n_fine * fine_dx / coarse_dx))


@pytest.mark.parametrize("centred", [True, False])
@pytest.mark.parametrize(("n_fine", "fine_dx", "coarse_dx"), GRID_CASES)
def test_box_average_matrix_rows_are_weighted_averages(
    n_fine: int, fine_dx: float, coarse_dx: float, centred: bool
) -> None:
    """Every coarse bin is an average of the fine cells it covers.

    The weights of a bin sum to one, except for bins that hang off the end of
    the fine grid, which sum to the covered fraction of the bin.
    """
    n_coarse = n_coarse_for(n_fine, fine_dx, coarse_dx)
    matrix = _box_average_matrix(
        n_fine, n_coarse, fine_dx, coarse_dx, centred=centred
    ).toarray()
    assert matrix.shape == (n_coarse, n_fine)
    assert (matrix >= 0).all()
    assert matrix.sum(axis=1) == pytest.approx(
        covered_fraction(n_coarse, coarse_dx, n_fine * fine_dx, centred=centred)
    )


@pytest.mark.parametrize(("n_fine", "fine_dx", "coarse_dx"), GRID_CASES)
def test_box_average_matrix_is_centred(
    n_fine: int, fine_dx: float, coarse_dx: float
) -> None:
    """A centred coarse grid overhangs both ends of the fine grid equally."""
    n_coarse = n_coarse_for(n_fine, fine_dx, coarse_dx)
    matrix = _box_average_matrix(
        n_fine, n_coarse, fine_dx, coarse_dx, centred=True
    ).toarray()
    # Reversing both the bins and the cells they cover is the same grid.
    assert matrix == pytest.approx(matrix[::-1, ::-1])


def test_box_average_matrix_uncentred_starts_with_the_fine_grid() -> None:
    """An uncentred coarse grid starts where the fine grid does.

    This is the down-dip case: the HF code hangs the stoch grid from the top
    edge of the plane, so for a 3.5km wide plane on a 2km grid the first row
    must cover [0, 2]km and the second [2, 3.5]km, with the 0.5km of padding
    below the bottom of the plane.
    """
    matrix = _box_average_matrix(7, 2, 0.5, 2.0, centred=False).toarray()
    assert matrix == pytest.approx(
        np.array(
            [
                [1 / 4, 1 / 4, 1 / 4, 1 / 4, 0, 0, 0],
                [0, 0, 0, 0, 1 / 4, 1 / 4, 1 / 4],
            ]
        )
    )


@pytest.mark.parametrize("centred", [True, False])
@pytest.mark.parametrize(("n_fine", "fine_dx", "coarse_dx"), GRID_CASES)
def test_box_average_matrix_conserves_mass(
    n_fine: int, fine_dx: float, coarse_dx: float, centred: bool
) -> None:
    """Averaging then re-integrating over the coarse cells preserves the integral."""
    n_coarse = n_coarse_for(n_fine, fine_dx, coarse_dx)
    matrix = _box_average_matrix(n_fine, n_coarse, fine_dx, coarse_dx, centred=centred)
    values = np.random.default_rng(2).uniform(0, 10, n_fine)
    coarse = matrix @ values
    assert (coarse.sum() * coarse_dx) == pytest.approx(values.sum() * fine_dx)


# --- Moment preservation -----------------------------------------------------

STOCH_RESOLUTIONS = [(2.0, 2.0), (1.0, 1.0), (0.7, 1.3), (0.5, 0.5)]


@pytest.mark.parametrize(("dx", "dy"), STOCH_RESOLUTIONS)
def test_convert_srf_to_stoch_preserves_moment(
    synthetic_srf: SrfFile, dx: float, dy: float
) -> None:
    """Total moment (slip x area) of each plane survives the down-sampling.

    The stoch cells are physically larger than the SRF patches, so the
    box average must be weighted by the overlap between the two grids for
    the sum of slip x area to be unchanged.
    """
    assert_moment_preserved(synthetic_srf, convert_srf_to_stoch(synthetic_srf, dx, dy))


def test_convert_srf_to_stoch_preserves_uniform_slip(synthetic_srf: SrfFile) -> None:
    """A uniform slip distribution down-samples to the same uniform slip."""
    dx = dy = 2.0
    synthetic_srf.points["slip"] = 42.0
    stoch_file = convert_srf_to_stoch(synthetic_srf, dx, dy)
    for i, plane in enumerate(stoch_file.data):
        header = synthetic_srf.header.iloc[i]
        # Cells the plane only partially covers are scaled down by the
        # covered fraction of the cell, which is what keeps the moment
        # (rather than the slip value) constant. Along strike the grid is
        # centred on the plane, so the partial cells are at both ends.
        # Down-dip it hangs from the top edge, so they are the bottom row.
        covered_x = covered_fraction(plane.header.nx, dx, header["len"], centred=True)
        covered_y = covered_fraction(plane.header.ny, dy, header["wid"], centred=False)
        assert plane.slip == pytest.approx(
            42.0 * np.outer(covered_y, covered_x), rel=1e-5
        )


def test_convert_srf_to_stoch_grid_covers_the_plane(synthetic_srf: SrfFile) -> None:
    """The stoch grid is the smallest dx by dy grid covering the SRF plane."""
    dx, dy = 2.0, 2.0
    stoch_file = convert_srf_to_stoch(synthetic_srf, dx, dy)
    for i, plane in enumerate(stoch_file.data):
        header = synthetic_srf.header.iloc[i]
        assert plane.header.nx == int(np.ceil(header["len"] / dx))
        assert plane.header.ny == int(np.ceil(header["wid"] / dy))
        assert plane.slip.shape == (plane.header.ny, plane.header.nx)
        assert plane.rise.shape == plane.slip.shape
        assert plane.trup.shape == plane.slip.shape


def test_convert_srf_to_stoch_rise_is_slip_weighted(two_patch_srf: SrfFile) -> None:
    """Rise time is averaged in proportion to slip, not by area."""
    # All of the slip is on the patch with a rise time of 3s, so the cell
    # rise time must be 3s.
    two_patch_srf.points["slip"] = [0.0, 10.0]
    two_patch_srf.points["rise"] = [7.0, 3.0]

    (plane,) = convert_srf_to_stoch(two_patch_srf, 2.0, 2.0).data
    assert plane.slip.shape == (1, 1)
    assert plane.rise.item() == pytest.approx(3.0)


@pytest.mark.parametrize(("dx", "dy"), STOCH_RESOLUTIONS)
def test_convert_srf_to_stoch_trup_is_not_scaled_by_coverage(
    synthetic_srf: SrfFile, dx: float, dy: float
) -> None:
    """Rupture time is a time, so partially covered cells must not dilute it.

    Slip is deliberately scaled down in the cells at the edge of the plane
    to conserve the moment. Applying the same scaling to the rupture time
    would make the rupture arrive early at the edges of every plane.
    """
    synthetic_srf.points["tinit"] = 5.0
    stoch_file = convert_srf_to_stoch(synthetic_srf, dx, dy)
    for plane in stoch_file.data:
        assert plane.trup == pytest.approx(5.0, rel=1e-5)


def test_convert_srf_to_stoch_trup_matches_a_uniform_average(
    two_patch_srf: SrfFile,
) -> None:
    """A cell covering the whole plane gets the mean rupture time of the plane."""
    two_patch_srf.points["tinit"] = [4.0, 6.0]

    (plane,) = convert_srf_to_stoch(two_patch_srf, 2.0, 2.0).data
    assert plane.trup.item() == pytest.approx(5.0)


def test_convert_srf_to_stoch_zero_slip_rise(synthetic_srf: SrfFile) -> None:
    """Cells with no slip get a nominal (non-zero) rise time."""
    synthetic_srf.points["slip"] = 0.0
    stoch_file = convert_srf_to_stoch(synthetic_srf, 2.0, 2.0)
    for plane in stoch_file.data:
        assert (plane.slip == 0).all()
        assert plane.rise == pytest.approx(1e-5)


def test_average_rake_is_in_degrees(synthetic_srf: SrfFile) -> None:
    """The stoch header rake is a bearing in degrees, not radians."""
    synthetic_srf.points["rake"] = 185.0
    stoch_file = convert_srf_to_stoch(synthetic_srf, 2.0, 2.0)
    for plane in stoch_file.data:
        assert plane.header.average_rake == pytest.approx(185.0, abs=1e-3)


# --- stoch_resolution --------------------------------------------------------


def cost(
    extents: np.ndarray, resolution: float, target: float, padding_weight: float
) -> float:
    """The drift + weighted padding that stoch_resolution minimises, in cells."""
    cells = extents / resolution
    padding = (np.maximum(1, np.ceil(cells - 1e-9)) - cells).sum()
    drift = np.abs(cells - extents / target).sum()
    return float(drift + padding_weight * padding)


def srf2stoch_resolution(extent: float, target: float) -> float:
    """srf2stoch's target_dx: the whole number of cells closest to the target."""
    return extent / int(extent / target + 0.5)


@pytest.mark.parametrize("extent", [23.7, 6.5, 15.3, 41.05])
@pytest.mark.parametrize("target", [1.0, 2.0, 2.2, 3.0])
def test_stoch_resolution_zero_weight_is_the_target(
    extent: float, target: float
) -> None:
    """A padding weight of 0 gives the target, however much it pads."""
    assert stoch_resolution(
        np.array([extent, extent / 3]), np.array([0.1, 0.1]), target, 0.0, 0.0, None
    ) == pytest.approx(target)


@pytest.mark.parametrize("extent", [23.7, 6.5, 15.3, 41.05])
@pytest.mark.parametrize("target", [1.0, 2.0, 2.2, 3.0])
@pytest.mark.parametrize("padding_weight", [1.01, 2.0, 100.0])
def test_stoch_resolution_single_plane_matches_srf2stoch(
    extent: float, target: float, padding_weight: float
) -> None:
    """For a single plane, a padding weight above 1 is srf2stoch's target_dx."""
    assert stoch_resolution(
        np.array([extent]), np.array([0.1]), target, padding_weight, 0.0, None
    ) == pytest.approx(srf2stoch_resolution(extent, target))


def test_stoch_resolution_trades_padding_for_drift() -> None:
    """A large enough weight moves off the target to a resolution that pads less."""
    extents = np.array([3.4, 1.9, 5.1, 5.6])
    srf_resolutions = np.full(4, 0.1)
    exact = stoch_resolution(extents, srf_resolutions, 2.0, 0.0, 0.0, None)
    traded = stoch_resolution(extents, srf_resolutions, 2.0, 2.0, 0.0, None)
    assert exact == pytest.approx(2.0)
    assert traded != pytest.approx(2.0)
    assert cost(extents, traded, 2.0, 2.0) < cost(extents, exact, 2.0, 2.0)


def test_stoch_resolution_ties_go_to_the_target() -> None:
    """When padding exactly pays for the drift, the target wins."""
    # At weight 1, 1.9 km saves exactly as much padding as it drifts.
    extents = np.array([3.4, 1.9, 5.1, 5.6])
    srf_resolutions = np.full(4, 0.1)
    assert cost(extents, 1.9, 2.0, 1.0) == pytest.approx(cost(extents, 2.0, 2.0, 1.0))
    assert stoch_resolution(
        extents, srf_resolutions, 2.0, 1.0, 0.0, None
    ) == pytest.approx(2.0)


def test_stoch_resolution_equal_bounds_force_the_resolution() -> None:
    """Setting the minimum equal to the maximum forces that resolution."""
    assert stoch_resolution(
        np.array([6.5, 2.7]), np.array([0.5, 0.3]), 2.0, 10.0, 2.0, 2.0
    ) == pytest.approx(2.0)


def test_stoch_resolution_forces_finer_than_the_srf() -> None:
    """An upper bound below the SRF resolution is respected."""
    assert stoch_resolution(
        np.array([6.5]), np.array([0.5]), 0.25, 0.0, 0.0, 0.25
    ) == pytest.approx(0.25)


def test_stoch_resolution_is_no_finer_than_the_srf() -> None:
    """The planes are never up-sampled past the coarsest SRF resolution."""
    assert stoch_resolution(
        np.array([6.0, 4.0]), np.array([0.5, 0.25]), 0.1, 0.0, 0.0, None
    ) == pytest.approx(0.5)


def test_stoch_resolution_finds_a_common_factor() -> None:
    """Planes are fit exactly by a common factor of their extents near the target."""
    assert stoch_resolution(
        np.array([6.0, 4.0]), np.array([0.5, 0.5]), 1.8, 10.0, 0.0, None
    ) == pytest.approx(2.0)


@pytest.mark.parametrize("seed", range(10))
def test_stoch_resolution_beats_a_grid_search(seed: int) -> None:
    """No resolution in the bounds costs less than the chosen one."""
    rng = np.random.default_rng(seed)
    extents = np.round(rng.uniform(1, 30, rng.integers(1, 5)), 1)
    srf_resolutions = extents / rng.integers(5, 50, len(extents))
    min_resolution, max_resolution = np.sort(rng.uniform(0.5, 5, 2))
    target = rng.uniform(min_resolution, max_resolution)
    padding_weight = rng.uniform(0, 3)
    resolution = stoch_resolution(
        extents,
        srf_resolutions,
        target,
        padding_weight,
        min_resolution,
        max_resolution,
    )
    lower = min(max(min_resolution, srf_resolutions.max()), max_resolution)
    assert lower - 1e-9 <= resolution <= max_resolution + 1e-9
    best = cost(extents, resolution, target, padding_weight)
    for trial in np.linspace(lower, max_resolution, 2000):
        assert best <= cost(extents, trial, target, padding_weight) + 1e-9


def test_convert_srf_to_stoch_exact_resolution_has_no_sliver_cell(
    synthetic_srf: SrfFile,
) -> None:
    """A resolution dividing the plane exactly gives exactly that many cells.

    Without a tolerance, float error in length / dx can round up to an extra,
    almost entirely empty, cell.
    """
    # 2.7 / (2.7 / 31) is 31.000000000000004 in floating point.
    (_, plane) = convert_srf_to_stoch(synthetic_srf, 2.7 / 31, 1.5 / 3).data
    assert plane.header.nx == 31
    assert plane.header.ny == 3


STOCH_CONFIG = {
    "stoch_target_dx": 2.0,
    "stoch_min_dx": 0.0,
    "stoch_max_dx": None,
    "stoch_target_dy": 2.0,
    "stoch_min_dy": 0.0,
    "stoch_max_dy": None,
    "stoch_padding_weight": 0.0,
}


def test_stoch_config_accepts_no_maximum() -> None:
    """A maximum of None means the resolution has no upper limit."""
    config = realisations.StochConfig.from_dict(
        STOCH_CONFIG | {"stoch_min_dx": 1.0, "stoch_max_dy": 2.0}
    )
    assert config.stoch_max_dx is None


@pytest.mark.parametrize(
    "bad",
    [
        {"stoch_min_dx": 3.0, "stoch_max_dx": 2.5},
        {"stoch_target_dx": 2.5, "stoch_max_dx": 2.0},
        {"stoch_target_dy": 1.0, "stoch_min_dy": 1.5},
        {"stoch_max_dx": float("inf")},
        {"stoch_target_dx": 0.0},
        {"stoch_padding_weight": -1.0},
    ],
)
def test_stoch_config_rejects_bad_values(bad: dict[str, float]) -> None:
    """The bounds must contain the target, and no limit is spelled None."""
    with pytest.raises(SchemaError):
        realisations.StochConfig.from_dict(STOCH_CONFIG | bad)


# --- Integration -------------------------------------------------------------


@pytest.fixture
def realisation_ffp(tmp_path: Path) -> Path:
    """A realisation using the default stoch configuration."""
    realisation_ffp = tmp_path / "realisation.json"
    realisations.RealisationMetadata(
        name="generate stoch test",
        version="1",
        defaults_version=defaults.DefaultsVersion.v24_2_2_1,
    ).write_to_realisation(realisation_ffp)
    return realisation_ffp


def test_generate_stoch_smoke(
    tmp_path: Path, realisation_ffp: Path, synthetic_srf: SrfFile
) -> None:
    """An SRF file on disk converts into a readable stoch file."""
    srf_ffp = tmp_path / "realisation.srf"
    stoch_ffp = tmp_path / "realisation.stoch"
    srf.write_srf(srf_ffp, synthetic_srf)

    result = CliRunner().invoke(
        app, [str(realisation_ffp), str(srf_ffp), str(stoch_ffp)]
    )
    assert result.exit_code == 0, result.output

    stoch_file = StochFile.from_file(stoch_ffp)
    assert len(stoch_file.data) == len(PLANE_SHAPES)

    srf_file = srf.read_srf(srf_ffp)
    config = realisations.StochConfig.read_from_realisation_or_defaults(
        realisation_ffp, defaults.DefaultsVersion.v24_2_2_1
    )
    lengths = srf_file.header["len"].to_numpy(dtype=np.float64)
    widths = srf_file.header["wid"].to_numpy(dtype=np.float64)
    dx = stoch_resolution(
        lengths,
        lengths / srf_file.header["nstk"].to_numpy(),
        config.stoch_target_dx,
        config.stoch_padding_weight,
        config.stoch_min_dx,
        config.stoch_max_dx,
    )
    dy = stoch_resolution(
        widths,
        widths / srf_file.header["ndip"].to_numpy(),
        config.stoch_target_dy,
        config.stoch_padding_weight,
        config.stoch_min_dy,
        config.stoch_max_dy,
    )
    for i, plane in enumerate(stoch_file.data):
        header = srf_file.header.iloc[i]
        # Every plane shares the configured stoch dx/dy, as the HF code
        # requires, to the 10 m the stoch header is written to. Planes
        # smaller than a cell round up to a single cell rather than
        # down-sampling to an empty grid.
        assert plane.header.dx == pytest.approx(dx, abs=0.005)
        assert plane.header.dy == pytest.approx(dy, abs=0.005)
        assert plane.slip.shape == (plane.header.ny, plane.header.nx)
        assert plane.header.dtop == pytest.approx(header["dtop"])
        assert plane.header.dip == pytest.approx(header["dip"])
        assert plane.header.strike == pytest.approx(header["stk"] % 360)
        assert (plane.slip >= 0).all()
        assert (plane.rise > 0).all()
    # The written file preserves the moment to the precision of the stoch
    # format: %e for slip, and dx and dy rounded to 10 m in the header (as
    # srf2stoch writes them), which scales the cell area by up to 0.005 / dx
    # and 0.005 / dy.
    assert_moment_preserved(srf_file, stoch_file, rel=1e-4 + 0.005 / dx + 0.005 / dy)
