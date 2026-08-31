import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from typer.testing import CliRunner

from source_modelling import sources, srf
from source_modelling.srf import SrfFile
from source_modelling.stoch import StochFile
from workflow import defaults, realisations
from workflow.scripts.generate_stoch import (
    _box_average_matrix,
    app,
    circular_mean,
    convert_srf_to_stoch,
)

# (nstk, ndip, len, wid) for each plane of the synthetic SRF. The first
# plane divides evenly into the 2km stoch grid used below (dstk = ddip =
# 0.5), the second does not (dstk = ddip = 0.3).
PLANE_SHAPES = [(13, 7, 6.5, 3.5), (9, 5, 2.7, 1.5)]

DT = 0.1

# A real (multi-segment) SRF, if a source_modelling checkout is available.
REAL_SRF_FFP = (
    Path(
        os.environ.get(
            "SOURCE_MODELLING_PATH", Path.home() / "src" / "source_modelling"
        )
    )
    / "tests"
    / "srfs"
    / "3366146.srf"
)


def make_srf(seed: int = 1) -> SrfFile:
    """Build a small synthetic (version 1.0) SRF with random slip."""
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
    n_points = int((header["nstk"] * header["ndip"]).sum())
    # The rise time of each point is a whole number of timesteps so that
    # the SRF round-trips through disk exactly (on reading, rise = nt * dt).
    nt = rng.integers(1, 6, n_points)
    points = pd.DataFrame(
        {
            "lon": rng.uniform(171, 173, n_points),
            "lat": rng.uniform(-44, -43, n_points),
            "dep": rng.uniform(1, 10, n_points),
            "stk": np.repeat(
                header["stk"].to_numpy(), (header["nstk"] * header["ndip"]).to_numpy()
            ),
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
def synthetic_srf() -> SrfFile:
    return make_srf()


def covered_fraction(n_coarse: int, coarse_dx: float, extent: float) -> np.ndarray:
    """Fraction of each coarse cell that lies over a plane of length `extent`.

    The coarse grid is centred on the plane, so the overhang is split
    evenly between the first and last cells.
    """
    overhang = (n_coarse * coarse_dx - extent) / 2
    edges = np.arange(n_coarse + 1) * coarse_dx - overhang
    return (np.minimum(edges[1:], extent) - np.maximum(edges[:-1], 0)) / coarse_dx


def fine_moment(srf_file: SrfFile, i: int) -> float:
    """Sum of slip * patch area (in km^2) for plane `i` of an SRF."""
    plane = srf_file.header.iloc[i]
    patch_area = (plane["len"] / plane["nstk"]) * (plane["wid"] / plane["ndip"])
    return float(srf_file.segments[i]["slip"].sum() * patch_area)


# --- _box_average_matrix -----------------------------------------------------


def test_box_average_matrix_matches_docstring_example() -> None:
    """Five fine cells of width 3 pooled into three coarse cells of width 5."""
    matrix = _box_average_matrix(5, 3, 3.0, 5.0).toarray()
    assert matrix == pytest.approx(
        np.array(
            [
                [3 / 5, 2 / 5, 0, 0, 0],
                [0, 1 / 5, 3 / 5, 1 / 5, 0],
                [0, 0, 0, 2 / 5, 3 / 5],
            ]
        )
    )


def test_box_average_matrix_is_identity_when_grids_agree() -> None:
    matrix = _box_average_matrix(7, 7, 0.5, 0.5).toarray()
    assert matrix == pytest.approx(np.eye(7))


@pytest.mark.parametrize(
    ("n_fine", "fine_dx", "coarse_dx"),
    [(100, 0.1, 2.0), (13, 0.5, 2.0), (9, 0.3, 2.0), (37, 0.2, 1.7), (5, 3.0, 5.0)],
)
def test_box_average_matrix_rows_are_weighted_averages(
    n_fine: int, fine_dx: float, coarse_dx: float
) -> None:
    """Every coarse bin is an average of the fine cells it covers.

    The weights of a bin sum to one, except for the first and last bins,
    which hang off either end of the fine grid by half the overhang each
    and sum to the covered fraction of the bin.
    """
    n_coarse = int(np.ceil(n_fine * fine_dx / coarse_dx))
    assert n_coarse >= 2, "the covered fraction below assumes two distinct end bins"
    matrix = _box_average_matrix(n_fine, n_coarse, fine_dx, coarse_dx).toarray()
    assert matrix.shape == (n_coarse, n_fine)
    assert (matrix >= 0).all()

    row_sums = matrix.sum(axis=1)
    overhang = (n_coarse * coarse_dx - n_fine * fine_dx) / 2
    covered = (coarse_dx - overhang) / coarse_dx
    assert row_sums[1:-1] == pytest.approx(np.ones(n_coarse - 2))
    assert row_sums[0] == pytest.approx(covered)
    assert row_sums[-1] == pytest.approx(covered)


@pytest.mark.parametrize(
    ("n_fine", "fine_dx", "coarse_dx"),
    [(100, 0.1, 2.0), (13, 0.5, 2.0), (9, 0.3, 2.0), (37, 0.2, 1.7), (5, 3.0, 5.0)],
)
def test_box_average_matrix_is_centred(
    n_fine: int, fine_dx: float, coarse_dx: float
) -> None:
    """The coarse grid is centred on the fine grid, not aligned to its start.

    The stoch format records a centre point and an ``nx * dx`` extent, so a
    coarse grid longer than the plane has to overhang both ends equally.
    Otherwise the slip would sit off-centre on the plane the HF code
    reconstructs from the header.
    """
    n_coarse = int(np.ceil(n_fine * fine_dx / coarse_dx))
    matrix = _box_average_matrix(n_fine, n_coarse, fine_dx, coarse_dx).toarray()
    # Reversing both the bins and the cells they cover is the same grid.
    assert matrix == pytest.approx(matrix[::-1, ::-1])


@pytest.mark.parametrize(
    ("n_fine", "fine_dx", "coarse_dx"),
    [(100, 0.1, 2.0), (13, 0.5, 2.0), (9, 0.3, 2.0), (37, 0.2, 1.7), (5, 3.0, 5.0)],
)
def test_box_average_matrix_conserves_mass(
    n_fine: int, fine_dx: float, coarse_dx: float
) -> None:
    """Averaging then re-integrating over the coarse cells preserves the integral."""
    n_coarse = int(np.ceil(n_fine * fine_dx / coarse_dx))
    matrix = _box_average_matrix(n_fine, n_coarse, fine_dx, coarse_dx)
    values = np.random.default_rng(2).uniform(0, 10, n_fine)
    coarse = matrix @ values
    assert (coarse.sum() * coarse_dx) == pytest.approx(values.sum() * fine_dx)


# --- Moment preservation -----------------------------------------------------


@pytest.mark.parametrize(("dx", "dy"), [(2.0, 2.0), (1.0, 1.0), (0.7, 1.3), (0.5, 0.5)])
def test_convert_srf_to_stoch_preserves_moment(
    synthetic_srf: SrfFile, dx: float, dy: float
) -> None:
    """Total moment (slip x area) of each plane survives the down-sampling.

    The stoch cells are physically larger than the SRF patches, so the
    box average must be weighted by the overlap between the two grids for
    the sum of slip x area to be unchanged.
    """
    stoch_file = convert_srf_to_stoch(synthetic_srf, dx, dy)
    assert len(stoch_file.data) == len(PLANE_SHAPES)
    for i, plane in enumerate(stoch_file.data):
        coarse_moment = float(plane.slip.sum()) * dx * dy
        assert coarse_moment == pytest.approx(fine_moment(synthetic_srf, i), rel=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_SRF_FFP.exists(), reason=f"{REAL_SRF_FFP} is not available"
)
def test_convert_srf_to_stoch_preserves_moment_real_srf() -> None:
    """Moment is preserved for a real multi-segment rupture."""
    srf_file = srf.read_srf(REAL_SRF_FFP)
    stoch_file = convert_srf_to_stoch(srf_file, 2.0, 2.0)
    for i, plane in enumerate(stoch_file.data):
        coarse_moment = float(plane.slip.sum()) * plane.header.dx * plane.header.dy
        assert coarse_moment == pytest.approx(fine_moment(srf_file, i), rel=1e-5)


def test_convert_srf_to_stoch_preserves_uniform_slip(synthetic_srf: SrfFile) -> None:
    """A uniform slip distribution down-samples to the same uniform slip."""
    dx = dy = 2.0
    synthetic_srf.points["slip"] = 42.0
    stoch_file = convert_srf_to_stoch(synthetic_srf, dx, dy)
    for i, plane in enumerate(stoch_file.data):
        header = synthetic_srf.header.iloc[i]
        # Cells the plane only partially covers are scaled down by the
        # covered fraction of the cell, which is what keeps the moment
        # (rather than the slip value) constant.
        covered_x = covered_fraction(plane.header.nx, dx, header["len"])
        covered_y = covered_fraction(plane.header.ny, dy, header["wid"])
        assert plane.slip == pytest.approx(
            42.0 * np.outer(covered_y, covered_x), rel=1e-5
        )
        # The partial cells are the two ends, not just the far end.
        assert plane.slip == pytest.approx(plane.slip[::-1, ::-1], rel=1e-5)


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


def test_convert_srf_to_stoch_rise_is_slip_weighted(synthetic_srf: SrfFile) -> None:
    """Rise time is averaged in proportion to slip, not by area."""
    # One plane, one stoch cell, two patches: all of the slip is on the
    # patch with a rise time of 3s, so the cell rise time must be 3s.
    srf_file = synthetic_srf
    srf_file.header = srf_file.header.iloc[:1].copy()
    srf_file.header.loc[0, ["nstk", "ndip", "len", "wid"]] = [2, 1, 1.0, 0.5]
    srf_file.points = srf_file.points.iloc[:2].copy()
    srf_file.points["slip"] = [0.0, 10.0]
    srf_file.points["rise"] = [7.0, 3.0]

    (plane,) = convert_srf_to_stoch(srf_file, 2.0, 2.0).data
    assert plane.slip.shape == (1, 1)
    assert plane.rise.item() == pytest.approx(3.0)


@pytest.mark.parametrize(("dx", "dy"), [(2.0, 2.0), (1.0, 1.0), (0.7, 1.3), (0.5, 0.5)])
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
        assert plane.trup == pytest.approx(np.full(plane.trup.shape, 5.0), rel=1e-5)


def test_convert_srf_to_stoch_trup_matches_a_uniform_average(
    synthetic_srf: SrfFile,
) -> None:
    """A cell covering the whole plane gets the mean rupture time of the plane."""
    srf_file = synthetic_srf
    srf_file.header = srf_file.header.iloc[:1].copy()
    srf_file.header.loc[0, ["nstk", "ndip", "len", "wid"]] = [2, 1, 1.0, 0.5]
    srf_file.points = srf_file.points.iloc[:2].copy()
    srf_file.points["tinit"] = [4.0, 6.0]

    (plane,) = convert_srf_to_stoch(srf_file, 2.0, 2.0).data
    assert plane.trup.item() == pytest.approx(5.0)


def test_convert_srf_to_stoch_zero_slip_rise(synthetic_srf: SrfFile) -> None:
    """Cells with no slip get a nominal (non-zero) rise time."""
    synthetic_srf.points["slip"] = 0.0
    stoch_file = convert_srf_to_stoch(synthetic_srf, 2.0, 2.0)
    for plane in stoch_file.data:
        assert (plane.slip == 0).all()
        assert plane.rise == pytest.approx(np.full(plane.rise.shape, 1e-5))


# --- circular_mean -----------------------------------------------------------


def test_circular_mean_wraps_around_zero() -> None:
    mean = circular_mean(np.array([350.0, 10.0]), np.array([1.0, 1.0]))
    # 0 and 360 are the same bearing.
    assert min(mean, 360 - mean) == pytest.approx(0.0, abs=1e-9)


def test_circular_mean_is_weighted() -> None:
    # Three quarters of the weight sits at 0 degrees, one quarter at 90.
    expected = np.degrees(np.arctan2(0.25, 0.75))
    assert circular_mean(np.array([0.0, 90.0]), np.array([3.0, 1.0])) == (
        pytest.approx(expected)
    )


def test_average_rake_is_in_degrees(synthetic_srf: SrfFile) -> None:
    """The stoch header rake is a bearing in degrees, not radians."""
    synthetic_srf.points["rake"] = 185.0
    stoch_file = convert_srf_to_stoch(synthetic_srf, 2.0, 2.0)
    for plane in stoch_file.data:
        assert plane.header.average_rake == pytest.approx(185.0, abs=1e-3)


# --- Integration -------------------------------------------------------------


@pytest.fixture
def realisation_ffp(tmp_path: Path, synthetic_srf: SrfFile) -> Path:
    """A realisation whose sources match the planes of the synthetic SRF."""
    realisation_ffp = tmp_path / "realisation.json"
    realisations.RealisationMetadata(
        name="generate stoch test",
        version="1",
        defaults_version=defaults.DefaultsVersion.v24_2_2_1,
    ).write_to_realisation(realisation_ffp)
    realisations.SourceConfig(
        source_geometries={
            f"plane_{i}": sources.Plane.from_centroid_strike_dip(
                np.array([plane["elat"], plane["elon"]]),
                plane["dip"],
                plane["len"],
                plane["wid"],
                dtop=plane["dtop"],
                strike=plane["stk"],
            )
            for i, plane in synthetic_srf.header.iterrows()
        }
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
    assert stoch_ffp.exists()

    stoch_file = StochFile.from_file(stoch_ffp)
    assert len(stoch_file.data) == len(PLANE_SHAPES)

    srf_file = srf.read_srf(srf_ffp)
    for i, plane in enumerate(stoch_file.data):
        header = srf_file.header.iloc[i]
        # Every plane uses the configured stoch dx/dy, as the HF code
        # requires. Planes smaller than a cell round up to a single cell
        # rather than down-sampling to an empty grid.
        assert plane.header.dx == pytest.approx(2.0)
        assert plane.header.dy == pytest.approx(2.0)
        assert plane.slip.shape == (plane.header.ny, plane.header.nx)
        assert plane.header.dtop == pytest.approx(header["dtop"])
        assert plane.header.dip == pytest.approx(header["dip"])
        assert plane.header.strike == pytest.approx(header["stk"] % 360)
        assert (plane.slip >= 0).all()
        assert (plane.rise > 0).all()
        # The written file preserves the moment to the precision of the
        # %e formatting used by the stoch format.
        coarse_moment = float(plane.slip.sum()) * plane.header.dx * plane.header.dy
        assert coarse_moment == pytest.approx(fine_moment(srf_file, i), rel=1e-4)
