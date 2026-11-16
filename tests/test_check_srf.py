"""Tests for the check_srf script."""

from pathlib import Path

import numpy as np
import pytest
import srf
import typer

from source_modelling import moment
from source_modelling.magnitude_scaling import BoldM
from workflow.defaults import DefaultsVersion
from workflow.realisations import Magnitudes, RealisationMetadata, VelocityModel1D
from workflow.scripts import check_srf

DEFAULTS_VERSION = DefaultsVersion.v26_7_1Hz
FAULT_AREA_CM2 = 1e12  # 10 km x 10 km
FAULT_DEPTH_KM = 10.0
MAGNITUDE = BoldM(6.0)
# A shear speed (cm/s) and density (g/cm^3) away from the 1D model's, as in
# the NZCVM medium that generate-rupture writes into an SRF.
OWN_VS = 1.5e5
OWN_DEN = 2.1


def _write_realisation(realisation_ffp: Path) -> None:
    RealisationMetadata(
        name="Test_REL01", version="1", defaults_version=DEFAULTS_VERSION
    ).write_to_realisation(realisation_ffp)
    Magnitudes(magnitudes={"fault": MAGNITUDE}).write_to_realisation(realisation_ffp)


def _points(slip_cm: float, **columns: float) -> srf.Points:
    """One point holding the whole fault, version 2.0 if given ``vs`` and ``density``."""
    values = {
        "lon": 172.6,
        "lat": -43.5,
        "dep": FAULT_DEPTH_KM,
        "stk": 0.0,
        "dip": 90.0,
        "area": FAULT_AREA_CM2,
        "tinit": 0.0,
        "dt": 0.1,
        "rake": 0.0,
        "slip1": slip_cm,
    } | columns
    return srf.Points(
        **{name: np.array([value], dtype=np.float32) for name, value in values.items()}
    )


def _check(realisation_ffp: Path, points: srf.Points) -> None:
    """Write the point as a one-plane SRF and check it."""
    srf_ffp = realisation_ffp.with_suffix(".srf")
    plane = srf.Plane(
        elon=172.6,
        elat=-43.5,
        nstk=1,
        ndip=1,
        len=10.0,
        wid=10.0,
        stk=0.0,
        dip=90.0,
        dtop=FAULT_DEPTH_KM - 5.0,
        shyp=0.0,
        dhyp=5.0,
    )
    srf.write(
        srf_ffp,
        srf.Srf(
            planes=[plane],
            points=points,
            slip_rate=srf.SlipRate.from_pulses([[0.0, 1.0, 0.0]]),
        ),
    )
    check_srf.check_srf(realisation_ffp, srf_ffp)


def _slip_for_magnitude(mu_dyne_cm2: float) -> float:
    moment_dyne_cm = moment.magnitude_to_moment(MAGNITUDE, bold_m=True) * 1e7
    return moment_dyne_cm / (FAULT_AREA_CM2 * mu_dyne_cm2)


def _mu_1d() -> float:
    model = VelocityModel1D.read_from_defaults(DEFAULTS_VERSION).model
    layer = np.searchsorted(model["thickness"].cumsum(), FAULT_DEPTH_KM)
    return float(model["Vs"].iloc[layer] ** 2 * model["rho"].iloc[layer] * 1e10)


def test_negative_depth_exits_cleanly(tmp_path: Path) -> None:
    """A negative SRF depth is reported with typer.Exit(1), not a KeyError."""
    realisation_ffp = tmp_path / "realisation.json"
    RealisationMetadata(
        name="Test_REL01", version="1", defaults_version=DefaultsVersion.v24_2_2_1
    ).write_to_realisation(realisation_ffp)

    with pytest.raises(typer.Exit) as exc_info:
        _check(realisation_ffp, _points(1.0, dep=-1.0))
    assert exc_info.value.exit_code == 1


def test_own_rigidity_matching_moment_passes(tmp_path: Path) -> None:
    """An SRF whose vs/density differ from the 1D model passes when its moment matches."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)
    mu = OWN_VS**2 * OWN_DEN
    assert not np.isclose(mu, _mu_1d(), rtol=0.1)

    _check(
        realisation_ffp,
        _points(_slip_for_magnitude(mu), vs=OWN_VS, density=OWN_DEN),
    )


def test_own_rigidity_wrong_moment_fails(tmp_path: Path) -> None:
    """An SRF with twice the realisation's moment fails under its own rigidity."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)
    mu = OWN_VS**2 * OWN_DEN

    with pytest.raises(typer.Exit) as exc_info:
        _check(
            realisation_ffp,
            _points(2 * _slip_for_magnitude(mu), vs=OWN_VS, density=OWN_DEN),
        )
    assert exc_info.value.exit_code == 1


def test_own_rigidity_ignores_1d_model(tmp_path: Path) -> None:
    """An SRF whose moment matches only under the 1D model fails if its vs/density disagree."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)

    with pytest.raises(typer.Exit) as exc_info:
        _check(
            realisation_ffp,
            _points(_slip_for_magnitude(_mu_1d()), vs=OWN_VS, density=OWN_DEN),
        )
    assert exc_info.value.exit_code == 1


def test_srf_without_rigidity_uses_1d_model(tmp_path: Path) -> None:
    """A version 1.0 SRF (no vs/density) is checked against the 1D model's rigidity."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)

    _check(realisation_ffp, _points(_slip_for_magnitude(_mu_1d())))

    with pytest.raises(typer.Exit) as exc_info:
        _check(realisation_ffp, _points(2 * _slip_for_magnitude(_mu_1d())))
    assert exc_info.value.exit_code == 1


def test_nan_point_column_fails(tmp_path: Path) -> None:
    """A NaN in any point column fails the check."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)

    with pytest.raises(typer.Exit) as exc_info:
        _check(realisation_ffp, _points(_slip_for_magnitude(_mu_1d()), rake=np.nan))
    assert exc_info.value.exit_code == 1


def test_non_planar_srf_passes(tmp_path: Path) -> None:
    """An SRF's points need not lie on its header's plane."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)
    mu = OWN_VS**2 * OWN_DEN
    # Four patches, each a quarter of the fault, at depths no plane fits.
    columns = {
        "lon": [172.6, 172.65, 172.6, 172.65],
        "lat": [-43.5, -43.5, -43.45, -43.45],
        "dep": [5.0, 9.0, 12.0, 6.0],
        "area": [FAULT_AREA_CM2 / 4] * 4,
        "slip1": [_slip_for_magnitude(mu)] * 4,
        "vs": [OWN_VS] * 4,
        "density": [OWN_DEN] * 4,
    }
    constants = {"stk": 0.0, "dip": 90.0, "tinit": 0.0, "dt": 0.1, "rake": 0.0}
    columns |= {name: [value] * 4 for name, value in constants.items()}
    points = srf.Points(
        **{name: np.array(column, dtype=np.float32) for name, column in columns.items()}
    )
    plane = srf.Plane(
        elon=172.625,
        elat=-43.475,
        nstk=2,
        ndip=2,
        len=10.0,
        wid=10.0,
        stk=0.0,
        dip=90.0,
        dtop=5.0,
        shyp=0.0,
        dhyp=5.0,
    )
    srf_ffp = tmp_path / "realisation.srf"
    srf.write(
        srf_ffp,
        srf.Srf(
            planes=[plane],
            points=points,
            slip_rate=srf.SlipRate.from_pulses([[0.0, 1.0, 0.0]] * 4),
        ),
    )

    check_srf.check_srf(realisation_ffp, srf_ffp)
