"""Tests for the check_srf script."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
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


def _points(slip_cm: float, **rigidity_columns: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [-43.5],
            "lon": [172.6],
            "dep": [FAULT_DEPTH_KM],
            "tinit": [0.0],
            "area": [FAULT_AREA_CM2],
            "slip": [slip_cm],
        }
        | {key: [value] for key, value in rigidity_columns.items()}
    )


def _check(
    realisation_ffp: Path, points: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        check_srf.srf,
        "read_srf",
        lambda _: SimpleNamespace(points=points, planes=[]),
    )
    check_srf.check_srf(realisation_ffp, realisation_ffp.with_suffix(".srf"))


def _slip_for_magnitude(mu_dyne_cm2: float) -> float:
    moment_dyne_cm = moment.magnitude_to_moment(MAGNITUDE, bold_m=True) * 1e7
    return moment_dyne_cm / (FAULT_AREA_CM2 * mu_dyne_cm2)


def _mu_1d() -> float:
    model = VelocityModel1D.read_from_defaults(DEFAULTS_VERSION).model
    layer = np.searchsorted(model["thickness"].cumsum(), FAULT_DEPTH_KM)
    return float(model["Vs"].iloc[layer] ** 2 * model["rho"].iloc[layer] * 1e10)


def test_negative_depth_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative SRF depth is reported with typer.Exit(1), not a KeyError."""
    realisation_ffp = tmp_path / "realisation.json"
    RealisationMetadata(
        name="Test_REL01", version="1", defaults_version=DefaultsVersion.v24_2_2_1
    ).write_to_realisation(realisation_ffp)
    points = pd.DataFrame(
        {
            "lat": [-43.5],
            "lon": [172.6],
            "dep": [-1.0],
            "tinit": [0.0],
            "area": [1e8],
            "slip": [1.0],
        }
    )
    monkeypatch.setattr(
        check_srf.srf, "read_srf", lambda _: SimpleNamespace(points=points)
    )

    with pytest.raises(typer.Exit) as exc_info:
        check_srf.check_srf(realisation_ffp, tmp_path / "realisation.srf")
    assert exc_info.value.exit_code == 1


def test_own_rigidity_matching_moment_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An SRF whose vs/den differ from the 1D model passes when its moment matches."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)
    mu = OWN_VS**2 * OWN_DEN
    assert not np.isclose(mu, _mu_1d(), rtol=0.1)

    _check(
        realisation_ffp,
        _points(_slip_for_magnitude(mu), vs=OWN_VS, den=OWN_DEN),
        monkeypatch,
    )


def test_own_rigidity_wrong_moment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An SRF with twice the realisation's moment fails under its own rigidity."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)
    mu = OWN_VS**2 * OWN_DEN

    with pytest.raises(typer.Exit) as exc_info:
        _check(
            realisation_ffp,
            _points(2 * _slip_for_magnitude(mu), vs=OWN_VS, den=OWN_DEN),
            monkeypatch,
        )
    assert exc_info.value.exit_code == 1


def test_own_rigidity_ignores_1d_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An SRF whose moment matches only under the 1D model fails if its vs/den disagree."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)

    with pytest.raises(typer.Exit) as exc_info:
        _check(
            realisation_ffp,
            _points(_slip_for_magnitude(_mu_1d()), vs=OWN_VS, den=OWN_DEN),
            monkeypatch,
        )
    assert exc_info.value.exit_code == 1


def test_srf_without_rigidity_uses_1d_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version 1.0 SRF (no vs/den) is checked against the 1D model's rigidity."""
    realisation_ffp = tmp_path / "realisation.json"
    _write_realisation(realisation_ffp)

    _check(realisation_ffp, _points(_slip_for_magnitude(_mu_1d())), monkeypatch)

    with pytest.raises(typer.Exit) as exc_info:
        _check(realisation_ffp, _points(2 * _slip_for_magnitude(_mu_1d())), monkeypatch)
    assert exc_info.value.exit_code == 1
