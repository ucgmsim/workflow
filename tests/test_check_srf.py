"""Tests for the check-srf script."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import typer
from scipy.sparse import csr_array

from source_modelling import moment, srf
from workflow import defaults
from workflow.realisations import Magnitudes, RealisationMetadata, VelocityModel1D
from workflow.scripts import check_srf

DEFAULTS_VERSION = defaults.DefaultsVersion.v24_2_2_1

VS_KM_S = 800.0
RHO_G_CM3 = 1800.0
# Mirrors the mu = Vs**2 * rho * 1e10 formula in check_srf.py so tests can pick
# a slip that lands on a known target magnitude.
MU = VS_KM_S**2 * RHO_G_CM3 * 1e10


def _write_realisation(
    realisation_ffp: Path, magnitudes: dict[str, moment.BoldM] | None = None
) -> None:
    RealisationMetadata(
        name="test", version="1", defaults_version=DEFAULTS_VERSION
    ).write_to_realisation(realisation_ffp)
    VelocityModel1D(
        model=pd.DataFrame(
            [
                {
                    "thickness": 100.0,
                    "Vp": 3000.0,
                    "Vs": VS_KM_S,
                    "rho": RHO_G_CM3,
                    "Qp": 500.0,
                    "Qs": 250.0,
                }
            ]
        )
    ).write_to_realisation(realisation_ffp)
    if magnitudes is not None:
        Magnitudes(magnitudes=magnitudes).write_to_realisation(realisation_ffp)


def _write_srf(
    srf_ffp: Path,
    lon: float = 172.0,
    lat: float = -43.5,
    dep: float = 1.0,
    tinit: float = 0.0,
    area: float = 1.0,
    slip: float = 1.0,
    wid: float = 10.0,
) -> None:
    """Write a single-plane, single-point SRF with the given point values."""
    header = pd.DataFrame(
        [
            {
                "elon": lon,
                "elat": lat,
                "nstk": 1,
                "ndip": 1,
                "len": 10.0,
                "wid": wid,
                "stk": 0.0,
                "dip": 90.0,
                "dtop": 0.0,
                "shyp": 0.0,
                "dhyp": 0.0,
            }
        ]
    )
    points = pd.DataFrame(
        [
            {
                "lon": lon,
                "lat": lat,
                "dep": dep,
                "stk": 0.0,
                "dip": 90.0,
                "area": area,
                "tinit": tinit,
                "dt": 0.1,
                "rake": 45.0,
                "slip": slip,
                "rise": 0.1,
            }
        ]
    )
    srf_file = srf.SrfFile(
        version="1.0",
        header=header,
        points=points,
        slipt1_array=csr_array(np.zeros((1, 1), dtype=np.float32)),
    )
    srf_file.write_srf(srf_ffp)


def _slip_for_magnitude(magnitude: float, area: float = 1.0) -> float:
    """The slip (cm) that, with MU and area, produces the given BoldM magnitude."""
    target_moment = moment.magnitude_to_moment(moment.BoldM(magnitude), bold_m=True)
    return target_moment / (area * MU)


def test_check_srf_rejects_nan_points(tmp_path: Path) -> None:
    realisation_ffp = tmp_path / "realisation.json"
    srf_ffp = tmp_path / "test.srf"
    _write_realisation(realisation_ffp)
    _write_srf(srf_ffp, slip=float("nan"))

    with pytest.raises(typer.Exit) as exc_info:
        check_srf.check_srf(realisation_ffp, srf_ffp)
    assert exc_info.value.exit_code == 1


def test_check_srf_rejects_rupture_not_starting_at_zero(tmp_path: Path) -> None:
    realisation_ffp = tmp_path / "realisation.json"
    srf_ffp = tmp_path / "test.srf"
    _write_realisation(realisation_ffp)
    _write_srf(srf_ffp, tinit=0.5)

    with pytest.raises(typer.Exit) as exc_info:
        check_srf.check_srf(realisation_ffp, srf_ffp)
    assert exc_info.value.exit_code == 1


def test_check_srf_rejects_out_of_bounds_latitude(tmp_path: Path) -> None:
    realisation_ffp = tmp_path / "realisation.json"
    srf_ffp = tmp_path / "test.srf"
    _write_realisation(realisation_ffp)
    _write_srf(srf_ffp, lat=95.0)

    with pytest.raises(typer.Exit) as exc_info:
        check_srf.check_srf(realisation_ffp, srf_ffp)
    assert exc_info.value.exit_code == 1


def test_check_srf_rejects_out_of_bounds_longitude(tmp_path: Path) -> None:
    realisation_ffp = tmp_path / "realisation.json"
    srf_ffp = tmp_path / "test.srf"
    _write_realisation(realisation_ffp)
    _write_srf(srf_ffp, lon=185.0)

    with pytest.raises(typer.Exit) as exc_info:
        check_srf.check_srf(realisation_ffp, srf_ffp)
    assert exc_info.value.exit_code == 1


def test_check_srf_rejects_broken_geometry(tmp_path: Path) -> None:
    """A zero-width plane collapses to a line, which is not a valid plane."""
    realisation_ffp = tmp_path / "realisation.json"
    srf_ffp = tmp_path / "test.srf"
    _write_realisation(realisation_ffp)
    _write_srf(srf_ffp, wid=0.0)

    with pytest.raises(typer.Exit) as exc_info:
        check_srf.check_srf(realisation_ffp, srf_ffp)
    assert exc_info.value.exit_code == 1


def test_check_srf_rejects_magnitude_mismatch(tmp_path: Path) -> None:
    realisation_ffp = tmp_path / "realisation.json"
    srf_ffp = tmp_path / "test.srf"
    _write_realisation(realisation_ffp, magnitudes={"fault": moment.BoldM(4.0)})
    _write_srf(srf_ffp, slip=_slip_for_magnitude(6.0))

    with pytest.raises(typer.Exit) as exc_info:
        check_srf.check_srf(realisation_ffp, srf_ffp)
    assert exc_info.value.exit_code == 1


def test_check_srf_accepts_matching_magnitude(tmp_path: Path) -> None:
    realisation_ffp = tmp_path / "realisation.json"
    srf_ffp = tmp_path / "test.srf"
    _write_realisation(realisation_ffp, magnitudes={"fault": moment.BoldM(6.0)})
    _write_srf(srf_ffp, slip=_slip_for_magnitude(6.0))

    check_srf.check_srf(realisation_ffp, srf_ffp)

    with open(realisation_ffp) as realisation_handle:
        assert "log_trail" in json.load(realisation_handle)


def test_check_srf_succeeds_without_magnitudes_config(tmp_path: Path) -> None:
    """Realisations need not declare fault magnitudes; the check is skipped."""
    realisation_ffp = tmp_path / "realisation.json"
    srf_ffp = tmp_path / "test.srf"
    _write_realisation(realisation_ffp)
    _write_srf(srf_ffp, slip=_slip_for_magnitude(6.0))

    check_srf.check_srf(realisation_ffp, srf_ffp)

    with open(realisation_ffp) as realisation_handle:
        assert "log_trail" in json.load(realisation_handle)
