"""Tests for the nshm2022-to-realisation script."""

from pathlib import Path

import numpy as np
import pytest

from source_modelling.sources import Fault, Plane
from workflow import defaults
from workflow.realisations import RealisationMetadata
from workflow.scripts import nshm2022_to_realisation


class _DatabaseReachedError(Exception):
    """Raised by the mocked database to stop generation once it is reached."""


def test_creates_missing_realisation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realisation's parent directory is created before the first write."""

    def stop(*_args: object, **_kwargs: object) -> None:
        raise _DatabaseReachedError

    monkeypatch.setattr(nshm2022_to_realisation.nshmdb, "NSHMDB", stop)
    realisation_ffp = tmp_path / "new_dir" / "realisation.json"

    with pytest.raises(_DatabaseReachedError):
        nshm2022_to_realisation.generate_realisation(
            tmp_path / "db.db",
            1,
            realisation_ffp,
            defaults.DefaultsVersion.v26_7_1Hz,
        )

    metadata = RealisationMetadata.read_from_realisation(realisation_ffp)
    assert metadata.name == "Rupture 1"


_X0, _Y0 = 1_570_000.0, 5_180_000.0
_TRACE = np.array([[_Y0, _X0], [_Y0 + 10_000, _X0]])


def _vertical_fault() -> Fault:
    return Fault([Plane.from_nztm_trace(_TRACE, 0, 15_000, 90, dip_dir_nztm=0)])


def _dipping_fault() -> Fault:
    return Fault([Plane.from_nztm_trace(_TRACE, 0, 15_000, 60, dip_dir_nztm=90)])


def test_find_hypocentre_on_vertical_fault_with_dhypo() -> None:
    """A lat/lon on a vertical fault is located when dhypo supplies the depth."""
    fault = _vertical_fault()
    lat, lon, _ = fault.fault_coordinates_to_wgs_depth_coordinates(np.array([0.5, 0.0]))
    fault_name, hypocentre = nshm2022_to_realisation.find_fault_and_hypocentre(
        {"vertical": fault}, lat, lon, dhypo=0.3
    )
    assert fault_name == "vertical"
    assert hypocentre == pytest.approx([0.5, 0.3], abs=1e-6)


def test_find_hypocentre_on_vertical_fault_without_dhypo() -> None:
    """Without dhypo, a lat/lon on a vertical fault asks for a depth."""
    fault = _vertical_fault()
    lat, lon, _ = fault.fault_coordinates_to_wgs_depth_coordinates(np.array([0.5, 0.0]))
    with pytest.raises(ValueError, match="dhypo"):
        nshm2022_to_realisation.find_fault_and_hypocentre({"vertical": fault}, lat, lon)


def test_find_hypocentre_skips_vertical_fault_not_containing_point() -> None:
    """A vertical fault the point is not on does not trigger the depth error."""
    other = Fault(
        [
            Plane.from_nztm_trace(
                _TRACE + np.array([0.0, 50_000.0]), 0, 15_000, 90, dip_dir_nztm=0
            )
        ]
    )
    dipping = _dipping_fault()
    lat, lon, _ = dipping.fault_coordinates_to_wgs_depth_coordinates(
        np.array([0.5, 0.0])
    )
    fault_name, hypocentre = nshm2022_to_realisation.find_fault_and_hypocentre(
        {"other": other, "dipping": dipping}, lat, lon
    )
    assert fault_name == "dipping"
    assert hypocentre == pytest.approx([0.5, 0.0], abs=1e-4)


def test_find_hypocentre_on_dipping_fault() -> None:
    """Dipping faults are located from lat/lon alone, as before."""
    fault = _dipping_fault()
    lat, lon, _ = fault.fault_coordinates_to_wgs_depth_coordinates(np.array([0.5, 0.0]))
    fault_name, hypocentre = nshm2022_to_realisation.find_fault_and_hypocentre(
        {"dipping": fault}, lat, lon
    )
    assert fault_name == "dipping"
    assert hypocentre == pytest.approx([0.5, 0.0], abs=1e-4)


def test_find_hypocentre_not_on_any_fault() -> None:
    """A point on no fault still reports that it is not on any fault."""
    with pytest.raises(ValueError, match="not on any fault"):
        nshm2022_to_realisation.find_fault_and_hypocentre(
            {"vertical": _vertical_fault(), "dipping": _dipping_fault()},
            -40.0,
            170.0,
        )
