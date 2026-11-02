"""Tests for the check_srf script."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import typer

from workflow.defaults import DefaultsVersion
from workflow.realisations import RealisationMetadata
from workflow.scripts import check_srf


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
