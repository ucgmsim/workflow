"""Tests for the nshm2022-to-realisation script."""

from pathlib import Path

import pytest

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
