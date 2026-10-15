"""Tests for workflow.defaults."""

from importlib import resources
from typing import Any
from unittest.mock import patch

from workflow import defaults


class _RecordingTraversable:
    """Wraps a resources.Traversable, recording the encoding used to open it."""

    def __init__(self, wrapped: Any, calls: list[str | None]) -> None:
        self._wrapped = wrapped
        self._calls = calls

    def __truediv__(self, name: str) -> "_RecordingTraversable":
        return _RecordingTraversable(self._wrapped / name, self._calls)

    def open(self, *args: Any, **kwargs: Any) -> Any:
        self._calls.append(kwargs.get("encoding"))
        return self._wrapped.open(*args, **kwargs)


def test_load_defaults_opens_yaml_files_with_utf8_encoding() -> None:
    calls: list[str | None] = []
    real_files = resources.files

    def fake_files(package: Any) -> _RecordingTraversable:
        return _RecordingTraversable(real_files(package), calls)

    with patch("workflow.defaults.resources.files", side_effect=fake_files):
        defaults.load_defaults(defaults.DefaultsVersion.v24_2_2_1)

    # Both the root defaults file and the version-specific defaults file must
    # be opened with an explicit UTF-8 encoding, rather than relying on the
    # platform's locale-dependent default encoding (encoding=None).
    assert len(calls) == 2
    for encoding in calls:
        assert encoding is not None
        assert encoding.casefold() == "utf-8"
