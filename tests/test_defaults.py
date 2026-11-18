"""Tests for workflow.defaults."""

from importlib import resources
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from workflow import defaults

# Regression tests for ucgmsim/workflow#285 use the constants below. The
# 26.7.1Hz defaults lost the `- name: "developer"` line, so the developer
# command's `parameters` became a second `parameters` key of the attenuation
# command. YAML kept the last one: attenuation ran with `{cfl, reporttiming}`
# and there was no developer command.

SW4_VERSIONS = [
    defaults.DefaultsVersion.v26_7_0_25Hz,
    defaults.DefaultsVersion.v26_7_0_5Hz,
    defaults.DefaultsVersion.v26_7_1Hz,
]

# The options SW4's `attenuation` command accepts.
SW4_ATTENUATION_OPTIONS = {
    "maxfreq",
    "minppw",
    "phasefreq",
    "nmech",
    "bandwidth",
    "qmultiplier",
}

# The 26.7.1Hz defaults as they were on pegasus when #285 was filed.
ISSUE_285_YAML = """
sw4:
  commands:
    - name: "attenuation"
      parameters: { maxfreq: 2.5, phasefreq: 0.5, nmech: 3 }
    # No `failonnan`: it scans seven full arrays for NaN every step, a tenth
      # of the run time. SW4's own health check still aborts on NaN, Inf or a
      # runaway solution every `printcycle` steps.
      parameters: { cfl: 1.3, reporttiming: true }
"""


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


def _sw4_commands(version: defaults.DefaultsVersion, name: str) -> list[dict[str, Any]]:
    commands = defaults.load_defaults(version)["sw4"]["commands"]
    return [command for command in commands if command["name"] == name]


def test_unique_key_loader_rejects_issue_285_yaml() -> None:
    # Guard against the test passing vacuously: the stock loader accepts this
    # document and silently keeps the developer command's parameters.
    silently_merged = yaml.safe_load(ISSUE_285_YAML)
    assert silently_merged["sw4"]["commands"] == [
        {"name": "attenuation", "parameters": {"cfl": 1.3, "reporttiming": True}}
    ]

    with pytest.raises(ValueError, match="Duplicate 'parameters' key"):
        yaml.load(ISSUE_285_YAML, defaults.UniqueKeyLoader)


def test_unique_key_loader_allows_merge_key_overrides() -> None:
    document = """
base: &base { maxfreq: 2.5, nmech: 3 }
derived:
  <<: *base
  nmech: 2
"""
    loaded = yaml.load(document, defaults.UniqueKeyLoader)
    assert loaded["derived"] == {"maxfreq": 2.5, "nmech": 2}


@pytest.mark.parametrize("version", list(defaults.DefaultsVersion))
def test_load_defaults_has_no_duplicate_keys(
    version: defaults.DefaultsVersion,
) -> None:
    # load_defaults raises ValueError on a duplicate key in either file.
    assert defaults.load_defaults(version)


@pytest.mark.parametrize("version", SW4_VERSIONS)
def test_sw4_attenuation_command_has_attenuation_parameters(
    version: defaults.DefaultsVersion,
) -> None:
    (attenuation,) = _sw4_commands(version, "attenuation")
    parameters = set(attenuation["parameters"])
    assert parameters <= SW4_ATTENUATION_OPTIONS
    assert {"maxfreq", "phasefreq", "nmech"} <= parameters


@pytest.mark.parametrize("version", SW4_VERSIONS)
def test_sw4_developer_command_sets_cfl(version: defaults.DefaultsVersion) -> None:
    (developer,) = _sw4_commands(version, "developer")
    assert developer["parameters"]["cfl"] == 1.3
    assert developer["parameters"]["reporttiming"] is True
