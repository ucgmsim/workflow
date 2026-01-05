"""Functions to load default parameters for EMOD-3D simulations."""

import importlib
from enum import StrEnum
from importlib import resources
from typing import Any

import yaml

import workflow.default_parameters.root as root


class DefaultsVersion(StrEnum):
    """Enum of versions that can be loaded by load_defaults."""

    v24_2_2_1 = "24.2.2.1"
    v24_2_2_2 = "24.2.2.2"
    v24_2_2_4 = "24.2.2.4"
    develop = "develop"


def _merge_defaults(defaults_a: dict[str, Any], defaults_b: dict[str, Any]) -> None:
    """Deep merge python dictionaries preferring the values of the second argument.

    Parameters
    ----------
    defaults_a : dict[str, Any]
        The first dictionary to merge from.
    defaults_b : dict[str, Any]
        The second dictionary to merge from. Keys in this dictionary
        are preferred.
    """

    for key, value in defaults_b.items():
        if (
            key in defaults_a
            and isinstance(defaults_a[key], dict)
            and isinstance(value, dict)
        ):
            _merge_defaults(defaults_a[key], defaults_b[key])
        else:
            defaults_a[key] = value


def load_defaults(version: DefaultsVersion) -> dict[str, int | float | str]:
    """Load default parameters for EMOD3D simulation from a YAML file.

    Parameters
    ----------
    version : str
        Version number of the EMOD3D parameters to load. This should be in the format 'YY.M.D.V'.

    Returns
    -------
    dict
        A dictionary containing the default parameters loaded from the YAML file.
        The keys are strings representing parameter names, and the values can be
        integers, floats, or strings depending on the parameter.
    """
    if version == DefaultsVersion.develop:
        defaults_package = importlib.import_module(
            "workflow.default_parameters.develop"
        )
    else:
        defaults_package = importlib.import_module(
            f"workflow.default_parameters.v{version.value.replace('.', '_')}"
        )
    root_defaults_path = resources.files(root) / "defaults.yaml"
    with root_defaults_path.open(encoding="utf-8") as root_defaults_handle:
        root_defaults = yaml.safe_load(root_defaults_handle)
    defaults_path = resources.files(defaults_package) / "defaults.yaml"
    with defaults_path.open(encoding="utf-8") as emod3d_defaults_file_handle:
        defaults = yaml.safe_load(emod3d_defaults_file_handle)
    _merge_defaults(root_defaults, defaults)
    return root_defaults
