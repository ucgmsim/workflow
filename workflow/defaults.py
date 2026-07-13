"""Functions to load default parameters for EMOD-3D simulations."""

import importlib
from enum import StrEnum
from importlib import resources

import yaml

import workflow.default_parameters.root as root
from workflow import utils


class DefaultsVersion(StrEnum):
    """Enum of versions that can be loaded by load_defaults."""

    v24_2_2_1 = "24.2.2.1"
    v24_2_2_2 = "24.2.2.2"
    v24_2_2_4 = "24.2.2.4"
    v26_7_1Hz = "26.7.1Hz"


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
    defaults_package = importlib.import_module(
        f"workflow.default_parameters.v{version.value.replace('.', '_')}"
    )
    root_defaults_path = resources.files(root) / "defaults.yaml"
    with root_defaults_path.open(encoding="utf-8") as root_defaults_handle:
        root_defaults = yaml.safe_load(root_defaults_handle)
    defaults_path = resources.files(defaults_package) / "defaults.yaml"
    with defaults_path.open(encoding="utf-8") as emod3d_defaults_file_handle:
        defaults = yaml.safe_load(emod3d_defaults_file_handle)
    utils.merge_dictionaries(root_defaults, defaults)
    return root_defaults
