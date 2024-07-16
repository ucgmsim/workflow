"""Functions to load default parameters for EMOD-3D simulations."""

import importlib
from importlib import resources

import yaml


def load_emod3d_defaults(version: str) -> dict[str, int | float | str]:
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
    emod3d_defaults_package = importlib.import_module(
        f'workflow.default_parameters.v{version.replace('.', '_')}'
    )
    emod3d_defaults_path = (
        resources.files(emod3d_defaults_package) / "emod3d_defaults.yaml"
    )
    with emod3d_defaults_path.open(encoding="utf-8") as emod3d_defaults_file_handle:
        return yaml.safe_load(emod3d_defaults_file_handle)
