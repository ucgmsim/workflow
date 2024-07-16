import importlib
from importlib import resources

import yaml


def load_emod3d_defaults(version: str) -> dict[str, int | float | str]:
    emod3d_defaults_package = importlib.import_module(
        f'workflow.default_parameters.v{version.replace('.', '_')}'
    )
    emod3d_defaults_path = (
        resources.files(emod3d_defaults_package) / "emod3d_defaults.yaml"
    )
    with emod3d_defaults_path.open(encoding="utf-8") as emod3d_defaults_file_handle:
        return yaml.safe_load(emod3d_defaults_file_handle)
