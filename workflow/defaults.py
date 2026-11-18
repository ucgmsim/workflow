"""Functions to load default parameters for EMOD-3D simulations."""

import importlib
from collections.abc import Hashable
from enum import StrEnum
from importlib import resources
from typing import Any

import yaml

from workflow import utils
from workflow.default_parameters import root

# The following code is poached from: wangshu1331 on GH. See:
# https://gist.github.com/pypt/94d747fe5180851196eb?permalink_comment_id=4653474#gistcomment-4653474
# This code is added to inject additional validation for duplicate keys (and in
# future, other parsing validation) to address issues like ucgmsim/workflow#285.


class UniqueKeyLoader(yaml.SafeLoader):
    """Subclassed YAML loader that throws a ValueError when encountering duplicate keys."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        """Construct a mapping from a YAML node, rejecting duplicate keys.

        Parameters
        ----------
        node : yaml.MappingNode
            The mapping node to construct.
        deep : bool, optional
            If True, construct nested objects eagerly. Defaults to False.

        Returns
        -------
        dict
            The constructed mapping.

        Raises
        ------
        ValueError
            If the same key appears more than once in the mapping. Merge
            keys (``<<``) are exempt, so a mapping may still override keys
            it merges in.
        """
        mapping: set[Hashable] = set()
        for key_node, _ in node.value:
            if ":merge" in key_node.tag:
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


class DefaultsVersion(StrEnum):
    """Enum of versions that can be loaded by load_defaults."""

    v24_2_2_1 = "24.2.2.1"
    v24_2_2_2 = "24.2.2.2"
    v24_2_2_4 = "24.2.2.4"
    v26_7_0_25Hz = "26.7.0.25Hz"  # noqa: N815 - mirrors the version string
    v26_7_0_5Hz = "26.7.0.5Hz"  # noqa: N815 - mirrors the version string
    v26_7_1Hz = "26.7.1Hz"  # noqa: N815 - mirrors the version string


def load_defaults(version: DefaultsVersion) -> dict[str, dict[str, Any]]:
    """Load default parameters for EMOD3D simulation from a YAML file.

    Parameters
    ----------
    version : DefaultsVersion
        Version of the default parameters to load.

    Returns
    -------
    dict
        The default parameters loaded from the YAML files, one section per
        realisation configuration (e.g. ``"nzcvm"``, ``"velocity_model"``),
        keyed by the configuration's name. Each section maps parameter
        names to values, which may themselves be nested lists and mappings.
        Version-specific values override the root defaults shared by all
        versions.

    Raises
    ------
    ValueError
        If either defaults file contains a duplicate key.
    """
    defaults_package = importlib.import_module(
        f"workflow.default_parameters.v{version.value.replace('.', '_')}"
    )
    root_defaults_path = resources.files(root) / "defaults.yaml"

    with root_defaults_path.open(encoding="utf-8") as root_defaults_handle:
        root_defaults = yaml.load(root_defaults_handle, UniqueKeyLoader)

    defaults_path = resources.files(defaults_package) / "defaults.yaml"
    with defaults_path.open(encoding="utf-8") as emod3d_defaults_file_handle:
        defaults = yaml.load(emod3d_defaults_file_handle, UniqueKeyLoader)
    utils.merge_dictionaries(root_defaults, defaults)
    return root_defaults
