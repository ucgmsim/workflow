"""This module defines the schema for the realisation file format.

The configuration schemas are contained in `_REALISATION_SCHEMAS`. The
schemas loosely validates the input data. Input bounds checking is
done directly with the schema. More complicated input checking (for
example, that the rupture propagation defines a tree with one root
node) should be done outside this module. This is to avoid having this
module become an "everything" module.

Classes:
--------
SourceConfig:
    Configuration for defining sources.
SRFConfig:
    Configuration for SRF generation.
RupturePropagationConfig:
    Configuration for rupture propagation.
DomainParameters:
    Parameters defining the spatial domain for simulation.
RealisationMetadata:
    Metadata for describing a realisation.

Functions
---------
read_config_from_realisation
    Read a configuration object from a realisation file.
write_config_to_realisation
    Write a configuration object to a realisation file.
"""

import dataclasses
import json
from pathlib import Path
from typing import Union

import numpy as np
from schema import And, Literal, Or, Schema, Use

from source_modelling import sources
from source_modelling.rupture_propagation import JumpPair
from source_modelling.sources import IsSource


def to_name_coordinate_dictionary(
    coordinate_array: np.ndarray,
    coordinate_names: list[str] = ["latitude", "longitude", "depth"],
) -> Union[dict[str, float], list[dict[str, float]]]:
    """Convert an array of coordinates values into a (list of)
    dictionaries tagged with coordinate names.

    Parameters
    ----------
    coordinate_array : np.ndarray
        The array of values. Should have shape (k,), (m, k) where k is at most the length of `coordinate_names`.
    coordinate_names : list[str]
        The names of the coordinates. Defaults to ['latitude', 'longitude', 'depth'].

    Returns
    -------
    dict[str, float] or list[dict[str, float]]
        Either a dictionary with keys 'latitude', 'longitude', 'depth'
        or a list of dictionaries with the same keys. The single
        dictionary is returned only if the input is one-dimensional.

    Examples
    --------

    >>> to_name_coordinate_dictionary(np.array([1, 0]), coordinate_names=['s', 'd'])
    {'s': 1, 'd': 0}
    >>> to_name_coordinate_dictionary(np.array([0, 0, 1000]))
    {'latitude': 0, 'longitude': 0, 'depth': 1000}
    """
    coordinate_dicts = [
        dict(
            zip(
                coordinate_names,
                [float(value) for value in coordinate_array],
            )
        )
        for coordinate_array in np.atleast_2d(coordinate_array)
    ]

    if len(coordinate_array.shape) == 1:
        return coordinate_dicts[0]

    return coordinate_dicts


@dataclasses.dataclass
class SourceConfig:
    """
    Configuration for defining sources.

    Attributes
    ----------
    sources : dict[str, "Source"]
        Dictionary mapping source names to their definitions.
    """

    source_geometries: dict[str, IsSource]

    def to_dict(self):
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """

        config_dict = {}
        for name, geometry in self.source_geometries.items():
            if isinstance(geometry, sources.Point):
                config_dict[name] = {
                    "type": "point",
                    "coordinates": to_name_coordinate_dictionary(geometry.coordinates),
                    "length": geometry.length_m,
                    "strike": geometry.strike,
                    "dip": geometry.dip,
                    "dip_dir": geometry.dip_dir,
                }
            elif isinstance(geometry, sources.Plane):
                config_dict[name] = {
                    "type": "plane",
                    "corners": to_name_coordinate_dictionary(geometry.corners),
                }
            elif isinstance(geometry, sources.Fault):
                config_dict[name] = {
                    "type": "fault",
                    "corners": to_name_coordinate_dictionary(geometry.corners()),
                }
        return config_dict


@dataclasses.dataclass
class SRFConfig:
    """
    Configuration for SRF generation.

    Attributes
    ----------
    genslip_dt : float
        The timestep for genslip (used to specify the resolution for the `TINIT` values).
    genslip_seed : int
        The random seed passed to genslip.
    genslip_version : str
        The version of genslip to use (currently supports "5.4.2").
    srfgen_seed : int
        A second random seed for genslip, used for specific purposes in the generation process.
    """

    genslip_dt: float
    genslip_seed: int
    genslip_version: str
    srfgen_seed: int

    def to_dict(self):
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RupturePropagationConfig:
    """
    Configuration for rupture propagation.

    Attributes
    ----------
    rupture_causality_tree: dict[str, str]
        A dict where the keys are faults and the values the parent
        fault (i.e. if fault a triggers fault b then
        rupture_causality_tree[fault b] = fault a).
    jump_points: dict[str, JumpPoint]
        A map from faults to pairs of fault-local coordinates
        representing jump points. If the rupture jumps from fault a at
        point a to point b on fault b then jump_points[fault a] =
        JumpPoint(point b, point a).
    rakes: dict[str, float]
        A map from faults to rakes.
    magnitudes: dict[str, float]
        A map from faults to the magnitude of the rupture for each fault.
    """

    rupture_causality_tree: dict[str, str]
    jump_points: dict[str, JumpPair]
    rakes: dict[str, float]
    magnitudes: dict[str, float]
    hypocentre: np.ndarray

    def to_dict(self):
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        config_dict = dataclasses.asdict(self)
        config_dict["jump_points"] = {
            fault: {
                "from": to_name_coordinate_dictionary(
                    jump_point.from_point, ["s", "d"]
                ),
                "to": to_name_coordinate_dictionary(jump_point.to_point, ["s", "d"]),
            }
            for fault, jump_point in self.jump_points.items()
        }
        config_dict["hypocentre"] = to_name_coordinate_dictionary(
            self.hypocentre, ["s", "d"]
        )
        return config_dict


@dataclasses.dataclass
class DomainParameters:
    """
    Parameters defining the spatial domain for simulation.

    Attributes
    ----------
    resolution : float
        The simulation resolution in kilometers.
    centroid : np.ndarray
        The centroid location of the model in latitude and longitude coordinates.
    width : float
        The width of the model in kilometers.
    length : float
        The length of the model in kilometers.
    depth : float
        The depth of the model in kilometers.
    """

    resolution: float
    centroid: np.ndarray
    width: float
    length: float
    depth: float

    def to_dict(self):
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        param_dict = dataclasses.asdict(self)
        param_dict["centroid"] = to_name_coordinate_dictionary(self.centroid)
        return param_dict


@dataclasses.dataclass
class RealisationMetadata:
    """
    Metadata for describing a realisation.

    Attributes
    ----------
    name : str
        The name of the realisation.
    version : str
        The version of the realisation format (currently supports version "5").
    """

    name: str
    version: str

    def to_dict(self):
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        return dataclasses.asdict(self)


_REALISATION_SCHEMAS = {
    SourceConfig: schemas.SOURCE_SCHEMA,
    SRFConfig: schemas.SRF_SCHEMA,
    DomainParameters: schemas.DOMAIN_SCHEMA,
    RupturePropagationConfig: schemas.RUPTURE_PROPAGATION_SCHEMA,
    RealisationMetadata: schemas.REALISATION_METADATA_SCHEMA,
}

_REALISATION_KEYS = {
    SourceConfig: "sources",
    SRFConfig: "srf",
    DomainParameters: "domain",
    RupturePropagationConfig: "rupture_propagation",
    RealisationMetadata: "metadata",
}


LoadableConfig = Union[
    SourceConfig,
    SRFConfig,
    DomainParameters,
    RupturePropagationConfig,
    RealisationMetadata,
]


class RealisationParseError(Exception):
    pass


def read_config_from_realisation(config: type, realisation_ffp: Path) -> LoadableConfig:
    """Read configuration from a realisation file.

    Parameters
    ----------
    config : type (one of the LoadableConfig types)
        The configuration to read.
    realisation_ffp : Path
        The filepath to read from.

    Returns
    -------
    LoadableConfig
        The configuration loaded from the realisation filepath. The
        configuration schema is looked up from `_REALISATION_SCHEMAS`
        and the key within the config is specified
        `_REALISATION_KEYS`.

    Raises
    ------
    RealisationParseError
        If the key in `_REALISATION_KEYS[config]` is not present in
        the realisation filepath.
    """
    with open(realisation_ffp, "r", encoding="utf-8") as realisation_file_handle:
        realisation_config = json.load(realisation_file_handle)
        config_key = _REALISATION_KEYS[config]
        schema = _REALISATION_SCHEMAS[config]
        if config_key not in realisation_config:
            raise RealisationParseError(
                f"No '{config_key}' key in realisation configuration."
            )
        return config(**schema.validate(realisation_config[config_key]))


def write_config_to_realisation(
    config: LoadableConfig, realisation_ffp: Path, update: bool = True
) -> None:
    """Write a configuration to a realisation file.

    The default beheviour will update the realisation and replace just
    the configuration keys specified by `config`. If `update` is set
    to False, then the realisation is completely overwritten and
    populated with only the section pertaining to the config.

    Parameters
    ----------
    config : LoadableConfig
        The configuration object to write.
    realisation_ffp : Path
        The realisation filepath to write to.
    update : bool
        If True, then the realisation is updated, rather than
        replaced. Default is True.
    """
    existing_realisation_configuration = {}
    if realisation_ffp.exists() and update:
        with open(realisation_ffp, "r", encoding="utf-8") as realisation_file_handle:
            existing_realisation_configuration = json.load(realisation_file_handle)
    config_key = _REALISATION_KEYS[config.__class__]
    existing_realisation_configuration.update({config_key: config.to_dict()})
    with open(realisation_ffp, "w", encoding="utf-8") as realisation_file_handle:
        json.dump(existing_realisation_configuration, realisation_file_handle)
