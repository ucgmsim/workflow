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
import sys
from pathlib import Path
from typing import Any, Protocol, Union

import numpy as np
from schema import And, Literal, Or, Schema, Use


def to_lat_lon_dictionary(
    lat_lon_array: np.ndarray,
) -> Union[dict[str, float], list[dict[str, float]]]:
    """Convert an array of lat, lon and optionally depth values into a serialisable dictionary of lat, lon, depth dictionaries.

    Parameters
    ----------
    lat_lon_array : np.ndarray
        The array of values. Should have shape (2,), (3,), (n, 2), or (n, 3)

    Returns
    -------
    dict[str, float] or list[dict[str, float]]
        Either a dictionary with keys 'latitude', 'longitude', 'depth'
        or a list of dictionaries with the same keys. The single
        dictionary is returned only if the input is one-dimensional.
    """
    lat_lon_dicts = [
        dict(
            zip(
                ["latitude", "longitude", "depth"],
                [float(value) for value in lat_lon_array],
            )
        )
        for lat_lon_pair in np.atleast_2d(lat_lon_array)
    ]

    if len(lat_lon_array.shape) == 1:
        return lat_lon_dicts[0]

    return lat_lon_dicts


@dataclasses.dataclass
class SourceConfig:
    """
    Configuration for defining sources.

    Attributes
    ----------
    sources : dict[str, "Source"]
        Dictionary mapping source names to their definitions.
    """

    sources: dict[str, "Source"]

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
    rupture_propagation : dict[str, Any]
        Dictionary defining rupture propagation parameters for different faults.
        Each key is a fault identifier and its value is a dictionary with:
        - 'parent': The parent fault triggering this fault (or null if the initial fault).
        - 'hypocentre': The hypocentre coordinates (or initial rupture point if not the initial fault).
        - 'magnitude': The total moment magnitude for the rupture on this fault.
        - 'rake': The fault rake angle.
    """

    rupture_propagation: dict[str, Any]

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
        param_dict["centroid"] = to_lat_lon_dictionary(self.centroid)
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


# NOTE: These functions seem silly and short, however there is a good
# reason for the choice to create functions like this. The reason is
# because when the schema library reports an error (such as the input
# file having a negative depth value) it prints the name of the
# function. So
#
# And(float, lambda x: x > 0).validate(-12)
#
# Would report the error
#
# schema.SchemaError: <lambda>(-12) should evaluate to True
#
# But using the function `is_positive` we instead have
#
# And(float, is_positive).validate(-12)
# schema.SchemaError: is_positive(-12) should evaluate to True
#
# So using short functions with names improves the error reporting
# from the library.
#
# Accordingly, the most trivial of these functions lack docstrings.


def is_positive(x: float) -> bool:
    return x > 0


def is_non_negative(x: float) -> bool:
    return x >= 0


def is_valid_latitude(latitude: float) -> bool:
    return -90 <= latitude <= 90


def is_valid_longitude(longitude: float) -> bool:
    return -180 <= longitude <= 180


def is_plausible_magnitude(magnitude: float) -> bool:
    return magnitude < 11


def is_valid_degrees(degrees: float) -> bool:
    return -360 <= degrees <= 360


def is_valid_local_coordinate(coordinate: float) -> bool:
    return 0 <= coordinate <= 1


def is_valid_bearing(bearing: float) -> bool:
    return 0 <= bearing <= 360


def is_correct_corner_shape(corners: np.ndarray) -> bool:
    """Check if the corner shape matches the corner shape for plane sources.

    Parameters
    ----------
    corners : np.ndarray
        The corners to validate

    Returns
    -------
    bool
        True if the corners has the shape (4, 3) (one for each point
        and three components lat, lon and depth).
    """
    return corners.shape == (4, 3)


def is_correct_fault_corner_shape(corners: np.ndarray) -> bool:
    """Check if the corner shape matches the definition of corners for a fault (multi-plane; type-4).

    Parameters
    ----------
    corners : np.ndarray
        The corners to validate

    Returns
    -------
    bool
        True if the corners have shape (n x 4 x 3).
    """
    if len(corners.shape) != 3:
        return False
    return corners.shape[1:] == (4, 3)


def has_non_negative_depth(corners: np.ndarray) -> bool:
    """Checks the depth component of corners array is non-negative.

    Parameters
    ----------
    corners : np.ndarray
        The corners to validate.

    Returns
    -------
    bool
        Returns true if the last column of the corners is
        non-negative.
    """
    return np.all(corners[:, -1] >= 0)


def corners_to_array(corners_spec: list[dict[str, float]]) -> np.ndarray:
    """Convert a list of coordinates to a numpy array in the corner format.

    Parameters
    ----------
    corners_spec : list[dict[str, float]]
        The corners to convert.

    Returns
    -------
    np.ndarray
        An array of shape (n x 3) where columns 0, 1, and 2 correspond
        to latitude, longitude, and depth respectively.
    """
    corners_array = []
    for corner in corners_spec:
        if "depth" in corner:
            corners_array.append(
                [corner["latitude"], corner["longitude"], corner["depth"]]
            )
        else:
            corners_array.append([corner["latitude"], corner["longitude"]])
    return np.array(corners_array)


FAULT_LOCAL_COORDINATES_SCHEMA = Schema(
    And(
        {
            Literal(
                "s",
                description="The `s` coordinate (fraction of length, in range [0, 1])",
            ): And(float, is_valid_local_coordinate),
            Literal(
                "d",
                description="The `d` coordinate (fraction of width, in range [0, 1])",
            ): And(float, is_valid_local_coordinate),
        },
        Use(lambda local_coords: np.array([local_coords["s"], local_coords["d"]])),
    )
)


LAT_LON_SCHEMA = Schema(
    And(
        {
            Literal("latitude", description="Latitude (in decimal degrees)"): And(
                float, is_valid_latitude
            ),
            Literal("longitude", description="Longitude (in decimal degrees)"): And(
                float, is_valid_longitude
            ),
        },
        Use(lambda latlon: np.array([latlon["latitude"], latlon["longitude"]])),
    ),
)

LAT_LON_DEPTH_SCHEMA = Schema(
    And(
        {
            Literal("latitude", description="Latitude (in decimal degrees)"): And(
                float, is_valid_latitude
            ),
            Literal("longitude", description="Longitude (in decimal degrees)"): And(
                float, is_valid_longitude
            ),
            Literal("depth", description="Depth (in metres)"): And(
                float, is_non_negative
            ),
        },
        Use(
            lambda latlondepth: np.array(
                [
                    latlondepth["latitude"],
                    latlondepth["longitude"],
                    latlondepth["depth"],
                ]
            )
        ),
    )
)

POINT_SCHEMA = Schema(
    {
        Literal(
            "type",
            description="The type of the source geometry (Point, Plane or Fault)",
        ): "point",
        Literal(
            "coordinates", description="The coordinates of the point source"
        ): LAT_LON_DEPTH_SCHEMA,
        Literal("strike", description="The strike bearing of the point source"): And(
            float, is_valid_bearing
        ),
        Literal("dip", description="The dip angle of the point source"): And(
            float, is_valid_bearing
        ),
        Literal(
            "dip_dir", description="The dip direction bearing of the point source"
        ): And(float, is_valid_bearing),
    }
)

PLANE_SCHEMA = Schema(
    {
        Literal(
            "type",
            description="The type of the source geometry (Point, Plane or Fault)",
        ): "plane",
        Literal(
            "corners",
            description="The corners of the plane (shape 4 x 3: lat, lon, depth)",
        ): And(Use(corners_to_array), is_correct_corner_shape, has_non_negative_depth),
    }
)

FAULT_SCHEMA = Schema(
    {
        Literal(
            "type",
            description="The type of the source geometry (Point, Plane, or Fault)",
        ): "fault",
        Literal(
            "corners",
            description="The corners of the plane (shape 4n x 3: lat, lon, depth)",
        ): And(
            Use(corners_to_array),
            Use(
                lambda corners: corners.reshape(
                    (-1, 4, 3), error="Corners cannot be reshaped to (n x 4 x 3)."
                )
            ),
            has_non_negative_depth,
        ),
    }
)


POINT_SCHEMA = Schema(
    {
        Literal(
            "type",
            description="The type of the source geometry (Point, Plane or Fault)",
        ): "point",
        Literal(
            "coordinates", description="The coordinates of the point source"
        ): LAT_LON_DEPTH_SCHEMA,
        Literal("strike", description="The strike bearing of the point source"): And(
            float, is_valid_bearing
        ),
        Literal("dip", description="The dip angle of the point source"): And(
            float, is_valid_bearing
        ),
        Literal(
            "dip_dir", description="The dip direction bearing of the point source"
        ): And(float, is_valid_bearing),
    }
)

PLANE_SCHEMA = Schema(
    {
        Literal(
            "type",
            description="The type of the source geometry (Point, Plane or Fault)",
        ): "plane",
        Literal(
            "corners",
            description="The corners of the plane (shape 4 x 3: lat, lon, depth)",
        ): And(Use(corners_to_array), is_correct_corner_shape, has_non_negative_depth),
    }
)

FAULT_SCHEMA = Schema(
    {
        Literal(
            "type",
            description="The type of the source geometry (Point, Plane, or Fault)",
        ): "fault",
        Literal(
            "corners",
            description="The corners of the plane (shape 4n x 3: lat, lon, depth)",
        ): And(
            Use(corners_to_array),
            Use(
                lambda corners: corners.reshape(
                    (-1, 4, 3), error="Corners cannot be reshaped to (n x 4 x 3)."
                )
            ),
            has_non_negative_depth,
        ),
    }
)


_REALISATION_SCHEMAS = {
    SourceConfig: Schema({str: Or(POINT_SCHEMA, PLANE_SCHEMA, FAULT_SCHEMA)}),
    SRFConfig: Schema(
        {
            Literal(
                "genslip_dt",
                description="The timestep for genslip (used to specify the resolution for the `TINIT` values)",
            ): And(float, is_positive),
            Literal(
                "genslip_seed", description="The random seed passed to genslip"
            ): And(int, is_non_negative),
            Literal("genslip_version", description="The version of genslip to use"): Or(
                "5.4.2"
            ),
            Literal(
                "srfgen_seed",
                description="A second random seed for genslip (TODO: how does genslip use this value?)",
            ): And(int, is_non_negative),
        }
    ),
    DomainParameters: Schema(
        {
            Literal("resolution", description="The simulation resolution (in km)"): And(
                float, is_positive
            ),
            Literal(
                "centroid", description="The centroid location of the model"
            ): LAT_LON_SCHEMA,
            Literal("width", description="The width of the model (in km)"): And(
                float, is_positive
            ),
            Literal("length", description="The length of the model (in km)"): And(
                float, is_positive
            ),
            Literal("depth", description="The depth of the model (in km)"): And(
                float, is_positive
            ),
        }
    ),
    RupturePropagationConfig: Schema(
        {
            str: {
                Literal(
                    "parent",
                    description="The parent fault that triggers this fault (or null if the initial fault)",
                ): Or(str, None),
                Literal(
                    "hypocentre",
                    description="The hypocentre coordinates (or initial rupture point if not the initial fault)",
                ): FAULT_LOCAL_COORDINATES_SCHEMA,
                Literal(
                    "magnitude",
                    description="The total moment magnitude for the rupture on this fault",
                ): And(float, is_plausible_magnitude),
                Literal("rake", description="The fault rake"): And(
                    float, is_valid_degrees
                ),
            }
        }
    ),
    RealisationMetadata: Schema(
        {
            Literal("name", description="The name of the realisation"): str,
            Literal("version", description="The version of the realisation format"): Or(
                "5"
            ),
        }
    ),
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
