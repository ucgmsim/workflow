"""This module defines the schema for the realisation file format.

The root level schema is contained in `REALISATION_SCHEMA`. The schema
loosely validates the input data and returns a Realisation object.
Input bounds checking is done directly with the schema. More
complicated input checking (for example, that the rupture propagation
defines a tree with one root node) should be done by the objects that
get passed this data. This is to avoid having this module become an
"everything" module.

Classes
-------
Realisation
    Object that holds all the realisation data.

Functions
---------
read_realisation_file:
    Read a realisation JSON file from a given filepath.
"""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from schema import And, Literal, Or, Schema, Use


@dataclasses.dataclass
class Realisation:
    """Object that holds all the realisation data."""

    name: str
    version: str
    # TODO: Replace with proper types
    sources: dict[str, Any]
    srf_generation_parameters: Any
    domain_parameters: Any
    rupture_propagation: Any

    @staticmethod
    def from_realisation_spec(realisation_spec: dict) -> "Realisation":
        """Convert a specification of a realisation into a realisation object.

        The output of REALISATION_SCHEMA.validate is formally called a
        realisation specification. The specification is simply a dict
        of validated schema properties. This function converts the
        dictionary into a Realisation object.

        Parameters
        ----------
        realisation_spec : dict
            The realisation specification to convert.

        Returns
        -------
        Realisation
            The converted specification.
        """
        return Realisation(**realisation_spec)


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

RUPTURE_PROPAGATION_SCHEMA = Schema(
    {
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
        Literal("rake", description="The fault rake"): And(float, is_valid_degrees),
    }
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

SOURCE_SCHEMA = Schema({str: Or(POINT_SCHEMA, PLANE_SCHEMA, FAULT_SCHEMA)})


SRF_GEN_SCHEMA = Schema(
    {
        Literal(
            "genslip_dt",
            description="The timestep for genslip (used to specify the resolution for the `TINIT` values)",
        ): And(float, is_positive),
        Literal("genslip_seed", description="The random seed passed to genslip"): And(
            int, is_non_negative
        ),
        Literal("genslip_version", description="The version of genslip to use"): Or(
            "5.4.2"
        ),
        Literal(
            "srfgen_seed",
            description="A second random seed for genslip (TODO: how does genslip use this value?)",
        ): And(int, is_non_negative),
    }
)

DOMAIN_PARAMETER_SCHEMA = Schema(
    {
        Literal("resolution", description="The simulation resolution (in km)"): And(
            float, is_positive
        ),
        Literal(
            "model_centroid", description="The centroid location of the model"
        ): LAT_LON_SCHEMA,
        Literal("model_width", description="The width of the model (in km)"): And(
            float, is_positive
        ),
        Literal("model_length", description="The length of the model (in km)"): And(
            float, is_positive
        ),
        Literal("model_depth", description="The depth of the model (in km)"): And(
            float, is_positive
        ),
    }
)

REALISATION_SCHEMA = Schema(
    And(
        {
            Literal("name", description="The name of the realisation"): str,
            Literal("version", description="The version of the realisation format"): Or(
                "5"
            ),
            Literal(
                "sources", description="The sources involved in the realisation"
            ): SOURCE_SCHEMA,
            Literal(
                "srf_generation_parameters",
                description="The parameters for SRF generation",
            ): SRF_GEN_SCHEMA,
            Literal(
                "domain_parameters",
                description="The parameters defining the simulation domain boundaries and resolution.",
            ): DOMAIN_PARAMETER_SCHEMA,
            Literal(
                "rupture_propagation",
                description="Information about how the rupture will propagate across the involved fault(s)",
            ): {str: RUPTURE_PROPAGATION_SCHEMA},
        },
        Use(Realisation.from_realisation_spec),
    ),
    description="Realisation Schema",
)


def read_realisation_file(filepath: Path) -> Realisation:
    """Read a realisation from a JSON file.

    Parameters
    ----------
    filepath : Path
        The file path to the realisation JSON.

    Returns
    -------
    Realisation
        The realisation object read from the JSON file path.
    """
    with open(filepath, "r", encoding="utf-8") as realisation_file_handle:
        return REALISATION_SCHEMA.validate(json.load(realisation_file_handle))
