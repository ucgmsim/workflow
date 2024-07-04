"""Module containing schema definitions for the realisation specification.

See the repository wiki page (specifically the Realisations page, and
the Realisations Proposal page) for a description of realisations and
the schemas.
"""

import numpy as np
from schema import And, Literal, Optional, Or, Schema, Use

from source_modelling import sources, rupture_propagation

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
    """Check the depth component of corners array is non-negative.

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
    And(
        {
            Literal(
                "type",
                description="The type of the source geometry (Point, Plane or Fault)",
            ): "point",
            Literal(
                "coordinates", description="The coordinates of the point source"
            ): LAT_LON_DEPTH_SCHEMA,
            Literal("length", description="The pseudo-length of the point source"): And(
                float, is_positive
            ),
            Literal(
                "strike", description="The strike bearing of the point source"
            ): And(float, is_valid_bearing),
            Literal("dip", description="The dip angle of the point source"): And(
                float, is_valid_bearing
            ),
            Literal(
                "dip_dir", description="The dip direction bearing of the point source"
            ): And(float, is_valid_bearing),
        },
        Use(
            lambda schema: sources.Point.from_lat_lon_depth(
                schema["coordinates"],
                length_m=schema["length"],
                strike=schema["strike"],
                dip=schema["dip"],
                dip_dir=schema["dip_dir"],
            )
        ),
    )
)

PLANE_SCHEMA = Schema(
    And(
        {
            Literal(
                "type",
                description="The type of the source geometry (Point, Plane or Fault)",
            ): "plane",
            Literal(
                "corners",
                description="The corners of the plane (shape 4 x 3: lat, lon, depth)",
            ): And(
                Use(corners_to_array), is_correct_corner_shape, has_non_negative_depth
            ),
        },
        Use(lambda schema: sources.Plane.from_corners(schema["corners"])),
    )
)

FAULT_SCHEMA = Schema(
    And(
        {
            Literal(
                "type",
                description="The type of the source geometry (Point, Plane, or Fault)",
            ): "fault",
            Literal(
                "corners",
                description="The corners of the plane (shape 4 x n x 3: lat, lon, depth)",
            ): And(
                Use(corners_to_array),
                has_non_negative_depth,
                Use(
                    (lambda corners: corners.reshape((-1, 4, 3))),
                    error="Corners cannot be reshaped to (n x 4 x 3).",
                ),
            ),
        },
        Use(lambda schema: sources.Fault.from_corners(schema["corners"])),
    )
)


SOURCE_SCHEMA = Schema(
    {"source_geometries": {str: Or(POINT_SCHEMA, PLANE_SCHEMA, FAULT_SCHEMA)}}
)

SRF_SCHEMA = Schema(
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

DOMAIN_SCHEMA = Schema(
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
)

RUPTURE_PROPAGATION_SCHEMA = Schema(
    {
        Literal(
            "hypocentre",
            description="The hypocentre coordinates (or initial rupture point if not the initial fault)",
        ): FAULT_LOCAL_COORDINATES_SCHEMA,
        Literal(
            "magnitudes",
            description="The total moment magnitude for the rupture on this fault",
        ): {str: And(float, is_plausible_magnitude)},
        Literal('jump_points', description='The jump points for the rupture'): {
            str: And({ 'from_point': FAULT_LOCAL_COORDINATES_SCHEMA, 'to_point': FAULT_LOCAL_COORDINATES_SCHEMA }, Use(lambda pts: rupture_propagation.JumpPair(**pts)))
        },
        Literal("rakes", description="The fault rakes"): {str: And(float, is_valid_degrees)},
        Literal('rupture_causality_tree', description="The fault propagation tree") : {str: Or(str, None)}
    }
)

REALISATION_METADATA_SCHEMA = Schema(
    {
        Literal("name", description="The name of the realisation"): str,
        Literal("version", description="The version of the realisation format"): Or(
            "1"
        ),
        Optional(
            Literal(
                "tag",
                description="Metadata tag for the realisation used to specify the origin or category of the realisation (e.g. NSHM, GCMT or custom).",
            )
        ): str,
    }
)
