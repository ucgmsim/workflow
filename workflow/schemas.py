"""Module containing schema definitions for the realisation specification.

See the `realisations` module, or repository wiki pages (specifically
the [Realisations page](https://github.com/ucgmsim/workflow/wiki/Realisations), and the
[Realisations Proposal page](https://github.com/ucgmsim/workflow/wiki/Realisation-Proposal))
for a description of realisations and the schemas.
"""

from enum import StrEnum

import numpy as np
from schema import And, Literal, Optional, Or, Schema, Use

from qcore import constants
from source_modelling import rupture_propagation, sources
from velocity_modelling.bounding_box import BoundingBox
from workflow.defaults import DefaultsVersion

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
        Literal("resolution", description="Subdivision resolution."): And(
            float, is_positive
        ),
    }
)

DOMAIN_SCHEMA = Schema(
    {
        Literal("resolution", description="The simulation resolution (in km)"): And(
            float, is_positive
        ),
        Literal("domain", description="The corners of the simulation domain."): And(
            Use(corners_to_array), Use(BoundingBox.from_wgs84_coordinates)
        ),
        Literal("depth", description="The depth of the model (in km)"): And(
            float, is_positive
        ),
        Literal(
            "duration", description="The duration of the simulation (in seconds)"
        ): And(float, is_positive),
        Literal("dt", "The resolution of the domain in time (in seconds)."): And(
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
        Literal("jump_points", description="The jump points for the rupture"): Or(
            {
                str: And(
                    {
                        "from_point": FAULT_LOCAL_COORDINATES_SCHEMA,
                        "to_point": FAULT_LOCAL_COORDINATES_SCHEMA,
                    },
                    Use(lambda pts: rupture_propagation.JumpPair(**pts)),
                )
            },
            {},
        ),
        Literal("rakes", description="The fault rakes"): {
            str: And(float, is_valid_degrees)
        },
        Literal("rupture_causality_tree", description="The fault propagation tree"): {
            str: Or(str, None)
        },
    }
)

VELOCITY_MODEL_SCHEMA = Schema(
    {
        Literal(
            "min_vs",
            description="The minimum velocity (km/s) produced in the velocity model.",
        ): And(float, is_positive),
        Literal("version", "Velocity model version"): "2.06",
        Literal("topo_type", "Velocity model topology type"): str,
        Literal("dt", "Velocity model timestep resolution"): And(float, is_positive),
        Literal("ds_multiplier", "Velocity model ds multiplier"): And(
            float, is_positive
        ),
        Literal("resolution", "Velocity model spatial resolution"): And(
            float, is_positive
        ),
        Literal("vs30", "VS30 value"): And(float, is_positive),
        Literal("s_wave_velocity", "S-wave velocity"): And(float, is_positive),
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
        ): Or(str, None),
        Literal(
            "defaults_version", description="Simulation default parameters version."
        ): And(str, Use(DefaultsVersion)),
    }
)

HF_CONFIG_SCHEMA = Schema(
    {
        Literal("nbu", description="Unknown!"): int,
        Literal("ift", description="Unknown!"): int,
        Literal("flo", description="Unknown!"): float,
        Literal("fhi", description="Unknown!"): float,
        Literal("nl_skip", description="Skip empty lines in input?"): int,
        Literal("vp_sig", description="Unknown!"): float,
        Literal("vsh_sig", description="Unknown!"): float,
        Literal("rho_sig", description="Unknown!"): float,
        Literal("qs_sig", description="Unknown!"): float,
        Literal("ic_flag", description="Unknown!"): bool,
        Literal("velocity_name", description="Unknown!"): str,
        Literal("dt", description="Time resolution for HF simulation"): And(
            float, is_positive
        ),
        Literal("t_sec", description="High frequency output start time."): And(
            float, is_non_negative
        ),
        Literal("sdrop", description="Stress drop average (bars)"): float,
        Literal("rayset", description="ray types 1: direct, 2: moho"): [Or(1, 2)],
        Literal(
            "no_siteamp", description="Disable BJ97 site amplification factors"
        ): bool,
        Literal("fmax", description="Max simulation frequency"): And(
            float, is_positive
        ),
        Literal("kappa", description="Unknown!"): float,
        Literal("qfexp", description="Q frequency exponent"): float,
        Literal("rvfac", description="Rupture velocity factor (rupture : Vs)"): And(
            float, is_non_negative
        ),
        Literal("rvfac_shal", description="rvfac shallow fault multiplier"): float,
        Literal("rvfac_deep", description="rvfac deep fault multiplier"): float,
        Literal("czero", description="C0 coefficient"): float,
        Literal("calpha", description="Ca coefficient"): float,
        Literal("mom", description="Seimic moment (or null, to infer value)"): Or(
            float, None
        ),
        Literal("rupv", description="Rupture velocity (or binary default)"): Or(
            float, None
        ),
        Literal("site_specific", description="Enable site-specific calculation"): bool,
        Literal("vs_moho", description="vs of moho layer"): float,
        Literal("fa_sig1", "Fourier amplitude uncertainty (1)"): float,
        Literal("fa_sig2", description="Fourier amplitude uncertainty (2)"): float,
        Literal("rv_sig1", description="Rupture velocity uncertainty"): And(
            float, is_non_negative
        ),
        Literal(
            "path_dur",
            description="path duration model. 0: GP2010, 1: WUS modification trail/errol, 2: ENA modificiation trial/error"
            ", 11: WUS formutian of BT2014, 12: ENA formulation of BT2015. Models 11 and 12 overpredict for multiple rays.",
        ): Or(0, 1, 2, 11, 12),
        Literal("dpath_pert", description="Log of path duration multiplier"): float,
        Literal(
            "stress_parameter_adjustment_tect_type",
            description="Adjustment option 0 = off, 1 = active tectonic, 2 = stable continent",
        ): Or(0, 1, 2),
        Literal(
            "stress_parameter_adjustment_target_magnitude",
            description="Target magnitude (or inferred if null)",
        ): Or(float, None),
        Literal(
            "stress_parameter_adjustment_fault_area", "Fault area (or inferred if null)"
        ): Or(float, None),
        Literal("seed", description="HF seed"): int,
        Literal("stoch_dx", description="Stoch file dx"): And(float, is_positive),
        Literal("stoch_dy", description="Stoch file dy"): And(float, is_positive),
    }
)

EMOD3D_PARAMETERS_SCHEMA = Schema(
    {
        "all_in_one": int,
        "bfilt": int,
        "bforce": int,
        "dampwidth": int,
        "dblcpl": int,
        "dmodfile": str,
        "dtts": int,
        "dump_itinc": int,
        "dxout": int,
        "dxts": int,
        "dyout": int,
        "dyts": int,
        "dzout": int,
        "dzts": int,
        "elas_only": int,
        "enable_output_dump": int,
        "enable_restart": int,
        "ffault": int,
        "fhi": float,
        "fmax": float,
        "fmin": float,
        "freesurf": int,
        "geoproj": int,
        "intmem": int,
        "ix_ts": int,
        "ix_ys": int,
        "ix_zs": int,
        "iy_ts": int,
        "iy_xs": int,
        "iy_zs": int,
        "iz_ts": int,
        "iz_xs": int,
        "iz_ys": int,
        "lonlat_out": int,
        "maxmem": int,
        "model_style": int,
        "nseis": int,
        "order": int,
        "pmodfile": str,
        "pointmt": int,
        "qbndmax": float,
        "qpfrac": float,
        "qpqs_factor": float,
        "qsfrac": float,
        "read_restart": int,
        "report": int,
        "restart_itinc": int,
        "scale": int,
        "smodfile": str,
        "span": int,
        "stype": str,
        "swap_bytes": int,
        "ts_inc": int,
        "ts_start": int,
        "ts_total": int,
        "ts_xy": int,
        "ts_xz": int,
        "ts_yz": int,
        "tzero": float,
        "vmodel_swapb": int,
        "xseis": int,
        "yseis": int,
        "zseis": int,
        "pertbfile": str,
    }
)

BROADBAND_PARAMETERS_SCHEMA = Schema(
    {
        Literal("flo", description="low/high frequency cutoff"): And(
            float, is_non_negative
        ),
        Literal("dt", description="simulation time resolution"): And(
            float, is_positive
        ),
        Literal("fmidbot", description="fmidbot for site amplification"): And(
            float, is_non_negative
        ),
        Literal("fmin", description="fmin for site amplification"): And(
            float, is_non_negative
        ),
        "site_amp_version": str,
    }
)


class IntensityMeasure(StrEnum):
    """Intensity Measures for IM Calculation."""

    PGA = "PGA"
    PGV = "PGV"
    CAV = "CAV"
    AI = "AI"
    DS575 = "Ds575"
    DS595 = "Ds595"
    MMI = "MMI"
    PSA = "pSA"
    SED = "SED"
    FAS = "FAS"
    SDI = "SDI"


class Component(StrEnum):
    """Component values for IM calculation."""

    C090 = "090"
    C000 = "000"
    VER = "ver"
    H1 = "H1"
    H2 = "H2"
    GEOM = "geom"
    ROTD50 = "rotd50"
    ROTD100 = "rotd100"
    ROTD100_50 = "rotd100_50"
    NORM = "norm"
    EAS = "EAS"


class Units(StrEnum):
    """Units for IM Calculation."""

    g = "g"
    cms2 = "cm/s^2"


INTENSITY_MEASURE_CALCUATION_PARAMETERS = Schema(
    {
        Literal("ims", description="Intensity measures to calculate"): [
            And(str, Use(IntensityMeasure))
        ],
        Literal(
            "components", description="Components to calculate intensity measures in"
        ): [And(str, Use(Component))],
        Literal("valid_periods", description="Valid periods to calculate for"): And(
            [And(float, is_positive)], Use(np.array)
        ),
        Literal("fas_frequencies", description="Fourier spectrum frequencies"): And(
            [And(float, is_positive)], Use(np.array)
        ),
        Literal("units", description="Units to calculate intensity measures in"): And(
            str, Use(Units)
        ),
    }
)
