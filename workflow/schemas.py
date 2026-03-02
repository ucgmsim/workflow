"""Module containing schema definitions for the realisation specification.

See the `realisations` module, or repository wiki pages (specifically
the [Realisations page](https://github.com/ucgmsim/workflow/wiki/Realisations), and the
[Realisations Proposal page](https://github.com/ucgmsim/workflow/wiki/Realisation-Proposal))
for a description of realisations and the schemas.
"""

import dataclasses
from enum import StrEnum

import numpy as np
import pandas as pd
from schema import And, Literal, Optional, Or, Schema, Use

from IM import im_calculation
from source_modelling import rupture_propagation, sources
from velocity_modelling.bounding_box import BoundingBox
from workflow.defaults import DefaultsVersion


class Stype(StrEnum):
    """Options for slip time function (stype) in generic_slip2srf."""

    esg2006 = "esg2006"
    urs = "urs"
    ucsb = "ucsb"
    ucsb2 = "ucsb2"
    ucsb_T = "ucsb-T"  # noqa: N815
    ucsb_varT1 = "ucsb-varT1"  # noqa: N815
    cos = "cos"
    seki = "seki"


@dataclasses.dataclass
class PointSourceParams:
    """Parameters for point source approximation."""

    stype: Stype
    """Slip type for generic_slip2srf"""

    risetime: float
    """Rise time for generic_slip2srf"""

    risetimefac: float
    """Rise time factor for generic_slip2srf"""

    risetimedep: float
    """Rise time depth dependency for generic_slip2srf"""

    inittime: float
    """Initial time for generic_slip2srf"""

    @classmethod
    def from_dict(cls, params_dict: dict) -> "PointSourceParams":
        """Create PointSourceParams from a dictionary.

        This class method is necessary because the schema library passes validated
        dictionaries to the Use() constructor, but dataclasses require keyword arguments
        to be unpacked with **kwargs. This method handles the unpacking automatically.

        Parameters
        ----------
        params_dict : dict
            The dictionary containing parameters for PointSourceParams.

        Returns
        -------
        PointSourceParams
            A new instance created from the dictionary.
        """
        return cls(**params_dict)


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


def _is_positive(x: float) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return x > 0


def _is_non_negative(x: float) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return x >= 0


def _is_valid_latitude(latitude: float) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return -90 <= latitude <= 90


def _is_valid_longitude(longitude: float) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return -180 <= longitude <= 180


def _is_plausible_magnitude(
    magnitude: float,
) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return magnitude < 11


def _is_valid_degrees(degrees: float) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return -360 <= degrees <= 360


def _is_valid_local_coordinate(
    coordinate: float,
) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return 0 <= coordinate <= 1


def _is_valid_bearing(bearing: float) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return 0 <= bearing <= 360


def _is_proportion(x: float | int) -> bool:  # noqa: D103 # numpydoc ignore=GL08
    return 0 <= x <= 1


def _is_correct_corner_shape(corners: np.ndarray) -> bool:
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


def _is_correct_fault_corner_shape(corners: np.ndarray) -> bool:
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


def _has_non_negative_depth(corners: np.ndarray) -> bool:
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
    return bool(np.all(corners[:, -1] >= 0))


def _corners_to_array(corners_spec: list[dict[str, float]]) -> np.ndarray:
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


NUMBER = And(Or(float, int), Use(float))

FAULT_LOCAL_COORDINATES_SCHEMA = Schema(
    And(
        {
            Literal(
                "s",
                description="The `s` coordinate (fraction of length, in range [0, 1])",
            ): And(NUMBER, _is_valid_local_coordinate),
            Literal(
                "d",
                description="The `d` coordinate (fraction of width, in range [0, 1])",
            ): And(NUMBER, _is_valid_local_coordinate),
        },
        Use(lambda local_coords: np.array([local_coords["s"], local_coords["d"]])),
    )
)


LAT_LON_SCHEMA = And(
    [
        {
            Literal("latitude", description="Latitude (in decimal degrees)"): And(
                NUMBER, _is_valid_latitude
            ),
            Literal("longitude", description="Longitude (in decimal degrees)"): And(
                NUMBER, _is_valid_longitude
            ),
        }
    ],
    Use(
        lambda latlon: np.array([[row["latitude"], row["longitude"]] for row in latlon])
    ),
)

POINT_COORD_SCHEMA = And(
    {
        Literal("latitude", description="Latitude (in decimal degrees)"): And(
            NUMBER, _is_valid_latitude
        ),
        Literal("longitude", description="Longitude (in decimal degrees)"): And(
            NUMBER, _is_valid_longitude
        ),
        Literal("depth", description="Depth (in metres)"): And(
            NUMBER, _is_non_negative
        ),
    },
    Use(
        lambda latlon: np.array(
            [latlon["latitude"], latlon["longitude"], latlon["depth"]]
        )
    ),
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
            ): POINT_COORD_SCHEMA,
            Literal("length", description="The pseudo-length of the point source"): And(
                NUMBER, _is_positive
            ),
            Literal("width", description="The pseudo-width of the point source"): And(
                NUMBER, _is_positive
            ),
            Literal(
                "strike", description="The strike bearing of the point source"
            ): And(NUMBER, _is_valid_bearing),
            Literal("dip", description="The dip angle of the point source"): And(
                NUMBER, _is_valid_bearing
            ),
            Literal(
                "dip_dir", description="The dip direction bearing of the point source"
            ): And(NUMBER, _is_valid_bearing),
        },
        Use(
            lambda schema: sources.Point.from_lat_lon_depth(
                schema["coordinates"],
                length_m=schema["length"],
                width_m=schema["width"],
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
                Use(_corners_to_array),
                _is_correct_corner_shape,
                _has_non_negative_depth,
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
                Use(_corners_to_array),
                _has_non_negative_depth,
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


POINT_SOURCE_PARAMS_SCHEMA = Schema(
    And(
        {
            Literal(
                "stype", description="Slip time function for generic_slip2srf"
            ): Use(Stype),
            Literal("risetime", description="Rise time for generic_slip2srf"): And(
                NUMBER, _is_positive
            ),
            Literal(
                "risetimefac", description="Rise time factor for generic_slip2srf"
            ): And(NUMBER, _is_positive),
            Literal(
                "risetimedep",
                description="Rise time depth dependency for generic_slip2srf",
            ): And(NUMBER, _is_non_negative),
            Literal("inittime", description="Initial time for generic_slip2srf"): And(
                NUMBER, _is_non_negative
            ),
        },
        Use(PointSourceParams.from_dict),
    )
)

RUPTURE_VELOCITY_SCHEMA = Schema(
    {
        Literal("rvfrac", description="Rupture velocity factor (rupture : Vs)"): And(
            NUMBER, _is_positive
        ),
        Literal("rvfrac_shal", description="Rupture velocity at shallow depths"): And(
            NUMBER, _is_positive
        ),
        Literal("rvfrac_deep", description="Rupture velocity at depth"): And(
            NUMBER, _is_positive
        ),
        Literal(
            "shallow_depth",
            description="Shallow transition depth.",
        ): And(NUMBER, _is_positive),
        Literal(
            "shallow_transition_range",
            description="Shallow transition depth transition range.",
        ): And(NUMBER, _is_positive),
        Literal(
            "deep_depth",
            description="Deep transition depth.",
        ): And(NUMBER, _is_positive),
        Literal(
            "deep_transition_range",
            description="Deep transition depth transition range.",
        ): And(NUMBER, _is_positive),
    }
)

SRF_SCHEMA = Schema(
    {
        Literal(
            "resolution", description="The resolution of the SRF discretisation."
        ): And(NUMBER, _is_positive),
        Optional(
            Literal(
                "point_source_params",
                description="Parameters for point source approximation, if applicable",
            )
        ): Or(None, POINT_SOURCE_PARAMS_SCHEMA),
        Literal(
            "side_taper",
            description="Side slip tapering, proportion of along-strike 0-1.",
        ): And(NUMBER, _is_proportion),
        Literal(
            "bot_taper",
            description="Bottom slip tapering proportion of down-dip 0-1.",
        ): And(NUMBER, _is_proportion),
        Literal(
            "top_taper",
            description="Top slip tapering proportion of down-dip 0-1.",
        ): And(NUMBER, _is_proportion),
        Literal(
            "alpha_rough",
            description="Roughness (0.0 = smooth)",
        ): And(NUMBER, _is_proportion),
        Literal(
            "gwid",
            description="Width of delay zone around segment boundaries",
        ): [And(NUMBER, _is_positive)],
        Literal(
            "rvfrac_seg",
            description="Rupture speed reduction (proportion) at segment boundaries.",
        ): [And(NUMBER, _is_proportion)],
        Literal(
            "seg_delay",
            description="If true, delay rupture across slip boundaries according to specifications of rvfac_seg and gwid.",
        ): bool,
        Literal(
            "ymag_exp",
            description="Corner magnitude exponent for along-strike slip correlation length. See genslip_v5.6.2c:1385",
        ): Or(NUMBER, None),
        Literal(
            "xmag_exp",
            description="Corner magnitude exponent for down-dip slip correlation length. See genslip_v5.6.2c:1385",
        ): Or(NUMBER, None),
        Literal(
            "kx_corner",
            description="Corner wavenumber for along-strike slip correlation length. See genslip_v5.6.2c:1385",
        ): Or(NUMBER, None),
        Literal(
            "ky_corner",
            description="Corner wavenumber for down-dip slip correlation length. See genslip_v5.6.2c:1385",
        ): Or(NUMBER, None),
        Literal(
            "slip_sigma",
            description="The stddev of slip distribution.",
        ): And(NUMBER, _is_non_negative),
        Literal(
            "risetime_coef",
            description="Risetime scaling coefficient.",
        ): And(NUMBER, _is_positive),
        # Rise time and source time function values
        Literal("beta_asp", description="Asperity beta parameter."): And(
            NUMBER, _is_positive
        ),
        Literal(
            "beta_deep", description="Deep beta parameter for rise time."
        ): And(NUMBER, _is_positive),
        Literal(
            "beta_mid", description="Mid-depth beta parameter for rise time."
        ): And(NUMBER, _is_positive),
        Literal(
            "beta_mid_depth",
            description="Mid-depth level for beta depth scaling.",
        ): And(NUMBER, _is_positive),
        Literal(
            "beta_mid_depth_range",
            description="Mid-depth range for beta depth scaling.",
        ): And(NUMBER, _is_positive),
        Literal(
            "beta_shal", description="Shallow beta parameter for rise time."
        ): And(NUMBER, _is_positive),
        Literal(
            "beta_shal_depth",
            description="Shallow depth level for beta depth scaling.",
        ): And(NUMBER, _is_positive),
        Literal(
            "beta_shal_depth_range",
            description="Shallow depth range for beta depth scaling.",
        ): And(NUMBER, _is_positive),
        Literal(
            "beta_subevt", description="Sub-event beta parameter for rise time."
        ): And(NUMBER, _is_positive),
        Literal(
            "deep_risetimedep",
            description="Deep rise time depth dependency level.",
        ): And(NUMBER, _is_positive),
        Literal(
            "deep_risetimedep_range",
            description="Deep rise time depth dependency range.",
        ): And(NUMBER, _is_positive),
        Literal(
            "deep_risetimefac",
            description="Deep rise time scaling factor.",
        ): And(NUMBER, _is_positive),
        Literal(
            "risetimedep",
            description="Rise time depth dependency level.",
        ): And(NUMBER, _is_positive),
        Literal(
            "risetimedep_range",
            description="Rise time depth dependency range.",
        ): And(NUMBER, _is_positive),
        Literal(
            "risetimefac",
            description="Rise time scaling factor.",
        ): And(NUMBER, _is_positive),
        Literal(
            "rt_rand",
            description="Rise time randomness factor (0.0 = deterministic).",
        ): And(NUMBER, _is_non_negative),
        Literal(
            "rt_scalefac",
            description="Rise time scale factor.",
        ): And(NUMBER, _is_positive),
        Literal(
            "stype",
            description="Slip time function type for genslip (null = use genslip default).",
        ): Or(str, None),
        # Hybrid Correlation Length (hyb_corlen) Parameters
        Literal(
            "hyb_corlen_deep_wt_end",
            description="Hybrid correlation length deep weight end.",
        ): NUMBER,
        Literal(
            "hyb_corlen_deep_wt_start",
            description="Hybrid correlation length deep weight start.",
        ): NUMBER,
        Literal(
            "hyb_corlen_dep",
            description="Hybrid correlation length depth scaling level.",
        ): And(NUMBER, _is_positive),
        Literal(
            "hyb_corlen_dep_range",
            description="Hybrid correlation length depth scaling range.",
        ): And(NUMBER, _is_positive),
        Literal(
            "hyb_corlen_fac",
            description="Hybrid correlation length factor.",
        ): And(NUMBER, _is_positive),
        Literal(
            "hyb_corlen_flag",
            description="Enable hybrid correlation length model.",
        ): bool,
        Literal(
            "hyb_corlen_kmodel",
            description="Hybrid correlation length k-model index.",
        ): And(NUMBER, _is_positive),
        Literal(
            "hyb_corlen_shal_wt_end",
            description="Hybrid correlation length shallow weight end.",
        ): NUMBER,
        Literal(
            "hyb_corlen_shal_wt_start",
            description="Hybrid correlation length shallow weight start.",
        ): NUMBER,
        Literal(
            "hyb_corlen_side_taper",
            description="Hybrid correlation length side taper.",
        ): And(NUMBER, _is_non_negative),
        # Rupture Velocity & Fault Dimensions
        Literal(
            "fdrup_scale_slip",
            description="Scale slip by fault rupture area.",
        ): bool,
        Literal(
            "fdrup_time",
            description="Use fault rupture time.",
        ): bool,
        Literal(
            "rupture_delay",
            description="Delay (in seconds) before rupture starts.",
        ): And(NUMBER, _is_non_negative),
        Literal(
            "rvfmax",
            description="Maximum rupture velocity fraction.",
        ): And(NUMBER, _is_positive),
        Literal(
            "rvfmin",
            description="Minimum rupture velocity fraction.",
        ): And(NUMBER, _is_positive),
        Literal(
            "rvfrac_slip_sig",
            description="Rupture velocity fraction slip sigma (null = disabled).",
        ): Or(NUMBER, None),
        # Tapering & Slip Level Adjustments
        Literal(
            "truncate_zero_slip",
            description="Truncate subfaults with zero slip.",
        ): bool,
        Literal(
            "slip_water_level",
            description="Water level for slip distribution (null = disabled).",
        ): Or(NUMBER, None),
        Literal(
            "rake_sigma",
            description="Standard deviation of rake perturbation (degrees).",
        ): And(NUMBER, _is_positive),
        Literal(
            "fractal_rake",
            description="Use fractal rake distribution.",
        ): bool,
        # Time Shift Factors (tsfac)
        Literal(
            "tsfac1_scor",
            description="Time shift factor 1 spatial correlation.",
        ): NUMBER,
        Literal(
            "tsfac1_sigma",
            description="Time shift factor 1 sigma.",
        ): And(NUMBER, _is_positive),
        Literal(
            "tsfac2_lambda_max",
            description="Time shift factor 2 maximum wavelength.",
        ): NUMBER,
        Literal(
            "tsfac2_lambda_min",
            description="Time shift factor 2 minimum wavelength (null = disabled).",
        ): Or(NUMBER, None),
        Literal(
            "tsfac2_scor",
            description="Time shift factor 2 spatial correlation.",
        ): NUMBER,
        Literal(
            "tsfac2_sigma",
            description="Time shift factor 2 sigma.",
        ): And(NUMBER, _is_positive),
        Literal(
            "tsfac_bzero",
            description="Time shift factor b0 coefficient.",
        ): NUMBER,
        Literal(
            "tsfac_coef",
            description="Time shift factor coefficient.",
        ): NUMBER,
        Literal(
            "tsfac_main",
            description="Main time shift factor (null = use computed value).",
        ): Or(NUMBER, None),
        Literal(
            "tsfac_slope",
            description="Time shift factor slope.",
        ): NUMBER,
        # Kinematic Models & Corner Frequencies
        Literal(
            "circular_average",
            description="Use circular average for slip correlation.",
        ): bool,
        Literal(
            "kmodel",
            description="Kinematic model index: 1=SOMERVILLE_FLAG, 2=MAI_FLAG, -1=INPUT_CORNERS_FLAG.",
        ): And(NUMBER, _is_positive),
        Literal(
            "kord",
            description="Wavenumber filter order (integer stored as float for schema compatibility).",
        ): And(NUMBER, _is_positive),
        Literal(
            "magC",
            description="Magnitude corner frequency coefficient.",
        ): NUMBER,
        Literal(
            "mag_area_Acoef",
            description="Magnitude-area scaling A coefficient (null = disabled).",
        ): Or(NUMBER, None),
        Literal(
            "mag_area_Bcoef",
            description="Magnitude-area scaling B coefficient (null = disabled).",
        ): Or(NUMBER, None),
        Literal(
            "mai_wt",
            description="MAI model weight.",
        ): And(NUMBER, _is_proportion),
        Literal(
            "modified_corners",
            description="Use modified corner frequencies.",
        ): bool,
        Literal(
            "somerville_wt",
            description="Somerville model weight.",
        ): And(NUMBER, _is_proportion),
        Literal(
            "stretch_kcorner",
            description="Stretch corner wavenumber.",
        ): bool,
        Literal(
            "use_gaus",
            description="Use Gaussian slip distribution.",
        ): bool,
        Literal(
            "use_median_mag",
            description="Use median magnitude for scaling.",
        ): bool,
        # Roughness & Wavelength Limits
        Literal(
            "lambda_max",
            description="Maximum wavelength for slip correlation (null = no limit).",
        ): Or(NUMBER, None),
        Literal(
            "lambda_min",
            description="Minimum wavelength for slip correlation (null = no limit).",
        ): Or(NUMBER, None),
        Literal(
            "roughnessfile",
            description="Path to roughness file (null = no file).",
        ): Or(str, None),
        Literal(
            "wavelength_max",
            description="Maximum wavelength limit (null = no limit).",
        ): Or(NUMBER, None),
        Literal(
            "wavelength_min",
            description="Minimum wavelength limit (null = no limit).",
        ): Or(NUMBER, None),
        # Miscellaneous Slip/Rupture/Rake Vars
        Literal(
            "asp_taper_fac",
            description="Asperity taper factor.",
        ): And(NUMBER, _is_non_negative),
        Literal(
            "extend_fac",
            description="Fault extension factor (null = disabled).",
        ): Or(NUMBER, None),
        Literal(
            "flen_max",
            description="Maximum fault length (null = no limit).",
        ): Or(NUMBER, None),
        Literal(
            "fwid_max",
            description="Maximum fault width (null = no limit).",
        ): Or(NUMBER, None),
        Literal(
            "init_slip_file",
            description="Path to initial slip file (null = no file).",
        ): Or(str, None),
        Literal(
            "moment_fraction",
            description="Fraction of seismic moment (null = use all).",
        ): Or(NUMBER, None),
        Literal(
            "perturb_subfault_location",
            description="Perturb subfault locations.",
        ): bool,
        Literal(
            "rand_rake_degs",
            description="Range of random rake perturbation (degrees).",
        ): And(NUMBER, _is_non_negative),
        Literal(
            "read_slip_file",
            description="Read slip distribution from file.",
        ): bool,
        Literal(
            "rtime1_depth",
            description="Rise time 1 depth level.",
        ): And(NUMBER, _is_positive),
        Literal(
            "rtime1_depth_range",
            description="Rise time 1 depth range.",
        ): And(NUMBER, _is_positive),
        Literal(
            "rtime1_scor",
            description="Rise time 1 spatial correlation.",
        ): NUMBER,
        Literal(
            "rtime1_sigma",
            description="Rise time 1 sigma.",
        ): And(NUMBER, _is_positive),
        Literal(
            "rtime2_scor",
            description="Rise time 2 spatial correlation.",
        ): NUMBER,
        Literal(
            "rtime2slip_exp",
            description="Rise time 2 slip exponent.",
        ): NUMBER,
        Literal(
            "rtime_rand",
            description="Rise time randomness (null = disabled).",
        ): Or(NUMBER, None),
        Literal(
            "set_rake",
            description="Fixed rake angle (null = use fault rake).",
        ): Or(NUMBER, None),
        Literal(
            "svr_wt",
            description="SVR model weight.",
        ): And(NUMBER, _is_non_negative),
        Literal(
            "target_savg",
            description="Target average slip (null = compute from magnitude).",
        ): Or(NUMBER, None),
        Literal(
            "use_Mw",
            description="Use moment magnitude for scaling.",
        ): bool,
        # Aseismic & Segment Settings
        Literal(
            "aseis_flag",
            description="Enable aseismic zone correction.",
        ): bool,
        Literal(
            "aseis_smooth",
            description="Enable smoothing of aseismic zones.",
        ): bool,
        Literal(
            "aseis_dep",
            description="Aseismic zone depth.",
        ): And(NUMBER, _is_positive),
        Literal(
            "aseis_fac",
            description="Aseismic zone factor (null = disabled).",
        ): Or(NUMBER, None),
        Literal(
            "xshift",
            description="Along-strike shift of slip distribution.",
        ): NUMBER,
        Literal(
            "yshift",
            description="Down-dip shift of slip distribution.",
        ): NUMBER,
    }
)

DOMAIN_SCHEMA = Schema(
    {
        Literal("domain", description="The corners of the simulation domain."): And(
            LAT_LON_SCHEMA, Use(BoundingBox.from_wgs84_coordinates)
        ),
        Literal("depth", description="The depth of the model (in km)"): And(
            NUMBER, _is_positive
        ),
        Literal(
            "duration", description="The duration of the simulation (in seconds)"
        ): And(NUMBER, _is_positive),
    }
)

RAKE_SCHEMA = Schema(
    {
        Literal("rakes", description="The fault rakes"): {
            str: And(NUMBER, _is_valid_degrees)
        },
    }
)

MAGNITUDE_SCHEMA = Schema(
    {
        Literal(
            "magnitudes",
            description="The total moment magnitude for the rupture on this fault",
        ): {str: And(NUMBER, _is_plausible_magnitude)},
    }
)


RUPTURE_PROPAGATION_SCHEMA = Schema(
    {
        Literal(
            "hypocentre",
            description="The hypocentre coordinates (or initial rupture point if not the initial fault)",
        ): FAULT_LOCAL_COORDINATES_SCHEMA,
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
        ): And(NUMBER, _is_positive),
        Literal("version", "Velocity model version"): str,
        Literal("topo_type", "Velocity model topology type"): str,
        Literal("ds_multiplier", "Velocity model ds multiplier"): And(
            NUMBER, _is_positive
        ),
        Literal("vs30", "VS30 value"): And(NUMBER, _is_positive),
        Literal("s_wave_velocity", "S-wave velocity"): And(NUMBER, _is_positive),
        Literal("rrup_interpolants", "RRup interpolants to estimate domain size"): And(
            [[And(NUMBER, _is_positive)]], Use(np.array)
        ),
        Literal("fault_buffer", "Buffer width (km) around sources in rupture."): And(
            NUMBER, _is_positive
        ),
    }
)

SEED_SCHEMA = Schema(
    {
        Literal("genslip_seed", description="The random seed passed to genslip."): int,
        Literal(
            "nshm_to_realisation_seed",
            description="The random seed passed for NSHM -> realisation.",
        ): int,
        Literal(
            "srfgen_seed",
            description="A second random seed for genslip, used for specific purposes in the generation process.",
        ): int,
        Literal(
            "rupture_propagation_seed", description="Seed for rupture propagation"
        ): int,
        Literal("hf_seed", description="HF seed."): int,
    }
)

VELOCITY_MODEL_1D_SCHEMA = Schema(
    {
        Literal("model", description="The 1D velocity model"): And(
            [
                {
                    "thickness": And(NUMBER, _is_positive),
                    "Vp": And(NUMBER, _is_positive),
                    "Vs": And(NUMBER, _is_positive),
                    "rho": And(NUMBER, _is_positive),
                    "Qp": And(NUMBER, _is_positive),
                    "Qs": And(NUMBER, _is_positive),
                }
            ],
            Use(pd.DataFrame),
        )
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
        Literal("flo", description="Unknown!"): NUMBER,
        Literal("fhi", description="Unknown!"): NUMBER,
        Literal("nl_skip", description="Skip empty lines in input?"): int,
        Literal("vp_sig", description="Unknown!"): NUMBER,
        Literal("vsh_sig", description="Unknown!"): NUMBER,
        Literal("rho_sig", description="Unknown!"): NUMBER,
        Literal("qs_sig", description="Unknown!"): NUMBER,
        Literal("ic_flag", description="Unknown!"): bool,
        Literal("velocity_name", description="Unknown!"): str,
        Literal("t_sec", description="High frequency output start time."): And(
            NUMBER, _is_non_negative
        ),
        Literal("sdrop", description="Stress drop average (bars)"): NUMBER,
        Literal("rayset", description="ray types 1: direct, 2: moho"): [Or(1, 2)],
        Literal(
            "no_siteamp", description="Disable BJ97 site amplification factors"
        ): bool,
        Literal("fmax", description="Max simulation frequency"): And(
            NUMBER, _is_positive
        ),
        Literal("kappa", description="Unknown!"): NUMBER,
        Literal("qfexp", description="Q frequency exponent"): NUMBER,
        Literal("czero", description="C0 coefficient"): NUMBER,
        Literal("calpha", description="Ca coefficient"): NUMBER,
        Literal("mom", description="Seimic moment (or null, to infer value)"): Or(
            NUMBER, None
        ),
        Literal("rupv", description="Rupture velocity (or binary default)"): Or(
            NUMBER, None
        ),
        Literal("site_specific", description="Enable site-specific calculation"): bool,
        Literal("vs_moho", description="vs of moho layer"): NUMBER,
        Literal("fa_sig1", "Fourier amplitude uncertainty (1)"): NUMBER,
        Literal("fa_sig2", description="Fourier amplitude uncertainty (2)"): NUMBER,
        Literal("rv_sig1", description="Rupture velocity uncertainty"): And(
            NUMBER, _is_non_negative
        ),
        Literal(
            "path_dur",
            description="path duration model. 0: GP2010, 1: WUS modification trail/errol, 2: ENA modificiation trial/error"
            ", 11: WUS formutian of BT2014, 12: ENA formulation of BT2015. Models 11 and 12 overpredict for multiple rays.",
        ): Or(0, 1, 2, 11, 12),
        Literal("dpath_pert", description="Log of path duration multiplier"): NUMBER,
        Literal(
            "stress_parameter_adjustment_tect_type",
            description="Adjustment option 0 = off, 1 = active tectonic, 2 = stable continent",
        ): Or(0, 1, 2),
        Literal(
            "stress_parameter_adjustment_target_magnitude",
            description="Target magnitude (or inferred if null)",
        ): Or(NUMBER, None),
        Literal(
            "stress_parameter_adjustment_fault_area", "Fault area (or inferred if null)"
        ): Or(NUMBER, None),
        Literal("stoch_dx", description="Stoch file dx"): And(NUMBER, _is_positive),
        Literal("stoch_dy", description="Stoch file dy"): And(NUMBER, _is_positive),
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
        "fhi": NUMBER,
        "fmax": NUMBER,
        "fmin": NUMBER,
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
        "qbndmax": NUMBER,
        "qpfrac": NUMBER,
        "qpqs_factor": NUMBER,
        "qsfrac": NUMBER,
        "read_restart": int,
        "report": int,
        "scale": int,
        "smodfile": str,
        "span": int,
        "stype": str,
        "swap_bytes": int,
        "ts_xy": int,
        "ts_xz": int,
        "ts_yz": int,
        "tzero": NUMBER,
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
            NUMBER, _is_non_negative
        ),
        Literal("fmidbot", description="fmidbot for site amplification"): And(
            NUMBER, _is_non_negative
        ),
        Literal("fmin", description="fmin for site amplification"): And(
            NUMBER, _is_non_negative
        ),
        "site_amp_version": str,
    }
)


INTENSITY_MEASURE_CALCUATION_PARAMETERS = Schema(
    {
        Literal("ims", description="Intensity measures to calculate"): [
            And(str, Use(im_calculation.IM))
        ],
        Literal("valid_periods", description="Valid periods to calculate for"): And(
            [And(NUMBER, _is_positive)], Use(np.array)
        ),
        Literal("fas_frequencies", description="Fourier spectrum frequencies"): And(
            [And(NUMBER, _is_positive)], Use(np.array)
        ),
    }
)


LOG_ENTRY_SCHEMA = Schema(
    {
        Literal(
            "utility",
            description="The name of the utility that produced this log entry.",
        ): str,
        Literal("version", description="The version of the utility."): str,
        Literal(
            "timestamp",
            description="The timestamp of when the utility was run.",
        ): str,
        Optional(
            Literal(
                "args",
                description="The arguments the utility was executed with.",
            )
        ): [str],
    }
)

LOG_TRAIL_SCHEMA = Schema(
    {
        Literal(
            "log",
            description="The utilities executed on this realisation.",
        ): [LOG_ENTRY_SCHEMA],
    }
)

RESOLUTION_SCHEMA = Schema(
    {
        Literal("resolution", description="Simulation spatial resolution."): And(
            NUMBER, _is_positive
        )
    }
)
