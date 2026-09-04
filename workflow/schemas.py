"""Module containing schema definitions for the realisation specification.

See the `realisations` module, or repository wiki pages (specifically
the [Realisations page](https://github.com/ucgmsim/workflow/wiki/Realisations), and the
[Realisations Proposal page](https://github.com/ucgmsim/workflow/wiki/Realisation-Proposal))
for a description of realisations and the schemas.
"""

from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from nzcvm.config.layers import LayerConfig
from nzcvm.coordinates import Coordinate
from schema import And, Literal, Optional, Or, Schema, Use

from IM import im_calculation
from source_modelling import rupture_propagation, sources
from velocity_modelling.bounding_box import BoundingBox
from workflow.defaults import DefaultsVersion


class SiteAmpModel(StrEnum):
    """Site amplification models for broadband simulation."""

    CB2014 = "cb2014"
    BA2018 = "ba2018"


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


def _is_positive(x: float) -> bool:  # numpydoc ignore=GL08
    return x > 0


def _is_non_negative(x: float) -> bool:  # numpydoc ignore=GL08
    return x >= 0


def _is_valid_latitude(latitude: float) -> bool:  # numpydoc ignore=GL08
    return -90 <= latitude <= 90


def _is_valid_longitude(longitude: float) -> bool:  # numpydoc ignore=GL08
    return -180 <= longitude <= 180


def _is_plausible_magnitude(
    magnitude: float,
) -> bool:  # numpydoc ignore=GL08
    return magnitude < 11


def _is_valid_degrees(degrees: float) -> bool:  # numpydoc ignore=GL08
    return -360 <= degrees <= 360


def _is_valid_local_coordinate(
    coordinate: float,
) -> bool:  # numpydoc ignore=GL08
    return 0 <= coordinate <= 1


def _is_valid_bearing(bearing: float) -> bool:  # numpydoc ignore=GL08
    return 0 <= bearing <= 360


def _is_proportion(x: float) -> bool:  # numpydoc ignore=GL08
    return 0 <= x <= 1


def _is_slip_spread(x: float) -> bool:
    """Check a slip coefficient of variation lies in the truncated exponential family.

    Slip is drawn from a truncated exponential fitted to SRCMOD (Thingbaijam & Mai
    2016), whose spread and largest-slip-to-mean ratio are one number. The family runs
    from 0.5774 to 1 and attains neither end.

    Parameters
    ----------
    x : float
        The coefficient of variation to validate.

    Returns
    -------
    bool
        True if `x` lies strictly inside the family's range.
    """
    return 3.0**-0.5 < x < 1


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
            "rvfrac_slip_sig",
            description="Rupture velocity fraction slip sigma (null = disabled).",
        ): Or(And(NUMBER, _is_non_negative), None),
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

RAMP_SCHEMA = Schema(
    {
        Literal("centre_km", description="Depth the ramp is centred on"): And(
            NUMBER, _is_non_negative
        ),
        Literal(
            "half_width_km",
            description="Half the depth range the ramp takes to cross over",
        ): And(NUMBER, _is_positive),
    }
)

SRF_SCHEMA = Schema(
    {
        Literal(
            "resolution",
            description="Subfault size (km) the fault is discretised at",
        ): And(NUMBER, _is_positive),
        Literal("source", description="How rise time scales with moment"): {
            Literal(
                "rise_time_coefficient",
                description="Rise time per cube-root dyne-centimetre (Graves & Pitarka)",
            ): And(NUMBER, _is_positive),
        },
        Literal("slip", description="The slip field"): {
            Literal("shape", description="Spectral falloff of the slip field"): str,
            Literal(
                "coefficient_of_variation",
                description="Spread of the slip field, dimensionless. The truncated exponential family runs from 0.5774 to 1 and attains neither end",
            ): And(NUMBER, _is_slip_spread),
            Literal(
                "rake_sigma_deg", description="Spread of the rake field, degrees"
            ): And(NUMBER, _is_positive),
            Literal(
                "side_taper", description="Along-strike taper, as a fraction of length"
            ): And(NUMBER, _is_proportion),
            Literal(
                "top_taper", description="Taper at the top edge, as a fraction of width"
            ): And(NUMBER, _is_proportion),
            Literal(
                "bottom_taper",
                description="Taper at the bottom edge, as a fraction of width",
            ): And(NUMBER, _is_proportion),
        },
        Literal("timing", description="Onsets, rise times and the slip-rate shape"): {
            Literal(
                "rupture_time_offset_s",
                description="The `offset` of sigma = offset + coefficient * 1e-9 * M0^(1/3). Both it and the coefficient zero is a coherent front",
            ): And(NUMBER, _is_non_negative),
            Literal(
                "rupture_time_coefficient",
                description="The `coefficient` of that relation, per cube-root dyne-centimetre",
            ): And(NUMBER, _is_non_negative),
            Literal(
                "rupture_time_correlation",
                description="Correlation of the onset perturbation with slip",
            ): NUMBER,
            Literal(
                "rupture_time_blend_sigma",
                description="How far the front travels before it carries the full onset perturbation, in units of its spread",
            ): And(NUMBER, _is_positive),
            Literal(
                "rupture_velocity_min_fraction",
                description="Floor on the perturbed rupture speed, as a fraction of Vs",
            ): And(NUMBER, _is_positive),
            Literal(
                "rupture_velocity_max_fraction",
                description="Ceiling on the perturbed rupture speed, as a fraction of Vs",
            ): And(NUMBER, _is_positive),
            Literal(
                "rupture_delay_s", description="Delay applied to every onset, seconds"
            ): And(NUMBER, _is_non_negative),
            Literal(
                "rise_time_correlation",
                description="Correlation of the rise-time perturbation with slip",
            ): NUMBER,
            Literal(
                "rise_time_sigma", description="Log-normal spread on the rise time"
            ): And(NUMBER, _is_positive),
            Literal(
                "slip_exponent",
                description="Exponent on slip in the rise-time scaling",
            ): NUMBER,
            Literal(
                "shallow_rise_factor",
                description="Rise time multiplier above `shallow_ramp`",
            ): And(NUMBER, _is_positive),
            Literal(
                "deep_rise_factor",
                description="Rise time multiplier below `deep_ramp`",
            ): And(NUMBER, _is_positive),
            Literal(
                "beta_shallow", description="Slip-rate pulse beta at the surface"
            ): And(NUMBER, _is_positive),
            Literal("beta_mid", description="Slip-rate pulse beta at mid depth"): And(
                NUMBER, _is_positive
            ),
            Literal("beta_deep", description="Slip-rate pulse beta at depth"): And(
                NUMBER, _is_positive
            ),
            Literal(
                "sample_interval_s", description="Slip-rate sample interval, seconds"
            ): And(NUMBER, _is_positive),
            Literal(
                "rise_time_blend",
                description="Depth ramp blending the shallow rise-time factor in",
            ): RAMP_SCHEMA,
            Literal(
                "beta_shallow_ramp",
                description="Depth ramp from `beta_shallow` to `beta_mid`",
            ): RAMP_SCHEMA,
            Literal(
                "beta_mid_ramp",
                description="Depth ramp from `beta_mid` to `beta_deep`",
            ): RAMP_SCHEMA,
        },
        Literal("field", description="The rake field's baseline"): {
            Literal(
                "base_rake_deg",
                description="Rake the field is drawn around where a fault does not state its own",
            ): NUMBER,
        },
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

NZCVM_SCHEMA = Schema(
    {
        Literal("layers"): [Use(LayerConfig.from_dict)],
        Literal("chunks"): Or({}, {Use(Coordinate): int}),
        Literal("surface"): Use(Path),
    }
)
SEED_SCHEMA = Schema(
    {
        Literal(
            "rupture_seed",
            description="The random seed passed to the rupture generator.",
        ): int,
        Literal(
            "nshm_to_realisation_seed",
            description="The random seed passed for NSHM -> realisation.",
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

HF_VELOCITY_MODEL_1D_SCHEMA = Schema(
    {
        **VELOCITY_MODEL_1D_SCHEMA.schema,
        Literal(
            "vs_moho",
            description="Shear velocity at which to truncate the model at the Moho (km/s)",
        ): And(NUMBER, _is_positive),
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
        Literal(
            "source",
            description="The earthquake source: radiation strength and rupture speed",
        ): {
            Literal("stress_drop_bars", description="Brune stress parameter"): And(
                NUMBER, _is_positive
            ),
            Literal(
                "corner_frequency_constant", description="c0 of Graves & Pitarka eq. 13"
            ): NUMBER,
            Literal(
                "corner_frequency_alpha",
                description="c_alpha of the alpha_T adjustment",
            ): NUMBER,
            Literal("rupture_velocity", description="Depth-dependent rupture taper"): {
                Literal(
                    "sigma", description="Log-normal scatter on the rupture factor"
                ): And(NUMBER, _is_non_negative),
            },
        },
        Literal("path", description="Which rays, and how the medium attenuates"): {
            Literal("rayset", description="ray types 1: direct, 2: moho"): [Or(1, 2)],
            Literal("q_frequency_exponent", description="x in Q(f) = Q0 f^x"): NUMBER,
            Literal(
                "path_duration_model",
                description="0: GP2010, 1: WUS, 2: ENA, 11: BT2014, 12: BT2015",
            ): Or(0, 1, 2, 11, 12),
        },
        Literal("site", description="The near-surface"): {
            Literal("kappa_s", description="Near-surface attenuation (s)"): NUMBER,
            Literal("fmax_hz", description="High-frequency cutoff"): And(
                NUMBER, _is_positive
            ),
        },
        Literal("record", description="The shape of the record to produce"): {
            Literal("dt", description="Sample interval (s)"): And(NUMBER, _is_positive),
        },
    }
)

STOCH_CONFIG_SCHEMA = Schema(
    {
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
        Literal("fhightop", description="fhightop for site amplification"): And(
            NUMBER, _is_non_negative
        ),
        Literal("fmax", description="fmax for site amplification"): And(
            NUMBER, _is_non_negative
        ),
        "site_amp_version": Use(SiteAmpModel),
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


EMPIRICAL_PARAMETERS = Schema(
    {
        Literal(
            "tect_type",
            description="Tectonic type of the source (one of oq_wrapper.constants.TectType)",
        ): str,
        Literal(
            "models",
            description=(
                "Ground motion models or ground motion model logic trees to "
                "evaluate (members of oq_wrapper.constants.GMM or "
                "oq_wrapper.constants.GMMLogicTree)"
            ),
        ): [str],
    }
)
# NOTE: The values of this schema are validated as plain strings rather than
# `oq_wrapper.constants` enum members. Importing `oq_wrapper.constants` pulls in
# OpenQuake, which is expensive (and must be precompiled), so the strings are
# only resolved to enum members inside the IM calculation stage.


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

REFINEMENT_SCHEMA = Schema(
    {
        Literal(
            "resolution", description="Vertical mesh resolution in this layer."
        ): And(NUMBER, _is_positive),
        Literal(
            "bottom", description="Bottom depth of this refinement layer (m)."
        ): And(NUMBER, _is_positive),
    }
)

REFINEMENTS_SCHEMA = Schema(
    {
        Literal(
            "refinements",
            description="List of vertical mesh refinements from top to bottom.",
        ): [REFINEMENT_SCHEMA],
        Literal(
            "unbounded_refinement_resolution",
            description="Resolution below the last refinement layer.",
        ): And(NUMBER, _is_positive),
    }
)

SW4_COMMAND_SCHEMA = Schema(
    {
        "name": str,
        "parameters": {str: Or(str, int, float, bool, None)},
    }
)

SW4_PARAMETERS_SCHEMA = Schema(
    {
        Literal("verbose", description="Fileio verbosity level."): int,
        Literal("printcycle", description="Output fileio print cycle."): int,
        Literal(
            "nz_min",
            description="Minimum vertical cells in each refinement layer.",
        ): int,
        Literal(
            "commands",
            description="List of SW4 input file commands (grid, attenuation, supergrid, developer, prefilter, topography, imagehdf5, or any other non-testing SW4 command).",
        ): [SW4_COMMAND_SCHEMA],
    }
)
