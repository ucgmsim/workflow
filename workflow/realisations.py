"""The realisations module defines the schema for the realisation file format.

The configuration schemas are contained in `_REALISATION_SCHEMAS`. The
schemas loosely validates the input data. Input bounds checking is
done directly with the schema. More complicated input checking (for
example, that the rupture propagation defines a tree with one root
node) should be done outside this module. This is to avoid having this
module become an "everything" module.
"""

import dataclasses
import datetime
import json
import random
import struct
import sys
from abc import ABC
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from nzcvm.config.layers import LayerConfig
from nzcvm.coordinates import Coordinate
from schema import Schema

from IM import im_calculation
from source_modelling import sources
from source_modelling.magnitude_scaling import BoldM
from source_modelling.rupture_propagation import JumpPair
from source_modelling.sources import IsSource
from velocity_modelling.bounding_box import BoundingBox
from workflow import defaults, schemas
from workflow.defaults import DefaultsVersion


def to_name_coordinate_dictionary(
    coordinate_array: npt.NDArray[np.float64],
    coordinate_names: list[str] = ["latitude", "longitude", "depth"],
) -> Union[dict[str, float], list[dict[str, float]]]:
    """Convert an array of coordinates values into a (list of) dictionaries tagged with coordinate names.

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
        dict(zip(coordinate_names, coordinate_array.tolist()))
        for coordinate_array in np.atleast_2d(coordinate_array)
    ]

    if len(coordinate_array.shape) == 1:
        return coordinate_dicts[0]

    return coordinate_dicts


class RealisationParseError(Exception):
    """Realisation JSON parse error."""

    pass


def path_serialiser(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclasses.dataclass
class RealisationConfiguration(ABC):
    """Abstract base class for RealisationConfiguration."""

    _config_key: ClassVar[str]
    """The configuration key to save and load from in the realisation."""
    _schema: ClassVar[Schema]
    """The reference schema to validate against when reading from a realisation."""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        return dataclasses.asdict(self)

    @classmethod
    def read_from_realisation(cls, realisation_ffp: Path | str) -> Self:
        """Read configuration from a realisation file.

        Parameters
        ----------
        realisation_ffp : Path-like
            The filepath to read from.

        Returns
        -------
        RealisationConfiguration
            The configuration loaded from the realisation filepath. The
            configuration schema is looked up from `cls._config_key`
            and the key within the config is specified
            `cls._schema`.

        Raises
        ------
        RealisationParseError
            If the key in `cls._config_key` is not present in
            the realisation filepath.
        """
        realisation_ffp = Path(realisation_ffp)
        with open(realisation_ffp, "r", encoding="utf-8") as realisation_file_handle:
            realisation_config = json.load(realisation_file_handle)
            if cls._config_key not in realisation_config:
                raise RealisationParseError(
                    f"No {cls._config_key} in realisation configuration"
                )
        return cls(**cls._schema.validate(realisation_config[cls._config_key]))

    @classmethod
    def read_from_defaults(cls, defaults_version: DefaultsVersion) -> Self:
        """Read default values for this configuration.

        Parameters
        ----------
        defaults_version : DefaultsVersion
            The default parameter version to load with.

        Returns
        -------
        RealisationConfiguration
            The configuration loaded from the defaults. The configuration
            schema is looked up from `cls._config_key` and the key within
            the config is specified `cls._schema`.

        Raises
        ------
        RealisationParseError
            If the key in `cls._config_key` is not present in the scientific
            defaults configuration.
        """
        default_config = defaults.load_defaults(defaults_version)
        if cls._config_key not in default_config:
            raise RealisationParseError(
                f"No {cls._config_key} in defaults configuration"
            )
        return cls(**cls._schema.validate(default_config[cls._config_key]))

    @classmethod
    def read_from_realisation_or_defaults(
        cls, realisation_ffp: Path | str, defaults_version: DefaultsVersion
    ) -> Self:
        """Read configuration from realisation, or read from defaults and write to realisation.

        Parameters
        ----------
        realisation_ffp : Path-like
                    The realisation filepath to read from.
        defaults_version : DefaultsVersion
            The default parameter version to load with.

        Returns
        -------
        RealisationConfiguration
            The configuration loaded from the realisation filepath, or the
            defaults if the realisation does not contain the configuration
            key. The configuration schema is looked up from `cls._config_key`
            and the key within the config is specified `cls._schema`.

        Raises
        ------
        RealisationParseError
            If the key in `cls._config_key` is not present in
            the realisation or scientific defaults configuration.
        """
        realisation_ffp = Path(realisation_ffp)
        try:
            return cls.read_from_realisation(realisation_ffp)
        except (RealisationParseError, FileNotFoundError):
            default_config = cls.read_from_defaults(defaults_version)
            default_config.write_to_realisation(realisation_ffp)
            return default_config

    def write_to_realisation(
        self, realisation_ffp: Path | str, update: bool = True
    ) -> None:
        """Write a configuration to a realisation file.

        The default behaviour will update the realisation and replace just
        the configuration keys specified by `config`. If `update` is set
        to False, then the realisation is completely overwritten and
        populated with only the section pertaining to the config.

        Parameters
        ----------
        realisation_ffp : Path-like
            The realisation filepath to write to.
        update : bool
            If True, then the realisation is updated, rather than
            replaced. Default is True.
        """
        realisation_ffp = Path(realisation_ffp)
        realisation_configuration = {}
        if realisation_ffp.exists() and update:
            with open(
                realisation_ffp, "r", encoding="utf-8"
            ) as realisation_file_handle:
                realisation_configuration = json.load(realisation_file_handle)
        realisation_configuration.update({self._config_key: self.to_dict()})
        with open(realisation_ffp, "w", encoding="utf-8") as realisation_file_handle:
            json.dump(realisation_configuration, realisation_file_handle, indent=4, default=path_serialiser)


@dataclasses.dataclass
class Resolution(RealisationConfiguration):
    """Configuration for spatial/temporal resolution."""

    _config_key: ClassVar[str] = "resolution"
    _schema: ClassVar[Schema] = schemas.RESOLUTION_SCHEMA

    resolution: float
    """Simulation spatial resolution."""

    @property
    def dt(self) -> float:  # numpydoc ignore=RT01
        """float: Simulation temporal resolution."""
        return self.resolution / 20


@dataclasses.dataclass
class Seeds(RealisationConfiguration):
    """Configuration block for random seeds."""

    _config_key: ClassVar[str] = "seeds"
    _schema: ClassVar[Schema] = schemas.SEED_SCHEMA

    nshm_to_realisation_seed: int
    """The random seed for NSHM -> realisation."""
    rupture_propagation_seed: int
    """The random seed for rupture propagation."""
    genslip_seed: int
    """The random seed passed to genslip."""
    srfgen_seed: int
    """A second random seed for genslip, used for specific purposes in the generation process."""
    hf_seed: int
    """HF seed."""

    @classmethod
    def read_from_realisation_or_random(
        cls, realisation_ffp: Path
    ) -> Self:  # *args is to maintain compat with superclass (remove this and see the error in mypy).
        """Read seeds configuration from a realisation file or generate random seeds if not present.

        This method attempts to read the seeds configuration from the specified
        realisation file. If the configuration is not present or the file is not found,
        it generates a new seeds configuration with random seeds and writes it to the
        realisation file.

        Parameters
        ----------
        realisation_ffp : Path
            The realisation filepath to read from.

        Returns
        -------
        Seeds
            The seeds configuration loaded from the realisation filepath, or a new
            configuration with random seeds if the realisation does not contain the
            configuration key.

        Raises
        ------
        RealisationParseError
            If the key in `cls._config_key` is not present in the realisation or
            scientific defaults configuration.
        """
        try:
            return cls.read_from_realisation(realisation_ffp)
        except (RealisationParseError, FileNotFoundError):
            config = cls.random_seeds()
            config.write_to_realisation(realisation_ffp)
            return config

    @classmethod
    def random_seeds(cls) -> Self:
        """Generate random seeds for the seeds configuration.

        Returns
        -------
        Self
            A new instance of the configuration with random seeds.
        """
        return cls(
            **{
                field.name: random.randint(
                    # The following bounds for a random integer
                    # are based on the maximum machine size
                    # integer with the "i" datatype used in
                    # genslip and HF.
                    # See:
                    # https://stackoverflow.com/questions/13795758/what-is-sys-maxint-in-python-3/13796364#13796364
                    0,
                    2 ** (struct.Struct("i").size * 8 - 1) - 1,
                )
                for field in dataclasses.fields(cls)
            }
        )


@dataclasses.dataclass
class SourceConfig(RealisationConfiguration):
    """Configuration for defining sources."""

    _config_key: ClassVar[str] = "sources"
    _schema: ClassVar[Schema] = schemas.SOURCE_SCHEMA

    source_geometries: dict[str, IsSource]
    """Dictionary mapping source names to their definitions."""

    def to_dict(self) -> dict[str, Any]:
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
                    "width": geometry.width_m,
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
                    "corners": to_name_coordinate_dictionary(geometry.corners),
                }
        return {"source_geometries": config_dict}


@dataclasses.dataclass
class SRFConfig(RealisationConfiguration):
    """Configuration for SRF generation."""

    _config_key: ClassVar[str] = "srf"
    _schema: ClassVar[Schema] = schemas.SRF_SCHEMA

    resolution: float
    """The resolution of the SRF discretisation (different, in general, from the simulation resolution)."""

    point_source_params: schemas.PointSourceParams | None
    """Parameters for point source approximation, if applicable."""

    alpha_rough: float
    """Scalar indicating fault roughness (0 = disabled)"""

    side_taper: float
    """Fraction of along-strike length used to taper slip to zero at the lateral fault edges"""
    bot_taper: float
    """Fraction of down-dip width used to taper slip to zero at the bottom edge."""
    top_taper: float
    """Fraction of down-dip width used to taper slip to zero at the top (shallow)"""

    gwid: list[float]
    """Width of per-segment delay zone, see rvfac_seg"""
    rvfac_seg: list[float]
    """Per-segment delays for boundaries of segments"""
    seg_delay: bool
    """If true, enable per segment delays"""

    slip_sigma: float
    """Target SRF slip CoV"""

    risetime_coef: float
    """Constant used to scale the average slip rise time with seismic moment."""

    ymag_exponent: float | None
    """Sets correlation widths according to custom relation: clen_d = exp(bigM*(ymag_exponent*mag - ky_corner));"""
    xmag_exponent: float | None
    """Sets correlation lengths according to custom relation: clen_s = exp(bigM*(xmag_exponent*mag - kx_corner))"""
    kx_corner: float | None
    """Note: variable depends on kmodel. Corner wavenumber for along-strike correlation lengths."""
    ky_corner: float | None
    """Note: variable depends on kmodel. Corner wavenumber for down-dip correlation lengths."""

    # Rise time and source time function values
    beta_asp: float
    """Minimum value of β applied to asperity patches. This functionality is not used (as asp_mask = 0)."""
    beta_deep: float
    """Fraction of the rise time used in the computation of the slip-rate function used for deep subfaults. Default value according to Eq. 5 in Graves and Pitarka (2022)."""
    beta_mid: float
    """Fraction of the rise time used in the computation of the slip-rate function for mid-crust depths. Default value according to Eq. 5 in Graves and Pitarka (2022)."""
    beta_mid_depth: float
    """Center depth of the mid-to-deep transition zone (km). Default value according to Eq. 5 in Graves and Pitarka (2022)."""
    beta_mid_depth_range: float
    """Half-width of the mid-to-deep transition zone (km)."""
    beta_shal: float
    """Fraction of the rise time used in the computation of the slip-rate function for shallow subfaults. Default value according to Eq. 5 in Graves and Pitarka (2022)."""
    beta_shal_depth: float
    """Center depth of the shallow transition zone (km)."""
    beta_shal_depth_range: float
    """Half-width of the shallow transition zone (km)."""
    beta_subevt: float
    """Minimum value of β enforced on subevent patches. This functionality is not used (as subevt_mask = 0)."""
    deep_risetimedep: float
    """Sets the midpoint for the deep rise time adjustment zone."""
    deep_risetimedep_range: float
    """Sets the range for the deep rise time adjustment zone."""
    deep_risetimefac: float
    """Risetime adjustment factor"""
    risetimedep: float
    """Center depth of the shallow transition zone (km) used for local rise time"""
    risetimedep_range: float
    """Half-width of the shallow transition zone (km) used for local rise time"""
    risetimefac: float
    """Depth-dependent scaling factor applied to the local rise time for shallow and deep depths. Value consistent with GP10 Eq. 7."""
    rt_rand: float
    """Scales perturbation of risetimes so they are not 1:1 correlated with slip. ln(sigma)=rt_rand*trise"""
    rt_scalefac: float
    """A value of 1 implies that local rise time is scaled with sqrt(slip), consistent with GP10 Eq.7."""
    stype: schemas.Stype | None
    """Slip function type"""

    # Hybrid Correlation Length (hyb_corlen) Parameters
    hyb_corlen_deep_wt_end: float
    """Parameter setting the weighting in the hybrid slip approach"""
    hyb_corlen_deep_wt_start: float
    """Parameter setting the weighting in the hybrid slip approach"""
    hyb_corlen_dep: float
    """Center depth of the shallow-to-deep transition zone (km) used for the hybrid slip model"""
    hyb_corlen_dep_range: float
    """Half-width of the shallow-to-deep transition zone (km) used for the hybrid slip model"""
    hyb_corlen_fac: float
    """When using the Mai and Beroza (2022) model for both the shallow and deep regions in the hybrid slip approach, hyb_corlen_fac is the multiplicative factor used for adjusting the correlation length in the shallow region"""
    hyb_corlen_flag: bool
    """If enabled, incorporate hybrid correlation lengths, using two different correlation models for shallow and deep regions."""
    hyb_corlen_kmodel: schemas.KModel
    """Model selected for the shallow region in the hybrid slip approach. Default=5 is the Suzuki (2022) model."""
    hyb_corlen_shal_wt_end: float
    """Parameter setting the weighting in the hybrid slip approach"""
    hyb_corlen_shal_wt_start: float
    """Parameter setting the weighting in the hybrid slip approach"""
    hyb_corlen_side_taper: float
    """Side taper of hybrid correlation structure."""

    # Rupture Velocity & Fault Dimensions
    fdrup_scale_slip: bool
    """Scale rslow with slip prior to computing FD times"""
    fdrup_time: bool
    """Calculate rupture initiation times with wavefront propagation"""
    rupture_delay: float
    """Scalar delay to all rupture initiation times (s)"""
    rvfmax: float
    """Upper bound on how much faster rupture velocity can be than the average rupture velocity. Rupture velocity is adjusted to be faster in regions of high slip."""
    rvfmin: float
    """Lower bound on how much faster rupture velocity can be than the average rupture velocity. Rupture velocity is adjusted to be faster in regions of high slip."""

    # Tapering & Slip Level Adjustments
    truncate_zero_slip: bool
    """If true, truncates negative slip. This can skew the spectral distribution of slip but removes non-physical values."""
    slip_water_level: float | None
    """Minimum background slip level given as a percentage of the average slip amount (basically fills-in very low/zero slip patches with long rise time, low slip)."""
    rake_sigma: float
    """Target rake std deviation (absolute value, degrees)"""
    fractal_rake: bool
    """If enabled, uses a von Karman filter for rake (producing self-similar fractal rake)."""

    # Time Shift Factors (tsfac)
    tsfac1_scor: float
    """Allow specification of correlation levels between slip and rupture time. Correlation can be between 0 (uncorrelated) and 1.0 (1:1 correlation)."""
    tsfac1_sigma: float
    """Adjusts stddev of risetime perturbations given by gaussian random numbers (via tsfac1_r array) with zero mean and sigma set to tsfac1_sigma*tsfac."""
    tsfac2_lambda_max: float
    """Maximum wavelength considered when band-pass filtering the wavenumber spectra associated with the fault roughness model, as explained in GP16. Value in km. (Not used as alpha_rough=0)"""
    tsfac2_lambda_min: float | None
    """Minimum wavelength considered when band-pass filtering the wavenumber spectra associated with the fault roughness model, as explained in GP16. Value in km. (Not used as alpha_rough=0)"""
    tsfac2_scor: float
    """Allow specification of correlation levels between roughness and rupture time. Correlation can be between 0 (uncorrelated) and 1.0 (1:1 correlation). (Not used as alpha_rough=0)"""
    tsfac2_sigma: float
    """The standard deviation of the rupture time perturbations due to roughness."""
    tsfac_bzero: float
    """Offset constant value used when scaling the rupture time perturbation with seismic moment"""
    tsfac_coef: float
    """Coefficient equal to 1.1 given in Eq. A2 from GP16 used to scale rupture time perturbation with seismic moment. Not used anymore in the code, as now this scaling is performed with tsfac_bzero and tsfac_slope"""
    tsfac_main: float | None
    """Depends on tsfac_bzero, tsfac_slope and moment"""
    tsfac_slope: float
    """Coefficient used to scale rupture time perturbation with seismic moment, similar to Eq A2 from GP16 but now adding an offset value of tsfac_bzero"""

    # Kinematic Models & Corner Frequencies
    circular_average: bool
    """if set, correlation lengths are equal in both directions. only used if KModel is Mai or Sommerville."""
    kmodel: schemas.KModel
    """Correlation lengths relationship. Defaults to Mai 2002"""
    kord: int
    """The von Karman filter order. I believe this determines the rolloff from the correlation lengths"""
    magC: float  # noqa: N815
    """magnitude clamping for down-dip correlation lengths. only used if the KModel is Suzuki (5)."""
    mag_area_Acoef: float | None  # noqa: N815
    """the 'A' in M = A + B log10(area). only used if use_median_mag is true"""
    mag_area_Bcoef: float | None  # noqa: N815
    """the 'B' in M = A + B log10(area). only used if use_median_mag is true"""
    mai_wt: float
    """The weighting for Mai correlation model. Only used if KModel is set to hybrid Mai-Sommerville"""
    modified_corners: bool
    """another correlation model of unknown origin. overrides the value of KModel"""
    somerville_wt: float
    """Parameter only used if kmodel is set as MAI_SOMERVILLE_HYBRID_FLAG, which is not the default value. weight of the Sommerville model in the hybrid Sommerville and Mai correlation model"""
    stretch_kcorner: bool
    """Appears to be unused, purpose unknown"""
    use_gaus: bool
    """Unused"""
    use_median_mag: bool
    """If true set the magnitude of the rupture according to area"""

    # Roughness & Wavelength Limits
    lambda_max: float | None
    """Maximum wavelength for slip correlation (None = no limit)."""
    lambda_min: float | None
    """Minimum wavelength for slip correlation (None = no limit)."""
    wavelength_max: float | None
    """Maximum wavelength considered when band-pass filtering the rake spectral distribution (only if fractal_rake=0). It is also used to bandpass filter the initial risetime, and rupture time (perturbations) in the spectral domain. When it is not set, it gets a value 80% of the Nyquist frequency, 2 * sqrt(dx * dy) / 0.8. Hardcoded, will be overridden if set."""
    wavelength_min: float | None
    """Minimum wavelength considered when band-pass filtering the rake spectral distribution (only if fractal_rake=0). It is also used to bandpass filter the initial risetime, and rupture time (perturbations) in the spectral domain. When it is not set, it gets a value 80% of the Nyquist frequency, 2 * sqrt(dx * dy) / 0.8."""

    # Miscellaneous Slip/Rupture/Rake Vars
    asp_taper_fac: float
    """Size (proportion) of the asperity patch taper width."""
    extend_fac: float | None
    """geometric scaling variable used during multi-segment fault simulations to adjust how spatial wavenumber spectra are evaluated. It scales individual segment dimensions up to the full aggregate dimensions of the parent multi-segment fault system, ensuring long-wavelength features are not artificially truncated."""
    flen_max: float | None
    """Seems to be a maximum along-strike length to generate SRF output"""
    fwid_max: float | None
    """Seems to be a maximum down-dip width to generate SRF output"""
    moment_fraction: float | None
    """Scales seismic moment by fraction given"""
    perturb_subfault_location: bool
    """Shift subfault location according to roughness (in addition to perturbing strike and dip). Only used if alpha_rough > 0"""
    rand_rake_degs: float
    """Appears to be unused in code, purpose unknown"""
    rtime1_depth: float | None
    """This value sets the midpoint of the transition range for risetime-slip scaling. For depths above rtime_depth - rtime1_depth_range, risetime is directly proportional to slip. For depths below rtime_depth + rtime1_depth_range, risetime is correlated with slip + stochastic perturbations. The default value is set to beta_shal_depth."""
    rtime1_depth_range: float | None
    """The default value is set to beta_shal_depth_range. See rtime1_depth for a description."""
    rtime1_scor: float
    """Correlation coefficient between slip and rise time. set between 0 (uncorrelated) and 1 (perfectly correlated)"""
    rtime1_sigma: float | None
    """The coefficient of variation of the risetime distribution. The default value is  slip_sigma."""
    rtime2_scor: float
    """Correlation coefficient for additional perturbation of rise time with roughness."""
    rtime2slip_exp: float
    """Adjusts the power for the correlation between rise time and slip. Rise time is correlated with slip^p, defaulting to sqrt(slip)."""
    rtime_rand: float | None
    """If true, correlate rise time with roughness"""
    set_rake: float | None
    """If set, fix rake at every location"""
    svr_wt: float
    """Unclear to me (Jake) anyway what this does. this and a couple of other variable set rt_scalefac but so far as I can tell that variable is unused. to review further."""
    target_savg: float | None
    """Target slip average"""
    use_Mw: bool  # noqa: N815
    """If true, use Hanks and Kanamori (1979) Eq 4 for magnitude ('Mw'). Otherwise use Eq 7 ('M')."""

    # Aseismic & Segment Settings
    aseis_flag: bool
    """If true, enable aseismogenic creep adjustments."""
    aseis_smooth: bool
    """If true, smooth aseismogenic factors with an averaging kernel of width 3."""
    aseis_dep: float
    """Terminal depth for aseismic scaling region. slip above this depth is scaled down linearly, according to asei_fac. Below this depth no adjust occurs."""
    aseis_fac: float | None
    """Global setting for aseismic slip adjustment. Can be per subfault if read_aseis is set. only used if aseis_flag is true"""
    xshift: float
    """Shift phase of slip field"""
    yshift: float
    """Shift phase of slip field"""

    # IO settings
    read_erf: bool
    """If true, read an ERF file."""
    read_gsf: bool
    """If set, read a geometry input definition"""
    srf_version: str
    """Version of SRF to output (1.0 = basic, 2.0 = rho and Vs, 3.0 = Vp, rho, Vs and optional moment tensor)"""
    write_gsf: bool
    """Output geometry definition"""
    write_srf: bool
    """Output SRF"""
    dump_last_seed: bool
    """If true, write the final state of the SRF random seed generator to seedfile"""
    print_command: bool
    """If set, output SRF command to file"""
    print_seed: bool
    """If set, output SRF seed to file"""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        config_dict = dataclasses.asdict(self)
        if self.point_source_params is not None:
            config_dict["point_source_params"] = dataclasses.asdict(
                self.point_source_params
            )
        return config_dict


@dataclasses.dataclass
class Rakes(RealisationConfiguration):
    """Configuration for fault rakes."""

    _config_key: ClassVar[str] = "rakes"
    _schema: ClassVar[Schema] = schemas.RAKE_SCHEMA

    rakes: dict[str, float]
    """A map from faults to their rake angles."""

    def __getitem__(self, key: str) -> float:
        """Get the rake for a fault name.

        Parameters
        ----------
        key : str
            The fault rake to retrieve.

        Returns
        -------
        float
            The rake.
        """
        return self.rakes[key]


@dataclasses.dataclass
class Magnitudes(RealisationConfiguration):
    """Configuration for fault magnitudes."""

    _config_key: ClassVar[str] = "magnitudes"
    _schema: ClassVar[Schema] = schemas.MAGNITUDE_SCHEMA

    magnitudes: dict[str, BoldM]
    """A map from faults to their magnitudes."""

    def __getitem__(self, key: str) -> BoldM:
        """Get the magnitude for a fault name.

        Parameters
        ----------
        key : str
            The fault magnitude to retrieve.

        Returns
        -------
        BoldM
            The magnitude.
        """
        return self.magnitudes[key]


@dataclasses.dataclass
class RupturePropagationConfig(RealisationConfiguration):
    """Configuration for rupture propagation."""

    _config_key: ClassVar[str] = "rupture_propagation"
    _schema: ClassVar[Schema] = schemas.RUPTURE_PROPAGATION_SCHEMA

    rupture_causality_tree: dict[str, str | None]
    """A dict where the keys are faults and the values the parent fault (i.e. if fault a triggers fault b then rupture_causality_tree[fault b] = fault a)."""
    jump_points: dict[str, JumpPair]
    """A map from faults to pairs of fault-local coordinates representing jump points. If the rupture jumps from fault a at point a to point b on fault b then jump_points[fault a] = JumpPoint(point b, point a)."""
    hypocentre: npt.NDArray[np.float64]
    """The hypocentre of the fault."""

    def to_dict(self) -> dict[str, Any]:
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
                "from_point": to_name_coordinate_dictionary(
                    jump_point.from_point, ["s", "d"]
                ),
                "to_point": to_name_coordinate_dictionary(
                    jump_point.to_point, ["s", "d"]
                ),
            }
            for fault, jump_point in self.jump_points.items()
        }
        config_dict["hypocentre"] = to_name_coordinate_dictionary(
            self.hypocentre, ["s", "d"]
        )
        return config_dict

    @property
    def hypocentres(self) -> dict[str, npt.NDArray[np.float64]]:  # numpydoc ignore=RT01
        """Dict from str to array: the hypocentres on each fault in the simulation."""
        hypocentres = {
            fault_name: jump_point.to_point
            for fault_name, jump_point in self.jump_points.items()
        }

        hypocentres[self.initial_fault] = self.hypocentre

        return hypocentres

    @property
    def initial_fault(self) -> str:
        """The initial fault in the rupture.

        Returns
        -------
        str
            The initial fault in the rupture.
        """
        return next(
            fault_name
            for fault_name, parent_name in self.rupture_causality_tree.items()
            if parent_name is None
        )


@dataclasses.dataclass
class DomainParameters(RealisationConfiguration):
    """Parameters defining the spatial and temporal domain for simulation."""

    _config_key: ClassVar[str] = "domain"
    _schema: ClassVar[Schema] = schemas.DOMAIN_SCHEMA

    domain: BoundingBox
    """The bounding box for the domain."""
    depth: float
    """The depth of the domain (in metres)."""
    duration: float
    """The simulation duration (in seconds)."""

    def nx(self, resolution: float) -> int:
        """Calculate the number of point in the x-direction in the simulation domain.

        Parameters
        ----------
        resolution : float
            The simulation resolution in km, e.g. 0.1 for 100m.

        Returns
        -------
        int
            The number of points in the x-direction in the simulation domain.
        """
        # The C NZVM code always rounds 0.5 up to 1.0 but Python does not always
        # round in the same way. So we manually replicate the C rounding behaviour
        # here for consistency.
        return int((self.domain.extent_x / resolution) + 0.5)

    def ny(self, resolution: float) -> int:
        """Calculate the number of point in the y-direction in the simulation domain.

        Parameters
        ----------
        resolution : float
            The simulation resolution in km, e.g. 0.1 for 100m.

        Returns
        -------
        int
            The number of points in the y-direction in the simulation domain.
        """
        return int((self.domain.extent_y / resolution) + 0.5)

    def nz(self, resolution: float) -> int:
        """Calculate the number of point in the z-direction in the simulation domain.

        Parameters
        ----------
        resolution : float
            The simulation resolution in km, e.g. 0.1 for 100m.

        Returns
        -------
        int
            The number of points in the z-direction in the simulation domain.
        """
        return int((self.depth / resolution) + 0.5)

    def to_dict(self) -> dict:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        param_dict = dataclasses.asdict(self)
        param_dict["domain"] = to_name_coordinate_dictionary(
            self.domain.corners,
        )
        return param_dict


@dataclasses.dataclass
class VelocityModelParameters(RealisationConfiguration):
    """Parameters defining the velocity model."""

    _config_key: ClassVar[str] = "velocity_model"
    _schema: ClassVar[Schema] = schemas.VELOCITY_MODEL_SCHEMA

    min_vs: float
    """The minimum velocity in the velocity model."""
    version: str
    """The velocity model version."""
    topo_type: str
    """The topology type of the velocity model."""
    ds_multiplier: float
    """The ds multiplier used to adjust simulation duration."""
    vs30: float
    """The reference vs30 value for duration estimation."""
    s_wave_velocity: float
    """The s-wave velocity."""
    rrup_interpolants: npt.NDArray[np.float32]
    """Target RRup values at specific magnitudes, used to estimate domain size."""
    fault_buffer: float
    """Buffer width (km) around sources in rupture. Domain edge is guaranteed not be within this distance from any source."""
    layers: list[LayerConfig] | None
    """nzcvm layer config"""
    chunks: dict[Coordinate, int]
    """nzcvm chunk configuration"""
    surface: Path | None
    

    def to_dict(self) -> dict:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        _dict = dataclasses.asdict(self)
        _dict["rrup_interpolants"] = _dict["rrup_interpolants"].tolist()
        
        return _dict


@dataclasses.dataclass
class VelocityModel1D(RealisationConfiguration):
    """1D Velocity Model for SRF and HF."""

    _config_key: ClassVar[str] = "velocity_model_1d"
    _schema: ClassVar[Schema] = schemas.VELOCITY_MODEL_1D_SCHEMA

    model: pd.DataFrame

    def write_velocity_model(self, velocity_model_path: Path) -> None:
        """Write a 1D velocity model to the specified path.

        Parameters
        ----------
        velocity_model_path : Path
            The path to write the 1D velocity model.
        """
        with open(velocity_model_path, "w") as velocity_model:
            velocity_model.write(f"{len(self.model)}\n")
            self.model.to_csv(velocity_model, header=False, index=False, sep=" ")

    def to_dict(self) -> dict:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        _dict = dataclasses.asdict(self)
        _dict["model"] = _dict["model"].to_dict("records")
        return _dict


@dataclasses.dataclass
class HFVelocityModel1D(VelocityModel1D):
    """1D Velocity Model for SRF and HF.

    Differs from the VelocityModel1D class in the default case with a minimum
    Vs of 500 m/s."""

    _config_key: ClassVar[str] = "hf_velocity_model_1d"
    _schema: ClassVar[Schema] = schemas.VELOCITY_MODEL_1D_SCHEMA


@dataclasses.dataclass
class RealisationMetadata(RealisationConfiguration):
    """Metadata for describing a realisation."""

    _config_key: ClassVar[str] = "metadata"
    _schema: ClassVar[Schema] = schemas.REALISATION_METADATA_SCHEMA

    name: str
    """The name of the realisation."""
    version: str
    """The version of the realisation format (currently supports version "1")."""
    defaults_version: DefaultsVersion
    """The version of the scientific defaults to use."""
    tag: str | None = None
    """Metadata tag for the realisation used to specify the origin or
    category of the realisation (e.g. NSHM, GCMT or custom)."""


@dataclasses.dataclass
class RuptureVelocity(RealisationConfiguration):
    """Rupture velocities for source and high-frequency simulation."""

    _config_key: ClassVar[str] = "rupture_velocity"
    _schema: ClassVar[Schema] = schemas.RUPTURE_VELOCITY_SCHEMA

    rvfrac: float
    """Rupture velocity factor (rupture : Vs)"""
    rvfrac_shal: float
    """Rupture velocity at shallow depths"""
    rvfrac_slip_sig: float | None
    """Rupture velocity fraction slip sigma (None = disabled)."""
    rvfrac_deep: float
    """Rupture velocity at depth"""
    shallow_depth: float
    """Shallow depth (km). Marks transition to rvfrac_shal from rvfrac."""
    shallow_transition_range: float
    """Shallow depth transition range (km)."""
    deep_depth: float
    """Deep depth (km). Marks transition to rvfrac_deep from rvfrac."""
    deep_transition_range: float
    """Deep depth transition range (km)."""


@dataclasses.dataclass
class HFConfig(RealisationConfiguration):
    """High frequency simulation configuration."""

    _config_key: ClassVar[str] = "hf"
    _schema: ClassVar[Schema] = schemas.HF_CONFIG_SCHEMA

    nbu: int
    """Unknown!"""
    ift: int
    """Unknown!"""
    flo: float
    """Unknown!"""
    fhi: float
    """Unknown!"""
    nl_skip: int
    """Skip empty lines in input?"""
    vp_sig: float
    """Unknown!"""
    vsh_sig: float
    """Unknown!"""
    qs_sig: float
    """Unknown!"""
    rho_sig: float
    """Unknown!"""
    ic_flag: bool
    """Unknown!"""
    velocity_name: str
    """Unknown"""
    t_sec: float
    """High frequency output start time."""
    sdrop: float
    """Stress drop average (bars)"""
    rayset: list[Literal[1, 2]]
    """ray types 1: direct, 2: moho"""
    no_siteamp: bool
    """Disable BJ97 site amplification factors"""
    fmax: float
    """Max simulation frequency"""
    kappa: float
    """Unknown!"""
    qfexp: float
    """Q frequency exponent"""
    czero: float
    """C0 coefficient"""
    calpha: float
    """Ca coefficient"""
    mom: float | None
    """Seismic moment for HF simulation (or None, to infer value)"""
    rupv: float | None
    """Rupture velocity (or binary default)"""
    site_specific: bool
    """Enable site-specific calculation"""
    vs_moho: float
    """vs of moho layer"""
    fa_sig1: float
    """Fourier amplitude uncertainty (1)"""
    fa_sig2: float
    """Fourier amplitude uncertainty (2)"""
    rv_sig1: float
    """Rupture velocity uncertainty"""
    path_dur: Literal[0, 1, 2, 11, 12]
    """path duration model.
        - 0: GP2010
        - 1: WUS modification trail/error
        - 2: ENA modification trial/error
        - 11: WUS formulation of BT2014
        - 12: ENA formulation of BT2015. Models 11 and 12 over predict for multiple rays."""
    dpath_pert: float
    """Log of path duration multiplier"""
    stress_parameter_adjustment_tect_type: Literal[0, 1, 2]
    """Adjustment option 0 = off, 1 = active tectonic, 2 = stable continent"""
    stress_parameter_adjustment_target_magnitude: float | None
    """Target magnitude (or inferred if None)"""
    stress_parameter_adjustment_fault_area: float | None
    """Target magnitude (or inferred if None)"""
    # these are used in stoch generation, rather than HF invocation
    stoch_dx: float
    """stoch file resolution in x."""
    stoch_dy: float
    """stoch file resolution in x."""


@dataclasses.dataclass
class EMOD3DParameters(RealisationConfiguration):
    """Parameters for EMOD3D LF simulation."""

    _config_key: ClassVar[str] = "emod3d"
    _schema: ClassVar[Schema] = schemas.EMOD3D_PARAMETERS_SCHEMA

    all_in_one: int
    """Unknown!"""
    bfilt: int
    """Unknown!"""
    bforce: int
    """Unknown!"""
    dampwidth: int
    """Width of damping region"""
    dblcpl: int
    """Unknown!"""
    dmodfile: str
    """Path to density file"""
    dtts: int
    """dt per timeslice"""
    dump_itinc: int
    """Dump iteration increment"""
    dxout: int
    """Unknown!"""
    dxts: int
    """dx per timeslice"""
    dyout: int
    """Unknown!"""
    dyts: int
    """dy per timeslice"""
    dzout: int
    """Unknown!"""
    dzts: int
    """dz per timeslice"""
    elas_only: int
    """If non-zero, perform elastic calculations"""
    enable_output_dump: int
    """Unknown!"""
    enable_restart: int
    """Enable checkpoints"""
    ffault: int
    """If non-zero, source is a finite fault"""
    fhi: float
    """High-frequency cutoff?"""
    fmax: float
    """Maximum simulation frequency"""
    fmin: float
    """Minimum simulation frequency"""
    freesurf: int
    """Damping boundary related, 0 for absorbing"""
    geoproj: int
    """Geographic projection to use"""
    intmem: int
    """Unknown!"""
    ix_ts: int
    """Timeslice offset for ix?"""
    ix_ys: int
    ix_zs: int
    iy_ts: int
    iy_xs: int
    iy_zs: int
    iz_ts: int
    iz_xs: int
    iz_ys: int
    lonlat_out: int
    """Unknown!"""
    maxmem: int
    """Maximum memory usage in Mb"""
    model_style: int
    """Model type for simulation, 0 = 1d, 1 = 3d VM, 2 = 1d VM with 3d pertubations, 3 = 3d VM with 3d perturbations"""
    nseis: int
    """Individual points? (from the EMOD3D wiki page)"""
    order: int
    """Spatial differencing order"""
    pmodfile: str
    """Point to Vp file."""
    pointmt: int
    """Unknown!"""
    qbndmax: float
    """Unknown!"""
    qpfrac: float
    """Multiplier from Vp to Qp"""
    qpqs_factor: float
    """Ratio between qpfrac and qsfrac"""
    qsfrac: float
    """Multiplier from Vs to Qs"""
    read_restart: int
    """Read from checkpoint files?"""
    report: int
    """Unknown!"""
    scale: int
    """Unknown!"""
    smodfile: str
    """Path to vs file"""
    span: int
    """Unknown!"""
    stype: str
    """Unknown!"""
    swap_bytes: int
    """Endianness?"""
    ts_xy: int
    """Unknown!"""
    ts_xz: int
    """Unknown!"""
    ts_yz: int
    """Unknown!"""
    tzero: float
    """Start time offset"""
    vmodel_swapb: int
    """Velocity model endianness"""
    xseis: int
    """Unknown!"""
    yseis: int
    """Unknown!"""
    zseis: int
    """Unknown!"""
    pertbfile: str
    """Path to pertubation file"""


@dataclasses.dataclass
class BroadbandParameters(RealisationConfiguration):
    """Parameters for broadband waveform merger."""

    _config_key: ClassVar[str] = "bb"
    _schema: ClassVar[Schema] = schemas.BROADBAND_PARAMETERS_SCHEMA

    flo: float
    """low/high frequency cutoff."""
    fmidbot: float
    """fmidbot for site amplification"""
    fmin: float
    """fmin for site amplification."""
    site_amp_version: str


@dataclasses.dataclass
class IntensityMeasureCalculationParameters(RealisationConfiguration):
    """Intensity measure calculation parameters."""

    _config_key: ClassVar[str] = "im"
    _schema: ClassVar[Schema] = schemas.INTENSITY_MEASURE_CALCUATION_PARAMETERS

    ims: list[im_calculation.IM]
    """Intensity measures to calculate."""
    valid_periods: npt.NDArray[np.float64]
    """Valid periods to calculate for, applicable for pSA and SDI."""
    fas_frequencies: npt.NDArray[np.float64]
    """Fourier spectrum frequencies."""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        _dict = dataclasses.asdict(self)
        _dict["valid_periods"] = self.valid_periods.tolist()
        _dict["fas_frequencies"] = self.fas_frequencies.tolist()
        return _dict


@dataclasses.dataclass
class LogEntry:
    """Log entry for workflow utilities."""

    utility: str
    """The name of the utility."""
    version: str
    """The version of the utility."""
    timestamp: datetime.datetime
    """The timestamp of when the utility was run."""
    args: list[str]
    """The arguments passed to the utility."""

    def __post_init__(self) -> None:
        """Post-initialisation of the log entry."""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.datetime.fromisoformat(self.timestamp)

    @classmethod
    def from_utility(cls, utility: str, args: list[str]) -> Self:
        """Create a log entry from a utility.

        Parameters
        ----------
        utility : str
            The name of the utility.
        args : list[str]
            The arguments passed to the utility.

        Returns
        -------
        LogEntry
            A log entry for the utility.

        Raises
        ------
        RuntimeError
            If package metadata cannot be retrieved.
        """
        if not __package__:
            raise RuntimeError(
                "Cannot determine package name (__package__ is not set)."
            )
        version = metadata.version(__package__)
        return cls(
            utility=utility,
            version=version,
            timestamp=datetime.datetime.now(),
            args=args,
        )


@dataclasses.dataclass
class LogTrail(RealisationConfiguration):
    """Log of workflow utilities executed on this file."""

    _config_key: ClassVar[str] = "log_trail"
    _schema: ClassVar[Schema] = schemas.LOG_TRAIL_SCHEMA

    log: list[LogEntry]

    def __post_init__(self) -> None:
        """Post-initialisation of the log trail."""
        if self.log is None:
            self.log = []
        if self.log and not isinstance(self.log[0], LogEntry):
            self.log = [LogEntry(**log_entry) for log_entry in self.log]  # type: ignore

    def log_entry(self, utility: str, args: list[str]) -> None:
        """Add a log entry to the log trail.

        Parameters
        ----------
        utility : str
            The name of the utility.
        args : list[str]
            The arguments passed to the utility.
        """
        self.log.append(LogEntry.from_utility(utility, args))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        config_dict = dataclasses.asdict(self)
        for entry in config_dict["log"]:
            entry["timestamp"] = entry["timestamp"].isoformat()
        return config_dict


def append_log_entry(realisation_ffp: Path | str) -> None:
    """Append a log entry to the realisation file.

    Parameters
    ----------
    realisation_ffp : Path-like
        The realisation filepath to write to.
    """
    realisation_ffp = Path(realisation_ffp)
    utility = Path(sys.argv[0]).name
    args = sys.argv[1:]
    log_entry = LogEntry.from_utility(utility, args)

    try:
        log_trail = LogTrail.read_from_realisation(realisation_ffp)
        log_trail.log_entry(utility, args)
    except (RealisationParseError, FileNotFoundError):
        log_trail = LogTrail(log=[log_entry])

    log_trail.write_to_realisation(realisation_ffp)
