"""The realisations module defines the schema for the realisation file format.

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
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Self, Union

import numpy as np
from schema import Schema

from qcore import coordinates
from source_modelling import sources
from source_modelling.rupture_propagation import JumpPair
from source_modelling.sources import IsSource
from velocity_modelling.bounding_box import BoundingBox
from workflow import defaults, schemas
from workflow.defaults import DefaultsVersion


def to_name_coordinate_dictionary(
    coordinate_array: np.ndarray,
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


@dataclasses.dataclass
class RealisationConfiguration(ABC):
    """Abstract base class for RealisationConfiguration."""

    _config_key: ClassVar[str]
    _schema: ClassVar[Schema]

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
    def read_from_realisation(cls, realisation_ffp: Path) -> Self:
        """Read configuration from a realisation file.

        Parameters
        ----------
        realisation_ffp : Path
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
        cls, realisation_ffp: Path, defaults_version: DefaultsVersion
    ) -> Self:
        """Read configuration from realisation, or read from defaults and write to realisation.

        Parameters
        ----------
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
        try:
            return cls.read_from_realisation(realisation_ffp)
        except RealisationParseError:
            default_config = cls.read_from_defaults(defaults_version)
            default_config.write_to_realisation(realisation_ffp)
            return default_config

    def write_to_realisation(self, realisation_ffp: Path, update: bool = True) -> None:
        """Write a configuration to a realisation file.

        The default behaviour will update the realisation and replace just
        the configuration keys specified by `config`. If `update` is set
        to False, then the realisation is completely overwritten and
        populated with only the section pertaining to the config.

        Parameters
        ----------
        realisation_ffp : Path
            The realisation filepath to write to.
        update : bool
            If True, then the realisation is updated, rather than
            replaced. Default is True.
        """
        realisation_configuration = {}
        if realisation_ffp.exists() and update:
            with open(
                realisation_ffp, "r", encoding="utf-8"
            ) as realisation_file_handle:
                realisation_configuration = json.load(realisation_file_handle)
        realisation_configuration.update({self._config_key: self.to_dict()})
        with open(realisation_ffp, "w", encoding="utf-8") as realisation_file_handle:
            json.dump(realisation_configuration, realisation_file_handle)


@dataclasses.dataclass
class SourceConfig(RealisationConfiguration):
    """
    Configuration for defining sources.

    Attributes
    ----------
    sources : dict[str, "Source"]
        Dictionary mapping source names to their definitions.
    """

    _config_key: ClassVar[str] = "sources"
    _schema: ClassVar[Schema] = schemas.SOURCE_SCHEMA

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
                    "corners": to_name_coordinate_dictionary(geometry.corners),
                }
        return {"source_geometries": config_dict}


@dataclasses.dataclass
class SRFConfig(RealisationConfiguration):
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

    _config_key: ClassVar[str] = "srf"
    _schema: ClassVar[Schema] = schemas.SRF_SCHEMA

    genslip_dt: float
    genslip_seed: int
    genslip_version: str
    resolution: float
    srfgen_seed: int


@dataclasses.dataclass
class RupturePropagationConfig(RealisationConfiguration):
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

    _config_key: ClassVar[str] = "rupture_propagation"
    _schema: ClassVar[Schema] = schemas.RUPTURE_PROPAGATION_SCHEMA

    rupture_causality_tree: dict[str, str]
    jump_points: dict[str, JumpPair]
    rakes: dict[str, float]
    magnitudes: dict[str, float]
    hypocentre: np.ndarray

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
    """
    Parameters defining the spatial and temporal domain for simulation.

    Attributes
    ----------
    resolution : float
        The simulation resolution in kilometres.
    domain : BoundingBox
        The bounding box for the domain.
    depth : float
        The depth of the domain (in metres).
    duration : float
        The simulation duration (in seconds).
    dt : float
        The resolution of the domain in time (in seconds).
    """

    _config_key: ClassVar[str] = "domain"
    _schema: ClassVar[Schema] = schemas.DOMAIN_SCHEMA

    resolution: float
    domain: BoundingBox
    depth: float
    duration: float
    dt: float

    @property
    def nx(self) -> int:
        """int: The number of x coordinate positions in the discretised domain."""
        return int(np.round(self.domain.extent_x / self.resolution))

    @property
    def ny(self) -> int:
        """int: The number of y coordinate positions in the discretised domain."""
        return int(np.round(self.domain.extent_y / self.resolution))

    @property
    def nz(self) -> int:
        """int: The number of z coordinate positions in the discretised domain."""
        return int(np.round(self.depth / self.resolution))

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
            coordinates.nztm_to_wgs_depth(self.domain.corners),
        )
        return param_dict


@dataclasses.dataclass
class VelocityModelParameters(RealisationConfiguration):
    """Parameters defining the velocity model.

    min_vs : float
        The minimum velocity in the velocity model.
    version : str
        The velocity model version.
    topo_type : str
        The topology type of the velocity model.
    """

    _config_key: ClassVar[str] = "velocity_model"
    _schema: ClassVar[Schema] = schemas.VELOCITY_MODEL_SCHEMA

    min_vs: float
    version: str
    topo_type: str
    dt: float
    ds_multiplier: float
    resolution: float
    vs30: float
    s_wave_velocity: float


@dataclasses.dataclass
class RealisationMetadata(RealisationConfiguration):
    """
    Metadata for describing a realisation.

    Attributes
    ----------
    name : str
        The name of the realisation.
    version : str
        The version of the realisation format (currently supports version "5").
    tag : Optional[str]
        Metadata tag for the realisation used to specify the origin or
        category of the realisation (e.g. NSHM, GCMT or custom).
    """

    _config_key: ClassVar[str] = "metadata"
    _schema: ClassVar[Schema] = schemas.REALISATION_METADATA_SCHEMA

    name: str
    version: str
    defaults_version: DefaultsVersion
    tag: Optional[str] = None


@dataclasses.dataclass
class HFConfig(RealisationConfiguration):
    """High frequency simulation configuration.

    Attributes
    ----------
    nbu: float
        Unknown!
    ift: float
        Unknown!
    flo: float
        Unknown!
    fhi: float
        Unknown!
    nl_skip: int
        Skip empty lines in input?
    vp_sig: float
        Unknown!
    vsh_sig: float
        Unknown!
    rho_sig: float
        Unknown!
    qs_sig: float
        Unknown!
    ic_flag: bool
        Unknown!
    velocity_name: str
        Unknown
    dt: float
        High frequency time resolution.
    t_sec: float
        High frequency output start time.
    sdrop: float
        Stress drop average (bars)
    rayset: list[Literal[1, 2]]
        ray types 1: direct, 2: moho
    no_siteamp: bool
        Disable BJ97 site amplification factors
    fmax: float
        Max simulation frequency
    kappa: float
        Unknown!
    qfexp: float
        Q frequency exponent
    rvfac: float
        Rupture velocity factor (rupture : Vs)
    rvfac_shal: float
        rvfac shallow fault multiplier
    rvfac_deep: float
        rvfac deep fault multiplier
    czero: float
        C0 coefficient
    calpha: float
        Ca coefficient
    mom: Optional[float]
        Seismic moment for HF simulation (or None, to infer value)
    rupv: Optional[float]
        Rupture velocity (or binary default)
    site_specific: bool
        Enable site-specific calculation
    vs_moho: float
        vs of moho layer
    fa_sig1: float
        Fourier amplitude uncertainty (1)
    fa_sig2: float
        Fourier amplitude uncertainty (2)
    rv_sig1: float
        Rupture velocity uncertainty
    seed: float
        HF seed.
    path_dur: Literal[0, 1, 2, 11, 12]
        path duration model.
        - 0: GP2010
        - 1: WUS modification trail/error
        - 2: ENA modification trial/error
        - 11: WUS formulation of BT2014
        - 12: ENA formulation of BT2015. Models 11 and 12 over predict for multiple rays.
    dpath_pert: float
        Log of path duration multiplier
    stress_parameter_adjustment_tect_type: Literal[0, 1, 2]
        Adjustment option 0 = off, 1 = active tectonic, 2 = stable continent
    stress_parameter_adjustment_target_magnitude: Optional[float]
        Target magnitude (or inferred if None)
    stress_parameter_adjustment_fault_area: Optional[float]
        Target magnitude (or inferred if None)
    stress_parameter_fault_area: Optional[float]
        Fault area (or inferred if None)
    """

    _config_key: ClassVar[str] = "hf"
    _schema: ClassVar[Schema] = schemas.HF_CONFIG_SCHEMA

    dt: float
    nbu: int
    ift: int
    flo: float
    fhi: float
    nl_skip: int
    vp_sig: float
    vsh_sig: float
    qs_sig: float
    rho_sig: float
    ic_flag: bool
    velocity_name: str
    t_sec: float
    sdrop: float
    rayset: list[Literal[1, 2]]
    no_siteamp: bool
    fmax: float
    kappa: float
    qfexp: float
    rvfac: float
    rvfac_shal: float
    rvfac_deep: float
    seed: int
    czero: float
    calpha: float
    mom: Optional[float]
    rupv: Optional[float]
    site_specific: bool
    vs_moho: float
    fa_sig1: float
    fa_sig2: float
    rv_sig1: float
    path_dur: Literal[0, 1, 2, 11, 12]
    dpath_pert: float
    stress_parameter_adjustment_tect_type: Literal[0, 1, 2]
    stress_parameter_adjustment_target_magnitude: Optional[float]
    stress_parameter_adjustment_fault_area: Optional[float]
    # these are used in stoch generation, rather than HF invocation
    stoch_dx: float
    stoch_dy: float


@dataclasses.dataclass
class EMOD3DParameters(RealisationConfiguration):
    """Parameters for EMOD3D LF simulation."""

    _config_key: ClassVar[str] = "emod3d"
    _schema: ClassVar[Schema] = schemas.EMOD3D_PARAMETERS_SCHEMA

    all_in_one: int
    bfilt: int
    bforce: int
    dampwidth: int
    dblcpl: int
    dmodfile: str
    dtts: int
    dump_dtinc: int
    dxout: int
    dxts: int
    dyout: int
    dyts: int
    dzout: int
    dzts: int
    elas_only: int
    enable_output_dump: int
    enable_restart: int
    ffault: int
    fhi: float
    fmax: float
    fmin: float
    freesurf: int
    geoproj: int
    intmem: int
    ix_ts: int
    ix_ys: int
    ix_zs: int
    iy_ts: int
    iy_xs: int
    iy_zs: int
    iz_ts: int
    iz_xz: int
    iz_ys: int
    lonlat_out: int
    maxmem: int
    model_style: int
    nseis: int
    order: int
    pmodfile: str
    pointmt: int
    qbndmax: float
    qpfrac: float
    qpqs_factor: float
    qsfrac: float
    read_restart: int
    report: int
    restart_itinc: int
    scale: int
    smodfile: str
    span: int
    stype: str
    swap_bytes: int
    ts_inc: int
    ts_start: int
    ts_total: int
    ts_xy: int
    ts_xz: int
    ts_yz: int
    tzero: float
    vmodel_swapb: int
    xseis: int
    yseis: int
    zseis: int
    pertbfile: str
