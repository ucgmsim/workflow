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
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from nzcvm.config.layers import LayerConfig
from nzcvm.coordinates import Coordinate
from schema import Schema

from IM import im_calculation
from source_modelling import moment, sources
from source_modelling.magnitude_scaling import BoldM
from source_modelling.rupture_propagation import JumpPair
from source_modelling.sources import IsSource
from velocity_modelling.bounding_box import BoundingBox
from workflow import defaults, schemas
from workflow.defaults import DefaultsVersion


def to_name_coordinate_dictionary(
    coordinate_array: npt.NDArray[np.float64],
    coordinate_names: Sequence[str] = ("latitude", "longitude", "depth"),
) -> dict[str, float] | list[dict[str, float]]:
    """Convert an array of coordinates values into a (list of) dictionaries tagged with coordinate names.

    Parameters
    ----------
    coordinate_array : np.ndarray
        The array of values. Should have shape (k,), (m, k) where k is at most the length of `coordinate_names`.
    coordinate_names : Sequence[str]
        The names of the coordinates. Defaults to ('latitude', 'longitude', 'depth').

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


def path_serialiser(obj: Any) -> Any:
    """Serialise `Path` values, which `json` does not handle natively.

    Parameters
    ----------
    obj : Any
        The object `json` could not serialise.

    Returns
    -------
    Any
        The string form of `obj` if it is a `Path`, and `obj` unchanged
        otherwise.
    """
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
            json.dump(
                realisation_configuration,
                realisation_file_handle,
                indent=4,
                default=path_serialiser,
            )


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
    rupture_seed: int
    """The random seed passed to the rupture generator."""
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
                    # HF.
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
    """Kinematic rupture generation.

    **Mirrors `rupture_generator.config.rupture.RuptureConfig` section for section**, so
    this block deserialises straight into it rather than being translated. Keep it that
    way: a translation layer between two descriptions of the same physics is the thing
    that drifts.

    The sections of that class this one does *not* carry are the ones the realisation
    already knows. `hypocentre`, `velocity_model`, `random`, the per-fault
    `source.magnitudes` and `source.rakes`, and `propagation.parents` all come from
    other blocks. So does the depth-dependent rupture-speed profile
    -- `field.velocity_fraction`, `timing.shallow_speed_factor`, `timing.deep_speed_factor`,
    `timing.shallow_ramp` and `timing.deep_ramp` -- which lives in :class:`RuptureVelocity`
    because `hf-sim` reads the same physical values. All of them are injected by
    `create-rupture-input`.
    """

    _config_key: ClassVar[str] = "srf"
    _schema: ClassVar[Schema] = schemas.SRF_SCHEMA

    resolution: float
    """Subfault size (km) the fault is discretised at. Different, in general, from the
    simulation resolution."""
    source: dict[str, Any]
    """The corner relation and rise-time scaling. See `rupture_generator.config.rupture.SourceConfig`."""
    slip: dict[str, Any]
    """The slip field. See `rupture_generator.config.rupture.SlipConfig`."""
    timing: dict[str, Any]
    """Onsets, rise times and the slip-rate shape. See `rupture_generator.config.rupture.TimingConfig`."""
    field: dict[str, Any]
    """The rake field's baseline. See `rupture_generator.config.rupture.FieldConfig`."""

    @property
    def dt(self) -> float:
        """float: Slip-rate sample interval, seconds."""
        return self.timing["sample_interval_s"]


@dataclasses.dataclass
class Rakes(RealisationConfiguration):
    """Configuration for fault rakes."""

    _config_key: ClassVar[str] = "rakes"
    _schema: ClassVar[Schema] = schemas.RAKE_SCHEMA

    rakes: dict[str, float]
    """A map from faults to their rake angles."""

    def as_vectors(self) -> dict[str, npt.NDArray[np.float64]]:
        """Represent each rake angle as a unit vector.

        Rakes are angles, so they cannot be averaged directly (the mean of
        -179 and 179 degrees is 0, not 180). Averaging the unit vectors and
        recovering the angle with `arctan2` avoids this.

        Returns
        -------
        dict
            A map from faults to the unit vector of their rake angle.
        """
        return {
            k: np.array([np.cos(np.radians(r)), np.sin(np.radians(r))])
            for k, r in self.rakes.items()
        }

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

    @property
    def moments(self) -> dict[str, float]:  # numpydoc ignore=RT01
        """dict: a map from faults to their moment."""
        return {
            k: moment.magnitude_to_moment(mag, bold_m=True)
            for k, mag in self.magnitudes.items()
        }

    def moment_averaged(self, values: dict[str, Any]) -> Any:
        """Average per-fault quantities, weighted by fault moment.

        Parameters
        ----------
        values : dict
            A map from faults to the quantity to average. Every fault in
            this realisation must be present. Values may be scalars or
            arrays, provided they all share the same shape.

        Returns
        -------
        Any
            The moment-weighted average of `values`, with the same shape as
            the individual values.
        """
        keys = list(self.magnitudes)
        moments = self.moments
        return np.average(
            [values[key] for key in keys],
            weights=[moments[key] for key in keys],
            axis=0,
        )

    @property
    def total_moment(self) -> float:  # numpydoc ignore=RT01
        """float: total moment of realisation"""
        return sum(
            moment.magnitude_to_moment(mag, bold_m=True)
            for mag in self.magnitudes.values()
        )

    @property
    def total_magnitude(self) -> BoldM:  # numpydoc ignore=RT01
        """BoldM: total magnitude of realisation"""
        return moment.moment_to_magnitude(self.total_moment, bold_m=True)


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
class Refinement:
    """One vertical mesh refinement layer."""

    resolution: float
    """Grid spacing within this layer (metres)."""
    bottom: float
    """Depth of the bottom of this layer (metres)."""


@dataclasses.dataclass
class Refinements(RealisationConfiguration):
    """The vertical mesh refinements, from the surface down.

    SW4 solves on a stack of grids that coarsen with depth. This describes
    that stack in the abstract -- the layers a domain of *any* depth would
    be given -- and `refinements_for_depth` resolves it against a
    particular domain.
    """

    _config_key: ClassVar[str] = "refinements"
    _schema: ClassVar[Schema] = schemas.REFINEMENTS_SCHEMA
    refinements: list[Refinement]
    unbounded_refinement_resolution: float

    def __post_init__(self) -> None:
        """Coerce refinements read from JSON into `Refinement` instances."""
        if self.refinements and not isinstance(self.refinements[0], Refinement):
            self.refinements = [Refinement(**item) for item in self.refinements]  # type: ignore

    def refinements_for_depth(self, depth: float) -> list[Refinement]:
        """Resolve the refinement stack against a domain of a given depth.

        Layers below `depth` are dropped and the last one is truncated to
        it. If the stack does not reach `depth`, a final layer at
        `unbounded_refinement_resolution` extends to the bottom. The last
        layer is always given at least two cells, so a domain that ends
        just past a refinement boundary does not produce a degenerate grid.

        Parameters
        ----------
        depth : float
            The domain depth, in kilometres.

        Returns
        -------
        list of Refinement
            The layers covering `depth`, from the surface down.
        """
        depth_m = depth * 1000.0
        refinements = []
        for refinement in self.refinements:
            refinements.append(
                dataclasses.replace(refinement, bottom=min(refinement.bottom, depth_m))
            )
            if refinement.bottom > depth_m:
                break
        else:
            # This block only runs when we finish the loop without breaking, i.e. we
            # exhaust the refinement list.
            refinements.append(
                Refinement(
                    resolution=self.unbounded_refinement_resolution, bottom=depth_m
                )
            )

        match refinements:
            case [*_, previous_layer, last_layer]:
                # Ensure a minimum amount in the last layer.
                last_layer.bottom = max(
                    previous_layer.bottom + last_layer.resolution * 2, last_layer.bottom
                )

        return refinements


@dataclasses.dataclass
class NZCVMSettings(RealisationConfiguration):
    _config_key: ClassVar[str] = "nzcvm"
    _schema: ClassVar[Schema] = schemas.NZCVM_SCHEMA

    layers: list[LayerConfig]
    """nzcvm layer config"""
    chunks: dict[Coordinate, int]
    """nzcvm chunk configuration"""
    surface: Path
    """nzcvm DEM surface"""


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
    Vs of 500 m/s, and in carrying `vs_moho` -- which is a property of the model, not of
    the simulation, and truncates it.
    """

    _config_key: ClassVar[str] = "hf_velocity_model_1d"
    _schema: ClassVar[Schema] = schemas.HF_VELOCITY_MODEL_1D_SCHEMA

    vs_moho: float = 999.9
    """Shear velocity at which to truncate the model at the Moho, km/s."""


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
    """High frequency simulation configuration.

    **Mirrors `hf_simulation.HfConfig` group for group and field for field**, so this
    section can be deserialised straight into it rather than translated. Keep it that way:
    a translation layer between two descriptions of the same physics is the thing that
    drifts.

    Two of that class's inputs are deliberately absent. `record.duration_s` is computed
    from the domain, and the three `source.rupture_velocity` multipliers live in
    :class:`RuptureVelocity` because SRF generation reads the same physical values. Both
    are injected by `hf-sim`.
    """

    _config_key: ClassVar[str] = "hf"
    _schema: ClassVar[Schema] = schemas.HF_CONFIG_SCHEMA

    source: dict[str, Any]
    """Radiation strength and rupture speed. See `hf_simulation.SourceParameters`."""
    path: dict[str, Any]
    """Rays and attenuation. See `hf_simulation.PathParameters`."""
    site: dict[str, Any]
    """The near-surface. See `hf_simulation.SiteParameters`."""
    record: dict[str, Any]
    """Duration and sample interval. See `hf_simulation.RecordParameters`."""

    @property
    def dt(self) -> float:
        """float: Sample interval, seconds."""
        return self.record["dt"]


@dataclasses.dataclass
class StochConfig(RealisationConfiguration):
    """Stoch file generation.

    Not part of :class:`HFConfig`: these size the stoch grid, which is an input to the
    high-frequency simulation rather than one of its parameters.
    """

    _config_key: ClassVar[str] = "stoch"
    _schema: ClassVar[Schema] = schemas.STOCH_CONFIG_SCHEMA

    stoch_dx: float
    """stoch file resolution in x."""
    stoch_dy: float
    """stoch file resolution in y."""


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
    """fmidbot for the site amplification lowpass taper."""
    fmin: float
    """fmin for the site amplification lowpass taper."""
    fhightop: float
    """fhightop for the site amplification highpass taper."""
    fmax: float
    """fmax for the site amplification highpass taper."""
    site_amp_version: schemas.SiteAmpModel
    """The site amplification model to apply."""


@dataclasses.dataclass
class SW4Command:
    """A single SW4 input file command, e.g. `attenuation maxfreq=10 nmech=3`."""

    name: str
    """The SW4 command name (e.g. `attenuation`, `supergrid`, `imagehdf5`)."""
    parameters: dict[str, str | int | float | bool | None] = dataclasses.field(
        default_factory=dict
    )
    """The command's key=value parameters. None values are omitted when rendered."""

    def render(self) -> str:
        """Render this command as a single SW4 input file line.

        Returns
        -------
        str
            The command name followed by its non-None `key=value` parameters.
            Booleans render as `0`/`1`, SW4's expected format, rather than
            Python's `False`/`True`.
        """
        parts = [self.name]
        for key, value in self.parameters.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = int(value)
            parts.append(f"{key}={value}")
        return " ".join(parts)

    def merged(self, **overrides: str | float | bool | None) -> "SW4Command":
        """Return a copy of this command with `overrides` merged into its parameters.

        Parameters
        ----------
        **overrides : str | int | float | bool | None
            Parameter values to overlay on top of the existing parameters.

        Returns
        -------
        SW4Command
            A new command with the merged parameters.
        """
        return dataclasses.replace(self, parameters={**self.parameters, **overrides})


def find_command(commands: list[SW4Command], name: str) -> SW4Command | None:
    """Find the first command with the given name.

    Parameters
    ----------
    commands : list[SW4Command]
        The commands to search.
    name : str
        The command name to look for.

    Returns
    -------
    SW4Command | None
        The first matching command, or None if no command has this name.
    """
    return next((command for command in commands if command.name == name), None)


@dataclasses.dataclass
class SW4Parameters(RealisationConfiguration):
    """Parameters for SW4 simulation."""

    _config_key: ClassVar[str] = "sw4"
    _schema: ClassVar[Schema] = schemas.SW4_PARAMETERS_SCHEMA

    verbose: int
    """Fileio verbosity level."""
    printcycle: int
    """Output fileio print cycle."""
    nz_min: int
    """Minimum vertical cells in each refinement layer."""
    commands: list[SW4Command]
    """All other SW4 input file commands (grid projection, attenuation, supergrid,
    developer, prefilter, topography, imagehdf5 outputs, and any other non-testing
    SW4 command). Add any SW4 command here as `{"name": ..., "parameters": {...}}`."""

    def __post_init__(self) -> None:
        """Coerce commands read from JSON into `SW4Command` instances."""
        if self.commands and not isinstance(self.commands[0], SW4Command):
            self.commands = [SW4Command(**item) for item in self.commands]  # type: ignore


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
class EmpiricalParameters(RealisationConfiguration):
    """Empirical (ground motion model) intensity measure parameters."""

    _config_key: ClassVar[str] = "empirical"
    _schema: ClassVar[Schema] = schemas.EMPIRICAL_PARAMETERS

    # Types here are not explicitly declared so we do not pay the openquake tax
    # importing this module.
    tect_type: Any
    """The tectonic type of the source (an `oq_wrapper.constants.TectType`)."""
    models: list[Any]
    """The ground motion models or logic trees to evaluate."""


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
            timestamp=datetime.datetime.now(tz=datetime.UTC),
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
