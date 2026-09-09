"""Validate SW4 sfile Script.

Description
-----------
Checks an SW4 sfile (HDF5 velocity model) for the structural and physical
defects that make SW4 abort or silently produce NaNs.

Every observation is recorded as a `Finding` with one of four severities:

error
    The model is unusable. SW4 aborts, or propagates NaNs.
warn
    The model is usable but suspicious, and worth a look.
info
    A measurement, reported so the model can be eyeballed. Suppressed by
    `--no-verbose`.
skip
    A check could not run because a prerequisite was missing. A run with skips
    is not a clean pass: something went unverified.

Recording is separate from reporting. Checks are generators over a parsed
`Sfile` and never print; `log_report` is the only thing that writes output, and
`Report.counts` tallies the four severities.

Boundaries are reported in NZTM2000. The sfile origin and azimuth define a
local grid, and the projection's false-origin offsets cancel over the
forward/inverse round trip, so these corners agree with the writer's.

Inputs
------
1. An SW4 sfile (HDF5 velocity model).

Outputs
-------
No direct outputs. Logs one record per finding and a closing tally, and exits
non-zero if any finding is an error.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `validate-sfile` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`validate-sfile [OPTIONS] SFILE`

For More Help
-------------
See the output of `validate-sfile --help`.
"""

import dataclasses
import itertools
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, NamedTuple, Protocol, Self

import h5py
import numpy as np
import numpy.typing as npt
import typer

from qcore import cli, coordinates
from workflow import log_utils

app = typer.Typer()

SURFACE_GROUP = "Z_interfaces"
MATERIAL_GROUP = "Material_model"
ATTENUATION_ATTR = "Attenuation"
NGRIDS_ATTR = "ngrids"
DEPTH_ATTR = "Min, max depth"
ORIGIN_ATTR = "Origin longitude, latitude, azimuth"
HORIZONTAL_ATTR = "Horizontal grid size"
COMPONENTS_ATTR = "Number of components"
SPACING_ATTRS = ("Finest horizontal grid spacing", "Coarsest horizontal grid spacing")
REQUIRED_ATTRS = (ATTENUATION_ATTR, NGRIDS_ATTR, DEPTH_ATTR, ORIGIN_ATTR)

SQRT2 = float(np.sqrt(2.0))
#: Cell-to-cell jump in the top interface above which the surface looks like a
#: fill-value boundary rather than terrain.
TERRAIN_JUMP_WARN_M = 500.0
#: Vertical cell size below which SW4's timestep becomes impractical.
THIN_CELL_WARN_M = 0.01
#: Fractional disagreement in horizontal extent tolerated between grids.
EXTENT_TOLERANCE = 0.001
#: Disagreement tolerated between the depth attribute and the interface data.
DEPTH_TOLERANCE_M = 1.0
#: Expected values of the "Number of components" attribute (with and without Q).
COMPONENT_COUNTS = (3, 5)


class Severity(StrEnum):
    """How much a finding matters."""

    ERROR = auto()
    WARN = auto()
    INFO = auto()
    SKIP = auto()


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """A single observation made about an sfile."""

    severity: Severity
    """How much this finding matters."""

    message: str
    """Human-readable description of the observation."""

    context: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    """Structured fields backing the message, logged alongside it."""

    check: str = ""
    """Name of the check that produced this finding, filled in by `validate`."""

    @classmethod
    def error(cls, message: str, **context: Any) -> "Finding":
        """Record that the model is unusable.

        Parameters
        ----------
        message : str
            Description of the defect.
        **context : Any
            Structured fields to log alongside the message.

        Returns
        -------
        Finding
            The finding.
        """
        return cls(Severity.ERROR, message, context)

    @classmethod
    def warn(cls, message: str, **context: Any) -> "Finding":
        """Record that the model is usable but suspicious.

        Parameters
        ----------
        message : str
            Description of the suspicion.
        **context : Any
            Structured fields to log alongside the message.

        Returns
        -------
        Finding
            The finding.
        """
        return cls(Severity.WARN, message, context)

    @classmethod
    def info(cls, message: str, **context: Any) -> "Finding":
        """Record a measurement of the model.

        Parameters
        ----------
        message : str
            Description of the measurement.
        **context : Any
            Structured fields to log alongside the message.

        Returns
        -------
        Finding
            The finding.
        """
        return cls(Severity.INFO, message, context)

    @classmethod
    def skip(cls, message: str, **context: Any) -> "Finding":
        """Record that a check could not run.

        Parameters
        ----------
        message : str
            What could not be checked, and why.
        **context : Any
            Structured fields to log alongside the message.

        Returns
        -------
        Finding
            The finding.
        """
        return cls(Severity.SKIP, message, context)


@dataclasses.dataclass(frozen=True, slots=True)
class Report:
    """The findings from one validation run."""

    findings: tuple[Finding, ...]
    """Every finding, in the order the checks produced them."""

    @property
    def counts(self) -> Counter[Severity]:  # numpydoc ignore=RT01
        """Counter[Severity]: how many findings of each severity."""
        return Counter(finding.severity for finding in self.findings)

    @property
    def failed(self) -> bool:  # numpydoc ignore=RT01
        """bool: whether any finding is an error."""
        return any(finding.severity is Severity.ERROR for finding in self.findings)


class UnavailableError(Exception):
    """A check's prerequisite is missing, so the check cannot run."""


@dataclasses.dataclass(frozen=True, slots=True)
class VarSpec:
    """How one material variable is checked and reported."""

    label: str
    """Name used in messages, e.g. `Vp` for the `Cp` dataset."""

    unit: str
    """Unit appended to reported ranges."""

    zero_allowed: bool
    """Whether zero is legal, as it is for Vs in fluid cells."""

    nonpositive_note: str
    """Why a non-positive value breaks SW4."""

    low_warn: float | None
    """Warn when the minimum is positive but below this, or None to not warn."""

    attenuation_only: bool = False
    """Whether the variable is only required when attenuation is enabled."""


VARS = {
    "Rho": VarSpec("Rho", "kg/m^3", False, "SW4 aborts on a zero density", 100.0),
    "Cp": VarSpec("Vp", "m/s", False, "SW4 requires a positive Vp", 200.0),
    "Cs": VarSpec("Vs", "m/s", True, "Vs may not be negative", None),
    "Qp": VarSpec("Qp", "", False, "Q must be > 0 when attenuation=1", None, True),
    "Qs": VarSpec("Qs", "", False, "Q must be > 0 when attenuation=1", None, True),
}


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetStats:
    """Summary of one material dataset, accumulated over chunks."""

    lo: float
    """Smallest finite value, or +inf if there are none."""

    hi: float
    """Largest finite value, or -inf if there are none."""

    n_nan: int
    """Count of NaN values."""

    n_inf: int
    """Count of infinite values."""

    n_zero: int
    """Count of exactly-zero values."""

    n_neg: int
    """Count of negative values."""

    @property
    def n_nonpositive(self) -> int:  # numpydoc ignore=RT01
        """int: count of values that are zero or negative."""
        return self.n_zero + self.n_neg

    def merge(self, other: Self) -> Self:
        """Combine these statistics with those of another chunk.

        Parameters
        ----------
        other : DatasetStats
            Statistics from a further chunk of the same dataset.

        Returns
        -------
        DatasetStats
            Statistics covering both chunks.
        """
        return dataclasses.replace(
            self,
            lo=min(self.lo, other.lo),
            n_nan=self.n_nan + other.n_nan,
            n_inf=self.n_inf + other.n_inf,
            n_zero=self.n_zero + other.n_zero,
            n_neg=self.n_neg + other.n_neg,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RatioStats:
    """Summary of the Vp/Vs ratio over the solid (Vs > 0) cells of a grid."""

    n_solid: int
    """Number of cells with Vs > 0."""

    lo: float
    """Smallest ratio, or +inf if there are no solid cells."""

    hi: float
    """Largest ratio, or -inf if there are no solid cells."""

    n_below_one: int
    """Count of cells with Vp/Vs < 1, which is physically impossible."""

    n_below_sqrt2: int
    """Count of cells with Vp/Vs < sqrt(2). Contains `n_below_one`."""

    def merge(self, other: Self) -> Self:
        """Combine these statistics with those of another chunk.

        Parameters
        ----------
        other : RatioStats
            Statistics from a further chunk of the same grid.

        Returns
        -------
        RatioStats
            Statistics covering both chunks.
        """
        # TODO: obvious replace implementation here...
        return RatioStats(
            self.n_solid + other.n_solid,
            min(self.lo, other.lo),
            max(self.hi, other.hi),
            self.n_below_one + other.n_below_one,
            self.n_below_sqrt2 + other.n_below_sqrt2,
        )


EMPTY_RATIO = RatioStats(0, np.inf, -np.inf, 0, 0)


def _sorted_keys(group: h5py.Group) -> list[str]:
    """Sort a group's keys by the integer index in their suffix.

    Parameters
    ----------
    group : h5py.Group
        The group whose keys to sort.

    Returns
    -------
    list of str
        The keys, ordered by trailing index rather than lexically, so
        `z_values_10` sorts after `z_values_9`.
    """
    return sorted(group.keys(), key=lambda key: int(key.rsplit("_", 1)[-1]))


def _scalar(value: npt.ArrayLike) -> float:
    """Read the first element of `value` as a float.

    Parameters
    ----------
    value : array_like
        An HDF5 attribute value, which may be a scalar or a 1-element array.

    Returns
    -------
    float
        The first element.
    """
    return float(np.asarray(value).flat[0])


def _vector(value: npt.ArrayLike, length: int) -> tuple[float, ...] | None:
    """Read `value` as a flat tuple of exactly `length` floats.

    Parameters
    ----------
    value : array_like
        An HDF5 attribute value.
    length : int
        The number of elements the attribute must hold.

    Returns
    -------
    tuple of float or None
        The elements, or None if the attribute is the wrong length.
    """
    flat = np.asarray(value).ravel()
    return tuple(float(element) for element in flat) if len(flat) == length else None


def _resample_to(
    src: npt.NDArray[np.floating], shape: tuple[int, int]
) -> npt.NDArray[np.floating]:
    """Nearest-neighbour resample 2-D `src` onto `shape`.

    Parameters
    ----------
    src : numpy.ndarray
        The 2-D array to resample.
    shape : tuple of int
        The target `(rows, columns)`.

    Returns
    -------
    numpy.ndarray
        `src` resampled onto `shape`, or `src` itself if it already has that
        shape.
    """
    if src.shape == shape:
        return src
    rows = np.rint(np.linspace(0.0, src.shape[0] - 1, shape[0])).astype(int)
    columns = np.rint(np.linspace(0.0, src.shape[1] - 1, shape[1])).astype(int)
    return src[np.ix_(rows, columns)]


# TODO: remove and subsume into cached property: only one callsite
def _align(
    a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], bool]:
    """Put `a` and `b` on a common grid so they compare elementwise.

    Parameters
    ----------
    a, b : numpy.ndarray
        The 2-D arrays to align. The coarser is resampled onto the finer one's
        grid.

    Returns
    -------
    tuple
        `(a, b, resampled)`, where `resampled` says whether either array had to
        be resampled.
    """
    if a.shape == b.shape:
        return a, b, False
    target = a.shape if a.size >= b.size else b.shape
    return _resample_to(a, target), _resample_to(b, target), True


# TODO: remove and subsume into caller: only one callsite
def _worst_of(
    field: npt.NDArray[np.floating], selected: npt.NDArray[np.bool_]
) -> tuple[int, int, float]:
    """Locate the smallest selected element of a 2-D field.

    Masking to the selected points before searching keeps NaNs elsewhere in
    the field from displacing the answer, since a NaN compares false against
    every threshold and so is never selected.

    Parameters
    ----------
    field : numpy.ndarray
        The 2-D field to search.
    selected : numpy.ndarray
        Boolean mask of the points to consider, which must select at least
        one point.

    Returns
    -------
    tuple
        `(i, j, value)`: the index of the smallest selected point, and its
        value.
    """
    flat_index = int(np.argmin(np.where(selected, field, np.inf)))
    i, j = np.unravel_index(flat_index, field.shape)
    return int(i), int(j), float(field.flat[flat_index])


# TODO: remove and subsume into caller: only one callsite
def _chunk_stats(chunk: npt.NDArray[np.floating]) -> DatasetStats:
    """Summarise one slab of a material dataset.

    Parameters
    ----------
    chunk : numpy.ndarray
        A slab of the dataset.

    Returns
    -------
    DatasetStats
        The statistics for this slab.
    """
    finite = np.isfinite(chunk)
    n_nan = int(np.count_nonzero(np.isnan(chunk)))
    return DatasetStats(
        float(np.min(chunk, where=finite, initial=np.inf)),
        float(np.max(chunk, where=finite, initial=-np.inf)),
        n_nan,
        chunk.size - int(np.count_nonzero(finite)) - n_nan,
        int(np.count_nonzero(chunk == 0.0)),
        int(np.count_nonzero(chunk < 0.0)),
    )


# TODO: remove and subsume into caller: only one callsite
def _chunk_ratio(
    cp: npt.NDArray[np.floating], cs: npt.NDArray[np.floating]
) -> RatioStats:
    """Summarise the Vp/Vs ratio over the solid cells of one slab.

    Parameters
    ----------
    cp : numpy.ndarray
        A slab of the Cp dataset.
    cs : numpy.ndarray
        The matching slab of the Cs dataset.

    Returns
    -------
    RatioStats
        The ratio statistics for this slab.
    """
    solid = cs > 0.0
    if not solid.any():
        return EMPTY_RATIO
    # Select before dividing, so the division runs over the solid cells only.
    ratio = cp[solid].astype(np.float64) / cs[solid]
    return RatioStats(
        int(ratio.size),
        float(ratio.min()),
        float(ratio.max()),
        int(np.count_nonzero(ratio < 1.0)),
        int(np.count_nonzero(ratio < SQRT2)),
    )


class GridExtent(NamedTuple):
    """The horizontal domain one material grid spans."""

    h: float
    """Horizontal grid spacing in metres."""

    x: float
    """Extent along the grid's x-axis in metres."""

    y: float
    """Extent along the grid's y-axis in metres."""


@dataclasses.dataclass
class MaterialGrid:
    """One refinement level of an sfile's material model."""

    name: str
    """The grid's name within the material group."""

    group: h5py.Group
    """The HDF5 group holding this grid's datasets."""

    h: float | None
    """Horizontal grid spacing in metres, or None if the attribute is absent."""

    n_components: int | None
    """Declared component count, or None if the attribute is absent."""

    shape: tuple[int, ...] | None
    """Shape of the Cp dataset, or None if Cp is absent."""

    @property
    def extent(self) -> Self | None:  # numpydoc ignore=RT01
        """GridExtent or None: the domain this grid spans, if it is known."""
        if self.h is None or self.shape is None or len(self.shape) != 3:
            return None
        # TODO: obvious replace implementation
        return GridExtent(
            self.h, (self.shape[0] - 1) * self.h, (self.shape[1] - 1) * self.h
        )


@dataclasses.dataclass
class Sfile:
    """An sfile parsed into the values the checks assert against.

    Parsing records what the file says; it does not judge it. A malformed or
    absent value becomes None here and is reported by `check_attributes`, so
    one bad attribute does not silently disable unrelated checks.
    """

    path: Path
    """Path the model was read from."""

    handle: h5py.File
    """The open HDF5 file. Datasets are read lazily through it."""

    chunk_rows: int
    """Number of `i` rows read at a time when scanning material datasets."""

    ngrids: int | None
    """Declared number of material grids."""

    attenuation: int | None
    """Declared attenuation flag."""

    depth_range: tuple[float, ...] | None
    """Declared `(min, max)` depth in metres, depth-positive."""

    origin: tuple[float, ...] | None
    """Declared `(longitude, latitude, azimuth)` of the grid origin."""

    spacings: dict[str, float]
    """The horizontal spacing attributes that are present, by attribute name."""

    interfaces: dict[str, h5py.Dataset]
    """Depth interface datasets, shallowest first."""

    grids: dict[str, MaterialGrid]
    """Material grids, in refinement order."""

    @cached_property
    def interface_arrays(self) -> dict[str, npt.NDArray[np.float64]]:
        """Mapping of interface arrays.

        Returns
        -------
        dict
            Each 2-D interface by name, shallowest first.
        """
        return {
            name: dataset[()].astype(np.float64)
            for name, dataset in self.interfaces.items()
            if dataset.ndim == 2
        }

    @cached_property
    def layers(self) -> dict[tuple[str, str], tuple[npt.NDArray[np.float64], bool]]:
        """Compute the thickness of every layer between consecutive interfaces.

        A non-positive thickness is both a monotonicity violation (the deeper
        interface is not below the shallower one) and a zero-thickness layer,
        so both checks read one array rather than each computing its own.

        Returns
        -------
        dict
            Keyed by `(top name, bottom name)`, holding `(thickness,
            resampled)`, where `resampled` says whether the pair had to be put
            on a common grid first.
        """
        arrays = self.interface_arrays
        layers = {}
        for top, bottom in itertools.pairwise(arrays):
            top_z, bottom_z, resampled = _align(arrays[top], arrays[bottom])
            layers[top, bottom] = (bottom_z - top_z, resampled)
        return layers

    def require_interfaces(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return the interface arrays, or refuse to run the check.

        Returns
        -------
        dict
            Each 2-D interface by name, shallowest first.

        Raises
        ------
        UnavailableError
            If the file holds no readable 2-D interfaces.
        """
        if not self.interface_arrays:
            raise UnavailableError(f"no readable 2-D datasets in '{SURFACE_GROUP}'")
        return self.interface_arrays

    def require_grids(self) -> dict[str, MaterialGrid]:
        """Return the material grids, or refuse to run the check.

        Returns
        -------
        dict
            Material grids, in refinement order.

        Raises
        ------
        UnavailableError
            If the file holds no material grids.
        """
        if not self.grids:
            raise UnavailableError(f"no grids in '{MATERIAL_GROUP}'")
        return self.grids

    def require_origin(self) -> tuple[float, ...]:
        """Return the grid origin, or refuse to run the check.

        Returns
        -------
        tuple of float
            The `(longitude, latitude, azimuth)` of the grid origin.

        Raises
        ------
        UnavailableError
            If the origin attribute is absent or malformed.
        """
        if self.origin is None:
            raise UnavailableError(f"'{ORIGIN_ATTR}' is absent or malformed")
        return self.origin


def read_sfile(path: Path, handle: h5py.File, chunk_rows: int) -> Sfile:
    """Parse an open sfile into the model the checks run against.

    Parameters
    ----------
    path : Path
        The path the file was opened from, used only in messages.
    handle : h5py.File
        The open sfile.
    chunk_rows : int
        Number of `i` rows to read at a time when scanning material datasets.

    Returns
    -------
    Sfile
        The parsed model.
    """
    attrs = handle.attrs
    interfaces: dict[str, h5py.Dataset] = {}
    if SURFACE_GROUP in handle:
        group = handle[SURFACE_GROUP]
        interfaces = {name: group[name] for name in _sorted_keys(group)}

    grids: dict[str, MaterialGrid] = {}
    if MATERIAL_GROUP in handle:
        group = handle[MATERIAL_GROUP]
        for name in _sorted_keys(group):
            grid = group[name]
            grids[name] = MaterialGrid(
                name=name,
                group=grid,
                h=_scalar(grid.attrs[HORIZONTAL_ATTR])
                if HORIZONTAL_ATTR in grid.attrs
                else None,
                n_components=int(_scalar(grid.attrs[COMPONENTS_ATTR]))
                if COMPONENTS_ATTR in grid.attrs
                else None,
                shape=tuple(grid["Cp"].shape) if "Cp" in grid else None,
            )

    return Sfile(
        path=path,
        handle=handle,
        chunk_rows=chunk_rows,
        ngrids=int(_scalar(attrs[NGRIDS_ATTR])) if NGRIDS_ATTR in attrs else None,
        attenuation=int(_scalar(attrs[ATTENUATION_ATTR]))
        if ATTENUATION_ATTR in attrs
        else None,
        depth_range=_vector(attrs[DEPTH_ATTR], 2) if DEPTH_ATTR in attrs else None,
        origin=_vector(attrs[ORIGIN_ATTR], 3) if ORIGIN_ATTR in attrs else None,
        spacings={
            name: _scalar(attrs[name]) for name in SPACING_ATTRS if name in attrs
        },
        interfaces=interfaces,
        grids=grids,
    )


def check_attributes(model: Sfile) -> Iterator[Finding]:
    """Root attributes are present, well-formed and in range.

    Parameters
    ----------
    model : Sfile
        The parsed sfile to check.

    Yields
    ------
    Finding
        One finding per observation, and a skip for anything it could not
        check.
    """
    for name in REQUIRED_ATTRS:
        if name not in model.handle.attrs:
            yield Finding.error(f"missing required attribute '{name}'", attr=name)

    if not model.spacings:
        yield Finding.error(
            f"missing a horizontal spacing attribute; need one of {list(SPACING_ATTRS)}"
        )
    for name, spacing in model.spacings.items():
        yield Finding.info(f"{name}: {spacing} m", attr=name, spacing=spacing)
        if not np.isfinite(spacing) or spacing <= 0:
            yield Finding.error(
                f"'{name}' must be finite and > 0, got {spacing}", attr=name
            )

    if model.attenuation is not None:
        yield Finding.info(
            f"attenuation: {model.attenuation}", attenuation=model.attenuation
        )
        if model.attenuation not in (0, 1):
            yield Finding.error(
                f"attenuation must be 0 or 1, got {model.attenuation}",
                attenuation=model.attenuation,
            )

    if model.ngrids is not None:
        yield Finding.info(f"ngrids: {model.ngrids}", ngrids=model.ngrids)
        if model.ngrids <= 0:
            yield Finding.error(
                f"ngrids must be > 0, got {model.ngrids}", ngrids=model.ngrids
            )

    yield from _check_depth_attr(model)
    yield from _check_origin_attr(model)


def _check_depth_attr(model: Sfile) -> Iterator[Finding]:
    """Check the declared depth range.

    Parameters
    ----------
    model : Sfile
        The parsed sfile.

    Yields
    ------
    Finding
        One finding per observation about the depth attribute.
    """
    if DEPTH_ATTR not in model.handle.attrs:
        return
    if model.depth_range is None:
        yield Finding.error(f"'{DEPTH_ATTR}' must have 2 values")
        return

    zmin, zmax = model.depth_range
    yield Finding.info(
        f"depth range [{zmin:.2f}, {zmax:.2f}] m, depth-positive "
        f"(negative is above sea level)",
        zmin=zmin,
        zmax=zmax,
    )
    if not (np.isfinite(zmin) and np.isfinite(zmax)):
        yield Finding.error(f"'{DEPTH_ATTR}' contains NaN or Inf")
    elif zmin >= zmax:
        yield Finding.error(
            f"min depth ({zmin:.2f} m) must be < max depth ({zmax:.2f} m)"
        )


def _check_origin_attr(model: Sfile) -> Iterator[Finding]:
    """Check the declared grid origin.

    Parameters
    ----------
    model : Sfile
        The parsed sfile.

    Yields
    ------
    Finding
        One finding per observation about the origin attribute.
    """
    if ORIGIN_ATTR not in model.handle.attrs:
        return
    if model.origin is None:
        yield Finding.error(f"'{ORIGIN_ATTR}' must have 3 values")
        return

    lon, lat, azimuth = model.origin
    yield Finding.info(
        f"origin lon={lon:.6f}° lat={lat:.6f}° azimuth={azimuth:.4f}°",
        lon=lon,
        lat=lat,
        azimuth=azimuth,
    )
    if not -180 <= lon <= 180:
        yield Finding.error(f"origin longitude {lon} out of range [-180, 180]", lon=lon)
    if not -90 <= lat <= 90:
        yield Finding.error(f"origin latitude {lat} out of range [-90, 90]", lat=lat)
    if not np.isfinite(azimuth):
        yield Finding.error(f"origin azimuth {azimuth} is not finite", azimuth=azimuth)


def check_structure(model: Sfile) -> Iterator[Finding]:
    """Required groups exist and hold the counts that ngrids implies.

    Parameters
    ----------
    model : Sfile
        The parsed sfile to check.

    Yields
    ------
    Finding
        One finding per observation, and a skip for anything it could not
        check.
    """
    for group in (SURFACE_GROUP, MATERIAL_GROUP):
        if group not in model.handle:
            yield Finding.error(f"missing '{group}' group", group=group)

    n_interfaces, n_grids = len(model.interfaces), len(model.grids)
    yield Finding.info(
        f"{n_interfaces} interface dataset(s), {n_grids} material grid(s)",
        interfaces=n_interfaces,
        grids=n_grids,
    )

    # An sfile has exactly one more interface than it has grids (top and bottom sandwich each grid), and ngrids
    # must agree with both.
    if (
        SURFACE_GROUP in model.handle
        and MATERIAL_GROUP in model.handle
        and n_interfaces != n_grids + 1
    ):
        yield Finding.error(
            f"{n_interfaces} interface dataset(s) for {n_grids} grid(s); expected "
            f"one more interface than grids",
            interfaces=n_interfaces,
            grids=n_grids,
        )
    if model.ngrids is None:
        yield Finding.skip(f"cannot cross-check group sizes: '{NGRIDS_ATTR}' is absent")
    elif model.ngrids != n_grids and MATERIAL_GROUP in model.handle:
        yield Finding.error(
            f"ngrids={model.ngrids} but '{MATERIAL_GROUP}' holds {n_grids} grid(s)",
            ngrids=model.ngrids,
            grids=n_grids,
        )


def check_interfaces(model: Sfile) -> Iterator[Finding]:
    """Depth interfaces are finite, strictly ordered, and match the depth attribute.

    Parameters
    ----------
    model : Sfile
        The parsed sfile to check.

    Yields
    ------
    Finding
        One finding per observation, and a skip for anything it could not
        check.
    """
    arrays = model.require_interfaces()

    for name, dataset in model.interfaces.items():
        if dataset.ndim != 2:
            yield Finding.error(
                f"{name}: expected a 2-D dataset, got shape {dataset.shape}",
                interface=name,
            )
            yield Finding.skip(
                f"{name}: not checked for NaN, ordering or extent", interface=name
            )

    for name, array in arrays.items():
        n_nan = int(np.count_nonzero(np.isnan(array)))
        n_inf = int(np.count_nonzero(np.isinf(array)))
        if n_nan:
            yield Finding.error(
                f"{name}: {n_nan} NaN value(s)", interface=name, n_nan=n_nan
            )
        if n_inf:
            yield Finding.error(
                f"{name}: {n_inf} Inf value(s)", interface=name, n_inf=n_inf
            )
        lo, hi = float(np.nanmin(array)), float(np.nanmax(array))
        yield Finding.info(
            f"{name}: shape={array.shape} z=[{lo:.2f}, {hi:.2f}] m "
            f"elev=[{-hi:.1f}, {-lo:.1f}] m ASL",
            interface=name,
            zmin=lo,
            zmax=hi,
        )

    yield from _check_layer_ordering(model)
    yield from _check_terrain(arrays)
    yield from _check_depth_agreement(model, arrays)


def _check_layer_ordering(model: Sfile) -> Iterator[Finding]:
    """Check that each interface lies strictly below the one above it.

    Parameters
    ----------
    model : Sfile
        The parsed sfile.

    Yields
    ------
    Finding
        One finding per layer whose thickness is not everywhere positive.
    """
    for (top, bottom), (thickness, resampled) in model.layers.items():
        violating = thickness <= 0.0
        n_bad = int(np.count_nonzero(violating))
        if not n_bad:
            continue
        i, j, worst = _worst_of(thickness, violating)
        note = " (after nearest resample to a common grid)" if resampled else ""
        yield Finding.error(
            f"{bottom} is not below {top} at {n_bad} point(s){note}; thinnest "
            f"{worst:.2f} m at (i, j)=({i}, {j}). SW4 patch selection requires "
            f"strict depth ordering, and computes hv=thickness/(nk-1)",
            top=top,
            bottom=bottom,
            n_bad=n_bad,
            min_thickness=worst,
            i=i,
            j=j,
        )


def _check_terrain(arrays: Mapping[str, npt.NDArray[np.float64]]) -> Iterator[Finding]:
    """Check the top interface for implausible cell-to-cell jumps.

    Parameters
    ----------
    arrays : Mapping
        The readable interface arrays, shallowest first.

    Yields
    ------
    Finding
        The largest gradient found, or a warning if it is implausible.
    """
    name = next(iter(arrays))
    topography = arrays[name]
    if min(topography.shape) < 2:
        yield Finding.skip(
            f"{name}: too small to measure a terrain gradient", interface=name
        )
        return

    gradient = max(
        float(np.abs(np.diff(topography, axis=axis)).max()) for axis in (0, 1)
    )
    if gradient > TERRAIN_JUMP_WARN_M:
        yield Finding.warn(
            f"{name}: max cell-to-cell z jump {gradient:.1f} m; check for "
            f"fill-value boundaries",
            interface=name,
            gradient_m=gradient,
        )
    else:
        yield Finding.info(
            f"{name}: max terrain gradient {gradient:.1f} m/cell",
            interface=name,
            gradient_m=gradient,
        )


def _check_depth_agreement(
    model: Sfile, arrays: Mapping[str, npt.NDArray[np.float64]]
) -> Iterator[Finding]:
    """Check the depth attribute against the shallowest and deepest interfaces.

    Parameters
    ----------
    model : Sfile
        The parsed sfile.
    arrays : Mapping
        The readable interface arrays, shallowest first.

    Yields
    ------
    Finding
        One finding per end of the depth range that disagrees with the data.
    """
    if model.depth_range is None:
        yield Finding.skip(
            f"cannot compare interfaces against '{DEPTH_ATTR}': absent or malformed"
        )
        return
    if len(arrays) < 2:
        yield Finding.skip(
            f"cannot compare interfaces against '{DEPTH_ATTR}': need 2 interfaces"
        )
        return

    names = list(arrays)
    for label, declared, actual in (
        ("min", model.depth_range[0], float(np.nanmin(arrays[names[0]]))),
        ("max", model.depth_range[1], float(np.nanmax(arrays[names[-1]]))),
    ):
        if abs(declared - actual) > DEPTH_TOLERANCE_M:
            yield Finding.warn(
                f"'{DEPTH_ATTR}' {label}={declared:.2f} m but the data {label} is "
                f"{actual:.2f} m ({actual - declared:+.2f} m)",
                bound=label,
                declared=declared,
                actual=actual,
            )


def check_material(model: Sfile) -> Iterator[Finding]:
    """Material variables are present, finite and physically plausible.

    Parameters
    ----------
    model : Sfile
        The parsed sfile to check.

    Yields
    ------
    Finding
        One finding per observation, and a skip for anything it could not
        check.
    """
    for grid in model.require_grids().values():
        yield from _check_grid(model, grid)


def _check_grid(model: Sfile, grid: MaterialGrid) -> Iterator[Finding]:
    """Check one material grid's attributes and datasets.

    Parameters
    ----------
    model : Sfile
        The parsed sfile.
    grid : MaterialGrid
        The grid to check.

    Yields
    ------
    Finding
        One finding per observation about the grid.
    """
    if grid.h is None:
        yield Finding.error(f"{grid.name}: missing '{HORIZONTAL_ATTR}'", grid=grid.name)
    else:
        yield Finding.info(f"{grid.name}: h = {grid.h} m", grid=grid.name, h=grid.h)
        if not np.isfinite(grid.h) or grid.h <= 0:
            yield Finding.error(
                f"{grid.name}: h must be finite and > 0, got {grid.h}", grid=grid.name
            )

    if grid.n_components is not None and grid.n_components not in COMPONENT_COUNTS:
        yield Finding.warn(
            f"{grid.name}: unexpected {COMPONENTS_ATTR}={grid.n_components}, expected "
            f"one of {list(COMPONENT_COUNTS)}",
            grid=grid.name,
            n_components=grid.n_components,
        )

    expected = [
        name
        for name, spec in VARS.items()
        if not spec.attenuation_only or model.attenuation
    ]
    for name in expected:
        if name not in grid.group:
            reason = (
                "attenuation=1 requires it"
                if VARS[name].attenuation_only
                else "it is required"
            )
            yield Finding.error(
                f"{grid.name}: missing dataset '{name}'; {reason}",
                grid=grid.name,
                dataset=name,
            )

    if grid.shape is None:
        yield Finding.skip(
            f"{grid.name}: cannot scan datasets or check nk without Cp", grid=grid.name
        )
        return
    if len(grid.shape) != 3:
        yield Finding.error(
            f"{grid.name}/Cp: expected a 3-D dataset, got shape {grid.shape}",
            grid=grid.name,
        )
        yield Finding.skip(
            f"{grid.name}: cannot scan datasets or check nk", grid=grid.name
        )
        return

    yield Finding.info(
        f"{grid.name}: shape (ni, nj, nk) = {grid.shape}",
        grid=grid.name,
        shape=grid.shape,
    )
    nk = grid.shape[2]
    if nk < 2:
        yield Finding.error(
            f"{grid.name}: nk={nk}; SW4 computes hv=thickness/(nk-1), so nk<2 gives NaN",
            grid=grid.name,
            nk=nk,
        )

    present = {name: grid.group[name] for name in expected if name in grid.group}
    scannable = {
        name: dataset
        for name, dataset in present.items()
        if tuple(dataset.shape) == grid.shape
    }
    for name in present.keys() - scannable.keys():
        yield Finding.error(
            f"{grid.name}/{name}: shape {tuple(present[name].shape)} does not match "
            f"Cp's {grid.shape}",
            grid=grid.name,
            dataset=name,
        )
        yield Finding.skip(
            f"{grid.name}/{name}: not scanned", grid=grid.name, dataset=name
        )

    stats, ratio = _scan_grid(scannable, model.chunk_rows)
    for name, summary in stats.items():
        yield from _report_variable(grid.name, name, summary)
    yield from _report_ratio(grid.name, ratio)


# NOTE: This function is complex enough to warrant its existence despite being a single callsite function.
def _scan_grid(
    datasets: Mapping[str, h5py.Dataset], chunk_rows: int
) -> tuple[dict[str, DatasetStats], RatioStats]:
    """Summarise a grid's datasets in a single chunked pass.

    Every dataset is read once, in matching row slabs, so the Vp/Vs ratio comes
    out of the same pass that produces the per-variable statistics rather than
    re-reading Cp and Cs.

    Parameters
    ----------
    datasets : Mapping
        The datasets to scan, all of the same shape.
    chunk_rows : int
        Number of `i` rows to read at a time.

    Returns
    -------
    tuple
        `(stats, ratio)`: the per-dataset statistics by name, and the Vp/Vs
        ratio statistics over solid cells.
    """
    stats: dict[str, DatasetStats] = {}
    ratio = EMPTY_RATIO
    if not datasets:
        return stats, ratio

    n_rows = next(iter(datasets.values())).shape[0]
    for start in range(0, n_rows, chunk_rows):
        chunks = {
            name: dataset[start : start + chunk_rows]
            for name, dataset in datasets.items()
        }
        for name, chunk in chunks.items():
            summary = _chunk_stats(chunk)
            stats[name] = stats[name].merge(summary) if name in stats else summary
        if "Cp" in chunks and "Cs" in chunks:
            ratio = ratio.merge(_chunk_ratio(chunks["Cp"], chunks["Cs"]))
    return stats, ratio


def _report_variable(
    grid_name: str, name: str, stats: DatasetStats
) -> Iterator[Finding]:
    """Turn one dataset's statistics into findings, per its `VarSpec`.

    Parameters
    ----------
    grid_name : str
        The grid the dataset belongs to.
    name : str
        The dataset name, a key of `VARS`.
    stats : DatasetStats
        The statistics gathered for the dataset.

    Yields
    ------
    Finding
        One finding per observation about the dataset.
    """
    spec = VARS[name]
    label = f"{grid_name}/{name}"
    unit = f" {spec.unit}" if spec.unit else ""
    where = {"grid": grid_name, "dataset": name}

    yield Finding.info(
        f"{label}: {spec.label} in [{stats.lo:.4g}, {stats.hi:.4g}]{unit}, "
        f"zeros={stats.n_zero} nan={stats.n_nan} inf={stats.n_inf}",
        **where,
        **stats._asdict(),
    )
    if stats.n_nan:
        yield Finding.error(f"{label}: {stats.n_nan} NaN value(s)", **where)
    if stats.n_inf:
        yield Finding.error(f"{label}: {stats.n_inf} Inf value(s)", **where)

    if spec.zero_allowed:
        if stats.n_neg:
            yield Finding.error(
                f"{label}: {stats.n_neg} negative value(s); {spec.nonpositive_note}",
                **where,
            )
        if stats.n_zero:
            yield Finding.error(
                f"{label}: {stats.n_zero} zero value(s)",
                **where,
            )
    elif stats.n_nonpositive:
        yield Finding.error(
            f"{label}: {stats.n_nonpositive} non-positive value(s) "
            f"(zeros={stats.n_zero}, negative={stats.n_neg}); {spec.nonpositive_note}",
            **where,
        )

    if spec.low_warn is not None and 0 < stats.lo < spec.low_warn:
        yield Finding.warn(
            f"{label}: suspiciously low {spec.label} minimum {stats.lo:.4g}{unit}",
            **where,
        )


def _report_ratio(grid_name: str, ratio: RatioStats) -> Iterator[Finding]:
    """Turn a grid's Vp/Vs statistics into findings.

    Parameters
    ----------
    grid_name : str
        The grid the ratio was measured over.
    ratio : RatioStats
        The ratio statistics over solid cells.

    Yields
    ------
    Finding
        One finding per observation about the ratio.
    """
    if not ratio.n_solid:
        yield Finding.skip(
            f"{grid_name}: no solid (Vs > 0) cells, so Vp/Vs was not checked",
            grid=grid_name,
        )
        return

    yield Finding.info(
        f"{grid_name}: Vp/Vs over solid cells in [{ratio.lo:.3f}, {ratio.hi:.3f}]",
        grid=grid_name,
        n_solid=ratio.n_solid,
    )
    if ratio.n_below_one:
        yield Finding.error(
            f"{grid_name}: {ratio.n_below_one} solid point(s) with Vp/Vs < 1, which "
            f"is physically impossible",
            grid=grid_name,
            n_below_one=ratio.n_below_one,
        )
    # n_below_sqrt2 contains n_below_one, so report only the marginal band.
    # Reporting both totals would count the same points twice.
    marginal = ratio.n_below_sqrt2 - ratio.n_below_one
    if marginal:
        yield Finding.warn(
            f"{grid_name}: {marginal} solid point(s) with 1 <= Vp/Vs < √2 ≈ 1.414",
            grid=grid_name,
            n_marginal=marginal,
        )


def check_grid_consistency(model: Sfile) -> Iterator[Finding]:
    """Grids span the same horizontal domain and resolve the layers they span.

    Parameters
    ----------
    model : Sfile
        The parsed sfile to check.

    Yields
    ------
    Finding
        One finding per observation, and a skip for anything it could not
        check.
    """
    grids = model.require_grids()

    known = [extent for grid in grids.values() if (extent := grid.extent) is not None]
    if not known:
        yield Finding.skip(
            "no grid has both a spacing and a Cp shape, so extents are unknown"
        )
    for axis in ("x", "y"):
        spans = [getattr(extent, axis) for extent in known]
        if not spans:
            continue
        if max(spans) - min(spans) > max(1.0, max(spans) * EXTENT_TOLERANCE):
            yield Finding.warn(
                f"inconsistent {axis}-extents across grids: "
                f"{[f'{span:.0f}m' for span in spans]}",
                axis=axis,
                extents=spans,
            )

    yield from _check_vertical_resolution(model, grids)


def _check_vertical_resolution(
    model: Sfile, grids: Mapping[str, MaterialGrid]
) -> Iterator[Finding]:
    """Check each grid's vertical cell size against the layer it spans.

    Parameters
    ----------
    model : Sfile
        The parsed sfile.
    grids : Mapping
        The material grids, in refinement order.

    Yields
    ------
    Finding
        One finding per grid whose layer is unresolvable or extremely thin.
    """
    layers = list(model.layers.items())
    if not layers:
        yield Finding.skip(
            "cannot check vertical resolution: fewer than 2 readable interfaces"
        )
        return

    for index, grid in enumerate(grids.values()):
        if grid.shape is None or len(grid.shape) != 3:
            yield Finding.skip(
                f"{grid.name}: cannot check vertical resolution without a 3-D Cp",
                grid=grid.name,
            )
            continue
        if index >= len(layers):
            yield Finding.skip(
                f"{grid.name}: no bracketing interface pair", grid=grid.name
            )
            continue

        (top, bottom), (thickness, _) = layers[index]
        nk = grid.shape[2]
        lo, hi = float(np.nanmin(thickness)), float(np.nanmax(thickness))
        yield Finding.info(
            f"{grid.name} ({top} - {bottom}): thickness in [{lo:.1f}, {hi:.1f}] m, nk={nk}",
            grid=grid.name,
            min_thickness=lo,
            max_thickness=hi,
            nk=nk,
        )
        # Non-positive thickness is reported by _check_layer_ordering; here it
        # matters only through the vertical cell size it implies.
        if lo > 0 and nk >= 2 and lo / (nk - 1) < THIN_CELL_WARN_M:
            yield Finding.warn(
                f"{grid.name}: minimum vertical cell size {lo / (nk - 1):.4f} m is "
                f"extremely thin",
                grid=grid.name,
                cell_m=lo / (nk - 1),
            )


def describe_boundaries(model: Sfile) -> Iterator[Finding]:
    """Report the model's extent, depth range and geographic corners.

    Parameters
    ----------
    model : Sfile
        The parsed sfile to check.

    Yields
    ------
    Finding
        One finding per observation, and a skip for anything it could not
        check.
    """
    lon, lat, azimuth = model.require_origin()
    grids = model.require_grids()

    measured = [
        (grid, extent) for grid in grids.values() if (extent := grid.extent) is not None
    ]
    if not measured:
        yield Finding.skip(
            "cannot compute an extent: no grid has a spacing and a shape"
        )
        return

    finest, extent = min(measured, key=lambda pair: pair[1].h)
    yield Finding.info(
        f"using {finest.name} (h={extent.h:.1f} m, shape {finest.shape}): "
        f"{extent.x / 1e3:.3f} km × {extent.y / 1e3:.3f} km, "
        f"x at azimuth {azimuth:.2f}° from north",
        grid=finest.name,
        x_km=extent.x / 1e3,
        y_km=extent.y / 1e3,
    )
    if model.depth_range is not None:
        zmin, zmax = model.depth_range
        yield Finding.info(
            f"depth {zmin:.0f} m to {zmax:.0f} m "
            f"(~{(zmax - max(0.0, zmin)) / 1e3:.1f} km rock column)",
            zmin=zmin,
            zmax=zmax,
        )
    if model.interface_arrays:
        name, topography = next(iter(model.interface_arrays.items()))
        lo, hi = float(np.nanmin(topography)), float(np.nanmax(topography))
        yield Finding.info(
            f"topography {name}: z=[{lo:.2f}, {hi:.2f}] m, elev=[{-hi:.1f}, {-lo:.1f}] m ASL",
            interface=name,
        )

    yield from _describe_corners(lon, lat, azimuth, extent.x, extent.y)


def _describe_corners(
    lon: float, lat: float, azimuth: float, x_m: float, y_m: float
) -> Iterator[Finding]:
    """Report the geographic position of the model's four corners.

    Parameters
    ----------
    lon : float
        Longitude of the grid origin, in degrees.
    lat : float
        Latitude of the grid origin, in degrees.
    azimuth : float
        Clockwise angle from north to the grid's x-axis, in degrees.
    x_m : float
        Extent along the grid's x-axis, in metres.
    y_m : float
        Extent along the grid's y-axis, in metres.

    Yields
    ------
    Finding
        One finding per corner, then the bounding box.
    """
    try:
        northing, easting = coordinates.wgs_depth_to_nztm(np.array([lat, lon]))
    except ValueError as exc:
        yield Finding.skip(
            f"cannot project the origin to NZTM: {exc}", lon=lon, lat=lat
        )
        return

    # SW4 azimuth is the clockwise angle from north to the x-axis, so the x
    # unit vector in (east, north) is (sin, cos) and y's is (cos, -sin).
    sin_azimuth, cos_azimuth = np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))
    offsets = np.array([(0.0, 0.0), (x_m, 0.0), (0.0, y_m), (x_m, y_m)])
    eastings = easting + offsets[:, 0] * sin_azimuth + offsets[:, 1] * cos_azimuth
    northings = northing + offsets[:, 0] * cos_azimuth - offsets[:, 1] * sin_azimuth

    corners = coordinates.nztm_to_wgs_depth(np.column_stack([northings, eastings]))
    labels = ("origin (SW)", "far-x", "far-y", "far corner")
    for label, (corner_lat, corner_lon), corner_e, corner_n in zip(
        labels, corners, eastings, northings, strict=True
    ):
        yield Finding.info(
            f"corner {label}: lon={corner_lon:.6f}° lat={corner_lat:.6f}° "
            f"easting={corner_e:.1f} northing={corner_n:.1f}",
            corner=label,
            lon=float(corner_lon),
            lat=float(corner_lat),
        )

    lats, lons = corners[:, 0], corners[:, 1]
    yield Finding.info(
        f"bounding box lon [{lons.min():.6f}°, {lons.max():.6f}°] "
        f"lat [{lats.min():.6f}°, {lats.max():.6f}°]",
        lon_min=float(lons.min()),
        lon_max=float(lons.max()),
        lat_min=float(lats.min()),
        lat_max=float(lats.max()),
    )


class Check(Protocol):
    """One validation pass over a parsed sfile.

    The name is part of the interface: `validate` tags every finding with it,
    so a consumer can tell which pass produced a record.
    """

    __name__: str

    def __call__(self, model: Sfile) -> Iterator[Finding]:
        """Run the check.

        Parameters
        ----------
        model : Sfile
            The parsed sfile to check.

        Returns
        -------
        Iterator[Finding]
            One finding per observation the check makes.
        """
        ...


CHECKS: Sequence[Check] = (
    check_attributes,
    check_structure,
    check_interfaces,
    check_material,
    check_grid_consistency,
    describe_boundaries,
)


def _run_check(check: Check, model: Sfile) -> Iterator[Finding]:
    """Run one check, recording a skip if its prerequisites are missing.

    Parameters
    ----------
    check : Check
        The check to run.
    model : Sfile
        The parsed sfile to check.

    Yields
    ------
    Finding
        The check's findings, tagged with its name, or a single skip.
    """
    try:
        for finding in check(model):
            yield dataclasses.replace(finding, check=check.__name__)
    except UnavailableError as exc:
        yield Finding(
            Severity.SKIP, f"{check.__name__} did not run: {exc}", {}, check.__name__
        )


def validate(model: Sfile) -> Report:
    """Run every check against a parsed sfile.

    Parameters
    ----------
    model : Sfile
        The parsed sfile to check.

    Returns
    -------
    Report
        Every finding the checks produced.
    """
    return Report(
        tuple(finding for check in CHECKS for finding in _run_check(check, model))
    )


#: structlog level to report each severity at. A skipped check is a caveat on
#: the result, so it is not quietly filed under info.
LOG_LEVELS = {
    Severity.ERROR: "error",
    Severity.WARN: "warning",
    Severity.SKIP: "warning",
    Severity.INFO: "info",
}


def log_report(report: Report, verbose: bool) -> None:
    """Write a report out, one structured record per finding plus a tally.

    Parameters
    ----------
    report : Report
        The findings to write.
    verbose : bool
        Whether to include info findings, which are measurements rather than
        problems.
    """
    logger = log_utils.get_logger(__name__)
    for finding in report.findings:
        if verbose or finding.severity is not Severity.INFO:
            log = getattr(logger, LOG_LEVELS[finding.severity])
            log(
                finding.message,
                severity=str(finding.severity),
                check=finding.check,
                **finding.context,
            )

    counts = report.counts
    logger.info(
        "validation complete",
        result="fail" if report.failed else "pass",
        **{str(severity): counts[severity] for severity in Severity},
    )


@log_utils.log_call()
def validate_path(sfile: Path, chunk_rows: int, verbose: bool) -> bool:
    """Validate one sfile and write its report.

    Parameters
    ----------
    sfile : Path
        The sfile (HDF5 velocity model) to check.
    chunk_rows : int
        Number of `i` rows to read at a time when scanning material datasets.
    verbose : bool
        Whether to report measurements as well as problems.

    Returns
    -------
    bool
        Whether any finding was an error.
    """
    try:
        with h5py.File(sfile, "r") as handle:
            report = validate(read_sfile(sfile, handle, chunk_rows))
    except (OSError, KeyError, ValueError) as exc:
        report = Report(
            (Finding(Severity.ERROR, f"cannot read sfile: {exc}", {}, "read_sfile"),)
        )

    log_report(report, verbose)
    return report.failed


@cli.from_docstring(app)
def validate_sfile(
    sfile: Annotated[Path, typer.Argument(exists=True, readable=True, dir_okay=False)],
    chunk_rows: Annotated[int, typer.Option(min=1)] = 20,
    verbose: bool = True,
) -> None:
    """Check an SW4 sfile for issues, exiting non-zero if any are errors.

    Parameters
    ----------
    sfile : Path
        The sfile (HDF5 velocity model) to check.
    chunk_rows : int
        Number of `i` rows to read at a time when scanning material datasets,
        which do not fit in memory.
    verbose : bool
        Report measurements as well as problems.

    Raises
    ------
    typer.Exit
        If any check reports an error.
    """
    # Raised outside the logged call: a model that fails validation is an
    # expected outcome, not an exception worth a logged traceback.
    if validate_path(sfile, chunk_rows, verbose):
        raise typer.Exit(code=1)
