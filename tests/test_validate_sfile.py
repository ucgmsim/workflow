"""Tests for the SW4 sfile validator.

The validator only ever asks an open sfile for membership, indexing, `keys()`
and `attrs`, and only ever asks a dataset for `shape`, `ndim` and slicing. A
`dict` carrying an `attrs` mapping covers the first set and a numpy array
covers the second, so these tests hand `read_sfile` a synthetic tree instead of
an HDF5 file, and one test writes the same trees out with h5py to check the two
routes agree.

The claims here are the ones a user relies on -- a broken model is rejected, a
merely suspicious one is not, and an incomplete one is skipped rather than
crashed on -- so tests assert on severity and never on message wording, which
is rendering rather than contract.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import structlog.testing
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from typer.testing import CliRunner

from workflow.scripts import validate_sfile
from workflow.scripts.validate_sfile import (
    ATTENUATION_ATTR,
    COMPONENT_COUNTS,
    COMPONENTS_ATTR,
    DEPTH_ATTR,
    HORIZONTAL_ATTR,
    MATERIAL_GROUP,
    NGRIDS_ATTR,
    ORIGIN_ATTR,
    REQUIRED_ATTRS,
    SPACING_ATTRS,
    SURFACE_GROUP,
    VARS,
    Report,
    Severity,
    read_sfile,
    validate,
)

SHAPE = (5, 4, 3)  # (ni, nj, nk)
SPACING = 100.0
LAYER_THICKNESS = 1000.0
ORIGIN = (172.0, -43.5, 0.0)  # lon, lat, azimuth
CHUNK_ROWS = 20
# Depth-positive top surface: gentle terrain from 100 m ASL down to 100 m BSL.
TOPOGRAPHY = np.linspace(-100.0, 100.0, int(np.prod(SHAPE[:2]))).reshape(SHAPE[:2])
TOPOGRAPHY.setflags(write=False)  # so a test cannot leak a mutation into the next
# Uniform values that break none of the physical checks: Vp/Vs is 2, and every
# value is well above its "suspiciously low" threshold.
MATERIAL = {"Rho": 2500.0, "Cp": 3000.0, "Cs": 1500.0, "Qp": 100.0, "Qs": 50.0}


class Node(dict[str, Any]):
    """An h5py-shaped container: named members plus a bag of attributes."""

    def __init__(
        self,
        members: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(members or {})
        self.attrs: dict[str, Any] = dict(attrs or {})


Mutation = Callable[[Node], Any]
"""A change made to an otherwise clean tree."""


def material_grid(shape: tuple[int, int, int] = SHAPE, h: float = SPACING) -> Node:
    """One material grid holding uniform, physically plausible values."""
    # Keyed off VARS, so a new material variable fails here rather than going
    # silently unmodelled.
    return Node(
        {name: np.full(shape, MATERIAL[name]) for name in VARS},
        {HORIZONTAL_ATTR: h, COMPONENTS_ATTR: COMPONENT_COUNTS[-1]},
    )


def sfile_tree(n_grids: int = 1) -> Node:
    """A structurally and physically clean sfile, with evenly stacked interfaces."""
    surfaces = {
        f"z_values_{index}": TOPOGRAPHY + index * LAYER_THICKNESS
        for index in range(n_grids + 1)
    }
    grids = {f"material_{index}": material_grid() for index in range(n_grids)}
    return sync_depth_attr(
        Node(
            {SURFACE_GROUP: Node(surfaces), MATERIAL_GROUP: Node(grids)},
            {
                ATTENUATION_ATTR: 1,
                NGRIDS_ATTR: n_grids,
                ORIGIN_ATTR: ORIGIN,
                SPACING_ATTRS[0]: SPACING,
            },
        )
    )


def sync_depth_attr(tree: Node) -> Node:
    """Declare the depth range that the interface data actually spans."""
    surfaces = list(tree[SURFACE_GROUP].values())
    tree.attrs[DEPTH_ATTR] = (
        float(np.nanmin(surfaces[0])),
        float(np.nanmax(surfaces[-1])),
    )
    return tree


@pytest.fixture
def tree() -> Node:
    """A clean single-grid sfile."""
    return sfile_tree()


def report_of(tree: Node, chunk_rows: int = CHUNK_ROWS) -> Report:
    """Every finding the checks produce for a synthetic tree."""
    return validate(
        read_sfile(Path("synthetic.sfile"), cast(h5py.File, tree), chunk_rows)
    )


def severities(report: Report) -> set[Severity]:
    """Which severities a report contains."""
    return {finding.severity for finding in report.findings}


def fingerprint(report: Report) -> list[tuple[Severity, str, str]]:
    """A report reduced to what it said, comparably: NaN in a context dict is
    never equal to itself."""
    return [
        (finding.severity, finding.check, finding.message)
        for finding in report.findings
    ]


def interfaces(tree: Node) -> Node:
    """The interface group of a tree."""
    return tree[SURFACE_GROUP]


def grid(tree: Node, name: str = "material_0") -> Node:
    """One material grid of a tree."""
    return tree[MATERIAL_GROUP][name]


def attr(node: Node, name: str, value: Any) -> None:
    """Set one HDF5 attribute in place."""
    node.attrs[name] = value


def poke(node: Node, name: str, index: tuple[int, ...], value: float) -> None:
    """Overwrite one element of a dataset in place."""
    node[name][index] = value


def reshape(node: Node, name: str, shape: tuple[int, ...]) -> None:
    """Replace one dataset with a differently shaped one, same fill value."""
    node[name] = np.full(shape, np.ravel(node[name])[0])


def write_sfile(path: Path, tree: Node) -> Path:
    """Write a synthetic tree out as a real HDF5 hierarchy."""

    def write(node: Node, target: h5py.Group) -> None:
        target.attrs.update(node.attrs)
        for name, member in node.items():
            if isinstance(member, Node):
                write(member, target.create_group(name))
            else:
                target.create_dataset(name, data=member)

    with h5py.File(path, "w") as handle:
        write(tree, handle)
    return path


# fmt: off
#: Models SW4 cannot run: it aborts, or propagates NaNs.
BROKEN: dict[str, Mutation] = {
    "no-spacing-attribute":  lambda t: t.attrs.pop(SPACING_ATTRS[0]),
    "zero-spacing":          lambda t: attr(t, SPACING_ATTRS[0], 0.0),
    "bad-attenuation":       lambda t: attr(t, ATTENUATION_ATTR, 2),
    "ngrids-disagrees":      lambda t: attr(t, NGRIDS_ATTR, 3),
    "nonpositive-ngrids":    lambda t: attr(t, NGRIDS_ATTR, 0),
    "inverted-depth-range":  lambda t: attr(t, DEPTH_ATTR, t.attrs[DEPTH_ATTR][::-1]),
    "malformed-depth-range": lambda t: attr(t, DEPTH_ATTR, (0.0, 1.0, 2.0)),
    "nan-depth-range":       lambda t: attr(t, DEPTH_ATTR, (np.nan, 1000.0)),
    "origin-off-the-globe":  lambda t: attr(t, ORIGIN_ATTR, (172.0, 91.0, 0.0)),
    "longitude-out-of-range": lambda t: attr(t, ORIGIN_ATTR, (181.0, -43.5, 0.0)),
    "malformed-origin":      lambda t: attr(t, ORIGIN_ATTR, ORIGIN[:2]),
    "no-interfaces":         lambda t: t.pop(SURFACE_GROUP),
    "no-material":           lambda t: t.pop(MATERIAL_GROUP),
    "too-few-interfaces":    lambda t: interfaces(t).pop("z_values_1"),
    "nan-in-interface":      lambda t: poke(interfaces(t), "z_values_0", (1, 2), np.nan),
    "inf-in-interface":      lambda t: poke(interfaces(t), "z_values_0", (1, 2), np.inf),
    "interfaces-out-of-order": lambda t: interfaces(t).update({"z_values_1": TOPOGRAPHY - 1.0}),
    "interface-not-2d":      lambda t: reshape(interfaces(t), "z_values_1", SHAPE),
    "no-grid-spacing":       lambda t: grid(t).attrs.pop(HORIZONTAL_ATTR),
    "cp-not-3d":             lambda t: reshape(grid(t), "Cp", SHAPE[:2]),
    "one-cell-deep":         lambda t: [reshape(grid(t), n, (*SHAPE[:2], 1)) for n in VARS],
    "mismatched-shapes":     lambda t: reshape(grid(t), "Rho", (*SHAPE[:2], SHAPE[2] - 1)),
    "nan-in-cp":             lambda t: poke(grid(t), "Cp", (0, 0, 0), np.nan),
    "inf-in-cp":             lambda t: poke(grid(t), "Cp", (0, 0, 0), np.inf),
    "vs-faster-than-vp":     lambda t: grid(t)["Cs"].fill(4_000.0),
}

#: Models that are usable, but odd enough to be worth a look.
SUSPICIOUS: dict[str, Mutation] = {
    "unexpected-components": lambda t: attr(grid(t), COMPONENTS_ATTR, 4),
    "marginal-vp-vs":        lambda t: grid(t)["Cs"].fill(2_500.0),
    "very-low-density":      lambda t: grid(t)["Rho"].fill(50.0),
    "depth-attr-disagrees":  lambda t: attr(t, DEPTH_ATTR, np.add(t.attrs[DEPTH_ATTR], (-400.0, 900.0))),
    "terrain-cliff":         lambda t: sync_depth_attr(bump_corner(t, 5_000.0)),
    "unresolvable-layer":    lambda t: sync_depth_attr(squash_layer(t, 0.01)),
}

#: Models missing something a check needs, which must be skipped, not crashed on.
INCOMPLETE: dict[str, Mutation] = {
    "no-grids":         lambda t: t[MATERIAL_GROUP].clear(),
    "no-2d-interfaces": lambda t: [reshape(interfaces(t), n, SHAPE) for n in interfaces(t)],
    "no-origin":        lambda t: t.attrs.pop(ORIGIN_ATTR),
    "nan-azimuth":      lambda t: attr(t, ORIGIN_ATTR, (*ORIGIN[:2], np.nan)),
    "nan-grid-spacing": lambda t: attr(grid(t), HORIZONTAL_ATTR, np.nan),
    "single-point-interfaces": lambda t: sync_depth_attr(flatten_interfaces(t)),
}
# fmt: on


def bump_corner(tree: Node, height: float) -> Node:
    """Raise one corner of every interface, leaving the layers as thick as they were."""
    for surface in interfaces(tree).values():
        surface[0, 0] += height
    return tree


def flatten_interfaces(tree: Node) -> Node:
    """Shrink every interface to a single point, too small to have a gradient."""
    for index, name in enumerate(interfaces(tree)):
        interfaces(tree)[name] = np.full((1, 1), index * LAYER_THICKNESS)
    return tree


def squash_layer(tree: Node, thickness: float) -> Node:
    """Bring the second interface up to `thickness` below the first."""
    interfaces(tree)["z_values_1"] = TOPOGRAPHY + thickness
    return tree


@pytest.mark.parametrize("n_grids", [1, 2, 3])
def test_a_clean_model_is_accepted_without_comment(n_grids: int) -> None:
    report = report_of(sfile_tree(n_grids=n_grids))

    assert severities(report) == {Severity.INFO}, [
        finding.message
        for finding in report.findings
        if finding.severity != Severity.INFO
    ]


@pytest.mark.parametrize("mutate", BROKEN.values(), ids=BROKEN)
def test_a_broken_model_is_rejected(tree: Node, mutate: Mutation) -> None:
    mutate(tree)

    assert report_of(tree).failed


@pytest.mark.parametrize("name", REQUIRED_ATTRS)
def test_a_missing_required_attribute_is_rejected(tree: Node, name: str) -> None:
    del tree.attrs[name]

    assert report_of(tree).failed


@pytest.mark.parametrize("name", VARS)
def test_a_missing_material_dataset_is_rejected(tree: Node, name: str) -> None:
    del grid(tree)[name]

    assert report_of(tree).failed


@pytest.mark.parametrize("name", VARS)
def test_a_nonpositive_material_value_is_rejected(tree: Node, name: str) -> None:
    grid(tree)[name].put(0, 0.0)

    assert report_of(tree).failed


def test_the_q_datasets_are_only_required_with_attenuation(tree: Node) -> None:
    tree.attrs[ATTENUATION_ATTR] = 0
    for name, spec in VARS.items():
        if spec.attenuation_only:
            del grid(tree)[name]

    assert severities(report_of(tree)) == {Severity.INFO}


@pytest.mark.parametrize("mutate", SUSPICIOUS.values(), ids=SUSPICIOUS)
def test_a_suspicious_model_warns_but_is_not_rejected(
    tree: Node, mutate: Mutation
) -> None:
    mutate(tree)

    report = report_of(tree)

    assert Severity.WARN in severities(report)
    assert not report.failed


@pytest.mark.parametrize("mutate", INCOMPLETE.values(), ids=INCOMPLETE)
def test_an_incomplete_model_is_skipped_not_crashed_on(
    tree: Node, mutate: Mutation
) -> None:
    mutate(tree)

    with np.errstate(all="ignore"):  # non-finite inputs are the point here
        report = report_of(tree)

    assert Severity.SKIP in severities(report)


def test_interfaces_at_different_resolutions_are_accepted(tree: Node) -> None:
    interfaces(tree)["z_values_1"] = np.full((3, 2), LAYER_THICKNESS)
    sync_depth_attr(tree)

    assert not report_of(tree).failed


@pytest.mark.parametrize(
    ("shape", "spacing", "warns"),
    [
        ((9, 7, 3), SPACING, True),
        # (5, 4) cells at 100 m spans 400 m x 300 m, and so does (9, 7) at 50 m.
        ((9, 7, 3), SPACING / 2, False),
    ],
    ids=["mismatched", "same-domain"],
)
def test_grids_covering_different_domains_warn(
    shape: tuple[int, int, int], spacing: float, warns: bool
) -> None:
    tree = sfile_tree(n_grids=2)
    tree[MATERIAL_GROUP]["material_1"] = material_grid(shape, spacing)

    report = report_of(tree)

    assert (Severity.WARN in severities(report)) == warns
    assert not report.failed


def test_interfaces_and_grids_are_read_in_depth_order(tmp_path: Path) -> None:
    # Real HDF5 groups come back lexically, which would put z_values_10 between
    # z_values_1 and z_values_2, so this has to go through h5py to mean anything.
    n_grids = 10
    path = write_sfile(tmp_path / "model.sfile", sfile_tree(n_grids=n_grids))

    with h5py.File(path, "r") as handle:
        model = read_sfile(path, handle, CHUNK_ROWS)

        assert list(model.interfaces) == [f"z_values_{i}" for i in range(n_grids + 1)]
        assert list(model.grids) == [f"material_{i}" for i in range(n_grids)]


@pytest.mark.parametrize(
    "mutate",
    [lambda t: None, *BROKEN.values()],
    ids=["clean", *BROKEN],
)
def test_a_synthetic_tree_reports_what_a_real_hdf5_file_does(
    tmp_path: Path, tree: Node, mutate: Mutation
) -> None:
    mutate(tree)
    path = write_sfile(tmp_path / "model.sfile", tree)

    with h5py.File(path, "r") as handle:
        from_file = validate(read_sfile(path, handle, CHUNK_ROWS))

    assert fingerprint(from_file) == fingerprint(report_of(tree))


@settings(deadline=None)
@given(
    values=arrays(np.float64, SHAPE),
    chunk_rows=st.integers(min_value=1, max_value=SHAPE[0] + 2),
)
def test_chunk_size_does_not_change_the_report(
    values: npt.NDArray[np.float64], chunk_rows: int
) -> None:
    tree = sfile_tree()
    grid(tree).update({"Cp": values, "Cs": np.abs(values), "Rho": values + 1.0})

    with np.errstate(all="ignore"):  # the values are deliberately extreme
        assert fingerprint(report_of(tree, chunk_rows)) == fingerprint(report_of(tree))


@pytest.mark.parametrize(
    ("azimuth", "along", "across"),
    [(0.0, "lat", "lon"), (90.0, "lon", "lat")],
    ids=["x-runs-north", "x-runs-east"],
)
def test_the_reported_corners_follow_the_azimuth(
    tree: Node, azimuth: float, along: str, across: str
) -> None:
    lon, lat, _ = ORIGIN
    tree.attrs[ORIGIN_ATTR] = (lon, lat, azimuth)

    corners = {
        finding.context["corner"]: finding.context
        for finding in report_of(tree).findings
        if "corner" in finding.context
    }
    origin, far_x = corners["origin (SW)"], corners["far-x"]

    assert (origin["lon"], origin["lat"]) == pytest.approx((lon, lat), abs=1e-6)
    assert far_x[along] - origin[along] > 1e-3
    assert far_x[across] == pytest.approx(origin[across], abs=1e-3)


@pytest.mark.parametrize(
    ("mutate", "exit_code"),
    [(lambda t: None, 0), (lambda t: grid(t)["Cs"].fill(0.0), 1)],
    ids=["clean", "broken"],
)
def test_cli_exit_code(
    tmp_path: Path, tree: Node, mutate: Mutation, exit_code: int
) -> None:
    mutate(tree)
    path = write_sfile(tmp_path / "model.sfile", tree)

    result = CliRunner().invoke(validate_sfile.app, [str(path)])

    assert result.exit_code == exit_code


def test_a_file_that_is_not_an_sfile_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not-really.sfile"
    path.write_text("this is not an HDF5 file")

    with structlog.testing.capture_logs() as entries:
        assert validate_sfile.validate_path(path, chunk_rows=CHUNK_ROWS, verbose=True)

    assert any(entry["log_level"] == "error" for entry in entries)


@pytest.mark.parametrize("verbose", [True, False], ids=["verbose", "quiet"])
def test_measurements_are_only_logged_when_verbose(tree: Node, verbose: bool) -> None:
    with structlog.testing.capture_logs() as entries:
        validate_sfile.log_report(report_of(tree), verbose=verbose)

    logged = [entry for entry in entries if entry["event"] != "validation complete"]
    assert bool(logged) == verbose
