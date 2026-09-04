"""Pure-function tests for the intensity measure calculation.

Nothing here runs OpenQuake or an IM kernel: what is tested is the metadata
plumbing that carries the SW4 supergrid (absorbing layer) penetration from the
waveform file into the intensity measure file, which is arithmetic-free and
where the silent failures live.
"""

import numpy as np
import pytest
import xarray as xr

from workflow.scripts import im_calc


def _waveform_dataset(
    n_stations: int = 3,
    supergrid_depth: list[float] | None = None,
    attrs: dict[str, float] | None = None,
) -> xr.Dataset:
    """A minimal stand-in for an opened broadband/LF waveform file."""
    stations = [f"ST{index:02d}" for index in range(n_stations)]
    coords: dict[str, object] = {"station": stations}
    if supergrid_depth is not None:
        coords["supergrid_depth"] = (
            "station",
            np.array(supergrid_depth, dtype=np.float32),
        )
        coords["supergrid_depth_gp"] = (
            "station",
            np.array(supergrid_depth, dtype=np.float32) / 400.0,
        )
    return xr.Dataset(
        {"waveform": (("station",), np.ones(n_stations, dtype=np.float32))},
        coords=coords,
        attrs=attrs or {},
    )


def test_a_solver_with_no_absorbing_layer_still_gets_the_coordinate() -> None:
    """Every IM file carries the coordinate, whatever the solver produced it.

    EMOD3D and the high-frequency simulation have no supergrid at all, so the
    value has to be NaN -- "not applicable / not reported" -- and never `0.0`,
    which would assert the station had been checked and found in the interior.
    """
    supergrid = im_calc.supergrid_coordinates(_waveform_dataset(n_stations=4))

    assert set(supergrid) == set(im_calc.SUPERGRID_COORDINATES)
    for values in supergrid.values():
        assert values.dims == ("station",)
        assert values.dtype == np.float32
        assert np.isnan(values.values).all()


def test_a_reported_penetration_is_passed_through_unchanged() -> None:
    """The three states survive verbatim: clean, flagged, and unknown."""
    dataset = _waveform_dataset(supergrid_depth=[0.0, 5750.0, np.nan])

    supergrid = im_calc.supergrid_coordinates(dataset)

    depth = supergrid["supergrid_depth"]
    assert depth.values[0] == 0.0
    assert depth.values[1] == pytest.approx(5750.0)
    assert np.isnan(depth.values[2])
    # `> 0` is the documented threshold; NaN must not satisfy it.
    np.testing.assert_array_equal(depth.values > 0, [False, True, False])


def test_the_coordinate_is_read_eagerly() -> None:
    """`im-calc` opens the waveform file chunked, so the coordinate arrives as
    a dask array whose "auto" station chunking differs from the waveform's.
    Loading it here keeps that mismatch out of the attached coordinates, the
    way `vs30` is loaded eagerly for the same reason.
    """
    dataset = _waveform_dataset(supergrid_depth=[0.0, 1.0, 2.0]).chunk({"station": 1})

    supergrid = im_calc.supergrid_coordinates(dataset)

    assert supergrid["supergrid_depth"].chunks is None


def test_nothing_is_claimed_about_a_run_that_reported_nothing() -> None:
    """An all-NaN flag must not put `absorbing_layer` in the root attributes.

    Writing it would claim the run had an absorbing layer that somebody
    measured, on the strength of a coordinate that says only "unknown".
    """
    dataset = _waveform_dataset(attrs={"SGWIDTH": 12000.0})
    supergrid = im_calc.supergrid_coordinates(dataset)

    assert im_calc.supergrid_attributes(dataset, supergrid) == {}


def test_the_sponge_width_comes_from_the_waveform_file() -> None:
    """Not from the realisation configuration.

    The configuration is editable after a run; the waveform file is what SW4
    actually wrote. Reading the config here would let the IM file's
    self-description drift away from the run it describes.
    """
    dataset = _waveform_dataset(
        supergrid_depth=[0.0, 5750.0, 0.0],
        attrs={"SGWIDTH": 12000.0, "SGWIDTHGP": 30.0},
    )
    supergrid = im_calc.supergrid_coordinates(dataset)

    attributes = im_calc.supergrid_attributes(dataset, supergrid)

    assert attributes["absorbing_layer"] == "sw4_supergrid"
    assert attributes["absorbing_layer_width_m"] == pytest.approx(12000.0)
    assert attributes["absorbing_layer_width_gp"] == pytest.approx(30.0)


def test_a_flag_without_a_width_still_names_the_layer() -> None:
    """A station file written with penetrations but no file-level width (or a
    broadband file whose width did not survive) must not lose the layer name.
    """
    dataset = _waveform_dataset(supergrid_depth=[0.0, 5750.0, 0.0])
    supergrid = im_calc.supergrid_coordinates(dataset)

    attributes = im_calc.supergrid_attributes(dataset, supergrid)

    assert attributes == {"absorbing_layer": "sw4_supergrid"}


def test_the_flag_is_attached_as_a_coordinate_on_every_leaf() -> None:
    """A data variable here would die at `bb_sim`'s `combined` dataset on the
    next run through the pipeline, and cannot be selected alongside an IM.
    The root deliberately has no station dimension, so it stays untouched.
    """
    dataset = _waveform_dataset(supergrid_depth=[0.0, 5750.0, np.nan])
    supergrid = im_calc.supergrid_coordinates(dataset)
    dtree = xr.DataTree.from_dict(
        {
            "PGA": xr.Dataset(
                {"rotd50": (("station",), np.ones(3))},
                coords={"station": dataset.station},
            ),
            "pSA": xr.Dataset(
                {"rotd50": (("station",), np.ones(3))},
                coords={"station": dataset.station},
            ),
        }
    )

    parameterised = im_calc.add_station_parameters(dtree, supergrid)

    for group in ("PGA", "pSA"):
        leaf = parameterised[group].dataset
        for name in im_calc.SUPERGRID_COORDINATES:
            assert name in leaf.coords
            assert name not in leaf.data_vars
    assert "supergrid_depth" not in parameterised.dataset.coords


def test_the_recommended_threshold_travels_with_the_data() -> None:
    """The coordinate's `description` is the only documentation a downstream
    user ever sees, so it has to carry the threshold and say what the trace
    is, or every tool invents its own cut.
    """
    for name in im_calc.SUPERGRID_COORDINATES:
        assert name in im_calc.COORDINATE_METADATA
        description = im_calc.COORDINATE_METADATA[name]["description"]
        assert "> 0" in description
        assert "NaN" in description

    dataset = xr.Dataset(
        {"rotd50": (("station",), np.ones(2))},
        coords={
            "station": ["ST00", "ST01"],
            "supergrid_depth": ("station", np.array([0.0, 1.0], dtype=np.float32)),
        },
    )
    annotated = im_calc.add_units(xr.DataTree.from_dict({"PGA": dataset}))

    attrs = annotated["PGA"].dataset["supergrid_depth"].attrs
    assert attrs["units"] == "m"
    assert "> 0" in attrs["description"]
