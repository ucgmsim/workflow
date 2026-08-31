"""Tests for reading SW4 station recordings, and in particular for the
supergrid (absorbing layer) penetration SW4 reports per station.

The station file fixture below is the first synthetic SW4 recording in the
suite; it is deliberately written against the layout documented in Section
12.9 of the SW4 User Guide (a root `DELTA`, one group per station holding
`NPTS`, `STLA,STLO,STDP` and the three geographic components) so that it is
reusable for any other SW4-read test.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from workflow.scripts import lf_to_xarray


def write_sw4_station_file(
    path: Path,
    stations: dict[str, dict[str, float] | None],
    npts: int = 8,
    dt: float = 0.05,
    widths: dict[str, float] | None = None,
) -> Path:
    """Write a synthetic SW4 HDF5 station recording.

    Parameters
    ----------
    path : Path
        Where to write the file.
    stations : dict
        Map from station name to either `None` (an old-style station, with no
        supergrid datasets at all) or a dict which may hold `SGDEPTH` and
        `SGDEPTHGP`. A dict holding only one of the two produces a
        deliberately corrupt station.
    npts : int
        Number of samples per component.
    dt : float
        Sample spacing, written to the root `DELTA`.
    widths : dict, optional
        File-level scalars (`SGWIDTH`, `SGWIDTHGP`) written beside `DELTA`.

    Returns
    -------
    Path
        The path written, for convenience.
    """
    with h5py.File(path, "w") as handle:
        handle.create_dataset("DELTA", data=np.array([dt]))
        for name, value in (widths or {}).items():
            handle.create_dataset(name, data=np.array([value]))
        for index, (station, supergrid) in enumerate(stations.items()):
            group = handle.create_group(station)
            group.create_dataset("NPTS", data=np.array([npts]))
            group.create_dataset(
                "STLA,STLO,STDP",
                data=np.array([-43.5 + index, 172.6 + index, 0.0]),
            )
            for component in ("EW", "NS", "UP"):
                group.create_dataset(
                    component, data=np.arange(npts, dtype=np.float32) + index
                )
            for key, value in (supergrid or {}).items():
                group.create_dataset(key, data=np.array([value]))
    return path


def test_supergrid_penetration_arrives_as_float32_coordinates(tmp_path: Path) -> None:
    """The flag must be a *coordinate*, and it must be floating point.

    Both halves are load bearing. A station-dimension coordinate rides
    through `bb-sim` and `im-calc` untouched, whereas a data variable is
    silently dropped by `bb_sim._process_bb_chunk`, so a data variable here
    would mean the flag never reaches the intensity measures. And the
    downstream consumer opens IM files with `mask_and_scale=False`, so an
    integer with a `_FillValue` would read back raw and become a plausible
    penetration depth; only a real float NaN survives that.
    """
    ffp = write_sw4_station_file(
        tmp_path / "stations.h5",
        {
            "AAAA": {"SGDEPTH": 0.0, "SGDEPTHGP": 0.0},
            "BBBB": {"SGDEPTH": 5750.0, "SGDEPTHGP": 14.375},
        },
    )

    dset = lf_to_xarray.read_station_metadata(ffp)

    for name in ("supergrid_depth", "supergrid_depth_gp"):
        assert name in dset.coords
        assert name not in dset.data_vars
        assert dset.coords[name].dims == ("station",)
        assert dset.coords[name].dtype == np.float32

    ordered = dset.sortby("station")
    np.testing.assert_array_equal(ordered["supergrid_depth"].values, [0.0, 5750.0])
    np.testing.assert_allclose(
        ordered["supergrid_depth_gp"].values, [0.0, 14.375], rtol=1e-6
    )


def test_an_old_station_file_converts_with_an_all_nan_flag(tmp_path: Path) -> None:
    """A file written before SW4 reported the supergrid must not raise.

    This is the common case for every recording made so far, so it has to be
    a no-op rather than an error. The value must be NaN and never `0.0`: `0.0`
    is the positive claim "this station was checked and is in the interior",
    which nobody checked here.
    """
    ffp = write_sw4_station_file(
        tmp_path / "old.h5", {"AAAA": None, "BBBB": None, "CCCC": None}
    )

    dset = lf_to_xarray.read_station_metadata(ffp)

    assert dset.sizes["station"] == 3
    for name in ("supergrid_depth", "supergrid_depth_gp"):
        assert name in dset.coords
        assert dset.coords[name].dtype == np.float32
        assert np.isnan(dset.coords[name].values).all()
    assert "SGWIDTH" not in dset.attrs
    assert "SGWIDTHGP" not in dset.attrs


def test_stations_missing_the_flag_are_nan_not_zero(tmp_path: Path) -> None:
    """Mixed groups: only the stations SW4 reported on get a number."""
    ffp = write_sw4_station_file(
        tmp_path / "mixed.h5",
        {
            "AAAA": {"SGDEPTH": 0.0, "SGDEPTHGP": 0.0},
            "BBBB": None,
            "CCCC": {"SGDEPTH": 1200.0, "SGDEPTHGP": 3.0},
        },
    )

    depth = lf_to_xarray.read_station_metadata(ffp).sortby("station")["supergrid_depth"]

    assert depth.values[0] == 0.0
    assert np.isnan(depth.values[1])
    assert depth.values[2] == 1200.0


def test_one_dataset_without_the_other_is_a_corrupt_file(tmp_path: Path) -> None:
    """`SGDEPTHGP` missing while `SGDEPTH` is present is corruption, not age.

    The back-compatibility guard is deliberately on `SGDEPTH` alone, so this
    raises rather than quietly reporting a metre depth with no grid-point
    depth beside it.
    """
    ffp = write_sw4_station_file(tmp_path / "corrupt.h5", {"AAAA": {"SGDEPTH": 900.0}})

    with pytest.raises(KeyError):
        lf_to_xarray.read_station_metadata(ffp)


def test_the_sponge_width_is_lifted_into_the_dataset_attributes(
    tmp_path: Path,
) -> None:
    """`SGWIDTH`/`SGWIDTHGP` make the file self-describing.

    They are what turns the penetration into a severity fraction downstream,
    and taking them from the file rather than from the realisation
    configuration is the point: the configuration can be edited after the run.
    """
    ffp = write_sw4_station_file(
        tmp_path / "width.h5",
        {"AAAA": {"SGDEPTH": 0.0, "SGDEPTHGP": 0.0}},
        widths={"SGWIDTH": 12000.0, "SGWIDTHGP": 30.0},
    )

    dset = lf_to_xarray.read_station_metadata(ffp)

    assert dset.attrs["SGWIDTH"] == pytest.approx(12000.0)
    assert dset.attrs["SGWIDTHGP"] == pytest.approx(30.0)
    # The pre-existing attributes must survive alongside them.
    assert dset.attrs["nt"] == 8
    assert dset.attrs["dt"] == pytest.approx(0.05)


def test_the_flag_survives_a_netcdf_round_trip(tmp_path: Path) -> None:
    """As a coordinate, and as NaN -- checked the way a consumer reads it.

    `mask_and_scale=False` is what `eqvis`'s `open_ims` passes, so this is the
    exact read path the flag has to survive: no fill-value decoding, NaN read
    straight off disk.
    """
    ffp = write_sw4_station_file(
        tmp_path / "roundtrip.h5",
        {
            "AAAA": {"SGDEPTH": 0.0, "SGDEPTHGP": 0.0},
            "BBBB": None,
            "CCCC": {"SGDEPTH": 5750.0, "SGDEPTHGP": 14.0},
        },
        widths={"SGWIDTH": 12000.0, "SGWIDTHGP": 30.0},
    )
    dset = lf_to_xarray.convert_sw4_station_recording(ffp)
    output = tmp_path / "lf.nc"
    # The same engine `lf-to-xarray` itself writes with.
    dset.to_netcdf(output, engine="h5netcdf")

    with xr.open_dataset(output, mask_and_scale=False) as reopened:
        assert "supergrid_depth" in reopened.coords
        assert "supergrid_depth" not in reopened.data_vars
        assert reopened["supergrid_depth"].dtype == np.float32
        depth = reopened.sortby("station")["supergrid_depth"].values
        assert depth[0] == 0.0
        assert np.isnan(depth[1])
        assert depth[2] == 5750.0
        assert reopened.attrs["SGWIDTH"] == pytest.approx(12000.0)
