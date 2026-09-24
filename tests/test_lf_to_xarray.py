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
    """Write a synthetic SW4 HDF5 station recording."""
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


def convert(sw4_ffp: Path) -> xr.Dataset:
    """Run `lf-to-xarray` on an SW4 station file and read the result back."""
    output = sw4_ffp.with_suffix(".nc")
    lf_to_xarray.convert_lf_to_xarray_dataset(
        sw4_ffp, output, format=lf_to_xarray.Format.SW4
    )
    with xr.open_dataset(output, mask_and_scale=False) as dataset:
        return dataset.load()


def test_supergrid_penetration_arrives_as_float32_coordinates(tmp_path: Path) -> None:
    """The flag must be a *coordinate*, and it must be floating point."""
    ffp = write_sw4_station_file(
        tmp_path / "stations.h5",
        {
            "AAAA": {"SGDEPTH": 0.0, "SGDEPTHGP": 0.0},
            "BBBB": {"SGDEPTH": 5750.0, "SGDEPTHGP": 14.375},
        },
    )

    dset = convert(ffp)

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
    """A file written before SW4 reported the supergrid must not raise."""
    ffp = write_sw4_station_file(
        tmp_path / "old.h5", {"AAAA": None, "BBBB": None, "CCCC": None}
    )

    dset = convert(ffp)

    assert dset.sizes["station"] == 3
    for name in ("supergrid_depth", "supergrid_depth_gp"):
        assert name in dset.coords
        assert dset.coords[name].dtype == np.float32
        assert np.isnan(dset.coords[name].values).all()
    assert "supergrid_width" not in dset.attrs
    assert "supergrid_width_gp" not in dset.attrs


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

    depth = convert(ffp).sortby("station")["supergrid_depth"]

    assert depth.values[0] == 0.0
    assert np.isnan(depth.values[1])
    assert depth.values[2] == 1200.0


def test_one_dataset_without_the_other_is_a_corrupt_file(tmp_path: Path) -> None:
    """`SGDEPTHGP` missing while `SGDEPTH` is present is corruption, not age."""
    ffp = write_sw4_station_file(tmp_path / "corrupt.h5", {"AAAA": {"SGDEPTH": 900.0}})

    with pytest.raises(KeyError):
        convert(ffp)


def test_the_sponge_width_is_lifted_into_the_dataset_attributes(
    tmp_path: Path,
) -> None:
    """SW4's `SGWIDTH`/`SGWIDTHGP` are written under the names `im-calc` uses."""
    ffp = write_sw4_station_file(
        tmp_path / "width.h5",
        {"AAAA": {"SGDEPTH": 0.0, "SGDEPTHGP": 0.0}},
        widths={"SGWIDTH": 12000.0, "SGWIDTHGP": 30.0},
    )

    dset = convert(ffp)

    assert dset.attrs["supergrid_width"] == pytest.approx(12000.0)
    assert dset.attrs["supergrid_width_gp"] == pytest.approx(30.0)
    assert "SGWIDTH" not in dset.attrs
    assert "SGWIDTHGP" not in dset.attrs
    # The pre-existing attributes must survive alongside them.
    assert dset.attrs["nt"] == 8
    assert dset.attrs["dt"] == pytest.approx(0.05)
