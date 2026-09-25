"""Low-frequency output merger.

Description
-----------
Merges low-frequency outputs into one xarray dataset.

Inputs
------
1. A low-frequency output directory.

Outputs
-------
1. A combined LF waveform output containing ground acceleration data for each station.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `lf-to-xarray` command which is installed after running `pip install ucgmsim-workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`lf-to-xarray [OPTIONS] OUTBIN_DIRECTORY OUTPUT_FFP`

For More Help
-------------
See the output of `lf-to-xarray --help`.
"""

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import dask.array as da
import h5py
import numpy as np
import typer
import xarray as xr

from qcore import cli, timeseries
from workflow import log_utils, sw4
from workflow.waveforms import Component

app = typer.Typer()

CMS = 100.0
"""Unit to convert m/s to cm/s"""

SW4_COMPONENTS = {Component.NORTH: "NS", Component.EAST: "EW", Component.UP: "UP"}
"""SW4's station-recording dataset for each workflow component."""

TARGET_CHUNK_BYTES = 128 * 2**20
"""Target size of a dask chunk (all components for a batch of stations)."""


def _read_station_batch(
    stations: xr.DataArray,
    sw4_ffp: Path,
    time: xr.DataArray,
    component: xr.DataArray,
) -> xr.DataArray:
    """Read velocity waveforms (m/s) for a batch of stations."""
    # SW4 labels these as displacement, but for SRF sources it is given the slip
    # *rate*, so the output is really velocity (SW4 User Guide, Section 11.2.2).
    recordings = np.empty((len(component), len(stations), len(time)), dtype=np.float32)
    with h5py.File(sw4_ffp, "r") as handle:
        for i, station_name in enumerate(stations):
            group = handle[station_name.item()]
            if "NS" not in group:
                raise RuntimeError(
                    f"Station {station_name.item()} has no EW/NS/UP datasets."
                    " The SW4 rechdf5 command must output geographic (NSEW)"
                    " displacement-mode components."
                )
            for j, label in enumerate(component.values):
                recordings[j, i] = group[SW4_COMPONENTS[label]][:]
    # Doing in-place multiplication here saves one batch copy
    recordings *= CMS
    return xr.DataArray(
        recordings,
        dims=["component", "station", "time"],
        coords={"time": time, "component": component, "station": stations.values},
    )


def _read_station_metadata(sw4_ffp: Path) -> xr.Dataset:
    """Build the dataset's coordinates and attributes from an SW4 station file."""
    global_npts = None
    stations = []
    latitudes = []
    longitudes = []
    supergrid_depths: dict[str, list[float]] = {
        name: [] for name in sw4.SUPERGRID_DEPTH_COORDINATES.values()
    }

    with h5py.File(sw4_ffp, "r") as handle:
        dt = np.float32(handle["DELTA"][:].squeeze())

        attrs: dict[str, np.float32 | float] = {"dt": dt}
        for sw4_name, attribute_name in sw4.SUPERGRID_WIDTH_ATTRIBUTES.items():
            if sw4_name in handle:
                attrs[attribute_name] = float(handle[sw4_name][:].squeeze())
        for station_name, group in handle.items():
            if not isinstance(group, h5py.Group) or "NPTS" not in group:
                continue
            npts = int(group["NPTS"][:].squeeze())
            if global_npts is not None and npts != global_npts:
                raise RuntimeError(
                    f"SW4 output is corrupted: {npts=} but {global_npts=}"
                )
            global_npts = npts
            stations.append(station_name)

            latitude, longitude, _ = group["STLA,STLO,STDP"][:]
            latitudes.append(latitude)
            longitudes.append(longitude)

            # SW4 *may* record the SGDEPTH (master will not, we have a fork that
            # does). So we conservatively check for this.
            # SGDEPTH without SGDEPTHGP is a corrupt file, so let it raise.
            has_depth = "SGDEPTH" in group
            for sw4_name, coordinate_name in sw4.SUPERGRID_DEPTH_COORDINATES.items():
                supergrid_depths[coordinate_name].append(
                    float(group[sw4_name][:].squeeze()) if has_depth else np.nan
                )

    if global_npts is None:
        raise RuntimeError(
            "No valid station recordings found in file. Are you sure this is an SW4 station file? Use `h5ls` to check the file structure."
        )

    time = np.arange(global_npts) * dt
    return xr.Dataset(
        {
            "latitude": ("station", latitudes),
            "longitude": ("station", longitudes),
        },
        coords={
            "station": stations,
            "component": list(Component),
            "time": time,
            **{
                name: ("station", np.array(depths, dtype=np.float32))
                for name, depths in supergrid_depths.items()
            },
        },
        attrs=attrs | {"nt": global_npts},
    )


def _template_waveform(dset: xr.Dataset, batch_size: int) -> xr.DataArray:
    ncomponent = len(dset.coords["component"])
    nstation = len(dset.coords["station"])
    ntime = len(dset.coords["time"])
    return xr.DataArray(
        # Chunks must match the batches `_read_station_batch` returns.
        da.empty(
            (ncomponent, nstation, ntime),
            dtype=np.float32,
            chunks=(ncomponent, batch_size, ntime),
        ),
        dims=["component", "station", "time"],
        # Dimension coordinates only, to match `_read_station_batch`. The
        # others (e.g. `supergrid_depth`) come back when assigned onto `dset`.
        coords={dim: dset.coords[dim] for dim in ("component", "station", "time")},
    )


def _convert_sw4_station_recording(sw4_ffp: Path) -> xr.Dataset:
    """Lazily convert an SW4 station recording to an xarray dataset."""
    dset = _read_station_metadata(sw4_ffp)
    batch_size = max(
        1,
        TARGET_CHUNK_BYTES
        // (
            len(dset.coords["component"])
            * len(dset.coords["time"])
            * np.float32().itemsize
        ),
    )
    # Named "station" to line up with the template, but with no coordinate, as
    # an index coordinate cannot be chunked.
    chunked_stations = xr.DataArray(dset["station"].values, dims=["station"]).chunk(
        {"station": batch_size}
    )
    waveform = xr.map_blocks(
        _read_station_batch,
        chunked_stations,
        kwargs={
            "time": dset["time"],
            "component": dset["component"],
            "sw4_ffp": sw4_ffp,
        },
        template=_template_waveform(dset, batch_size),
    )
    # Copying here is not easily avoidable given that the gradient is a central
    # difference operator so overwriting values corrupts the derivative. Could
    # do some complicated ufunc shenanigans and masks but I have not noticed
    # issues with this portion of the code OOM'ing.
    waveform = waveform.differentiate("time")
    dset["waveform"] = waveform
    dset.attrs["units"] = "cm/s^2"
    # SW4 station recordings begin at simulation time zero.
    dset.attrs["start_sec"] = 0.0

    return dset


class Format(StrEnum):
    """Input low frequency file format."""

    SW4 = auto()
    """SW4 HDF5 station recording."""
    EMOD3D = auto()
    """EMOD3D LFSeis directory."""


@cli.from_docstring(app)
@log_utils.log_call()
def convert_lf_to_xarray_dataset(
    low_frequency_path: Annotated[Path, typer.Argument(exists=True)],
    output_ffp: Annotated[Path, typer.Argument(writable=True, dir_okay=False)],
    format: Format = Format.EMOD3D,
) -> None:
    """Merge low-frequency outputs into an xarray dataset.

    Parameters
    ----------
    low_frequency_path : Path
        Station seismogram outputs.
    output_ffp : Path
        Path to write the xarray dataset
    format : Format, optional
        Format for the low-frequency inputs (EMOD3D or SW4). If format is SW4,
        the low frequency path should be an HDF5 file in the SW4 station format
        (Section 12.9 of the SW4 User Guide). If format is instead EMOD3D, the
        low frequency path should be a directory containing LFSeis files.
        Defaults to EMOD3D.
    """
    match format:
        case Format.EMOD3D if low_frequency_path.is_dir():
            # qcore names the station coordinates lat/lon, and labels the
            # components x (east), y (north) and z (up). Its docstring says x
            # points north, but EMOD3D's y axis is rotated from south by
            # `modelrot`, and qcore's rotation turns that into east/north/up.
            lf_dataset = (
                timeseries.read_lfseis_directory(low_frequency_path)
                .rename({"lat": "latitude", "lon": "longitude"})
                .assign_coords(
                    component=[Component.EAST, Component.NORTH, Component.UP]
                )
                .sel(component=list(Component))
            )
            lf_dataset.to_netcdf(output_ffp, engine="h5netcdf")
        case Format.EMOD3D:
            raise ValueError("EMOD3D format requires directory containing LFSeis files")
        case Format.SW4 if low_frequency_path.is_file():
            lf_dataset = _convert_sw4_station_recording(low_frequency_path)
            lf_dataset.to_netcdf(output_ffp, engine="h5netcdf")
        case Format.SW4:
            raise ValueError("SW4 format requires station recording file.")


if __name__ == "__main__":
    app()
