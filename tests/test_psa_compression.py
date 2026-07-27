from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from workflow import psa_compression


def _sample_rotd180(n_stations: int = 3, n_periods: int = 5) -> xr.DataArray:
    """A smooth, physically-plausible pSA rotd180 curve for testing."""
    rng = np.random.default_rng(0)
    angle = np.deg2rad(np.arange(180))
    base = rng.uniform(0.05, 2.0, size=(n_stations, n_periods, 1))
    # Smooth in theta, like a real rotated-peak curve, strictly positive.
    curve = base * (1.2 + 0.8 * np.cos(2 * angle))
    return xr.DataArray(
        curve,
        dims=("station", "period", "angle"),
        coords={
            "station": [f"stat_{i}" for i in range(n_stations)],
            "period": np.linspace(0.1, 5.0, n_periods),
            "angle": np.arange(180),
        },
    )


def test_roundtrip_within_error_bound() -> None:
    rotd180 = _sample_rotd180()
    encoded = psa_compression.encode_psa_rotd180(rotd180, rel_step=0.02)
    decoded = psa_compression.decode_psa_rotd180(encoded)

    assert encoded.dtype == np.int16
    assert decoded.dims == rotd180.dims
    assert decoded.shape == rotd180.shape

    relative_error = np.abs(decoded - rotd180) / rotd180
    bound = encoded.attrs["max_relative_error_bound"]
    assert float(relative_error.max()) <= bound * 1.01  # small numerical slack


@pytest.mark.parametrize("rel_step", [0.01, 0.02, 0.05])
def test_error_bound_matches_attrs(rel_step: float) -> None:
    rotd180 = _sample_rotd180()
    encoded = psa_compression.encode_psa_rotd180(rotd180, rel_step=rel_step)
    decoded = psa_compression.decode_psa_rotd180(encoded)

    expected_bound = np.expm1(np.log1p(rel_step) / 2)
    assert encoded.attrs["max_relative_error_bound"] == pytest.approx(expected_bound)

    relative_error = np.abs(decoded - rotd180) / rotd180
    assert float(relative_error.max()) <= expected_bound * 1.01


def test_encoding_is_lazy_for_dask_input() -> None:
    rotd180 = _sample_rotd180(n_stations=6)
    lazy = rotd180.chunk({"station": 2})
    assert lazy.chunks is not None

    encoded = psa_compression.encode_psa_rotd180(lazy)
    assert encoded.chunks is not None

    decoded = psa_compression.decode_psa_rotd180(encoded)
    assert decoded.chunks is not None

    computed = decoded.compute()
    relative_error = np.abs(computed - rotd180) / rotd180
    bound = encoded.attrs["max_relative_error_bound"]
    assert float(relative_error.max()) <= bound * 1.01


def test_lazy_encoding_matches_eager() -> None:
    rotd180 = _sample_rotd180(n_stations=6)
    lazy_input = rotd180.copy()
    lazy_input.data = da.from_array(rotd180.data, chunks=(2, -1, -1))

    eager_encoded = psa_compression.encode_psa_rotd180(rotd180)
    lazy_encoded = psa_compression.encode_psa_rotd180(lazy_input)

    np.testing.assert_array_equal(eager_encoded.values, lazy_encoded.compute().values)


def test_default_complevel_is_not_max() -> None:
    """Regression test: gzip level 9 measured ~100x slower than level 4 for a
    ~2% smaller file on realistic delta-encoded data (writes go through
    h5py's global lock, so this cost is not parallelised across dask
    workers). The default must stay off level 9, or a real run silently
    regresses to the multi-hour write this was fixed to avoid.
    """
    encoded = psa_compression.encode_psa_rotd180(_sample_rotd180())
    assert psa_compression.rotd180_netcdf_encoding(encoded)["complevel"] <= 6


def test_netcdf_encoding_with_independently_chunked_coords() -> None:
    """Regression test: `latitude`/`longitude` ride along as station-dimension
    coordinates on the real broadband pipeline, and `Dataset.chunk({"station":
    "auto"})` sizes each variable's "auto" chunks independently -- a small
    float64 coordinate gets a different station chunk count than the much
    larger `waveform`/`rotd180` array. `rotd180_netcdf_encoding` must not call
    `.chunksizes` (which cross-validates every attached coordinate and raises
    "inconsistent chunks along dimension station" the moment they differ);
    it must only ever look at the data variable's own chunking.
    """
    # Deliberately different chunkings along the same "station" dimension --
    # exactly what `Dataset.chunk({"station": "auto"})` produces in the real
    # pipeline, where "auto" is sized per variable (a big float32 waveform
    # splits into many station chunks; a tiny float64 lat/lon coordinate
    # easily fits in one).
    n_stations = 400
    waveform_station_chunks = (150, 150, 100)
    latitude = xr.DataArray(
        np.arange(n_stations, dtype=np.float64), dims="station"
    ).chunk({"station": n_stations})
    longitude = xr.DataArray(
        -np.arange(n_stations, dtype=np.float64), dims="station"
    ).chunk({"station": n_stations})
    assert latitude.variable.chunks != (waveform_station_chunks,)  # the actual mismatch

    # A rotd180-shaped array whose data is chunked like `waveform` above, but
    # whose `latitude`/`longitude` coordinates carry the mismatched chunking
    # -- exactly what `apply_ufunc` hands back in the real pipeline, since it
    # never rechunks coordinates it just carries along.
    rotd180 = xr.DataArray(
        da.zeros((n_stations, 3, 180), chunks=(waveform_station_chunks, -1, -1)),
        dims=("station", "period", "angle"),
        coords={"latitude": latitude, "longitude": longitude, "angle": np.arange(180)},
    )
    with pytest.raises(ValueError, match="inconsistent chunks"):
        _ = rotd180.chunksizes

    encoded = psa_compression.encode_psa_rotd180(rotd180)
    encoding = psa_compression.rotd180_netcdf_encoding(encoded)
    assert encoding["chunksizes"][0] == waveform_station_chunks[0]


def test_netcdf_roundtrip_with_blosc_compression(tmp_path: Path) -> None:
    """The encoded variable must actually apply the Blosc filter and shrink
    relative to storing the raw floats, and decode back within the bound."""
    rotd180 = _sample_rotd180(n_stations=20, n_periods=10)
    encoded = psa_compression.encode_psa_rotd180(rotd180)

    dataset = xr.Dataset({"rotd180": encoded})
    dtree = xr.DataTree.from_dict({"pSA": dataset}, nested=True)

    compressed_path = tmp_path / "compressed.nc"
    dtree.to_netcdf(
        compressed_path,
        engine="h5netcdf",
        encoding={"/pSA": {"rotd180": psa_compression.rotd180_netcdf_encoding(encoded)}},
    )

    raw_path = tmp_path / "raw.nc"
    xr.DataTree.from_dict({"pSA": xr.Dataset({"rotd180": rotd180})}, nested=True).to_netcdf(
        raw_path, engine="h5netcdf"
    )

    assert compressed_path.stat().st_size < raw_path.stat().st_size

    reopened = xr.open_datatree(compressed_path, engine="h5netcdf")
    reopened_encoded = reopened["pSA"]["rotd180"]
    assert reopened_encoded.attrs["ln_step"] == encoded.attrs["ln_step"]

    decoded = psa_compression.decode_psa_rotd180(reopened_encoded.load())
    relative_error = np.abs(decoded - rotd180) / rotd180
    bound = encoded.attrs["max_relative_error_bound"]
    assert float(relative_error.max()) <= bound * 1.01
