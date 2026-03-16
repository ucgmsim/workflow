from pathlib import Path

import numpy as np
import xarray as xr

from workflow.scripts.compress_waveform import compress_waveform, decompress_waveform

N_COMPONENTS = 3
N_STATIONS = 20
N_TIME = 20_000
DT = 0.005


def _make_broadband_dataset(rng: np.random.Generator) -> xr.Dataset:
    """Create a synthetic broadband waveform dataset resembling bb_sim output.

    The waveform mimics real seismic traces by combining a few
    sinusoidal components with different frequencies and adding a
    small amount of noise.  Real broadband waveforms have strong
    temporal autocorrelation, so this synthetic data is a reasonable
    stand-in for testing compression behaviour.
    """
    time = np.arange(N_TIME) * DT

    # Build smooth, correlated waveforms (sine sweep + harmonics)
    freqs = rng.uniform(0.5, 5.0, size=(N_COMPONENTS, N_STATIONS, 5))
    phases = rng.uniform(0, 2 * np.pi, size=(N_COMPONENTS, N_STATIONS, 5))
    amplitudes = rng.uniform(0.001, 0.05, size=(N_COMPONENTS, N_STATIONS, 5))

    waveform = np.zeros((N_COMPONENTS, N_STATIONS, N_TIME), dtype=np.float32)
    for k in range(5):
        waveform += (
            amplitudes[..., k : k + 1]
            * np.sin(
                2 * np.pi * freqs[..., k : k + 1] * time[np.newaxis, np.newaxis, :]
                + phases[..., k : k + 1]
            )
        ).astype(np.float32)

    # Small additive noise
    waveform += (rng.standard_normal(waveform.shape) * 1e-4).astype(np.float32)

    return xr.Dataset(
        {"waveform": (["component", "station", "time"], waveform)},
        coords={
            "component": ("component", ["x", "y", "z"]),
            "station": ("station", [f"STA{i:03d}" for i in range(N_STATIONS)]),
            "time": ("time", time),
            "latitude": ("station", -43.5 + np.arange(N_STATIONS) * 0.1),
            "longitude": ("station", 172.5 + np.arange(N_STATIONS) * 0.1),
        },
        attrs={"units": "g"},
    )


def test_compress_decompress_roundtrip(tmp_path: Path) -> None:
    """Verify that compress → decompress preserves the dataset."""
    rng = np.random.default_rng(12345)
    ds = _make_broadband_dataset(rng)

    input_ffp = tmp_path / "broadband.nc"
    compressed_ffp = tmp_path / "broadband_compressed.h5"

    ds.to_netcdf(input_ffp, engine="h5netcdf")
    compress_waveform(input_ffp, compressed_ffp)

    restored = decompress_waveform(compressed_ffp)

    # Coordinates and attributes are preserved.
    assert set(restored.coords) == set(ds.coords)
    assert restored.attrs == ds.attrs
    assert list(restored.waveform.dims) == list(ds.waveform.dims)
    assert restored.waveform.dtype == ds.waveform.dtype
    assert restored.waveform.shape == ds.waveform.shape

    # Numerical accuracy: int16 quantisation error is at most 0.5 LSB.
    max_abs = float(np.abs(ds.waveform.values).max())
    scale_factor = max_abs / np.iinfo(np.int16).max
    max_err = float(np.abs(restored.waveform.values - ds.waveform.values).max())
    assert max_err <= scale_factor, (
        f"Roundtrip error {max_err} exceeds one LSB ({scale_factor})"
    )


def test_compression_is_efficient(tmp_path: Path) -> None:
    """Verify that the compressed file is meaningfully smaller than the original.

    Smooth, correlated waveforms (like real seismic data) should
    compress well with int16-scaling + delta-encoding + FLAC.
    """
    rng = np.random.default_rng(12345)
    ds = _make_broadband_dataset(rng)

    input_ffp = tmp_path / "broadband.nc"
    compressed_ffp = tmp_path / "broadband_compressed.h5"

    ds.to_netcdf(input_ffp, engine="h5netcdf")
    compress_waveform(input_ffp, compressed_ffp)

    original_bytes = input_ffp.stat().st_size
    compressed_bytes = compressed_ffp.stat().st_size
    ratio = original_bytes / compressed_bytes

    # With smooth waveforms, expect at least 3× compression.
    assert ratio > 3.0, (
        f"Compression ratio {ratio:.2f}x is too low "
        f"(original={original_bytes}, compressed={compressed_bytes})"
    )


def test_metadata_preserved(tmp_path: Path) -> None:
    """Verify that all coordinate variables round-trip exactly."""
    rng = np.random.default_rng(12345)
    ds = _make_broadband_dataset(rng)

    input_ffp = tmp_path / "broadband.nc"
    compressed_ffp = tmp_path / "broadband_compressed.h5"

    ds.to_netcdf(input_ffp, engine="h5netcdf")
    compress_waveform(input_ffp, compressed_ffp)
    restored = decompress_waveform(compressed_ffp)

    for coord_name in ds.coords:
        np.testing.assert_array_equal(
            restored.coords[coord_name].values,
            ds.coords[coord_name].values,
            err_msg=f"Coordinate {coord_name!r} differs after roundtrip",
        )
