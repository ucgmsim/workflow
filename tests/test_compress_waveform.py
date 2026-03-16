from pathlib import Path

import numpy as np
import xarray as xr

from workflow.scripts.compress_waveform import (
    _SCALE_LIMIT,
    compress_waveform,
    decompress_waveform,
)

N_COMPONENTS = 3
N_STATIONS = 20
N_TIME = 20_000
N_HARMONICS = 5
DT = 0.005


def _make_broadband_dataset(rng: np.random.Generator) -> xr.Dataset:
    """Create a synthetic broadband waveform dataset with correlated components.

    The waveform mimics real seismic traces where x, y, z components
    share a dominant signal with small per-component perturbations.
    This tests that the component-delta encoding exploits
    cross-component correlation effectively.
    """
    time = np.arange(N_TIME) * DT

    # Base signal shared by all components (dominant seismic motion).
    base_freqs = rng.uniform(0.5, 5.0, size=(1, N_STATIONS, N_HARMONICS))
    base_phases = rng.uniform(0, 2 * np.pi, size=(1, N_STATIONS, N_HARMONICS))
    base_amps = rng.uniform(0.001, 0.05, size=(1, N_STATIONS, N_HARMONICS))

    base = np.zeros((1, N_STATIONS, N_TIME), dtype=np.float32)
    for k in range(N_HARMONICS):
        base += (
            base_amps[..., k : k + 1]
            * np.sin(
                2 * np.pi * base_freqs[..., k : k + 1]
                * time[np.newaxis, np.newaxis, :]
                + base_phases[..., k : k + 1]
            )
        ).astype(np.float32)

    # Components share the base signal with ~10% perturbation.
    waveform = np.tile(base, (N_COMPONENTS, 1, 1))
    for c in range(N_COMPONENTS):
        pert_freqs = rng.uniform(0.5, 5.0, size=(1, N_STATIONS, 2))
        pert_phases = rng.uniform(0, 2 * np.pi, size=(1, N_STATIONS, 2))
        pert_amps = rng.uniform(0.0001, 0.005, size=(1, N_STATIONS, 2))
        for k in range(2):
            waveform[c : c + 1] += (
                pert_amps[..., k : k + 1]
                * np.sin(
                    2 * np.pi * pert_freqs[..., k : k + 1]
                    * time[np.newaxis, np.newaxis, :]
                    + pert_phases[..., k : k + 1]
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

    # Numerical accuracy: quantisation error is at most 0.5 LSB.
    max_abs = float(np.abs(ds.waveform.values).max())
    scale_factor = max_abs / _SCALE_LIMIT
    max_err = float(np.abs(restored.waveform.values - ds.waveform.values).max())
    assert max_err <= scale_factor, (
        f"Roundtrip error {max_err} exceeds one LSB ({scale_factor})"
    )


def test_compression_is_efficient(tmp_path: Path) -> None:
    """Verify that the compressed file is meaningfully smaller than the original.

    Smooth, correlated waveforms (like real seismic data) should
    compress well with int32-scaling + component-delta-encoding + FLAC.
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

    # With smooth waveforms, expect at least 1.5× compression.
    assert ratio > 1.5, (
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


def test_no_drift(tmp_path: Path) -> None:
    """Verify that the decompressed waveform has no accumulated drift.

    When the waveform is quiet (near zero) before and after a burst of
    seismic activity, the decompressed tail must return to near zero
    rather than drifting away—a symptom of cumulative errors in the
    reconstruction path.
    """
    rng = np.random.default_rng(54321)

    # Build a seismic-like waveform: quiet → active → quiet.
    time = np.arange(N_TIME) * DT
    envelope = np.exp(-0.5 * ((time - 50.0) / 10.0) ** 2)

    waveform = np.zeros((N_COMPONENTS, N_STATIONS, N_TIME), dtype=np.float32)
    for c in range(N_COMPONENTS):
        for s in range(N_STATIONS):
            for freq in [0.5, 1.0, 3.0, 7.0]:
                amp = rng.uniform(0.005, 0.03)
                phase = rng.uniform(0, 2 * np.pi)
                waveform[c, s, :] += (
                    amp * envelope * np.sin(2 * np.pi * freq * time + phase)
                ).astype(np.float32)
    waveform += (rng.standard_normal(waveform.shape) * 1e-5).astype(np.float32)

    ds = xr.Dataset(
        {"waveform": (["component", "station", "time"], waveform)},
        coords={
            "component": ("component", ["x", "y", "z"]),
            "station": ("station", [f"STA{i:03d}" for i in range(N_STATIONS)]),
            "time": ("time", time),
        },
    )

    input_ffp = tmp_path / "broadband.nc"
    compressed_ffp = tmp_path / "broadband_compressed.h5"

    ds.to_netcdf(input_ffp, engine="h5netcdf")
    compress_waveform(input_ffp, compressed_ffp)
    restored = decompress_waveform(compressed_ffp)

    orig = ds.waveform.values
    rest = restored.waveform.values

    # The last 20 % of the trace is the quiet tail; its mean should be
    # essentially zero (matching the original) rather than drifted.
    tail_start = int(N_TIME * 0.8)
    tail_mean_orig = np.mean(orig[:, :, tail_start:], axis=-1)
    tail_mean_rest = np.mean(rest[:, :, tail_start:], axis=-1)
    drift = tail_mean_rest - tail_mean_orig

    max_abs = float(np.abs(orig).max())
    scale_factor = max_abs / _SCALE_LIMIT

    # Drift must stay within one quantisation step.
    assert np.abs(drift).max() <= scale_factor, (
        f"Drift {np.abs(drift).max():.3e} exceeds one LSB ({scale_factor:.3e})"
    )
