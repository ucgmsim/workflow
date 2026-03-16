from pathlib import Path

import numpy as np
import xarray as xr

from workflow.scripts.compress_waveform import (
    compress_waveform,
    decompress_waveform,
)

# Constants for test data generation
N_COMPONENTS, N_STATIONS, N_TIME = 3, 5, 1000
DT = 0.05


def _make_test_dataset() -> xr.Dataset:
    """Create a simple synthetic waveform dataset for testing."""
    time = np.arange(N_TIME) * DT
    waveform = (
        np.sin(time * 2 * np.pi * 1.0)
        + np.random.default_rng(42).standard_normal((N_COMPONENTS, N_STATIONS, N_TIME))
        * 0.1
    )

    return xr.Dataset(
        {"waveform": (["component", "station", "time"], waveform.astype(np.float32))},
        coords={
            "component": ["x", "y", "z"],
            "station": [f"STA{i:02d}" for i in range(N_STATIONS)],
            "time": time,
            "lat": ("station", np.linspace(-45, -43, N_STATIONS)),
        },
        attrs={"units": "m/s", "source": "test_gen"},
    )


def test_waveform_roundtrip_integrity(tmp_path: Path) -> None:
    """Verify waveform values and metadata survive the compression roundtrip."""
    with _make_test_dataset() as ds:
        input_path = tmp_path / "input.h5"
        original_attrs = ds.attrs
        ds.to_netcdf(input_path, engine="h5netcdf")
    output_path = tmp_path / "output.h5"

    compress_waveform(input_path, output_path)
    restored = decompress_waveform(output_path)

    restored_subset = {k: v for k, v in restored.attrs.items() if k in original_attrs}
    assert restored_subset == original_attrs, (
        "Restored attributes do not match original attributes."
    )

    for coord in ds.coords:
        np.testing.assert_array_equal(restored[coord].values, ds[coord].values)

    xr.testing.assert_allclose(restored, ds, atol=5e-4)


def test_compression_efficiency(tmp_path: Path) -> None:
    """Verify the compressed file is actually smaller than the raw values."""
    input_path = tmp_path / "input.h5"
    output_path = tmp_path / "output.h5"

    with _make_test_dataset() as ds:
        ds.to_netcdf(input_path, engine="h5netcdf")

    compress_waveform(input_path, output_path)

    raw_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size

    assert compressed_size < raw_size, (
        f"Compression failed to reduce size: {compressed_size} >= {raw_size}"
    )
