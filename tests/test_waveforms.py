import numpy as np
import xarray as xr

from IM import ims
from workflow.waveforms import Component


def test_im_package_reads_the_stored_component_order() -> None:
    """The IM package reads 000, 090 and ver by position.

    Waveform files store `Component` in that same order, so the
    per-component IMs come out under the right labels.
    """
    peaks = {Component.NORTH: 1.0, Component.EAST: 2.0, Component.UP: 3.0}
    waveform = np.zeros((3, 1, 10), dtype=np.float32)
    for i, component in enumerate(Component):
        waveform[i, 0, 5] = peaks[component]
    broadband = xr.DataArray(
        waveform,
        dims=("component", "station", "time"),
        coords={"component": list(Component), "station": ["S1"]},
    )

    pga = ims.peak_ground_acceleration(broadband)

    for component, peak in peaks.items():
        assert float(pga[component].item()) == peak
