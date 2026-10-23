import datetime

import numpy as np
import xarray as xr

from IM import ims
from workflow.waveforms import Component, time_axis


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


def test_time_axis_without_an_event_time_is_seconds() -> None:
    seconds = np.arange(4) * 0.005
    np.testing.assert_array_equal(time_axis(seconds, None), seconds)


def test_time_axis_stamps_seconds_onto_the_event_time_in_utc() -> None:
    nzst = datetime.timezone(datetime.timedelta(hours=12))
    event_time = datetime.datetime(2010, 9, 4, 4, 35, 46, tzinfo=nzst)

    stamps = time_axis(np.arange(3) * 0.005, event_time)

    np.testing.assert_array_equal(
        stamps,
        np.array(
            [
                "2010-09-03T16:35:46.000",
                "2010-09-03T16:35:46.005",
                "2010-09-03T16:35:46.010",
            ],
            dtype="datetime64[ns]",
        ),
    )
